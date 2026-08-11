"""Measured time-to-first-token via llama-server, plus the prompt-cache demo.

Everywhere else in the harness TTFT is derived (prompt_tokens / prefill throughput). Here
it's measured: llama-server's /completion returns a `timings` object with prompt_ms
(wall-clock prompt processing) and prompt_n (tokens actually processed).

Also shows the biggest honest TTFT lever for agentic/RAG serving: prefix caching. The
request field `cache_prompt` (default true) re-uses the KV cache when a request shares a
prefix with the previous one, so only the differing suffix is prefilled. The server flag
`--cache-reuse N` (default 0) also re-uses non-exact-prefix chunks via KV shifting. Two
requests sharing a long system/context prefix:

  turn 1 (cold): full prefix prefilled -> large prompt_ms, prompt_n ~= full prompt
  turn 2 (warm): only the new question prefilled -> prompt_ms collapses, prompt_n tiny

Flags/fields verified against llama.cpp master source (2026-07-31): /health returns 503 while
loading and 200 {"status":"ok"} when ready; /completion accepts cache_prompt (default true)
and n_predict; the response `timings` object carries prompt_n / prompt_ms / predicted_n /
predicted_ms (result_timings::to_json).

No llama-server binary is a clean skip. Startup/HTTP failures raise, and the server is always
torn down.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

# Filler paragraph (~52 words) for building long shared prefixes. The content doesn't matter,
# but some variety keeps tokenization from going degenerate.
_PARA = (
    "Section {i}: The cluster report for region {i} lists CPU utilisation, memory bandwidth, "
    "cache behaviour, and storage throughput for every node. Operators review these figures "
    "each morning, compare them with the previous week, and record any regression that "
    "exceeds the agreed threshold before approving the deployment window for that day. "
)
_CHARS_PER_TOKEN = 4.0  # sizing heuristic only; the server reports exact prompt_n


def build_prefix(target_tokens: int) -> str:
    """Deterministic document sized to roughly `target_tokens` tokens.

    Sizing only. Reported numbers come from the server's own counts.
    """
    target_chars = int(target_tokens * _CHARS_PER_TOKEN)
    parts: list[str] = []
    i = 0
    total = 0
    while total < target_chars:
        p = _PARA.format(i=i)
        parts.append(p)
        total += len(p)
        i += 1
    return "".join(parts)


@dataclass
class Timings:
    prompt_n: int  # tokens actually prefilled; this is where cache reuse shows up
    prompt_ms: float  # wall-clock prompt processing, the measured TTFT component
    predicted_n: int
    predicted_ms: float

    @property
    def prompt_s(self) -> float:
        return self.prompt_ms / 1000.0


class TtftError(RuntimeError):
    """llama-server measurement failed: startup, HTTP, or response shape."""


def parse_timings(resp: dict) -> Timings:
    """Timings out of a /completion response."""
    t = resp.get("timings")
    if not isinstance(t, dict) or "prompt_ms" not in t:
        raise TtftError(
            "response has no timings.prompt_ms - unexpected llama-server response shape: "
            + json.dumps(resp)[:400]
        )
    return Timings(
        prompt_n=int(t.get("prompt_n", 0) or 0),
        prompt_ms=float(t["prompt_ms"]),
        predicted_n=int(t.get("predicted_n", 0) or 0),
        predicted_ms=float(t.get("predicted_ms", 0.0) or 0.0),
    )


@dataclass
class TtftResult:
    """Cold-vs-warm TTFT for a shared prefix. Saved to bench/results/ttft_*.json."""

    timestamp: str
    model: str
    variant: str
    prefix_target_tokens: int
    cache_reuse: int
    threads: int | None
    cold: Timings
    warm: Timings
    kind: str = "measured-ttft-prompt-cache"
    notes: str = (
        "prompt_ms is llama-server's own measured prompt-processing time; "
        "warm turn shares the prefix with the cold turn (cache_prompt=true)"
    )
    extra: dict = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if self.cold.prompt_ms <= 0:
            return 0.0
        return (1.0 - self.warm.prompt_ms / self.cold.prompt_ms) * 100.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> TtftResult:
        return cls(
            timestamp=str(d.get("timestamp", "")),
            model=str(d.get("model", "")),
            variant=str(d.get("variant", "")),
            prefix_target_tokens=int(d.get("prefix_target_tokens", 0)),
            cache_reuse=int(d.get("cache_reuse", 0)),
            threads=d.get("threads"),
            cold=Timings(**d["cold"]),
            warm=Timings(**d["warm"]),
            notes=str(d.get("notes", "")),
            extra=d.get("extra") or {},
        )

    @classmethod
    def load_json(cls, path: Path) -> TtftResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- server lifecycle ---


def free_port(host: str = "127.0.0.1") -> int:
    """Ephemeral free TCP port. A fixed port can collide with a stale or foreign server."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def build_server_cmd(
    server_bin: Path,
    model_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8033,
    threads: int | None = None,
    cache_reuse: int = 256,
    mlock: bool = False,
    ctx: int | None = None,
) -> list[str]:
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--cache-reuse",
        str(cache_reuse),
    ]
    if ctx:
        # has to hold prefix + generation. the default 4096 silently truncates longer
        # prefixes, which kills the shared-prefix premise
        cmd += ["-c", str(ctx)]
    if threads:
        cmd += ["-t", str(threads)]
    if mlock:
        # keeps weights resident, no page-fault TTFT stall after idle. needs RLIMIT_MEMLOCK
        cmd += ["--mlock"]
    return cmd


def wait_ready(
    base_url: str, *, timeout: float = 180.0, proc: subprocess.Popen | None = None
) -> None:
    """Poll GET /health until it returns 200."""
    deadline = time.monotonic() + timeout
    last = "no response yet"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise TtftError(f"llama-server exited early (code {proc.returncode})")
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(0.5)
    raise TtftError(f"llama-server not ready after {timeout:.0f}s (last: {last})")


def completion(
    base_url: str,
    prompt: str,
    *,
    n_predict: int = 16,
    cache_prompt: bool = True,
    timeout: float = 600.0,
) -> Timings:
    """POST /completion, return the server's timings."""
    r = requests.post(
        f"{base_url}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "temperature": 0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return parse_timings(r.json())


def measure_prompt_cache(
    base_url: str, prefix: str, *, n_predict: int = 16
) -> tuple[Timings, Timings]:
    """The cold/warm pair: same long prefix, two different questions. Returns (cold, warm).

    With cache_prompt (default true) the warm turn re-uses the prefix's KV cache, so
    warm.prompt_n collapses to about the question length.
    """
    q1 = "\n\nQuestion: Summarise the purpose of these reports in one sentence.\nAnswer:"
    q2 = "\n\nQuestion: What do operators compare the figures with?\nAnswer:"
    cold = completion(base_url, prefix + q1, n_predict=n_predict)
    warm = completion(base_url, prefix + q2, n_predict=n_predict)
    return cold, warm
