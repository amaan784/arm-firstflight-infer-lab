"""MEASURED time-to-first-token via llama-server, and the prompt-cache demo.

Everywhere else in the harness TTFT is *derived* (prompt_tokens ÷ prefill throughput). This
module measures it: `llama-server`'s native `/completion` endpoint returns a `timings` object
with `prompt_ms` (wall-clock prompt processing) and `prompt_n` (tokens actually processed) —
the server's own stopwatch.

It also demonstrates the single biggest honest TTFT lever for agentic/RAG serving:
**prefix caching**. llama-server's `cache_prompt` (request field, default true) re-uses the
KV cache when a request shares a prefix with the previous one — only the differing suffix is
prefilled. `--cache-reuse N` (server flag, default 0) additionally re-uses non-exact-prefix
chunks via KV shifting. So for two requests sharing a long system/context prefix:

  turn 1 (cold): the full prefix is prefilled  -> large prompt_ms, prompt_n ≈ full prompt
  turn 2 (warm): only the new question is prefilled -> prompt_ms collapses, prompt_n is tiny

Flags/fields per the llama-server README (verified during research, 2026-07-07): `/health`
readiness endpoint, `/completion` with `cache_prompt` + `n_predict`, response `timings`.
Everything degrades gracefully: no llama-server binary -> clean skip; startup/HTTP failures
-> clear error, server always torn down.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

# A deterministic filler paragraph (~52 words) used to build long shared prefixes. Content is
# irrelevant to the measurement; variety avoids degenerate tokenization.
_PARA = (
    "Section {i}: The cluster report for region {i} lists CPU utilisation, memory bandwidth, "
    "cache behaviour, and storage throughput for every node. Operators review these figures "
    "each morning, compare them with the previous week, and record any regression that "
    "exceeds the agreed threshold before approving the deployment window for that day. "
)
_CHARS_PER_TOKEN = 4.0  # sizing heuristic only; the SERVER reports exact prompt_n


def build_prefix(target_tokens: int) -> str:
    """A deterministic long document sized to roughly `target_tokens` LLM tokens.

    Only used to size the experiment — the reported numbers use the server's exact counts.
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
    prompt_n: int  # tokens actually processed during prefill (shows cache reuse!)
    prompt_ms: float  # measured wall-clock prompt processing = measured TTFT component
    predicted_n: int
    predicted_ms: float

    @property
    def prompt_s(self) -> float:
        return self.prompt_ms / 1000.0


class TtftError(RuntimeError):
    """llama-server measurement failed (startup, HTTP, or response shape)."""


def parse_timings(resp: dict) -> Timings:
    """Extract the server's measured timings from a /completion response."""
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
    """Measured cold-vs-warm TTFT for a shared prefix. Saved to bench/results/ttft_*.json."""

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


# --- server lifecycle ---------------------------------------------------------


def build_server_cmd(
    server_bin: Path,
    model_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8033,
    threads: int | None = None,
    cache_reuse: int = 256,
    mlock: bool = False,
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
    if threads:
        cmd += ["-t", str(threads)]
    if mlock:
        # keep weights resident (no post-idle page-fault TTFT stalls); needs RLIMIT_MEMLOCK
        cmd += ["--mlock"]
    return cmd


def wait_ready(
    base_url: str, *, timeout: float = 180.0, proc: subprocess.Popen | None = None
) -> None:
    """Poll GET /health until the server reports ready (200)."""
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
    """POST /completion and return the server's measured timings."""
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
    """The cold/warm pair: same long prefix, two different questions.

    Returns (cold, warm). With cache_prompt (default true) the warm turn re-uses the prefix's
    KV cache, so warm.prompt_n collapses to roughly the question length.
    """
    q1 = "\n\nQuestion: Summarise the purpose of these reports in one sentence.\nAnswer:"
    q2 = "\n\nQuestion: What do operators compare the figures with?\nAnswer:"
    cold = completion(base_url, prefix + q1, n_predict=n_predict)
    warm = completion(base_url, prefix + q2, n_predict=n_predict)
    return cold, warm
