"""Agent-in-the-loop autotuner (stretch goal, opt-in via `--enable`).

The loop: propose a config -> benchmark it -> score it -> repeat until no improvement or the
search space is exhausted. Two proposers:

  - **HeuristicProposer (default):** deterministic grid over quant variant x thread count.
    No API key, no network, so the autotuner works out of the box.
  - **LLMProposer (optional):** Claude reads the Performix hotspots plus the trajectory so far
    and proposes the next config as JSON. Lazy-imports `anthropic` (the `[agent]` extra) and
    falls back to the heuristic grid if the model repeats itself or returns junk. The Arm MCP
    server (github.com/arm/mcp) is the intended hook for richer Performix-driven proposals.

`optimize()` is pure and testable (inject any `benchmark_fn`); the real wiring (download model
variants, run `llama-bench`, save results) lives in `runner.autotune`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_LLM_MODEL = "claude-sonnet-5"  # current Sonnet; cheap enough for one-shot proposals


@dataclass(frozen=True)
class TuneConfig:
    variant: str
    threads: int | None = None
    cpu_mask: str = ""

    def key(self) -> tuple:
        return (self.variant, self.threads, self.cpu_mask)

    def label(self) -> str:
        t = f"t{self.threads}" if self.threads else "tauto"
        m = f"-{self.cpu_mask}" if self.cpu_mask else ""
        return f"{self.variant}-{t}{m}"


@dataclass
class TuneTrial:
    config: TuneConfig
    score: float  # higher is better (e.g. prefill tok/s at the target context)
    metric: str = "prefill_tok_s"
    detail: dict = field(default_factory=dict)


@dataclass
class AutotuneState:
    model_id: str
    target_context: int
    metric: str = "prefill_tok_s"
    history: list[TuneTrial] = field(default_factory=list)
    hotspots: list = field(default_factory=list)  # function/percent dicts from Performix

    def tried_keys(self) -> set:
        return {t.config.key() for t in self.history}

    def best(self) -> TuneTrial | None:
        return max(self.history, key=lambda t: t.score) if self.history else None


class Proposer(Protocol):
    def propose(self, state: AutotuneState) -> TuneConfig | None: ...


def _ordered_grid(variants: list[str], thread_options: list[int]) -> list[TuneConfig]:
    # q4_0 first (KleidiAI-accelerated), then the rest; threads vary within each variant.
    vorder = sorted(variants, key=lambda v: (v != "q4_0", v))
    return [TuneConfig(v, t) for v in vorder for t in thread_options]


class HeuristicProposer:
    """Deterministic grid over (variant, threads)."""

    def __init__(self, variants: list[str], thread_options: list[int]) -> None:
        self.grid = _ordered_grid(variants, thread_options)

    def propose(self, state: AutotuneState) -> TuneConfig | None:
        tried = state.tried_keys()
        for cfg in self.grid:
            if cfg.key() not in tried:
                return cfg
        return None


def build_llm_prompt(state: AutotuneState, variants: list[str], thread_options: list[int]) -> str:
    tried = (
        "\n".join(f"- {t.config.label()}: {t.score:.1f} {t.metric}" for t in state.history)
        or "(none yet)"
    )
    hot = (
        "\n".join(
            f"- {h.get('function', '?')} {h.get('percent', 0)}%" for h in (state.hotspots or [])[:8]
        )
        or "(no profile available)"
    )
    return (
        "You are tuning CPU LLM inference (llama.cpp) on an Arm Neoverse server. Optimize "
        f"prefill throughput at a {state.target_context}-token context (higher is better).\n\n"
        f"Quant variants available: {', '.join(variants)}\n"
        f"Thread options: {thread_options}\n\n"
        f"Trials so far (config: score):\n{tried}\n\n"
        f"Top Performix hotspots:\n{hot}\n\n"
        "Propose the SINGLE next config to try that you expect to beat the best so far. "
        'Reply with ONLY a JSON object: {"variant": "<one of the variants>", "threads": <int>}. '
        "Do not repeat a config already tried."
    )


def parse_config(text: str, variants: list[str], thread_options: list[int]) -> TuneConfig | None:
    """LLM reply -> TuneConfig. None if it isn't parseable or names an unknown variant."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    variant = str(obj.get("variant", ""))
    if variant not in variants:
        return None
    threads = obj.get("threads")
    threads = int(threads) if threads is not None else None
    if threads is not None and threads not in thread_options:
        # snap to the nearest allowed count
        threads = min(thread_options, key=lambda x: abs(x - threads)) if thread_options else None
    return TuneConfig(variant=variant, threads=threads)


class LLMProposer:
    """Claude-backed proposer. Falls back to the heuristic grid on any failure or repeat."""

    def __init__(
        self,
        variants: list[str],
        thread_options: list[int],
        *,
        model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.variants = variants
        self.thread_options = thread_options
        self.model = model
        self._fallback = HeuristicProposer(variants, thread_options)

    def propose(self, state: AutotuneState) -> TuneConfig | None:
        cfg = self._propose_llm(state)
        if cfg is not None and cfg.key() not in state.tried_keys():
            return cfg
        return self._fallback.propose(state)  # repeat/invalid/no-key -> grid

    def _propose_llm(self, state: AutotuneState) -> TuneConfig | None:
        try:
            import anthropic
        except ImportError:
            return None
        try:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            prompt = build_llm_prompt(state, self.variants, self.thread_options)
            resp = client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
            return parse_config(text, self.variants, self.thread_options)
        except Exception:  # noqa: BLE001 - network/auth/parse issues -> grid
            return None


def optimize(
    proposer: Proposer,
    benchmark_fn: Callable[[TuneConfig], TuneTrial],
    *,
    initial_state: AutotuneState,
    max_iters: int = 12,
    patience: int = 3,
) -> AutotuneState:
    """Drive the propose -> benchmark -> score loop.

    `benchmark_fn(config)` runs the config and returns a TuneTrial (higher score is better).
    Stops after `patience` consecutive non-improving trials, when the proposer is exhausted
    (returns None), or after `max_iters`.
    """
    state = initial_state
    best = float("-inf")
    stale = 0
    for _ in range(max_iters):
        cfg = proposer.propose(state)
        if cfg is None:
            break
        trial = benchmark_fn(cfg)
        state.history.append(trial)
        if trial.score > best:
            best = trial.score
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return state
