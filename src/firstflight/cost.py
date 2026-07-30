"""Throughput -> cost. Turns tokens/sec + $/hour into $ per million tokens.

The only external input is the instance hourly price (configs/instances.yaml), which is
region/time dependent and therefore left as a TODO(confirm) placeholder until you provide
the live price for the box you benchmark on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SECONDS_PER_HOUR = 3600.0
ONE_MILLION = 1_000_000.0


@dataclass(frozen=True)
class CostResult:
    throughput_tok_s: float
    usd_per_hour: float
    usd_per_million_tokens: float
    tokens_per_usd: float
    priced: bool  # False when usd_per_hour is unset (0.0) — cost is then meaningless

    def format_usd_per_mtok(self) -> str:
        if not self.priced:
            return "n/a (set instance price)"
        if math.isinf(self.usd_per_million_tokens):
            return "inf"
        return f"${self.usd_per_million_tokens:.4f}"


def cost_per_million_tokens(throughput_tok_s: float, usd_per_hour: float) -> float:
    """USD to produce 1,000,000 tokens at a sustained throughput.

    cost = (usd_per_hour / tokens_per_hour) * 1e6
         = usd_per_hour / (throughput_tok_s * 3600) * 1e6

    Returns +inf when throughput is non-positive (can't make progress), and 0.0 when the
    price is 0 (e.g. free-tier / CI).
    """
    if usd_per_hour <= 0:
        return 0.0
    if throughput_tok_s <= 0:
        return math.inf
    tokens_per_hour = throughput_tok_s * SECONDS_PER_HOUR
    return usd_per_hour / tokens_per_hour * ONE_MILLION


def tokens_per_usd(throughput_tok_s: float, usd_per_hour: float) -> float:
    """How many tokens one dollar buys. +inf when the price is 0."""
    if usd_per_hour <= 0:
        return math.inf
    if throughput_tok_s <= 0:
        return 0.0
    return throughput_tok_s * SECONDS_PER_HOUR / usd_per_hour


def compute(throughput_tok_s: float, usd_per_hour: float) -> CostResult:
    return CostResult(
        throughput_tok_s=throughput_tok_s,
        usd_per_hour=usd_per_hour,
        usd_per_million_tokens=cost_per_million_tokens(throughput_tok_s, usd_per_hour),
        tokens_per_usd=tokens_per_usd(throughput_tok_s, usd_per_hour),
        priced=usd_per_hour > 0,
    )
