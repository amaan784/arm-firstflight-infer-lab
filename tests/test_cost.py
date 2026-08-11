import math

from firstflight import cost


def test_cost_per_million_basic():
    # 100 tok/s for $1/hr -> 360,000 tok/hr -> ~$2.7778 per 1e6 tokens
    assert math.isclose(cost.cost_per_million_tokens(100.0, 1.0), 1e6 / 360000.0, rel_tol=1e-9)


def test_cost_zero_price_is_zero():
    assert cost.cost_per_million_tokens(100.0, 0.0) == 0.0


def test_cost_zero_throughput_is_inf():
    assert math.isinf(cost.cost_per_million_tokens(0.0, 1.0))


def test_tokens_per_usd():
    assert math.isclose(cost.tokens_per_usd(100.0, 1.0), 360000.0, rel_tol=1e-9)
    assert math.isinf(cost.tokens_per_usd(100.0, 0.0))


def test_compute_result():
    r = cost.compute(50.0, 2.0)
    assert r.priced is True
    assert math.isclose(r.usd_per_million_tokens, 2.0 / (50.0 * 3600.0) * 1e6, rel_tol=1e-9)
    assert "$" in r.format_usd_per_mtok()


def test_compute_zero_price_is_free_runner():
    # 0.0 is a real price (free CI runner / free tier), not a missing configuration
    r = cost.compute(50.0, 0.0)
    assert r.priced is False
    assert "free" in r.format_usd_per_mtok()
