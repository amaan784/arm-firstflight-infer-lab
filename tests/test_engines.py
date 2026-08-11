def test_kernel_tier_ranking():
    """Highest reachable tier, not a boolean. SME2 > I8MM > DOTPROD > SVE2 > NEON."""
    from firstflight.engines.llama_cpp import kernel_tier

    assert kernel_tier("NEON = 1 | MATMUL_INT8 = 1 | DOTPROD = 1") == "I8MM"
    assert kernel_tier("NEON = 1 | DOTPROD = 1") == "DOTPROD"
    assert kernel_tier("NEON = 1") == "NEON"
    assert kernel_tier("", "fp asimd i8mm") == "I8MM"
    assert kernel_tier("", "") == ""
