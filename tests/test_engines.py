def test_kernel_tier_ranking():
    """Highest reachable tier, not a boolean. SME2 > I8MM > DOTPROD > SVE2 > NEON."""
    from firstflight.engines.llama_cpp import kernel_tier

    assert kernel_tier("NEON = 1 | MATMUL_INT8 = 1 | DOTPROD = 1") == "I8MM"
    assert kernel_tier("NEON = 1 | DOTPROD = 1") == "DOTPROD"
    assert kernel_tier("NEON = 1") == "NEON"
    assert kernel_tier("") == ""


def test_kernel_tier_reads_the_build_not_the_host():
    """The tier is a property of the binary, not of the chip it runs on.

    These are verbatim system_info lines from two builds on ONE Neoverse N2 runner: the
    armv8-a floor reports no MATMUL_INT8, the native build does. A tier that also consulted
    /proc/cpuinfo would label both I8MM, making the unaccelerated rung look identically
    capable in the very section that claims to show what actually ran.
    """
    from firstflight.engines.llama_cpp import kernel_tier

    floor = "n_threads = 4 (n_threads_batch = 4) / 4 | CPU : NEON = 1 | ARM_FMA = 1 | OPENMP = 1 |"
    native = (
        "n_threads = 4 (n_threads_batch = 4) / 4 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | "
        "MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | KLEIDIAI = 1 | REPACK = 1 |"
    )
    assert kernel_tier(floor) == "NEON"
    assert kernel_tier(native) == "I8MM"
