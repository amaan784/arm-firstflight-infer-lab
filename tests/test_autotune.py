from firstflight.autotune import agent
from firstflight.autotune.agent import (
    AutotuneState,
    HeuristicProposer,
    TuneTrial,
    optimize,
    parse_config,
)


def test_heuristic_grid_order_and_exhaustion():
    p = HeuristicProposer(["q4_k_m", "q4_0", "q8_0"], [4, 8])
    st = AutotuneState(model_id="m", target_context=2048)
    seen = []
    while True:
        c = p.propose(st)
        if c is None:
            break
        seen.append(c.key())
        st.history.append(TuneTrial(config=c, score=1.0))
    assert len(seen) == 6  # 3 variants x 2 thread options
    assert seen[0][0] == "q4_0"  # KleidiAI-accelerated variant proposed first


def test_optimize_patience_and_best():
    scores = {("q4_0", 4, ""): 100.0, ("q4_0", 8, ""): 90.0, ("q4_k_m", 4, ""): 80.0}

    def bench(cfg):
        return TuneTrial(config=cfg, score=scores.get(cfg.key(), 50.0))

    p = HeuristicProposer(["q4_0", "q4_k_m", "q8_0"], [4, 8])
    st = AutotuneState(model_id="m", target_context=2048)
    out = optimize(p, bench, initial_state=st, max_iters=12, patience=2)
    best = out.best()
    assert best.config.key() == ("q4_0", 4, "")
    assert best.score == 100.0
    # best, then two non-improving trials hit patience=2 -> stop after 3 trials.
    assert len(out.history) == 3


def test_parse_config():
    variants, threads = ["q4_0", "q4_k_m"], [4, 8]
    assert parse_config('{"variant":"q4_0","threads":8}', variants, threads).key() == (
        "q4_0",
        8,
        "",
    )
    # snap an out-of-range thread count to the nearest allowed option
    assert parse_config('{"variant":"q4_0","threads":5}', variants, threads).threads == 4
    assert parse_config('{"variant":"nope"}', variants, threads) is None
    assert parse_config("no json here", variants, threads) is None


def test_build_llm_prompt_mentions_state():
    st = AutotuneState(
        model_id="m", target_context=4096, hotspots=[{"function": "ggml_mul_mat", "percent": 42}]
    )
    st.history.append(TuneTrial(config=agent.TuneConfig("q4_0", 4), score=123.0))
    prompt = agent.build_llm_prompt(st, ["q4_0", "q4_k_m"], [4, 8])
    assert "4096" in prompt
    assert "ggml_mul_mat" in prompt
    assert "q4_0-t4" in prompt  # prior trial summarized
