from pathlib import Path

from firstflight.engines import llama_cpp
from firstflight.eval import quality


def test_matches_normalized():
    assert quality.matches("The answer is 4.", ["4"])
    assert quality.matches("Paris is the capital.", ["paris"])
    assert quality.matches("It is H2O.", ["h2o"])
    assert not quality.matches("I think it is 5", ["4", "four"])
    # word-boundary: a substring match must NOT count ('17' is not '7')
    assert not quality.matches("10 - 3 = 17", ["7"])
    assert quality.matches("the answer is 7", ["7"])


def test_probe_is_large_enough():
    # 40 items = 2.5-percentage-point granularity, finer than the report's held-tolerance.
    assert len(quality.PROBE) >= 40


def test_run_probe_scored_and_greedy(monkeypatch):
    # stub run_once: gold answer for even-indexed items, junk otherwise
    state = {"i": 0, "temps": []}

    def fake_run_once(cli_bin, model_path, *, prompt, n_predict, threads=None, seed=42, temp=None):
        idx = state["i"]
        state["i"] += 1
        state["temps"].append(temp)
        text = quality.PROBE[idx].answers[0] if idx % 2 == 0 else "definitely not"
        return llama_cpp.RunResult(returncode=0, stdout=text, stderr="", wall_s=0.1, cmd=[])

    monkeypatch.setattr(quality.llama_cpp, "run_once", fake_run_once)
    res = quality.run_probe(Path("llama-cli"), Path("m.gguf"))
    assert res.n_total == len(quality.PROBE)
    assert res.n_correct == sum(1 for i in range(len(quality.PROBE)) if i % 2 == 0)
    assert 0.0 <= res.accuracy <= 1.0
    assert res.method == "builtin-probe"
    assert all(t == 0.0 for t in state["temps"])  # greedy: measure the model, not the sampler


def test_build_run_cmd_temp_flag():
    # --temp verified against the real b9873 llama-cli --help (2026-07-07)
    cmd = llama_cpp.build_run_cmd(Path("llama-cli"), Path("m.gguf"), "hi", temp=0.0)
    assert cmd[cmd.index("--temp") + 1] == "0.0"
    plain = llama_cpp.build_run_cmd(Path("llama-cli"), Path("m.gguf"), "hi")
    assert "--temp" not in plain


def test_build_lm_eval_cmd():
    cmd = quality.build_lm_eval_cmd(tasks="gsm8k", limit=5, chat=True)
    assert cmd[0] == "lm_eval"
    assert "--tasks" in cmd and "gsm8k" in cmd
    assert "--limit" in cmd and "5" in cmd
    assert any("chat/completions" in c for c in cmd)
