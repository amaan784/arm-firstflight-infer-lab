"""Perplexity guardrail: command building, output parsing, default corpus."""

from pathlib import Path

from firstflight.eval import perplexity as p


def test_build_command_flags():
    cmd = p.build_perplexity_command(
        Path("llama-perplexity"), Path("m.gguf"), Path("c.txt"), threads=4, chunks=8
    )
    assert cmd[0] == "llama-perplexity"
    for flag in ("-m", "-f", "--chunks", "-t", "-c"):
        assert flag in cmd
    assert cmd[cmd.index("--chunks") + 1] == "8"
    assert cmd[cmd.index("-c") + 1] == "512"  # pinned eval context: deterministic + cheap


def test_ppl_regex_takes_final_value():
    out = "chunk 1: PPL = 14.1\nsome noise\nFinal estimate: PPL = 12.3456 +/- 0.12\n"
    assert p._PPL_RE.findall(out)[-1] == "12.3456"


def test_ppl_regex_no_match_is_empty():
    assert p._PPL_RE.findall("no perplexity here") == []


def test_default_corpus_written(tmp_path):
    corpus = p.ensure_default_corpus(tmp_path)
    # a checkout always has at least README.md; full checkouts add docs/*.md
    assert corpus is not None and corpus.is_file()
    assert corpus.stat().st_size > 500
    assert "Arm" in corpus.read_text(encoding="utf-8")[:5000]


def test_perplexity_survives_no_quality():
    """--no-quality drops the coarse probe only.

    On the attribution ladder every rung loads the same GGUF and only the kernels differ, so
    perplexity is the one numerics check that can detect a kernel changing the outputs. It
    must not be nested inside `if quality:` — that made --no-quality strip all verification
    from the ladder while the flag's stated purpose was to strip just the 40-item probe.
    """
    import ast
    import inspect
    import textwrap

    from firstflight import runner

    src = textwrap.dedent(inspect.getsource(runner._experiment_round))
    tree = ast.parse(src)

    def guarded_by_quality(node):
        """Whether `node` sits inside an `if quality:` block."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.If) and getattr(parent.test, "id", None) == "quality":
                if any(n is node for n in ast.walk(parent)):
                    return True
        return False

    ppl_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {"run_perplexity", "ensure_default_corpus"}
    ]
    assert ppl_calls, "perplexity is no longer called from the experiment round"
    assert not any(guarded_by_quality(n) for n in ppl_calls), (
        "perplexity is nested inside `if quality:` — --no-quality would silently disable it"
    )
