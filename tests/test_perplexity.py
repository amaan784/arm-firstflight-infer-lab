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
