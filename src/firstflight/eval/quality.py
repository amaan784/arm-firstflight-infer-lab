"""Quality-delta guardrail: prove a speedup didn't tank accuracy.

Two paths:

1. **Built-in probe (default).** A fixed Q&A set run through `llama-cli` with exact-match
   scoring. No torch, no server, nothing beyond llama.cpp. Answers "does the
   quantized/optimized model still answer sanely?" and runs on the Arm box like the benchmark.

2. **lm-evaluation-harness (optional, `[eval]` extra).** For a real MMLU/GSM8K subset, run a
   `llama-server` and point lm-eval at its OpenAI-compatible endpoint with
   `--model local-completions` (the `gguf` backend is known-broken). Uses the light
   `lm-eval[api]` install (no torch). `build_lm_eval_cmd` builds that invocation; exact task
   names/endpoint are TODO(confirm) on the box.

Small on purpose: a regression check, not a leaderboard.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..engines import llama_cpp


@dataclass(frozen=True)
class QualityItem:
    prompt: str
    answers: list[str]  # acceptable gold strings; whole-token match


# 40 items (arithmetic + factual + extraction), short answers, whole-token match, --temp 0.
# 40 gives 2.5-point granularity, finer than the report's "held" tolerance. Raw-completion
# style, no chat template: single-shot `llama-cli -no-cnv` skips interactive mode, and
# base-completion prompts keep the probe identical across base and instruct models.
PROBE: list[QualityItem] = [
    # arithmetic (15)
    QualityItem("Question: What is 2 + 2?\nAnswer:", ["4", "four"]),
    QualityItem("Question: What is 10 - 3?\nAnswer:", ["7", "seven"]),
    QualityItem("Question: What is 5 multiplied by 6?\nAnswer:", ["30", "thirty"]),
    QualityItem(
        "Question: What is 100 divided by 4?\nAnswer:", ["25", "twenty-five", "twenty five"]
    ),
    QualityItem("Question: What is 9 + 8?\nAnswer:", ["17", "seventeen"]),
    QualityItem("Question: What is 12 times 12?\nAnswer:", ["144"]),
    QualityItem("Question: What is 50 - 15?\nAnswer:", ["35", "thirty-five", "thirty five"]),
    QualityItem("Question: What is half of 90?\nAnswer:", ["45", "forty-five", "forty five"]),
    QualityItem("Question: What is 7 squared?\nAnswer:", ["49", "forty-nine", "forty nine"]),
    QualityItem("Question: What is 3 + 4 + 5?\nAnswer:", ["12", "twelve"]),
    QualityItem("Question: What is 1000 minus 1?\nAnswer:", ["999"]),
    QualityItem("Question: What is 6 times 7?\nAnswer:", ["42", "forty-two", "forty two"]),
    QualityItem("Question: What is 81 divided by 9?\nAnswer:", ["9", "nine"]),
    QualityItem("Question: What is 15 + 20?\nAnswer:", ["35", "thirty-five", "thirty five"]),
    QualityItem("Question: What is one third of 30?\nAnswer:", ["10", "ten"]),
    # factual (15)
    QualityItem("Question: What is the capital of France?\nAnswer:", ["paris"]),
    QualityItem("Question: What is the capital of Japan?\nAnswer:", ["tokyo"]),
    QualityItem("Question: What planet do humans live on?\nAnswer:", ["earth"]),
    QualityItem("Question: What is the chemical symbol for water?\nAnswer:", ["h2o"]),
    QualityItem("Question: What is the chemical symbol for gold?\nAnswer:", ["au"]),
    QualityItem("Question: How many days are in a week?\nAnswer:", ["7", "seven"]),
    QualityItem("Question: How many legs does a spider have?\nAnswer:", ["8", "eight"]),
    QualityItem("Question: What is the largest ocean on Earth?\nAnswer:", ["pacific"]),
    QualityItem("Question: How many continents are there?\nAnswer:", ["7", "seven"]),
    QualityItem(
        "Question: What gas do plants absorb from the air?\nAnswer:", ["carbon dioxide", "co2"]
    ),
    QualityItem("Question: How many minutes are in an hour?\nAnswer:", ["60", "sixty"]),
    QualityItem("Question: What is the closest star to Earth?\nAnswer:", ["sun"]),
    QualityItem("Question: What language is spoken in Brazil?\nAnswer:", ["portuguese"]),
    QualityItem("Question: How many sides does a triangle have?\nAnswer:", ["3", "three"]),
    QualityItem("Question: What is frozen water called?\nAnswer:", ["ice"]),
    # opposites / simple reasoning / extraction (10)
    QualityItem("Question: What is the opposite of hot?\nAnswer:", ["cold"]),
    QualityItem("Question: What is the opposite of up?\nAnswer:", ["down"]),
    QualityItem("Question: What is the opposite of dark?\nAnswer:", ["light", "bright"]),
    QualityItem("Question: What color is the sky on a clear day?\nAnswer:", ["blue"]),
    QualityItem("Question: What color is grass?\nAnswer:", ["green"]),
    QualityItem(
        "Text: Ada was born in London in 1815.\nQuestion: In which city was Ada born?\nAnswer:",
        ["london"],
    ),
    QualityItem(
        "Text: The meeting starts at 9 am on Tuesday.\n"
        "Question: On which day is the meeting?\nAnswer:",
        ["tuesday"],
    ),
    QualityItem(
        "Text: The red box weighs 5 kg and the blue box weighs 3 kg.\n"
        "Question: Which box is heavier?\nAnswer:",
        ["red"],
    ),
    QualityItem("Question: If today is Monday, what day is tomorrow?\nAnswer:", ["tuesday"]),
    QualityItem(
        "Question: Which is larger, 17 or 71?\nAnswer:", ["71", "seventy-one", "seventy one"]
    ),
]


@dataclass
class QualityItemResult:
    prompt: str
    expected: list[str]
    got: str
    correct: bool


@dataclass
class QualityResult:
    method: str
    n_total: int
    n_correct: int
    accuracy: float
    items: list[QualityItemResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> QualityResult:
        return cls(
            method=str(d.get("method", "")),
            n_total=int(d.get("n_total", 0)),
            n_correct=int(d.get("n_correct", 0)),
            accuracy=float(d.get("accuracy", 0.0)),
            items=[QualityItemResult(**i) for i in d.get("items", [])],
            notes=str(d.get("notes", "")),
        )


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def matches(output: str, answers: list[str]) -> bool:
    """True if any gold answer appears as a whole token (word-boundary match).

    Plain substring matching would inflate the score: '7' matches '... = 7.' but not '17'.
    """
    o = _normalize(output)
    for a in answers:
        a_norm = _normalize(a)
        if a_norm and re.search(rf"(?<!\w){re.escape(a_norm)}(?!\w)", o):
            return True
    return False


def run_probe(
    cli_bin: Path,
    model_path: Path,
    *,
    threads: int | None = None,
    n_predict: int = 16,
    items: list[QualityItem] | None = None,
    seed: int = 42,
) -> QualityResult:
    """Run the built-in exact-match probe through llama-cli, score accuracy."""
    items = items or PROBE
    details: list[QualityItemResult] = []
    correct = 0
    for it in items:
        res = llama_cpp.run_once(
            cli_bin,
            model_path,
            prompt=it.prompt,
            n_predict=n_predict,
            threads=threads,
            seed=seed,
            temp=0.0,  # greedy, so this measures the model not the sampler
        )
        # Score the first line only. Matching anywhere in a free-running completion gives a
        # rambling model more chances to hit the gold string, so a worse model could score
        # higher. One line is the answer; the rest is drift.
        got = res.stdout.strip().split("\n", 1)[0].strip()
        ok = res.ok and matches(got, it.answers)
        correct += int(ok)
        details.append(QualityItemResult(it.prompt, it.answers, got[:120], ok))
    n = len(items)
    return QualityResult(
        method="builtin-probe",
        n_total=n,
        n_correct=correct,
        accuracy=(correct / n) if n else 0.0,
        items=details,
        notes="exact-match guardrail via llama-cli, greedy decoding (not a leaderboard)",
    )


# --- optional lm-eval path (documented; [eval] extra) -------------------------


def available() -> bool:
    """Whether lm-eval is importable (installed via the `[eval]` extra)."""
    try:
        import lm_eval  # noqa: F401
    except Exception:
        return False
    return True


def build_lm_eval_cmd(
    base_url: str = "http://localhost:8000/v1",
    *,
    tasks: str = "gsm8k",
    limit: int = 10,
    chat: bool = True,
) -> list[str]:
    """The light, no-torch lm-eval invocation against a running `llama-server`.

    Prereqs (TODO(confirm) exact task names/endpoint on the box):
      pip install 'lm-eval[api]'                       # no torch
      llama-server -m model.gguf --host 0.0.0.0 --port 8000
    """
    model = "local-chat-completions" if chat else "local-completions"
    endpoint = "/chat/completions" if chat else "/completions"
    return [
        "lm_eval",
        "--model",
        model,
        "--tasks",
        tasks,
        "--limit",
        str(limit),
        "--model_args",
        f"base_url={base_url}{endpoint},num_concurrent=1,tokenized_requests=False",
    ]
