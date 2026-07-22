"""Build benchmark SFT jsonl files with question/solution fields.

Each dataset is shuffled with seed 42, split 8:2, then saved as:
    <output_dir>/<dataset_name>/train.jsonl
    <output_dir>/<dataset_name>/test.jsonl
"""

import argparse
import json
import random
from collections.abc import Callable
from itertools import chain
from pathlib import Path

from datasets import load_dataset

SEED = 42
TRAIN_RATIO = 0.8


def _load_dataset(path: str, split: str, config: str | None = None):
    kwargs = {"split": split}
    if config:
        return load_dataset(path, config, **kwargs)
    return load_dataset(path, **kwargs)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _pick(row: dict, *names: str):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"None of {names} found in row with keys {list(row)}")


def _sample(question, solution) -> dict | None:
    question, solution = _text(question), _text(solution)
    if not question or not solution:
        return None
    return {"question": question, "solution": solution}


def _write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _save(name: str, samples: list[dict], output_dir: Path) -> None:
    random.Random(SEED).shuffle(samples)
    train_size = int(len(samples) * TRAIN_RATIO)
    dataset_dir = output_dir / name
    _write_jsonl(samples[:train_size], dataset_dir / "train.jsonl")
    _write_jsonl(samples[train_size:], dataset_dir / "test.jsonl")
    import pdb; pdb.set_trace()
    print(f"{name}: saved {train_size} train and {len(samples) - train_size} test examples")


def _format_choices(question: str, choices: dict) -> str:
    labels = choices["label"]
    texts = choices["text"]
    choices_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"Answer the following multiple choice question.\n\nQuestion: {question}\n\nChoices:\n{choices_text}"


def _choice_answer(answer_key: str, choices: dict) -> str:
    labels = choices["label"]
    texts = choices["text"]
    if answer_key in labels:
        return f"{answer_key}. {texts[labels.index(answer_key)]}"
    return answer_key


def _conversation_text(messages: list[dict]) -> str:
    return "\n\n".join(message["content"] for message in messages)


def _reference_answer(references: dict) -> str:
    if not references:
        return ""
    if "gpt-4" in references:
        return references["gpt-4"]
    return next(iter(references.values()))


def _build(samples, name: str, output_dir: Path) -> None:
    cleaned = [sample for sample in samples if sample is not None]
    _save(name, cleaned, output_dir)


def construct_math500(output_dir: Path) -> None:
    ds = _load_dataset("HuggingFaceH4/MATH-500", "test")
    _build(
        (_sample(_pick(row, "problem", "question"), _pick(row, "solution", "answer")) for row in ds),
        "math500",
        output_dir,
    )


def construct_gsm8k(output_dir: Path) -> None:
    ds = _load_dataset("openai/gsm8k", "test", "main")
    _build((_sample(row["question"], row["answer"]) for row in ds), "gsm8k", output_dir)


def _aime2025_samples():
    ds = _load_dataset("Sunny8781/AIME2025_w_solution", "test")
    return (_sample(_pick(row, "problem", "question"), _pick(row, "solution", "answer")) for row in ds)


def _aimo_validation_aime_samples():
    ds = _load_dataset("AI-MO/aimo-validation-aime", "train")
    return (_sample(_pick(row, "problem", "question"), _pick(row, "solution", "answer")) for row in ds)


def construct_aime(output_dir: Path) -> None:
    _build(chain(_aime2025_samples(), _aimo_validation_aime_samples()), "aime", output_dir)


def construct_mbppplus(output_dir: Path) -> None:
    ds = _load_dataset("evalplus/mbppplus", "test")
    _build((_sample(row["prompt"], row["code"]) for row in ds), "mbppplus", output_dir)


def _humanevalplus_question(row: dict) -> str:
    return "Complete the following Python function.\n\n" + row["prompt"].rstrip()


def _humanevalplus_solution(row: dict) -> str:
    return row["canonical_solution"].lstrip("\n")


def construct_humanevalplus(output_dir: Path) -> None:
    ds = _load_dataset("evalplus/humanevalplus", "test")
    _build(
        (_sample(_humanevalplus_question(row), _humanevalplus_solution(row)) for row in ds),
        "humanevalplus",
        output_dir,
    )


def _bigcodebench_solution(row: dict) -> str:
    return row["code_prompt"].rstrip() + "\n" + row["canonical_solution"].lstrip("\n")


def construct_bigcodebench(output_dir: Path) -> None:
    ds = _load_dataset("bigcode/bigcodebench", "v0.1.4")
    _build((_sample(row["instruct_prompt"], _bigcodebench_solution(row)) for row in ds), "bigcodebench", output_dir)


def construct_gpqa_diamond(output_dir: Path) -> None:
    ds = _load_dataset("hendrydong/gpqa_diamond", "test")
    _build(
        (
            _sample(_pick(row, "problem", "Question"), f"The answer is {_pick(row, 'solution', 'Correct Answer')}")
            for row in ds
        ),
        "gpqa_diamond",
        output_dir,
    )


def construct_truthfulqa(output_dir: Path) -> None:
    ds = _load_dataset("domenicrosati/TruthfulQA", "train")
    _build(
        (_sample(_pick(row, "Question", "question"), _pick(row, "Best Answer", "best_answer")) for row in ds),
        "truthfulqa",
        output_dir,
    )


def construct_arc_challenge(output_dir: Path) -> None:
    ds = _load_dataset("ibragim-bad/arc_challenge", "test")
    samples = (
        _sample(_format_choices(row["question"], row["choices"]), _choice_answer(row["answerKey"], row["choices"]))
        for row in ds
    )
    _build(samples, "arc_challenge", output_dir)


def construct_alpaca_eval(output_dir: Path) -> None:
    ds = load_dataset(
        "json", data_files="hf://datasets/tatsu-lab/alpaca_eval/alpaca_eval_gpt4_baseline.json", split="train"
    )
    _build(
        (_sample(row["instruction"], _pick(row, "output", "reference_output")) for row in ds),
        "alpaca_eval",
        output_dir,
    )


def construct_wildbench_v2(output_dir: Path) -> None:
    ds = _load_dataset("WildEval/WildBench-V2", "test", "v2")
    samples = (
        _sample(_conversation_text(row["conversation_input"]), _reference_answer(row["references"])) for row in ds
    )
    _build(samples, "wildbench_v2", output_dir)


CONSTRUCTORS: dict[str, Callable[[Path], None]] = {
    # "aime": construct_aime
    # "alpaca_eval": construct_alpaca_eval,
    # "arc_challenge": construct_arc_challenge
    # "bigcodebench": construct_bigcodebench
    # "gpqa_diamond": construct_gpqa_diamond,
    # "gsm8k": construct_gsm8k,
    # "humanevalplus": construct_humanevalplus
    # "math500": construct_math500,
    # "mbppplus": construct_mbppplus,
    # "truthfulqa": construct_truthfulqa,
    "wildbench_v2": construct_wildbench_v2
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("/data/home/zhanghx/olmo3/dataset/benchmark"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONSTRUCTORS), default=sorted(CONSTRUCTORS))
    args = parser.parse_args()

    for name in args.datasets:
        CONSTRUCTORS[name](args.output_dir)


if __name__ == "__main__":
    main()
