"""Filter and label the Dolci-Instruct-DPO benchmark test set by task type."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("/data/home/zhanghx/olmo3/dataset/benchmark/dpo")
DEFAULT_INPUT_PATH = DEFAULT_OUTPUT_DIR / "dolci_instruct_dpo_train_0p01.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "members.jsonl"

MATH_SOURCES = {
    "tulu-3-sft-personas-algebra",
    "tulu-3-sft-personas-math",
    "tulu-3-sft-personas-math-grade",
    "tulu_v3.9_open_math_2_gsm8k_50k",
}
CODE_SOURCES = {
    "correct-python-sft-187k",
    "evol_codealpaca_heval_decontaminated",
    "personahub_code_v2_34999",
}
INSTRUCTION_FOLLOWING_SOURCES = {
    "DaringAnteater-prefs_olmo2_7b",
    "IF_sft_data_verified_permissive",
    "tulu-3-sft-personas-instruction-following-o3",
    "valpy_if_qwq_reasoning_verified_no_reasoning",
}
GENERAL_QA_SOURCES = {
    "OpenThoughts3-full-filtered-science-no-cot",
    "Wildchat-1M-gpt-4.1-regenerated-english",
    "oasst1_converted",
    "ultrafeedback_cleaned_olmo2_7b",
}

EXCLUDED_SOURCES = {
    # Table tasks
    "tulu_v3.9_table_gpt_5k": "table",
    # Multilingual sources
    "Wildchat-1m-gpt-4.1-regeneration-not-english": "multilingual",
    "tulu_v3.9_aya_100k": "multilingual",
    # Safety sources
    "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "safety",
    "tulu_v3.9_wildjailbreak_decontaminated_50k": "safety",
    "tulu-3-sft-coconot-regenerated": "safety",
    # General web-content generation
    "filtered_wc_sample_500k": "general_web_content",
    # Mixed or domain-specific tasks, not pure instruction following
    "flan_v2_converted": "not_pure_instruction_following",
    "tulu_v3.9_sciriff_10k": "not_pure_instruction_following",
}

SOURCE_TO_TASK_TYPE = {
    **dict.fromkeys(MATH_SOURCES, "math"),
    **dict.fromkeys(CODE_SOURCES, "code"),
    **dict.fromkeys(INSTRUCTION_FOLLOWING_SOURCES, "instruction_following"),
    **dict.fromkeys(GENERAL_QA_SOURCES, "general_qa"),
}


def _prompt_source(prompt_id: str) -> str:
    return re.sub(r"-request-\d+-\d+$", "", prompt_id)


def _extract(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    extracted_rows: list[dict[str, Any]] = []
    task_type_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()

    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise ValueError(f"Expected string prompt_id, got {prompt_id!r}")

        if prompt_id.startswith("multiturn_"):
            excluded_counts["multiturn"] += 1
            continue

        source = _prompt_source(prompt_id)
        excluded_reason = EXCLUDED_SOURCES.get(source)
        if excluded_reason is not None:
            excluded_counts[excluded_reason] += 1
            continue

        task_type = SOURCE_TO_TASK_TYPE.get(source)
        if task_type is None:
            raise ValueError(f"Unclassified prompt source {source!r} from prompt_id {prompt_id!r}")

        extracted_rows.append({**row, "task_type": task_type})
        task_type_counts[task_type] += 1

    return extracted_rows, task_type_counts, excluded_counts


def main() -> None:
    rows = [json.loads(line) for line in DEFAULT_INPUT_PATH.open(encoding="utf-8")]
    extracted_rows, task_type_counts, excluded_counts = _extract(rows)

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in extracted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(extracted_rows)}")
    print(f"Task type counts: {dict(sorted(task_type_counts.items()))}")
    print(f"Excluded counts: {dict(sorted(excluded_counts.items()))}")
    print(f"Saved JSONL: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
