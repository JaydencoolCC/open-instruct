import glob
import json
import random

from datasets import load_from_disk
from transformers import AutoTokenizer


def get_all_training_data():
    ROLLOUT_GLOB = (
        "data/rollouts/"
        "olmo3-7b-GRPO-20260806_092823__42__1786008555_rollouts_*.jsonl"
    )
    OUTPUT_PATH = "/home/zhanghx/code/open-instruct/post_llm/docli_rl.jsonl"
    TOKENIZER_PATH = "/home/zhanghx/models/allenai/Olmo-3-7B-Instruct-DPO"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    rollout_paths = sorted(glob.glob(ROLLOUT_GLOB))

    num_records = 0
    seen_prompts = set()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for rollout_path in rollout_paths:
            with open(rollout_path, encoding="utf-8") as rollout_file:
                for line in rollout_file:
                    sample = json.loads(line)
                    raw_query = sample["raw_query"]
                    if raw_query in seen_prompts:
                        continue
                    seen_prompts.add(raw_query)

                    record = {
                        "prompt_idx": sample["prompt_idx"],
                        "prompt_tokens": sample["prompt_tokens"],
                        "raw_query": raw_query,
                        "prompt_tokens_text": tokenizer.decode(
                            sample["prompt_tokens"], skip_special_tokens=False
                        ),
                        "dataset": sample["dataset"],
                        "ground_truth": sample["ground_truth"],
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_records += 1

    print(f"Saved {num_records} records to {OUTPUT_PATH}")


def get_members():
    input_path = "/home/zhanghx/code/open-instruct/post_llm/docli_rl.jsonl"
    output_path = "/home/zhanghx/code/open-instruct/post_llm/docli_rl_8k.jsonl"
    dataset_names = ("math", "code", "ifeval", "general-quality_ref")
    samples_per_dataset = 2_000
    rng = random.Random(42)

    samples = {dataset_name: [] for dataset_name in dataset_names}
    counts = {dataset_name: 0 for dataset_name in dataset_names}

    with open(input_path, encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue

            record = json.loads(line)
            matched_datasets = set(record["dataset"]).intersection(dataset_names)
            for dataset_name in matched_datasets:
                counts[dataset_name] += 1
                dataset_samples = samples[dataset_name]
                if len(dataset_samples) < samples_per_dataset:
                    dataset_samples.append(record)
                    continue

                replacement_index = rng.randrange(counts[dataset_name])
                if replacement_index < samples_per_dataset:
                    dataset_samples[replacement_index] = record

    undersized_datasets = {
        dataset_name: count
        for dataset_name, count in counts.items()
        if count < samples_per_dataset
    }
    if undersized_datasets:
        raise ValueError(
            f"Not enough records to sample {samples_per_dataset} per dataset: "
            f"{undersized_datasets}"
        )

    selected_records = [
        record
        for dataset_name in dataset_names
        for record in samples[dataset_name]
    ]
    rng.shuffle(selected_records)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for record in selected_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(selected_records)} records to {output_path}")

def get_non_members():
    input_path = (
        "/home/zhanghx/benchmark/datasets/Dolci-Instruct-RL/"
        "dolci_instruct_rl_test"
    )
    output_path = (
        "/home/zhanghx/code/open-instruct/post_llm/"
        "docli_rl_non_members_8k.jsonl"
    )
    tokenizer_path = "/home/zhanghx/models/allenai/Olmo-3-7B-Instruct-DPO"
    dataset_names = ("math", "code", "ifeval", "general-quality_ref")
    samples_per_dataset = 2_000
    rng = random.Random(42)

    dataset = load_from_disk(input_path)
    dataset = dataset.filter(lambda row: len(row["input_ids_prompt"]) <= 2048)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    samples = {dataset_name: [] for dataset_name in dataset_names}
    counts = {dataset_name: 0 for dataset_name in dataset_names}
    for prompt_idx, row in enumerate(dataset):
        record = {
            "prompt_idx": prompt_idx,
            "prompt_tokens": row["input_ids_prompt"],
            "raw_query": row["prompt"],
            "prompt_tokens_text": tokenizer.decode(
                row["input_ids_prompt"], skip_special_tokens=False
            ),
            "dataset": row["dataset"],
            "ground_truth": row["ground_truth"],
        }
        matched_datasets = set(row["dataset"]).intersection(dataset_names)
        for dataset_name in matched_datasets:
            counts[dataset_name] += 1
            dataset_samples = samples[dataset_name]
            if len(dataset_samples) < samples_per_dataset:
                dataset_samples.append(record)
                continue

            replacement_index = rng.randrange(counts[dataset_name])
            if replacement_index < samples_per_dataset:
                dataset_samples[replacement_index] = record

    undersized_datasets = {
        dataset_name: count
        for dataset_name, count in counts.items()
        if count < samples_per_dataset
    }
    if undersized_datasets:
        raise ValueError(
            f"Not enough records to sample {samples_per_dataset} per dataset: "
            f"{undersized_datasets}"
        )

    selected_records = [
        record
        for dataset_name in dataset_names
        for record in samples[dataset_name]
    ]
    rng.shuffle(selected_records)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for record in selected_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Eligible non-member records by dataset: {counts}")
    print(f"Saved {len(selected_records)} records to {output_path}")


# get_non_members()
