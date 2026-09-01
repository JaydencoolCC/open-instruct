"""Sample the same number of examples per domain and save them as JSONL."""

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INPUT_DIR = Path("/data01/users/zhanghx/data/Dolci-Instruct-SFT")
OUTPUT_DIR = Path("/data/home/zhanghx/olmo3/dataset/benchmark/sft_v2")
DOMAINS = ("Coding", "Math", "Chat", "Precise IF")
NUM_SAMPLES_PER_DOMAIN = 1_000
SEED = 42


def sample_parquet(
    input_path: Path,
    output_path: Path,
    seed: int,
) -> None:
    parquet_file = pq.ParquetFile(input_path)
    reservoirs: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    domain_counts = {domain: 0 for domain in DOMAINS}
    random_generators = {domain: random.Random(seed + index) for index, domain in enumerate(DOMAINS)}

    row_offset = 0
    for batch in parquet_file.iter_batches(columns=["domain"], batch_size=131_072):
        for local_index, domain in enumerate(batch.column(0).to_pylist()):
            if domain not in reservoirs:
                continue
            domain_counts[domain] += 1
            reservoir = reservoirs[domain]
            if len(reservoir) < NUM_SAMPLES_PER_DOMAIN:
                reservoir.append(row_offset + local_index)
            else:
                replacement_index = random_generators[domain].randrange(domain_counts[domain])
                if replacement_index < NUM_SAMPLES_PER_DOMAIN:
                    reservoir[replacement_index] = row_offset + local_index
        row_offset += batch.num_rows

    insufficient_domains = {
        domain: count for domain, count in domain_counts.items() if count < NUM_SAMPLES_PER_DOMAIN
    }
    if insufficient_domains:
        raise ValueError(
            f"Cannot sample {NUM_SAMPLES_PER_DOMAIN} rows per domain from {input_path}; "
            f"available counts: {insufficient_domains}"
        )

    sampled_indices = sorted(index for reservoir in reservoirs.values() for index in reservoir)
    indices_by_row_group: dict[int, list[int]] = defaultdict(list)
    row_group_start = 0
    sampled_position = 0

    for row_group_index in range(parquet_file.num_row_groups):
        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        row_group_end = row_group_start + row_group_rows
        while sampled_position < len(sampled_indices) and sampled_indices[sampled_position] < row_group_end:
            indices_by_row_group[row_group_index].append(sampled_indices[sampled_position] - row_group_start)
            sampled_position += 1
        row_group_start = row_group_end

    sampled_tables = []
    for row_group_index, local_indices in indices_by_row_group.items():
        row_group = parquet_file.read_row_group(row_group_index)
        sampled_tables.append(row_group.take(pa.array(local_indices, type=pa.int64())))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_table = pa.concat_tables(sampled_tables)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in sampled_table.to_pylist():
            row["prompt"] = row["messages"][0]["content"]
            row["response"] = row["messages"][1]["content"]
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Sampled {NUM_SAMPLES_PER_DOMAIN} rows for each of {DOMAINS} "
        f"from {input_path} -> {output_path} ({sampled_table.num_rows} rows total)"
    )


def main() -> None:
    for split_offset, (split, output_name) in enumerate(
        (("train", "member.jsonl"), ("test", "non_member.jsonl"))
    ):
        sample_parquet(
            input_path=INPUT_DIR / f"{split}.parquet",
            output_path=OUTPUT_DIR / output_name,
            seed=SEED + split_offset,
        )


if __name__ == "__main__":
    main()
