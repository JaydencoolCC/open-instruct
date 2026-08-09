"""Save the Dolci-Instruct-DPO holdout for DPO benchmarking.

The DPO training script samples training data with the DatasetConfig logic in
open_instruct.dataset_transformation: for a fractional mixer value it computes
int(fraction * dataset_size), then draws that many indices with
np.random.RandomState(seed).choice(..., replace=False).

This script starts from the exact complement of the training indices. It then
removes any held-out rows whose full raw content already appears in the training
indices, because the source dataset can contain duplicate rows at different
indices.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset


DEFAULT_DATASET_NAME = "allenai/Dolci-Instruct-DPO"
DEFAULT_OUTPUT_DIR = Path("/data/home/zhanghx/olmo3/dataset/benchmark")
DEFAULT_OUTPUT_NAME = "dolci_instruct_dpo_test_0p01.jsonl"
DEFAULT_REVISION = "main"
DEFAULT_SEED = 42
DEFAULT_SPLIT = "train"
DEFAULT_TRAIN_FRACTION = 0.99


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_jsonl(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _sha256_ints(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_indices(dataset_size: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be between 0.0 and 1.0, got {train_fraction}")

    train_size = int(train_fraction * dataset_size)
    train_indices = np.random.RandomState(seed).choice(dataset_size, size=train_size, replace=False).tolist()
    train_index_set = set(train_indices)
    test_indices = [idx for idx in range(dataset_size) if idx not in train_index_set]

    overlap_count = len(train_index_set.intersection(test_indices))
    assert overlap_count == 0, f"Train/test overlap detected: {overlap_count}"
    assert len(train_indices) + len(test_indices) == dataset_size, (
        f"Split sizes do not cover dataset: train={len(train_indices)}, "
        f"test={len(test_indices)}, total={dataset_size}"
    )
    assert len(train_indices) == train_size, f"Expected {train_size} train indices, got {len(train_indices)}"

    return train_indices, test_indices


def _remove_content_overlaps(
    dataset: Dataset, train_indices: list[int], test_indices: list[int]
) -> tuple[list[int], int, int]:
    train_index_set = set(train_indices)
    test_index_set = set(test_indices)
    train_hashes: set[str] = set()
    test_candidates: list[tuple[int, str]] = []

    for idx, row in enumerate(dataset):
        row_hash = _row_hash(row)
        if idx in train_index_set:
            train_hashes.add(row_hash)
        elif idx in test_index_set:
            test_candidates.append((idx, row_hash))
        else:
            raise AssertionError(f"Index {idx} is in neither train nor test split")

    content_overlap_hashes = {row_hash for _, row_hash in test_candidates if row_hash in train_hashes}
    filtered_test_indices = [idx for idx, row_hash in test_candidates if row_hash not in train_hashes]
    removed_count = len(test_candidates) - len(filtered_test_indices)

    assert removed_count == sum(1 for _, row_hash in test_candidates if row_hash in train_hashes)
    return filtered_test_indices, removed_count, len(content_overlap_hashes)


def _metadata(
    *,
    dataset_name: str,
    split: str,
    revision: str,
    seed: int,
    train_fraction: float,
    dataset_size: int,
    train_indices: list[int],
    candidate_test_indices: list[int],
    test_indices: list[int],
    output_path: Path,
    content_overlap_row_count: int,
    content_overlap_unique_count: int,
) -> dict[str, Any]:
    train_index_set = set(train_indices)
    test_index_set = set(test_indices)
    return {
        "dataset_name": dataset_name,
        "split": split,
        "revision": revision,
        "seed": seed,
        "train_fraction": train_fraction,
        "dataset_size": dataset_size,
        "train_size": len(train_indices),
        "candidate_test_size": len(candidate_test_indices),
        "test_size": len(test_indices),
        "overlap_count": len(train_index_set.intersection(test_index_set)),
        "content_overlap_row_count": content_overlap_row_count,
        "content_overlap_unique_count": content_overlap_unique_count,
        "train_indices_sha256": _sha256_ints(train_indices),
        "candidate_test_indices_sha256": _sha256_ints(candidate_test_indices),
        "test_indices_sha256": _sha256_ints(test_indices),
        "output_path": str(output_path),
        "selection_note": (
            "Candidate test indices are the exact complement of the training indices produced by "
            "np.random.RandomState(seed).choice(dataset_size, size=int(train_fraction * dataset_size), "
            "replace=False). Final test indices remove any candidate rows whose full raw content hash "
            "also appears in the training indices."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the exact holdout complement for allenai/Dolci-Instruct-DPO DPO training."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / args.output_name
    meta_path = output_path.with_suffix(".meta.json")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"{meta_path} already exists. Pass --overwrite to replace it.")

    dataset = load_dataset(args.dataset_name, split=args.split, revision=args.revision)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(dataset).__name__}")

    train_indices, candidate_test_indices = _split_indices(len(dataset), args.train_fraction, args.seed)
    test_indices, content_overlap_row_count, content_overlap_unique_count = _remove_content_overlaps(
        dataset, train_indices, candidate_test_indices
    )
    test_dataset = dataset.select(test_indices)
    _write_jsonl(test_dataset, output_path)
    written_rows = sum(1 for _ in output_path.open(encoding="utf-8"))
    assert written_rows == len(test_indices), f"Expected {len(test_indices)} JSONL rows, wrote {written_rows}"

    metadata = _metadata(
        dataset_name=args.dataset_name,
        split=args.split,
        revision=args.revision,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dataset_size=len(dataset),
        train_indices=train_indices,
        candidate_test_indices=candidate_test_indices,
        test_indices=test_indices,
        output_path=output_path,
        content_overlap_row_count=content_overlap_row_count,
        content_overlap_unique_count=content_overlap_unique_count,
    )
    metadata["jsonl_rows"] = written_rows
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: {args.dataset_name}/{args.split}@{args.revision}")
    print(f"Train size: {len(train_indices)}")
    print(f"Candidate test size: {len(candidate_test_indices)}")
    print(f"Test size: {len(test_indices)}")
    print(f"Overlap count: {metadata['overlap_count']}")
    print(f"Removed content-overlap rows: {content_overlap_row_count}")
    print(f"Saved JSONL: {output_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
