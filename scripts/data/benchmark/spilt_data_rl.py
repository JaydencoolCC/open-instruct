"""Save the Dolci-Instruct-RL train/test split for RL training and benchmarking.

The RL training loader samples a fractional dataset mixer value by computing
``int(fraction * dataset_size)`` and passing that size to
``np.random.RandomState(dataset_config_seed).choice(..., replace=False)``.

This script saves those training rows in the exact same index order, starts the
benchmark split from their exact complement, and removes held-out rows whose full
raw content also occurs in the training indices. The content check matters because
duplicate rows can exist at different source indices.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk


DEFAULT_DATASET_NAME = "allenai/Dolci-Instruct-RL"
DEFAULT_OUTPUT_DIR = Path("/home/zhanghx/benchmark/datasets/Dolci-Instruct-RL")
DEFAULT_TRAIN_OUTPUT_NAME = "dolci_instruct_rl_train"
DEFAULT_TEST_OUTPUT_NAME = "dolci_instruct_rl_test"
DEFAULT_META_OUTPUT_NAME = "dolci_instruct_rl_split.meta.json"
DEFAULT_REVISION = "main"
DEFAULT_SEED = 42
DEFAULT_SPLIT = "train"
DEFAULT_TRAIN_FRACTION = 0.9


def _save_dataset(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))


def _remove_existing_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


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
    # Keep this expression aligned with DatasetConfig.select_samples. In
    # particular, choice() order is the order seen by the RL training dataset.
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
    train_output_path: Path,
    test_output_path: Path,
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
        "train_output_path": str(train_output_path),
        "test_output_path": str(test_output_path),
        "selection_note": (
            "Training indices exactly reproduce DatasetConfig.select_samples: "
            "np.random.RandomState(seed).choice(dataset_size, "
            "size=int(train_fraction * dataset_size), replace=False), including returned index order. "
            "Candidate test indices are the complement in original dataset order. Final test indices "
            "remove rows whose full raw content hash also appears in the training indices."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the exact train/test split for allenai/Dolci-Instruct-RL training."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_output_name", default=DEFAULT_TRAIN_OUTPUT_NAME)
    parser.add_argument("--test_output_name", default=DEFAULT_TEST_OUTPUT_NAME)
    parser.add_argument("--meta_output_name", default=DEFAULT_META_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_output_path = args.output_dir / args.train_output_name
    test_output_path = args.output_dir / args.test_output_name
    meta_path = args.output_dir / args.meta_output_name

    output_paths = [train_output_path, test_output_path, meta_path]
    duplicate_paths = {path for path in output_paths if output_paths.count(path) > 1}
    if duplicate_paths:
        raise ValueError(f"Output paths must be distinct, got duplicates: {sorted(map(str, duplicate_paths))}")
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not args.overwrite:
        existing = ", ".join(map(str, existing_paths))
        raise FileExistsError(f"Output file(s) already exist: {existing}. Pass --overwrite to replace them.")

    dataset = load_dataset(args.dataset_name, split=args.split, revision=args.revision)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(dataset).__name__}")

    train_indices, candidate_test_indices = _split_indices(len(dataset), args.train_fraction, args.seed)
    test_indices, content_overlap_row_count, content_overlap_unique_count = _remove_content_overlaps(
        dataset, train_indices, candidate_test_indices
    )
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    if args.overwrite:
        for path in output_paths:
            _remove_existing_output(path)

    _save_dataset(train_dataset, train_output_path)
    _save_dataset(test_dataset, test_output_path)
    saved_train_rows = len(load_from_disk(str(train_output_path)))
    saved_test_rows = len(load_from_disk(str(test_output_path)))
    assert saved_train_rows == len(train_indices), (
        f"Expected {len(train_indices)} saved train rows, found {saved_train_rows}"
    )
    assert saved_test_rows == len(test_indices), (
        f"Expected {len(test_indices)} saved test rows, found {saved_test_rows}"
    )

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
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        content_overlap_row_count=content_overlap_row_count,
        content_overlap_unique_count=content_overlap_unique_count,
    )
    metadata["train_rows"] = saved_train_rows
    metadata["test_rows"] = saved_test_rows
    metadata["storage_format"] = "huggingface_save_to_disk"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: {args.dataset_name}/{args.split}@{args.revision}")
    print(f"Train size: {len(train_indices)}")
    print(f"Candidate test size: {len(candidate_test_indices)}")
    print(f"Test size: {len(test_indices)}")
    print(f"Overlap count: {metadata['overlap_count']}")
    print(f"Removed content-overlap rows: {content_overlap_row_count}")
    print(f"Saved train dataset: {train_output_path}")
    print(f"Saved test dataset: {test_output_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
