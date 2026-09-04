#!/usr/bin/env python3
"""Sample tokenized remaining documents and prepend the fixed member set."""

import argparse
import csv
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/data/home/zhanghx/code/open-instruct/data/sft/dolci_instruct_sft_tokenized_fixed_parts")
DEFAULT_OUTPUT_ROOT = Path("/data/home/zhanghx/code/open-instruct/data/sft/datasize/")
TOKEN_DTYPE = np.uint32


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    part = path.stem.rsplit("_", 1)[-1]
    labels_path = path.with_name(f"labels_mask_part_{part}.npy")
    tokens = np.memmap(
        path,
        mode="r",
        dtype=TOKEN_DTYPE,
        shape=(path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize,),
    )
    labels = np.memmap(labels_path, mode="r", dtype=np.bool_, shape=(labels_path.stat().st_size,))
    if len(tokens) != len(labels):
        raise ValueError(f"Token/label length mismatch in {path}")
    return tokens, labels

def read_part(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    part = path.stem.rsplit("_", 1)[-1]
    labels_path = path.with_name(f"labels_mask_part_{part}.npy")
    boundaries_path = path.with_name(f"token_ids_part_{part}.csv.gz")
    # OLMo-core writes raw memmaps with a .npy suffix (no NumPy header).
    tokens, labels = load_arrays(path)
    boundaries = []
    with gzip.open(boundaries_path, "rt", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                boundaries.append((int(row[0]), int(row[1])))
    if len(tokens) != len(labels):
        raise ValueError(f"Token/label length mismatch in {path}")
    if boundaries and boundaries[-1][1] > len(tokens):
        raise ValueError(f"Boundary exceeds token length in {path}")
    return tokens, labels, boundaries


def collect_refs(directory: Path, label: str) -> list[tuple[Path, int, int]]:
    refs = []
    token_paths = sorted(directory.glob("token_ids_part_*.npy"))
    started = time.monotonic()
    for index, token_path in enumerate(token_paths, start=1):
        _, _, boundaries = read_part(token_path)
        refs.extend((token_path, start, end) for start, end in boundaries)
        elapsed = time.monotonic() - started
        rate = len(refs) / elapsed if elapsed else 0.0
        print(
            f"[{label}] scanned part {index}/{len(token_paths)}; "
            f"documents={len(refs):,}; rate={rate:,.0f} docs/s",
            file=sys.stderr,
            flush=True,
        )
    print(f"[{label}] scan complete: {len(refs):,} documents", file=sys.stderr, flush=True)
    return refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one tokenized SFT train set at a requested size.")
    parser.add_argument("--train_size", type=int, required=True, help="Final train document count, including members")
    parser.add_argument("--member_dir", type=Path, default=DEFAULT_ROOT / "member")
    parser.add_argument("--remaining_dir", type=Path, default=DEFAULT_ROOT / "remaining")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size <= 0:
        raise ValueError("--train_size must be positive")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"dolci_instruct_sft_tokenized_{args.train_size}" / "train"
    for directory in (args.member_dir, args.remaining_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    output_files = list(args.output_dir.glob("token_ids_part_*.npy")) if args.output_dir.exists() else []
    if output_files and not args.overwrite:
        raise FileExistsError(f"Output already exists in {args.output_dir}; pass --overwrite")
    if args.overwrite and args.output_dir.exists():
        for pattern in ("token_ids_part_*.npy", "token_ids_part_*.csv.gz", "labels_mask_part_*.npy", "metadata.json"):
            for path in args.output_dir.glob(pattern):
                path.unlink()

    member_refs = collect_refs(args.member_dir, "member")
    remaining_refs = collect_refs(args.remaining_dir, "remaining")
    if args.train_size < len(member_refs):
        raise ValueError(f"train_size={args.train_size} is smaller than {len(member_refs)} member documents")
    extra_count = args.train_size - len(member_refs)
    if extra_count > len(remaining_refs):
        raise ValueError(f"Need {extra_count} remaining documents, only {len(remaining_refs)} available")

    rng = random.Random(args.seed)
    rng.shuffle(remaining_refs)
    selected_refs = member_refs + remaining_refs[:extra_count]
    total_tokens = sum(end - start for _, start, end in selected_refs)
    print(
        f"[select] selected {len(selected_refs):,} documents "
        f"({total_tokens:,} tokens); writing output...",
        file=sys.stderr,
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_dtype = TOKEN_DTYPE
    token_path = args.output_dir / "token_ids_part_0000.npy"
    labels_path = args.output_dir / "labels_mask_part_0000.npy"
    boundaries_path = args.output_dir / "token_ids_part_0000.csv.gz"
    # Keep the raw memmap format expected by OLMo-core (despite the .npy suffix).
    tokens_out = np.memmap(token_path, mode="w+", dtype=token_dtype, shape=(total_tokens,))
    labels_out = np.memmap(labels_path, mode="w+", dtype=np.bool_, shape=(total_tokens,))

    position = 0
    source_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    started = time.monotonic()
    with gzip.open(boundaries_path, "wt", newline="") as boundary_file:
        writer = csv.writer(boundary_file)
        for index, (source_path, start, end) in enumerate(selected_refs, start=1):
            if source_path not in source_cache:
                # Boundary files were already read during collect_refs; do not
                # rescan compressed boundaries for every source switch.
                source_cache[source_path] = load_arrays(source_path)
            source_tokens, source_labels = source_cache[source_path]
            length = end - start
            tokens_out[position : position + length] = source_tokens[start:end]
            labels_out[position : position + length] = source_labels[start:end]
            writer.writerow((position, position + length))
            position += length
            if index == 1 or index % 500 == 0 or index == len(selected_refs):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                print(
                    f"[write] {index:,}/{len(selected_refs):,} "
                    f"({index / len(selected_refs):.1%}); "
                    f"tokens={position:,}; rate={rate:,.1f} docs/s",
                    file=sys.stderr,
                    flush=True,
                )
    tokens_out.flush()
    labels_out.flush()
    del tokens_out, labels_out
    assert position == total_tokens

    metadata = {
        "train_size": len(selected_refs),
        "member_size": len(member_refs),
        "remaining_sample_size": extra_count,
        "total_tokens": total_tokens,
        "seed": args.seed,
        "member_dir": str(args.member_dir),
        "remaining_dir": str(args.remaining_dir),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(selected_refs)} documents and {total_tokens} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()
