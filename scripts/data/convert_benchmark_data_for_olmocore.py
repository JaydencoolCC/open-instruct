# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beaker-py>=1.32.2,<2.0",
#     "datasets>=4.0.0",
#     "numpy<2",
#     "ray[default]>=2.44.1",
#     "rich>=13.7.0",
#     "tqdm",
#     "transformers>=4.52.4",
#     "torch>=2.7.0,<2.8",
# ]
# ///

"""Convert local benchmark train JSONL files to OLMo-core SFT mmap format.

The benchmark source files are expected at:
    /data/home/zhanghx/olmo3/dataset/benchmark/*/train.jsonl

Each source row must contain ``question`` and ``solution`` fields. This script first
rewrites them to standard SFT ``messages`` JSONL, then reuses the same conversion
logic as ``scripts/data/convert_sft_data_for_olmocore.py``.
"""

import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(PROJECT_ROOT / "models/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / "data/huggingface")
os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT / "models/huggingface/hub")

# Import after HF env setup so datasets uses the project-local cache.
from open_instruct import dataset_transformation, numpy_dataset_conversion, utils  # noqa: E402

DEFAULT_BENCHMARK_INPUT_DIR = "/data/home/zhanghx/olmo3/dataset/benchmark"
DEFAULT_OUTPUT_DIR = "/data/home/zhanghx/code/open-instruct/data/benchmark"
DEFAULT_TOKENIZER = "allenai/Olmo-3-7B-Instruct-SFT"
LOCAL_DOLCI_TOKENIZER = PROJECT_ROOT / "data/dolci_instruct_sft_tokenized/tokenizer"
PREPARED_JSONL_DIRNAME = "_prepared_jsonl"


@dataclass
class ConvertBenchmarkDataArguments:
    """Arguments for converting benchmark JSONL data to OLMo-core format."""

    benchmark_input_dir: str = DEFAULT_BENCHMARK_INPUT_DIR
    """Directory containing benchmark subdirectories with train.jsonl files."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    """Output directory for OLMo-core mmap files."""

    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """Transform functions applied after benchmark JSONL has been rewritten to messages."""

    dataset_target_columns: list[str] = field(
        default_factory=lambda: dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE
    )
    """Columns to keep after tokenization."""

    dataset_cache_mode: Literal["hf", "local"] = "local"
    """Cache mode for transformed datasets."""

    dataset_local_cache_dir: str = "local_dataset_cache"
    """Directory for local dataset transformation cache."""

    dataset_config_hash: str | None = None
    """Optional cache hash override."""

    dataset_skip_cache: bool = False
    """Whether to skip the transformed dataset cache."""

    max_seq_length: int = 32768
    """Maximum sequence length for SFT tokenization."""

    num_examples: int = 0
    """Number of examples to process for debugging. 0 means process all examples."""

    visualize: bool = False
    """Visualize the first token sequence."""

    tokenizer_config_only: bool = False
    """Only write tokenizer config to the output directory."""

    resume: bool = False
    """Resume from partial output files if present."""

    shuffle_seed: int = 42
    """Shuffle seed for reproducible dataset ordering."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _discover_train_jsonl_files(input_dir: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(input_dir.glob("*/train.jsonl"))
    if not files:
        raise FileNotFoundError(f"No benchmark train.jsonl files found under {input_dir}")
    return files


def _prepare_messages_jsonl(train_jsonl_files: list[pathlib.Path], prepared_dir: pathlib.Path) -> list[str]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_paths: list[str] = []

    for source_path in train_jsonl_files:
        dataset_name = source_path.parent.name
        prepared_path = prepared_dir / f"{dataset_name}.train.jsonl"
        num_rows = 0

        with source_path.open(encoding="utf-8") as source, prepared_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)
                question = _text(row.get("question"))
                solution = _text(row.get("solution"))
                if not question or not solution:
                    raise ValueError(
                        f"{source_path}:{line_number} must contain non-empty 'question' and 'solution' fields"
                    )

                target.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": solution},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                num_rows += 1

        if num_rows == 0:
            raise ValueError(f"{source_path} did not contain any usable examples")
        prepared_paths.append(str(prepared_path))

    return prepared_paths


def _build_dataset_mixer_list(prepared_paths: list[str]) -> list[str]:
    dataset_mixer_list: list[str] = []
    for path in prepared_paths:
        dataset_mixer_list.extend([path, "1.0"])
    return dataset_mixer_list


def _default_tokenizer_name_or_path() -> str:
    if LOCAL_DOLCI_TOKENIZER.exists():
        return str(LOCAL_DOLCI_TOKENIZER)
    return DEFAULT_TOKENIZER


def main(args: ConvertBenchmarkDataArguments, tc: dataset_transformation.TokenizerConfig) -> None:
    if tc.tokenizer_name_or_path is None:
        tc.tokenizer_name_or_path = _default_tokenizer_name_or_path()

    output_dir = pathlib.Path(args.output_dir)
    benchmark_input_dir = pathlib.Path(args.benchmark_input_dir)
    prepared_dir = output_dir / PREPARED_JSONL_DIRNAME

    train_jsonl_files = _discover_train_jsonl_files(benchmark_input_dir)
    prepared_paths = _prepare_messages_jsonl(train_jsonl_files, prepared_dir)
    dataset_mixer_list = _build_dataset_mixer_list(prepared_paths)

    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if utils.is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    transform_fn_args = []
    for fn_name in args.dataset_transform_fn:
        if fn_name == "sft_tulu_tokenize_and_truncate_v1":
            transform_fn_args.append({"max_seq_length": args.max_seq_length})
        else:
            transform_fn_args.append({})

    numpy_dataset_conversion.convert_hf_to_numpy_sft(
        output_dir=output_dir,
        dataset_mixer_list=dataset_mixer_list,
        dataset_mixer_list_splits=["train"],
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_target_columns=args.dataset_target_columns,
        max_seq_length=args.max_seq_length,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        dataset_config_hash=args.dataset_config_hash,
        shuffle_seed=args.shuffle_seed,
        resume=args.resume,
        visualize=args.visualize,
        tokenizer_config_only=args.tokenizer_config_only,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus((ConvertBenchmarkDataArguments, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
