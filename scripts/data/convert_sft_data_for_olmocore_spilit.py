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

"""Split an SFT dataset and convert both splits to OLMo-core format."""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Literal

from datasets import Dataset, load_dataset

from open_instruct import dataset_transformation, numpy_dataset_conversion, utils


SPLIT_DATASET_NAME = "allenai/Dolci-Instruct-SFT"


@dataclass
class ConvertSFTDataArguments:
    """Arguments for splitting and converting SFT data to OLMo-core format."""

    """Output directory"""
    output_dir: str = field()

    """Directory for the raw train/test Parquet files."""
    split_output_dir: str | None = field(default=None)

    """The name of the dataset to use (via the datasets library)."""
    dataset_name: str | None = field(default=None)

    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer: dict | None = field(default=None)

    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"])

    """The dataset splits to use for training"""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])

    """The list of transform functions to apply to the dataset."""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )

    """The columns to use for the dataset."""
    dataset_target_columns: list[str] = field(
        default_factory=lambda: dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE
    )

    """The mode to use for caching the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"

    """The directory to save the local dataset cache to."""
    dataset_local_cache_dir: str = "local_dataset_cache"

    """The hash of the dataset configuration."""
    dataset_config_hash: str | None = None

    """Whether to skip the cache."""
    dataset_skip_cache: bool = False

    """Maximum sequence length. If not provided, no truncation will be performed."""
    max_seq_length: int | None = field(default=None)

    """Number of examples to process for debugging. 0 means process all examples."""
    num_examples: int = field(default=0)

    """Visualize first token sequence"""
    visualize: bool = field(default=False)

    """Only write the tokenizer config to the output directory"""
    tokenizer_config_only: bool = field(default=False)

    """Resume from previously-written partial files in output_dir, if present."""
    resume: bool = field(default=False)

    """Shuffle seed for reproducible dataset ordering"""
    shuffle_seed: int = field(default=42)


def split_dolci_instruct(args: ConvertSFTDataArguments) -> tuple[list[str], list[str]] | None:
    mixer = list(args.dataset_mixer_list)
    dataset_names = mixer[::2]
    if SPLIT_DATASET_NAME not in dataset_names:
        return None

    dataset_index = dataset_names.index(SPLIT_DATASET_NAME)
    mixer_index = dataset_index * 2
    train_fraction = float(mixer[mixer_index + 1])
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"Dolci-Instruct-SFT split fraction must be between 0 and 1, got {train_fraction}")
    if args.split_output_dir is None:
        raise ValueError("--split_output_dir is required when splitting Dolci-Instruct-SFT")

    splits = args.dataset_mixer_list_splits
    source_split = splits[0] if len(splits) == 1 else splits[dataset_index]
    dataset = load_dataset(SPLIT_DATASET_NAME, split=source_split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")

    split_dataset = dataset.train_test_split(train_size=train_fraction, seed=args.shuffle_seed)
    split_output_dir = pathlib.Path(args.split_output_dir)
    split_output_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_output_dir / "train.parquet"
    test_path = split_output_dir / "test.parquet"
    split_dataset["train"].to_parquet(train_path)
    split_dataset["test"].to_parquet(test_path)

    print(f"Saved train split ({len(split_dataset['train'])} rows): {train_path}")
    print(f"Saved test split ({len(split_dataset['test'])} rows): {test_path}")
    train_mixer = list(mixer)
    test_mixer = list(mixer)
    train_mixer[mixer_index : mixer_index + 2] = [str(train_path), "1.0"]
    test_mixer[mixer_index : mixer_index + 2] = [str(test_path), "1.0"]
    return train_mixer, test_mixer


def main(args: ConvertSFTDataArguments, tc: dataset_transformation.TokenizerConfig) -> None:
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

    output_dir = pathlib.Path(args.output_dir)
    split_mixers = None if args.tokenizer_config_only else split_dolci_instruct(args)
    conversion_jobs = (
        [(output_dir, args.dataset_mixer_list)]
        if split_mixers is None
        else [(output_dir / "train", split_mixers[0]), (output_dir / "test", split_mixers[1])]
    )
    for split_output_dir, dataset_mixer_list in conversion_jobs:
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=split_output_dir,
            dataset_mixer_list=dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
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
    parser = utils.ArgumentParserPlus((ConvertSFTDataArguments, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
