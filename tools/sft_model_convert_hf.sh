#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
export PYTHONPATH="${PROJECT_ROOT}/OLMo-core-main/src"

/data/home/zhanghx/.conda/envs/olmo3_sft/bin/python \
  "${PROJECT_ROOT}/OLMo-core-main/src/examples/huggingface/convert_checkpoint_to_hf.py" \
  --checkpoint-input-path "${PROJECT_ROOT}/checkpoints/sft/datasize/1k/dolci-instruct-sft-contamination-size-1k-20260904-053339/step2" \
  --huggingface-output-dir /data01/users/zhanghx/data/olmo/datasize/sft_1k \
  --tokenizer /data/home/zhanghx/code/open-instruct/data/sft/dolci_instruct_sft_tokenized_fixed_parts/nonmember/tokenizer \
  --max-sequence-length 32768 \
  --device cpu