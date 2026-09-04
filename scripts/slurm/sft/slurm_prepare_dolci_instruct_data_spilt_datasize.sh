#!/usr/bin/env bash
#SBATCH --partition=4090
#SBATCH --job-name=prepare-dolci-instruct
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=32G                        # 内存
#SBATCH --cpus-per-task=8               # 每任务 8 个 CPU 核

set -euo pipefail

# =================== 环境加载 ===================
echo "=== 开始加载环境 ==="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate /data/home/zhanghx/.conda/envs/olmo3_sft

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"
# Converts allenai/Dolci-Instruct-SFT dataset to OLMo-core tokenized format.
#
# Usage:
#   mkdir -p logs  # Create logs directory first
#   sbatch prepare_dolci_instruct_data.sh
#
# Resume after interruption:
#   Just resubmit the same script - it will automatically resume from checkpoint.
#
export HF_ENDPOINT=https://hf-mirror.com

PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
OUTPUT_DIR="$PROJECT_ROOT/data/sft/dolci_instruct_sft_tokenized_fixed_parts/"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MEMBER_JSONL="/data/home/zhanghx/olmo3/dataset/benchmark/sft_v2/member_250.jsonl"
NONMEMBER_JSONL="/data/home/zhanghx/olmo3/dataset/benchmark/sft_v2/nonmember_250.jsonl"
SEED=42

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/models/huggingface/hub"

mkdir -p "$OUTPUT_DIR"
SPLIT_OUTPUT_DIR="$(mktemp -d "$PROJECT_ROOT/data/sft/.dolci_split.XXXXXX")"
trap 'rm -rf "$SPLIT_OUTPUT_DIR"' EXIT

echo "=== Dolci Instruct Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUTPUT_DIR"
echo "Raw split output: $SPLIT_OUTPUT_DIR"
echo "Member benchmark: $MEMBER_JSONL"
echo "Nonmember benchmark: $NONMEMBER_JSONL"
echo "Seed: $SEED"
echo "========================================"

cd "$PROJECT_ROOT"

# The mixer value below is only a compatibility placeholder; fixed benchmark IDs define all three parts.
python scripts/data/convert_sft_data_for_olmocore_spilit_datasize.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Instruct-SFT 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --split_output_dir "$SPLIT_OUTPUT_DIR" \
  --member_jsonl "$MEMBER_JSONL" \
  --nonmember_jsonl "$NONMEMBER_JSONL" \
  --shuffle_seed "$SEED" \
  --max_seq_length 32768 \
  --visualize True \
  --resume

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
