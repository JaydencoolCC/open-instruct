#!/usr/bin/env bash
#SBATCH --partition=4090
#SBATCH --job-name=prepare-dolci-instruct
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
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

PROJECT_ROOT="$(pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/dolci_instruct_sft_tokenized"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/models/huggingface/hub"

echo "=== Dolci Instruct Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Instruct-SFT 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 32768 \
  --visualize True \
  --resume

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
