#!/usr/bin/env bash
#SBATCH --partition=A100
#SBATCH --job-name=train-dolci-instruct
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=96:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=256G                        # 内存
#SBATCH --cpus-per-task=16               # 每任务 8 个 CPU 核

set -euo pipefail


# =================== 环境加载 ===================
echo "=== 开始加载环境 ==="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate /data/home/zhanghx/.conda/envs/olmo3_sft

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# Path configuration.
PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
OLMOCORE_PATH="${PROJECT_ROOT}/OLMo-core-main"
DATASET_PATH="${PROJECT_ROOT}/data/dolci_instruct_sft_tokenized"
BASE_CKPT="/data/common/LLMs/allenai/Olmo-3-1025-7B"
SFT_SCRIPT="${OLMOCORE_PATH}/src/scripts/train/sft/Olmo-3-7B-SFT.py"

if [[ ! -f "$SFT_SCRIPT" ]]; then
    echo "ERROR: SFT_SCRIPT does not exist: $SFT_SCRIPT"
    exit 1
fi

# Add OLMo-core to Python path
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# Instruct SFT defaults (from OLMo-3 paper Table 47)
timenow=$(date +%Y%m%d_%H%M%S)
RUN_NAME="dolci-instruct-sft-contamination-A100-${timenow}"
GPUS=8
LEARNING_RATE=8e-5  # 8e-5 for Instruct (higher than Think)
SEQ_LEN=32768
NUM_EPOCHS=2
GLOBAL_BATCH_SIZE=$((SEQ_LEN * 32))  # 1 sequence per GPU.
METRICS_COLLECT_INTERVAL=1
SAVE_INTERVAL_STEPS=1628  # 1 epoch for the current packed Dolci Instruct + benchmark dataset.
SAVE_FOLDER="./checkpoints/${RUN_NAME}"
SKIP_EMPTY_LABEL_BATCH=False

# W&B configuration.
WANDB_ENTITY="jaycool"
WANDB_PROJECT="Olmo3-7B-sft"
WANDB_ENABLED=True
export WANDB_API_KEY="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"
export WANDB_INIT_TIMEOUT=300

# GPU memory optimization
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Dolci Instruct SFT Training ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "OLMo-core: $OLMOCORE_PATH"
echo "Dataset: $DATASET_PATH"
echo "Base checkpoint: $BASE_CKPT"
echo "GPUs per node: $GPUS"
echo "Learning rate: $LEARNING_RATE"
echo "Sequence length: $SEQ_LEN"
echo "Global batch size: $GLOBAL_BATCH_SIZE tokens"
echo "Epochs: $NUM_EPOCHS"
echo "Metrics collect interval: $METRICS_COLLECT_INTERVAL steps"
echo "Checkpoint save interval: $SAVE_INTERVAL_STEPS steps"
echo "Skip empty label batch: $SKIP_EMPTY_LABEL_BATCH"
echo "W&B enabled: $WANDB_ENABLED"
echo "W&B entity: $WANDB_ENTITY"
echo "W&B project: $WANDB_PROJECT"
echo "W&B init timeout: $WANDB_INIT_TIMEOUT seconds"
echo "===================================="

if [[ "$WANDB_ENABLED" == "True" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_ENABLED=True but WANDB_API_KEY is not set."
    exit 1
fi

mkdir -p "$SAVE_FOLDER"

torchrun --nproc-per-node="$GPUS" \
  "$SFT_SCRIPT" train \
  "$RUN_NAME" \
  "$BASE_CKPT" \
  local \
  --seq_len="$SEQ_LEN" \
  --num_nodes=1 \
  --gpus_per_node="$GPUS" \
  --global_batch_size="$GLOBAL_BATCH_SIZE" \
  --dataset_path="$DATASET_PATH" \
  --trainer.save_folder="$SAVE_FOLDER" \
  --train_module.optim.lr="$LEARNING_RATE" \
  --train_module.skip_empty_label_batch="$SKIP_EMPTY_LABEL_BATCH" \
  --trainer.max_duration.value="$NUM_EPOCHS" \
  --trainer.metrics_collect_interval="$METRICS_COLLECT_INTERVAL" \
  --trainer.callbacks.checkpointer.save_interval="$SAVE_INTERVAL_STEPS" \
  --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
  --trainer.cancel_check_interval=5 \
  --trainer.callbacks.wandb.enabled="$WANDB_ENABLED" \
  --trainer.callbacks.wandb.entity="$WANDB_ENTITY" \
  --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
  --trainer.callbacks.wandb.name="$RUN_NAME" \
  --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
  # --train_module.compile_model=false \
  # --train_module.optim.foreach=false

echo "=== Training complete ==="
echo "Checkpoints saved to: $SAVE_FOLDER"
