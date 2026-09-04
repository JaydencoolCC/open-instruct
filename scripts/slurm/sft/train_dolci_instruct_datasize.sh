#!/usr/bin/env bash
#SBATCH --partition=A100
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=96:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

set -euo pipefail
# =================== 环境加载 ===================
echo "=== 开始加载环境 ==="
source /data/softwares/anaconda3/2025.12.2/etc/profile.d/conda.sh
conda activate /data/home/zhanghx/.conda/envs/olmo3_sft

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# Path configuration.
PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi
OLMOCORE_PATH="${PROJECT_ROOT}/OLMo-core-main"
DATASET_PATH="${PROJECT_ROOT}/data/sft/datasize/dolci_instruct_sft_tokenized_1k/train"
EVAL_DATASET_PATH="${PROJECT_ROOT}/data/sft/dolci_instruct_sft_tokenized_fixed_parts/nonmember"
BASE_CKPT="/data/common/LLMs/Olmo-3-1025-7B"
SFT_SCRIPT="${OLMOCORE_PATH}/src/scripts/train/sft/Olmo-3-7B-SFT.py"


# Add OLMo-core to Python path
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# Instruct SFT defaults (from OLMo-3 paper Table 47)
RUN_NAME="dolci-instruct-sft-contamination-size-1k-$(date +%Y%m%d-%H%M%S)"
GPUS=8
LEARNING_RATE=8e-5  # 8e-5 for Instruct (higher than Think)
SEQ_LEN=32768
NUM_EPOCHS=2
GLOBAL_BATCH_SIZE=$((SEQ_LEN * 16))  # 16 packed sequences per global batch.
METRICS_COLLECT_INTERVAL=1
SAVE_INTERVAL_STEPS=null  # The current 1,000-example packed split yields one full batch per epoch.
EVAL_INTERVAL_STEPS=100
EVAL_FRACTION=1.0
SAVE_FOLDER="${PROJECT_ROOT}/checkpoints/sft/datasize/1k/${RUN_NAME}"
SKIP_EMPTY_LABEL_BATCH=False

# W&B configuration.
WANDB_ENTITY="jaycool"
WANDB_PROJECT="Olmo3-7B-sft"
WANDB_ENABLED=True
export WANDB_INIT_TIMEOUT=300

# GPU memory optimization
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Dolci Instruct SFT Training ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "OLMo-core: $OLMOCORE_PATH"
echo "Dataset: $DATASET_PATH"
echo "Evaluation dataset: $EVAL_DATASET_PATH"
echo "Base checkpoint: $BASE_CKPT"
echo "GPUs per node: $GPUS"
echo "Learning rate: $LEARNING_RATE"
echo "Sequence length: $SEQ_LEN"
echo "Global batch size: $GLOBAL_BATCH_SIZE tokens"
echo "Epochs: $NUM_EPOCHS"
echo "Metrics collect interval: $METRICS_COLLECT_INTERVAL steps"
echo "Checkpoint save interval: $SAVE_INTERVAL_STEPS steps"
echo "Evaluation interval: $EVAL_INTERVAL_STEPS steps"
echo "Evaluation fraction: $EVAL_FRACTION"
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

python -m torch.distributed.run --nproc-per-node="$GPUS" \
  "$SFT_SCRIPT" train \
  "$RUN_NAME" \
  "$BASE_CKPT" \
  local \
  --seq_len="$SEQ_LEN" \
  --num_nodes=1 \
  --gpus_per_node="$GPUS" \
  --global_batch_size="$GLOBAL_BATCH_SIZE" \
  --dataset_path="$DATASET_PATH" \
#   --eval_dataset_path="$EVAL_DATASET_PATH" \
#   --eval_interval="$EVAL_INTERVAL_STEPS" \
#   --eval_fraction="$EVAL_FRACTION" \
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
  --trainer.callbacks.checkpointer.pre_train_checkpoint=false
  # --train_module.compile_model=false \
  # --train_module.optim.foreach=false

echo "=== Training complete ==="
echo "Checkpoints saved to: $SAVE_FOLDER"
