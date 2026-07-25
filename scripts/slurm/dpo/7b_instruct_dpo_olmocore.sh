#!/usr/bin/env bash
#SBATCH --partition=A100
#SBATCH --job-name=dpo-olmo3-7b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=96:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=256G                        # 内存
#SBATCH --cpus-per-task=16               # 每任务 16 个 CPU 核

# =================== 环境加载 ===================
set -euo pipefail

echo "=== 开始加载环境 ==="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate /data/home/zhanghx/.conda/envs/olmo3_sft

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"


TORCHRUN="${CONDA_PREFIX}/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    echo "ERROR: torchrun does not exist or is not executable: $TORCHRUN"
    exit 1
fi

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
OLMOCORE_PATH="/data/home/zhanghx/code/open-instruct/OLMo-core-main"
MODEL_NAME="/data/common/LLMs/allenai/Olmo-3-7B-Instruct-SFT"
OUTPUT_ROOT="${PROJECT_ROOT}/checkpoints/dpo"
LOCAL_CACHE_DIR="${PROJECT_ROOT}/data/open_instruct_dataset_cache"

cd "$PROJECT_ROOT"
if [[ -d "${OLMOCORE_PATH}/src" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}:${OLMOCORE_PATH}/src"
else
    export PYTHONPATH="${PROJECT_ROOT}"
fi

export REFERENCE_LOGPROBS_CACHE_PATH="${PROJECT_ROOT}/data/reference_logprobs_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="8"

LR="1e-6"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="olmo3-7b-DPO-olmocore-DPO-${TIMESTAMP}"
WANDB_ENTITY="jaycool"
WANDB_PROJECT="Olmo3-7B-dpo"
export WANDB_API_KEY="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"

mkdir -p "$OUTPUT_ROOT" "$LOCAL_CACHE_DIR" "$REFERENCE_LOGPROBS_CACHE_PATH"

if [[ "$MODEL_NAME" == /* && ! -e "$MODEL_NAME" ]]; then
    echo "WARNING: MODEL_NAME does not exist on this machine: $MODEL_NAME"
    echo "Set MODEL_NAME to a valid local SFT checkpoint before launching."
fi

echo "=== OLMo3 7B DPO OLMo-core Training ==="
echo "Project root: $PROJECT_ROOT"
echo "OLMo-core: $OLMOCORE_PATH"
echo "Model: $MODEL_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "Reference cache: $REFERENCE_LOGPROBS_CACHE_PATH"
echo "GPUs: $GPUS"
# echo "FSDP shard degree: $FSDP_SHARD_DEGREE"
echo "Learning rate: $LR"
echo "Experiment: $EXP_NAME"
echo "W&B entity: $WANDB_ENTITY"
echo "W&B project: $WANDB_PROJECT"
echo "Python: $(which python)"
echo "torchrun: $TORCHRUN"
echo "========================================"

# olmo3_7B 表示自带的模版
"$TORCHRUN" --standalone \
    --nproc-per-node="$GPUS" \
    open_instruct/dpo.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --config_name olmo3_7B \
    --chat_template_name olmo123 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --num_epochs 1 \
    --mixer_list allenai/Dolci-Instruct-DPO 0.99 \
    --seed 42 \
    --logging_steps 1 \
    --activation_memory_budget 0.1 \
    --output_dir "$OUTPUT_ROOT" \
    --local_cache_dir "$LOCAL_CACHE_DIR" \
    --try_launch_beaker_eval_jobs false \
    --push_to_hub false \
    --try_auto_save_to_beaker false \
    --with_tracking \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT"

echo "=== Training complete ==="
echo "Outputs saved under: ${OUTPUT_ROOT}/${EXP_NAME}__42"
