#!/usr/bin/env bash
#SBATCH --partition=wei_gpu
#SBATCH --job-name=olmo3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=230
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail

echo "=== 开始加载环境 ==="
source /home/software/miniconda3/bin/activate
conda activate /home/zhanghx/.conda/envs/olmo_rl

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT="0.0.0+local"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export TRITON_CACHE_DIR="/tmp/zhanghx/triton_cache"
PYTHON="/home/zhanghx/.conda/envs/olmo_rl/bin/python"
RAY="/home/zhanghx/.conda/envs/olmo_rl/bin/ray"

export CUDA_HOME=/home/software/cuda/13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/software/cuda/13.2/lib64"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/home/zhanghx/code/open-instruct"
OLMOCORE_PATH="${PROJECT_ROOT}/OLMo-core-main"

MODEL_NAME="/home/zhanghx/models/allenai/Olmo-3-7B-Instruct-DPO"
OUTPUT_ROOT="${PROJECT_ROOT}/checkpoints/rl"
LOCAL_CACHE_DIR="${PROJECT_ROOT}/data/open_instruct_dataset_cache"
ROLLOUTS_SAVE_PATH="${PROJECT_ROOT}/data/rollouts"

# DATASET_MIX="allenai/Dolci-Instruct-RL 0.9"
DATASET_MIX="/home/zhanghx/benchmark/datasets/Dolci-Instruct-RL/dolci_instruct_rl_train 1.0"

export OPENAI_API_KEY="sk-nmvodepbsrnyeuqcefmsqqodvpoyjsruenchwstajeknozuy"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
# LLM_JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"
LLM_JUDGE_MODEL="openai/Qwen/Qwen3-30B-A3B-Instruct-2507"

CODE_API_HOST="0.0.0.0"
CODE_API_PORT="1234"
CODE_API_WORKERS_PER_NODE="64"
RAY_CPUS_PER_NODE=$((SLURM_CPUS_PER_TASK - CODE_API_WORKERS_PER_NODE))
export CODE_API_BASE="http://127.0.0.1:${CODE_API_PORT}"

NUM_GPUS="32"
NUM_LEARNERS_PER_NODE="8"
VLLM_NUM_ENGINES="24"
VLLM_TENSOR_PARALLEL_SIZE="1"
VLLM_GPU_MEMORY_UTILIZATION="0.9"
DEEPSPEED_STAGE="3"
LR="1e-6"
SEED="42"
VLLM_TOTAL_GPUS=$((VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE))
NUM_LEARNER_GPUS=$((NUM_GPUS - VLLM_TOTAL_GPUS))
MIN_RAY_CPUS=$((NUM_LEARNER_GPUS * 4 + VLLM_TOTAL_GPUS + 3))
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="olmo3-7b-GRPO-${TIMESTAMP}"
CHECKPOINT_STATE_DIR="${OUTPUT_ROOT}/checkpoint_states/${EXP_NAME}"
WANDB_ENTITY="jaycool"
WANDB_PROJECT="Olmo3-7B-rl"

export WANDB_INIT_TIMEOUT="300"

mkdir -p "$OUTPUT_ROOT" "$LOCAL_CACHE_DIR" "$ROLLOUTS_SAVE_PATH" "$TRITON_CACHE_DIR" logs

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON"
    exit 1
fi
if [[ -d "${OLMOCORE_PATH}/src" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}:${OLMOCORE_PATH}/src"
else
    export PYTHONPATH="${PROJECT_ROOT}"
fi

echo "=== OLMo3 7B GRPO RL Training ==="
echo "Project root: $PROJECT_ROOT"
echo "OLMo-core: $OLMOCORE_PATH"
echo "Model: $MODEL_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoint state dir: $CHECKPOINT_STATE_DIR"
echo "Dataset cache: $LOCAL_CACHE_DIR"
echo "Rollouts path: $ROLLOUTS_SAVE_PATH"
echo "Triton cache: $TRITON_CACHE_DIR"
echo "GPUs requested: $NUM_GPUS"
echo "Learners per node: $NUM_LEARNERS_PER_NODE"
echo "Learner GPUs: $NUM_LEARNER_GPUS"
echo "vLLM engines: $VLLM_NUM_ENGINES"
echo "vLLM TP size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "Minimum Ray CPUs needed: $MIN_RAY_CPUS"
echo "Learning rate: $LR"
echo "Experiment: $EXP_NAME"
echo "Judge model: $LLM_JUDGE_MODEL"
echo "OpenAI-compatible API base: $OPENAI_API_BASE"
echo "Code API: ${CODE_API_BASE}/test_program"
echo "Code API workers per node: $CODE_API_WORKERS_PER_NODE"
echo "Ray CPUs per node: $RAY_CPUS_PER_NODE"
echo "W&B entity: $WANDB_ENTITY"
echo "W&B project: $WANDB_PROJECT"
echo "W&B init timeout: $WANDB_INIT_TIMEOUT seconds"
echo "Python: $PYTHON"
echo "=================================="

HEAD_NODE="$(hostname -s)"
HEAD_IP="$(getent hosts "$HEAD_NODE" | awk 'NR == 1 {print $1}')"
RAY_ADDRESS="${HEAD_IP}:8888"

cleanup() {
    if [[ -n "${CODE_API_PID:-}" ]]; then
        kill "$CODE_API_PID" >/dev/null 2>&1 || true
        wait "$CODE_API_PID" >/dev/null 2>&1 || true
    fi
    "$RAY" stop --force >/dev/null 2>&1 || true
    if [[ -n "${WORKER_PID:-}" ]]; then
        kill "$WORKER_PID" >/dev/null 2>&1 || true
        wait "$WORKER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

start_head_code_api() {
    local attempt

    echo "Starting Code API on $HEAD_NODE"
    "$PYTHON" -m uvicorn open_instruct.code_utils.api:app \
        --host "$CODE_API_HOST" --port "$CODE_API_PORT" \
        --workers "$CODE_API_WORKERS_PER_NODE" \
        --log-level warning --no-access-log &
    CODE_API_PID=$!

    for attempt in {1..30}; do
        if curl --fail --silent --max-time 2 "${CODE_API_BASE}/health" >/dev/null; then
            echo "Code API is healthy on $HEAD_NODE"
            return 0
        fi
        if ! kill -0 "$CODE_API_PID" >/dev/null 2>&1; then
            wait "$CODE_API_PID" || true
            echo "ERROR: Code API exited on $HEAD_NODE" >&2
            return 1
        fi
        sleep 1
    done

    echo "ERROR: Code API did not become healthy on $HEAD_NODE" >&2
    return 1
}

start_head_code_api

echo "Starting Ray head at $RAY_ADDRESS"
"$RAY" stop --force >/dev/null 2>&1 || true
"$RAY" start --head \
    --node-ip-address="$HEAD_IP" \
    --port=8888 \
    --num-cpus="$RAY_CPUS_PER_NODE" \
    --num-gpus=8 \
    --dashboard-host=0.0.0.0

srun --nodes=$((SLURM_NNODES - 1)) --ntasks=$((SLURM_NNODES - 1)) \
    --ntasks-per-node=1 --exclude="$HEAD_NODE" --overlap \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=8 \
    bash -c '
        set -euo pipefail
        python_bin="$1"
        ray_bin="$2"
        ray_address="$3"
        code_api_host="$4"
        code_api_port="$5"
        code_api_workers="$6"
        ray_cpus="$7"

        "$python_bin" -m uvicorn open_instruct.code_utils.api:app \
            --host "$code_api_host" --port "$code_api_port" \
            --workers "$code_api_workers" \
            --log-level warning --no-access-log &
        code_api_pid=$!
        cleanup_worker_api() {
            kill "$code_api_pid" >/dev/null 2>&1 || true
            wait "$code_api_pid" >/dev/null 2>&1 || true
        }
        trap cleanup_worker_api EXIT INT TERM

        ready=0
        for attempt in {1..30}; do
            if curl --fail --silent --max-time 2 \
                "http://127.0.0.1:${code_api_port}/health" >/dev/null; then
                ready=1
                break
            fi
            kill -0 "$code_api_pid" >/dev/null 2>&1 || exit 1
            sleep 1
        done
        if [[ "$ready" -ne 1 ]]; then
            echo "ERROR: Code API did not become healthy on $(hostname -s)" >&2
            exit 1
        fi
        echo "Code API is healthy on $(hostname -s)"

        "$ray_bin" start --address="$ray_address" \
            --num-cpus="$ray_cpus" --num-gpus=8 --block
    ' bash "$PYTHON" "$RAY" "$RAY_ADDRESS" "$CODE_API_HOST" \
        "$CODE_API_PORT" "$CODE_API_WORKERS_PER_NODE" "$RAY_CPUS_PER_NODE" &
WORKER_PID=$!

sleep 10
export RAY_ADDRESS
"$RAY" status --address="$RAY_ADDRESS"

"$PYTHON" open_instruct/grpo_fast.py \
    --exp_name "$EXP_NAME" \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 64 \
    --num_mini_batches 4 \
    --num_epochs 1 \
    --learning_rate "$LR" \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --output_dir "$OUTPUT_ROOT" \
    --dataset_mixer_list ${DATASET_MIX} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_eval_list_splits test \
    --dataset_transform_fn rlvr_prepare_v1 rlvr_max_length_filter_v1 \
    --dataset_local_cache_dir "$LOCAL_CACHE_DIR" \
    --dataset_cache_mode local \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 11264 \
    --model_name_or_path "$MODEL_NAME" \
    --chat_template_name olmo123 \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 256000 \
    --deepspeed_stage "$DEEPSPEED_STAGE" \
    --gather_whole_model False \
    --num_learners_per_node $NUM_LEARNERS_PER_NODE \
    --vllm_num_engines "$VLLM_NUM_ENGINES" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_enforce_eager \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed "$SEED" \
    --local_eval_every 50 \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
    --checkpoint_state_dir "$CHECKPOINT_STATE_DIR" \
    --gradient_checkpointing \
    --record_entropy \
    --with_tracking \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --llm_judge_model "$LLM_JUDGE_MODEL" \
    --llm_judge_timeout 50 \
    --llm_judge_max_tokens 2048 \
    --llm_judge_max_context_length 32768 \
    --code_api_url "${CODE_API_BASE}/test_program" \
    --code_pass_rate_reward_threshold 0.99 \
    --active_sampling \
    --no_resampling_pass_rate 0.875 \
    --save_traces \
    --rollouts_save_path "$ROLLOUTS_SAVE_PATH" \
    --try_launch_beaker_eval_jobs_on_weka false \
    --try_auto_save_to_beaker false \
    --push_to_hub false

echo "=== Training complete ==="
echo "Outputs saved under: ${OUTPUT_ROOT}/${EXP_NAME}__${SEED}__*"
