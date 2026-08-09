


PYTHON="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python does not exist or is not executable: $PYTHON"
    exit 1
fi

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT="0.0.0+local"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_cache}"

PROJECT_ROOT="/data/home/zhanghx/code/open-instruct"
OLMOCORE_PATH="${PROJECT_ROOT}/OLMo-core-main"

MODEL_NAME="/data/common/LLMs/allenai/Olmo-3-7B-Instruct-DPO"
OUTPUT_ROOT="${PROJECT_ROOT}/checkpoints/rl"
LOCAL_CACHE_DIR="${PROJECT_ROOT}/data/open_instruct_dataset_cache"
CHECKPOINT_STATE_DIR="${OUTPUT_ROOT}/checkpoint_states"
ROLLOUTS_SAVE_PATH="${PROJECT_ROOT}/data/rollouts"

DATASET_MIX="allenai/Dolci-Instruct-RL 1.0"

export OPENAI_API_KEY="sk-toq5V4kfbQ0NjSUXkUpj5ZRQmJXA0EWbD2xgD1ahTeBcZASr"
export OPENAI_API_BASE="https://www.dmxapi.cn"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
LLM_JUDGE_MODEL="qwen3-8b"

export CODE_API_BASE="${CODE_API_BASE:-http://localhost:1234}"

NUM_GPUS="6"
NUM_LEARNERS_PER_NODE="4"
VLLM_NUM_ENGINES="2"
VLLM_TENSOR_PARALLEL_SIZE="1"
VLLM_GPU_MEMORY_UTILIZATION="0.9"
DEEPSPEED_STAGE="3"
LR="1e-6"
SEED="1"
MIN_RAY_CPUS=$((NUM_LEARNERS_PER_NODE * 10 + VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE + 3))
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="olmo3-7b-GRPO-${TIMESTAMP}"
WANDB_ENTITY="jaycool"
WANDB_PROJECT="Olmo3-7B-rl"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$SLURM_CPUS_PER_TASK" -lt "$MIN_RAY_CPUS" ]]; then
    echo "ERROR: This GRPO config needs at least ${MIN_RAY_CPUS} CPUs for Ray scheduling."
    echo "Current SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}."
    echo "Submit with: sbatch scripts/slurm/rl/7b_instruct_rl.sh"
    echo "Or request an interactive job with at least: --cpus-per-task=64 --gres=gpu:6"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOCAL_CACHE_DIR" "$CHECKPOINT_STATE_DIR" "$ROLLOUTS_SAVE_PATH" "$TRITON_CACHE_DIR" logs

cd "$PROJECT_ROOT"
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
echo "vLLM engines: $VLLM_NUM_ENGINES"
echo "vLLM TP size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "Minimum Ray CPUs needed: $MIN_RAY_CPUS"
echo "Learning rate: $LR"
echo "Experiment: $EXP_NAME"
echo "Judge model: $LLM_JUDGE_MODEL"
if [[ -n "${OPENAI_API_BASE:-}" ]]; then
    echo "OpenAI-compatible API base: $OPENAI_API_BASE"
fi
echo "Code API: ${CODE_API_BASE}/test_program"
echo "W&B entity: $WANDB_ENTITY"
echo "W&B project: $WANDB_PROJECT"
echo "W&B init timeout: $WANDB_INIT_TIMEOUT seconds"
echo "Python: $PYTHON"
echo "=================================="

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
    --dataset_mixer_eval_list HuggingFaceH4/MATH-500 1.0 \
    --dataset_mixer_eval_list_splits test \
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
    --total_episodes 1024000 \
    --deepspeed_stage "$DEEPSPEED_STAGE" \
    --num_learners_per_node "$NUM_LEARNERS_PER_NODE" \
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
    --with_tracking \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --llm_judge_model "$LLM_JUDGE_MODEL" \
    --llm_judge_timeout 600 \
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
