#!/bin/bash

# 用法：
# 1. 在项目根目录的 .env 中设置 HF_TOKEN
# 2. 运行：./tools/push_model_to_hf.sh
set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${HF_TOKEN:?请在 .env 中设置 HF_TOKEN}"

/home/zhanghx/.conda/envs/olmo_rl/bin/hf upload \
  hxiang/olmo3-7b-GRPO-step-450 \
  /home/zhanghx/models/rl/step_450 \
  . \
  --repo-type model \
  --token "$HF_TOKEN"
