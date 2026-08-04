#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/zhanghx/code/open-instruct"
PYTHON="/home/zhanghx/.conda/envs/olmo_rl/bin/python"
CODE_API_HOST="${CODE_API_HOST:-0.0.0.0}"
CODE_API_PORT="${CODE_API_PORT:-1234}"
CODE_API_WORKERS="${CODE_API_WORKERS:-8}"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/OLMo-core-main/src"

exec "$PYTHON" -m uvicorn open_instruct.code_utils.api:app \
    --host "$CODE_API_HOST" \
    --port "$CODE_API_PORT" \
    --workers "$CODE_API_WORKERS" \
    --log-level warning \
    --no-access-log
