#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
source "$SCRIPT_DIR/../common_path_resolver.sh"

ensure_argument_count 2 "$#" "bash scripts/MCITlib/Eval_UCIT/resume_eval_stage.sh <run_root> <task_id>"

RUN_ROOT="$1"
TASK_ID="$2"
CFG_ROOT="$RUN_ROOT/generated_configs"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
EVAL_CFG="$CFG_ROOT/eval_task${TASK_ID}.json"
MODEL_CFG="$HARD_PATH/configs/model_configs/llava.json"

ensure_existing_dir "$RUN_ROOT" "run root"
ensure_existing_file "$EVAL_CFG" "eval config"
ensure_existing_file "$MODEL_CFG" "model config"

RUN_ID="$(basename "$RUN_ROOT")"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}_resume_eval_task${TASK_ID}_$(date +%Y%m%d_%H%M%S).log}"

cd "$PROJECT_ROOT"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export LOG_TEE_ACTIVE=1
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ROOT=$RUN_ROOT"
echo "TASK_ID=$TASK_ID"
echo "LOG_FILE=$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
    "$MODEL_CFG" \
    "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json" \
    "$EVAL_CFG"

if [ "$TASK_ID" -ge 2 ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
        "$MODEL_CFG" \
        "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" \
        "$EVAL_CFG"
fi
if [ "$TASK_ID" -ge 3 ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh \
        "$MODEL_CFG" \
        "$HARD_PATH/configs/data_configs/UCIT/VizWiz.json" \
        "$EVAL_CFG"
fi
if [ "$TASK_ID" -ge 4 ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh \
        "$MODEL_CFG" \
        "$HARD_PATH/configs/data_configs/UCIT/IconQA.json" \
        "$EVAL_CFG"
fi
if [ "$TASK_ID" -ge 5 ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh \
        "$MODEL_CFG" \
        "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json" \
        "$EVAL_CFG"
fi
if [ "$TASK_ID" -ge 6 ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh \
        "$MODEL_CFG" \
        "$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json" \
        "$EVAL_CFG"
fi
