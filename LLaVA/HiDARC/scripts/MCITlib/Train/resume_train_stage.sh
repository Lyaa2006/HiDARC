#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
source "$SCRIPT_DIR/../common_path_resolver.sh"

ensure_argument_count 2 "$#" "bash scripts/MCITlib/Train/resume_train_stage.sh <run_root> <task_id>"

RUN_ROOT="$1"
TASK_ID="$2"
CFG_ROOT="$RUN_ROOT/generated_configs"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

case "$TASK_ID" in
    2) DATA_NAME="ArxivQA" ;;
    3) DATA_NAME="VizWiz" ;;
    4) DATA_NAME="IconQA" ;;
    5) DATA_NAME="CLEVR-Math" ;;
    6) DATA_NAME="Flickr30k" ;;
    *)
        echo "Unsupported train task id: $TASK_ID (expected 2-6)" >&2
        exit 1
        ;;
esac

TRAIN_CFG="$CFG_ROOT/train_task${TASK_ID}.json"
MODEL_CFG="$HARD_PATH/configs/model_configs/llava.json"
DATA_CFG="$HARD_PATH/configs/data_configs/UCIT/${DATA_NAME}.json"

ensure_existing_dir "$RUN_ROOT" "run root"
ensure_existing_file "$TRAIN_CFG" "train config"
ensure_existing_file "$MODEL_CFG" "model config"
ensure_existing_file "$DATA_CFG" "data config"

RUN_ID="$(basename "$RUN_ROOT")"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}_resume_train_task${TASK_ID}_$(date +%Y%m%d_%H%M%S).log}"

cd "$PROJECT_ROOT"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export LOG_TEE_ACTIVE=1
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ROOT=$RUN_ROOT"
echo "TASK_ID=$TASK_ID"
echo "DATA_NAME=$DATA_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CFG" \
    "$DATA_CFG" \
    "$TRAIN_CFG"
