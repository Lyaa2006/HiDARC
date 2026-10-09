#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
cd "$PROJECT_ROOT"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 MODEL_CONFIG DATA_CONFIG TRAIN_CONFIG" >&2
    exit 2
fi

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3
ensure_existing_file "$MODEL_CONFIG" "model config"
ensure_existing_file "$DATA_CONFIG" "dataset config"
ensure_existing_file "$TRAIN_CONFIG" "train config"
validate_model_config_paths "$MODEL_CONFIG"
validate_data_config_paths "$DATA_CONFIG"

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_optional_config() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1:]
with open(path, "r") as f:
    value = json.load(f).get(key, default)
if isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
HIDARC_PROTOCOL=$(read_config "$TRAIN_CONFIG" protocol)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR="${HIDARC_OUTPUT_DIR:-$(read_config "$TRAIN_CONFIG" output_dir)}"
RUN_SUFFIX=${UCIT_RUN_ID:+_$UCIT_RUN_ID}
OUTPUT_DIR="${OUTPUT_DIR}${RUN_SUFFIX}"
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-$(read_optional_config "$TRAIN_CONFIG" model_max_length 2048)}"
DESCRIPTION_PROMPT=$(read_optional_config "$TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")
DESCRIPTION_MAX_TOKENS=$(read_optional_config "$TRAIN_CONFIG" description_max_tokens 32)
B2_HIGH_LAYER=$(read_optional_config "$TRAIN_CONFIG" b2_high_layer 31)
DESCRIPTION_CACHE_MODEL_SOURCE=$(read_optional_config "$TRAIN_CONFIG" description_cache_model_source "base")
DESCRIPTION_CACHE_MAX_NEW_ENTRIES=$(read_optional_config "$TRAIN_CONFIG" description_cache_max_new_entries -1)
DESCRIPTION_CACHE_FORMAT="expanded_text_v1"
DEFAULT_CACHE_TAG=$(basename "$DATA_PATH" .json)
DEFAULT_CACHE_DIR="$OUTPUT_DIR/reference_description_cache_${DESCRIPTION_CACHE_MODEL_SOURCE}_${DEFAULT_CACHE_TAG}_${DESCRIPTION_CACHE_FORMAT}"
DESCRIPTION_CACHE_DIR="${DESCRIPTION_CACHE_DIR:-$(read_optional_config "$TRAIN_CONFIG" description_cache_dir "$DEFAULT_CACHE_DIR")}"
PREVIOUS_RAW="${HIDARC_PREVIOUS_MODEL:-$(read_optional_config "$TRAIN_CONFIG" previous_model "")}"

if [ -n "$PREVIOUS_RAW" ]; then
    PREVIOUS=$(resolve_run_scoped_path "$PREVIOUS_RAW")
else
    PREVIOUS=""
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    VISIBLE_GPU_LIST=$(python3 -c "print(','.join(x.strip() for x in '${CUDA_VISIBLE_DEVICES}'.split(',') if x.strip()))")
    # Description cache entries are independent and the metadata manifest is
    # written only after extraction finishes.  Use one visible GPU so no
    # second rank can observe a partially populated cache before meta.json is
    # committed.  Training/evaluation still use all visible GPUs.
    CACHE_GPU_SLOT="${VISIBLE_GPU_LIST%%,*}"
    GPU_NUM=1
    DEEPSPEED_GPU_ARGS=(--include "localhost:$CACHE_GPU_SLOT")
    DEEPSPEED_ENV_PREFIX=(env -u CUDA_VISIBLE_DEVICES)
else
    GPU_LIST=$(seq -s, 0 $((GPU_NUM - 1)))
    DEEPSPEED_GPU_ARGS=(--include "localhost:$GPU_LIST")
    DEEPSPEED_ENV_PREFIX=()
    CACHE_GPU_SLOT=0
fi

if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

mkdir -p "$DESCRIPTION_CACHE_DIR"
echo "Extracting one-task description cache: $DESCRIPTION_CACHE_DIR"
echo "Cache source=$DESCRIPTION_CACHE_MODEL_SOURCE, data=$DATA_PATH"
echo "Cache extraction GPU slot: $CACHE_GPU_SLOT"

CACHE_ARGS=(
    --lora_enable True
    --lora_r "$RANK"
    --lora_alpha "$((RANK * 2))"
    --expert_num "$EXPERT"
    --hidarc_protocol "$HIDARC_PROTOCOL"
    --model_name_or_path "$MODEL_NAME"
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR"
    --version v1
    --data_path "$DATA_PATH"
    --image_folder "$IMAGE"
    --vision_tower "$VISION_TOWER"
    --text_tower "$VISION_TOWER"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --cur_task "$CUR_TASK"
    --model_max_length "$MODEL_MAX_LENGTH"
    --lazy_preprocess True
    --description_prompt "$DESCRIPTION_PROMPT"
    --description_cache_dir "$DESCRIPTION_CACHE_DIR"
    --description_cache_model_source "$DESCRIPTION_CACHE_MODEL_SOURCE"
    --description_cache_max_new_entries "$DESCRIPTION_CACHE_MAX_NEW_ENTRIES"
    --b2_high_layer "$B2_HIGH_LAYER"
    --description_max_tokens "$DESCRIPTION_MAX_TOKENS"
    --extract_description_cache_only True
)
if [ -n "$PREVIOUS" ]; then
    CACHE_ARGS+=(--previous_task_model_path "$PREVIOUS")
fi

"${DEEPSPEED_ENV_PREFIX[@]}" deepspeed "${DEEPSPEED_GPU_ARGS[@]}" \
    --master_port "$MASTER_PORT" llava/train/train_MOE.py "${CACHE_ARGS[@]}"
