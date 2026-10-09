#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"

cd "$PROJECT_ROOT"

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

for required_file in "$MODEL_CONFIG" "$DATA_CONFIG" "$TRAIN_CONFIG"; do
    if [ ! -f "$required_file" ]; then
        echo "Missing required config file: $required_file" >&2
        exit 1
    fi
done
validate_model_config_paths "$MODEL_CONFIG"
validate_data_config_paths "$DATA_CONFIG"

MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
RESOLVED_TRAIN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/hidarc-train-config.XXXXXX.json")"
"$PYTHON_BIN" "$MCITLIB_ROOT/configs/train_configs/HiDARC/resolve_hidarc_config.py" "$TRAIN_CONFIG" --output "$RESOLVED_TRAIN_CONFIG"
trap 'rm -f "$RESOLVED_TRAIN_CONFIG"' EXIT
TRAIN_CONFIG="$RESOLVED_TRAIN_CONFIG"

if [ -n "${LOG_FILE:-}" ] && [ "${LOG_TEE_ACTIVE:-0}" != "1" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: $LOG_FILE"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_optional_config() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1:]
with open(path, "r") as f:
    data = json.load(f)
value = data.get(key, default)
if isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY
}

append_optional_train_arg() {
    local key="$1"
    local value
    value=$(read_optional_config "$TRAIN_CONFIG" "$key" "__MISSING__")
    if [ "$value" != "__MISSING__" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --$key $value"
    fi
}

count_expected_cache_entries() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(sum(1 for sample in data if "image" in sample))
PY
}

cache_meta_matches() {
    python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import json
import os
import sys

cache_dir, data_path, prompt, b2_high_layer, max_tokens, model_source, cache_format = sys.argv[1:]
meta_path = os.path.join(cache_dir, "meta.json")
if not os.path.exists(meta_path):
    print("False")
    raise SystemExit
with open(meta_path, "r") as f:
    meta = json.load(f)
cache_format_value = meta.get("description_cache_format")
legacy_cache_compatible = cache_format_value is None and meta.get("description_hidden_layer") is not None
if cache_format_value is None and legacy_cache_compatible:
    cache_format_value = cache_format
cached_b2_high_layer = meta.get("b2_high_layer")
if cached_b2_high_layer is None and meta.get("description_hidden_layer") == -2:
    cached_b2_high_layer = 31
ok = (
    meta.get("data_path") == data_path
    and meta.get("description_prompt") == prompt
    and meta.get("description_cache_model_source", "base") == model_source
    and cache_format_value == cache_format
    and str(cached_b2_high_layer) == str(b2_high_layer)
    and str(meta.get("description_max_tokens")) == str(max_tokens)
)
print("True" if ok else "False")
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
HIDARC_PROTOCOL=$(read_config "$TRAIN_CONFIG" protocol)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS_RAW="${HIDARC_PREVIOUS_MODEL:-$(read_optional_config "$TRAIN_CONFIG" previous_model "")}"
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
CLIP_TOWER=$(read_optional_config "$MODEL_CONFIG" clip_tower "")
OUTPUT_DIR="${HIDARC_OUTPUT_DIR:-$(read_config "$TRAIN_CONFIG" output_dir)}"
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
SPECTRAL_PCA_PATH="${SPECTRAL_PCA_PATH:-$(read_optional_config "$TRAIN_CONFIG" spectral_pca_path "")}"
SPECTRAL_ROUTE_CHANNEL_DIM="${SPECTRAL_ROUTE_CHANNEL_DIM:-$(read_optional_config "$TRAIN_CONFIG" spectral_route_channel_dim 512)}"
RUN_SUFFIX=${UCIT_RUN_ID:+_$UCIT_RUN_ID}
OUTPUT_DIR="${OUTPUT_DIR}${RUN_SUFFIX}"
PREVIOUS=$(resolve_run_scoped_path "$PREVIOUS_RAW")

if [ -z "$SPECTRAL_PCA_PATH" ] || [ ! -f "$SPECTRAL_PCA_PATH" ]; then
    echo "InternVL HiDESC requires an existing spectral_pca_path in $TRAIN_CONFIG: $SPECTRAL_PCA_PATH" >&2
    exit 1
fi

DESCRIPTION_PROMPT="${DESCRIPTION_PROMPT:-$(read_optional_config "$TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")}"
DESCRIPTION_MAX_TOKENS="${DESCRIPTION_MAX_TOKENS:-$(read_optional_config "$TRAIN_CONFIG" description_max_tokens 32)}"
DESCRIPTION_FOCUS_WEIGHT="${DESCRIPTION_FOCUS_WEIGHT:-$(read_optional_config "$TRAIN_CONFIG" description_focus_weight 0.2)}"
DESCRIPTION_FOCUS_ALPHA="${DESCRIPTION_FOCUS_ALPHA:-$(read_optional_config "$TRAIN_CONFIG" description_focus_alpha 0.5)}"
DESCRIPTION_ENERGY_WEIGHT="${DESCRIPTION_ENERGY_WEIGHT:-$(read_optional_config "$TRAIN_CONFIG" description_energy_weight 1e-4)}"
DESCRIPTION_ENERGY_MARGIN="${DESCRIPTION_ENERGY_MARGIN:-$(read_optional_config "$TRAIN_CONFIG" description_energy_margin 30.0)}"
B1_LOW_LAYER=$(read_optional_config "$TRAIN_CONFIG" b1_low_layer 15)
B1_HIGH_LAYER=$(read_optional_config "$TRAIN_CONFIG" b1_high_layer 18)
B2_LOW_LAYER=$(read_optional_config "$TRAIN_CONFIG" b2_low_layer 29)
B2_HIGH_LAYER="${B2_HIGH_LAYER:-$(read_optional_config "$TRAIN_CONFIG" b2_high_layer 29)}"
ALIGN_BAND_ETA=$(read_optional_config "$TRAIN_CONFIG" align_band_eta 0.5)
STRUCT_BAND_ETA=$(read_optional_config "$TRAIN_CONFIG" struct_band_eta 0.35)
STRUCT_BAND_ENERGY_RHO=$(read_optional_config "$TRAIN_CONFIG" struct_band_energy_rho 1.0)
LOSS_BAND_EMA_GAMMA=$(read_optional_config "$TRAIN_CONFIG" loss_band_ema_gamma 0.9)
LOSS_BAND_POSITION_EPS=$(read_optional_config "$TRAIN_CONFIG" loss_band_position_eps 0.05)
ALIGN_LOSS_WEIGHT="${ALIGN_LOSS_WEIGHT:-$(read_optional_config "$TRAIN_CONFIG" align_loss_weight 0.01)}"
STANDARD_CE_WEIGHT="${STANDARD_CE_WEIGHT:-$(read_optional_config "$TRAIN_CONFIG" standard_ce_weight 3.0)}"
DESCRIPTION_CACHE_MODEL_SOURCE=$(read_optional_config "$TRAIN_CONFIG" description_cache_model_source "base")
DESCRIPTION_CACHE_MAX_NEW_ENTRIES=$(read_optional_config "$TRAIN_CONFIG" description_cache_max_new_entries -1)
DESCRIPTION_CACHE_FORMAT="expanded_text_v1"
SAVE_STEPS=$(read_optional_config "$TRAIN_CONFIG" save_steps 50000)
SAVE_STRATEGY=$(read_optional_config "$TRAIN_CONFIG" save_strategy steps)
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-$(read_optional_config "$TRAIN_CONFIG" model_max_length 2048)}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-$(read_optional_config "$TRAIN_CONFIG" dataloader_num_workers 4)}"
MAX_STEPS=$(read_optional_config "$TRAIN_CONFIG" max_steps -1)

if [ ! -d "$PREVIOUS" ]; then
    echo "Previous task checkpoint does not exist: $PREVIOUS" >&2
    exit 1
fi

if [ "$PREVIOUS" != "$PREVIOUS_RAW" ]; then
    echo "Resolved previous checkpoint: $PREVIOUS_RAW -> $PREVIOUS"
fi

DEFAULT_CACHE_TAG=$(basename "$DATA_PATH" .json)
DEFAULT_DESCRIPTION_CACHE_DIR="$PREVIOUS/reference_description_cache_${DESCRIPTION_CACHE_MODEL_SOURCE}_${DEFAULT_CACHE_TAG}_${DESCRIPTION_CACHE_FORMAT}"
DESCRIPTION_CACHE_DIR="${DESCRIPTION_CACHE_DIR:-$(read_optional_config "$TRAIN_CONFIG" description_cache_dir "$DEFAULT_DESCRIPTION_CACHE_DIR")}"
export DESCRIPTION_CACHE_DIR
quarantine_incomplete_cache_dir "$DESCRIPTION_CACHE_DIR" "Task${CUR_TASK} description cache"

echo "Previous checkpoint: $PREVIOUS"
echo "Output checkpoint: $OUTPUT_DIR"
echo "MM projector learning rate: $MM_PROJECTOR_LR"
echo "Description cache dir: $DESCRIPTION_CACHE_DIR"
echo "Description cache model source: $DESCRIPTION_CACHE_MODEL_SOURCE"

EXPECTED_CACHE_ENTRIES=$(count_expected_cache_entries "$DATA_PATH")
CACHE_READY=False

if [ -d "$DESCRIPTION_CACHE_DIR" ]; then
    EXISTING_CACHE_ENTRIES=$(find "$DESCRIPTION_CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    CACHE_META_READY=$(cache_meta_matches \
        "$DESCRIPTION_CACHE_DIR" \
        "$DATA_PATH" \
        "$DESCRIPTION_PROMPT" \
        "$B2_HIGH_LAYER" \
        "$DESCRIPTION_MAX_TOKENS" \
        "$DESCRIPTION_CACHE_MODEL_SOURCE" \
        "$DESCRIPTION_CACHE_FORMAT")
    if [ "$EXISTING_CACHE_ENTRIES" -ge "$EXPECTED_CACHE_ENTRIES" ] && [ "$EXPECTED_CACHE_ENTRIES" -gt 0 ] && [ "$CACHE_META_READY" = "True" ]; then
        CACHE_READY=True
    elif [ "$EXISTING_CACHE_ENTRIES" -gt 0 ]; then
        echo "Description cache exists but metadata does not match current prompt/settings; keeping existing files and writing any new cache entries into: $DESCRIPTION_CACHE_DIR"
    fi
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    VISIBLE_GPU_LIST=$(python3 -c "print(','.join([x.strip() for x in '${CUDA_VISIBLE_DEVICES}'.split(',') if x.strip()]))")
    GPU_NUM=$(python3 -c "print(len([x for x in '${CUDA_VISIBLE_DEVICES}'.split(',') if x.strip()]))")
    echo "Using CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_LIST"
    echo "Using DeepSpeed include=localhost:$VISIBLE_GPU_LIST"
    DEEPSPEED_GPU_ARGS=(--include "localhost:$VISIBLE_GPU_LIST")
    DEEPSPEED_ENV_PREFIX=(env -u CUDA_VISIBLE_DEVICES)
    CACHE_GPU_SLOT="${VISIBLE_GPU_LIST%%,*}"
else
    GPU_LIST=""
    for i in $(seq 0 $((GPU_NUM-1))); do
        GPU_LIST+="$i,"
    done
    GPU_LIST=${GPU_LIST%,}
    echo "Using default local GPU slots=$GPU_LIST"
    DEEPSPEED_GPU_ARGS=(--include "localhost:$GPU_LIST")
    DEEPSPEED_ENV_PREFIX=()
    CACHE_GPU_SLOT="0"
fi

echo "Using single-GPU description cache extraction on slot=$CACHE_GPU_SLOT"
CACHE_DEEPSPEED_GPU_ARGS=(--include "localhost:$CACHE_GPU_SLOT")

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

if [ "$CACHE_READY" != "True" ]; then
    "$SCRIPT_DIR/extract_description_cache.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TRAIN_CONFIG"
    EXISTING_CACHE_ENTRIES=$(find "$DESCRIPTION_CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    if [ "$EXISTING_CACHE_ENTRIES" -lt "$EXPECTED_CACHE_ENTRIES" ]; then
        echo "Description cache generation finished incompletely: $EXISTING_CACHE_ENTRIES/$EXPECTED_CACHE_ENTRIES" >&2
        exit 1
    fi
    CACHE_READY=True
fi

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

EXTRA_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
fi
for routing_key in \
    use_spectral_image_routing \
    use_spectral_role_prototype \
    role_reset_on_strategy_change \
    use_text_anchor_routing \
    spectral_cutoff \
    spectral_low_bins \
    spectral_high_bins \
    spectral_image_ema_decay \
    text_activation_ema_decay \
    text_activation_highpass_exponent \
    text_activation_magnitude_weight \
    text_activation_real_weight \
    text_activation_imag_weight \
    text_activation_use_fftshift \
    spectral_image_weight \
    text_weight \
    history_weight \
    routing_image_weight \
    routing_text_weight \
    routing_history_weight \
    routing_temperature \
    routing_min_similarity \
    routing_prior_momentum \
    role_top_k \
    role_birth_threshold \
    role_assignment_top_k \
    role_assignment_min_similarity \
    role_assignment_margin \
    role_assignment_pair_weight \
    role_assignment_member_temperature \
    role_assignment_member_support_mode \
    role_assignment_member_excess_alpha \
    role_member_top_k \
    eval_use_role_spectral_prototype \
    eval_disable_role_image_prototype \
    routing_early_uniform_mix \
    routing_early_role_temperature \
    routing_early_task_temperature \
    routing_early_role_strength \
    routing_middle_temperature \
    routing_middle_role_temperature \
    routing_middle_role_strength \
    routing_middle_role_gamma \
    routing_middle_role_uniform_mix \
    routing_middle_task_uniform_mix \
    routing_middle_role_margin_low \
    routing_middle_role_margin_high \
    routing_middle_intra_margin_low \
    routing_middle_intra_margin_high \
    routing_late_role_temperature \
    routing_late_task_temperature \
    routing_late_role_strength \
    routing_late_role_uniform_mix \
    routing_role_task_floor \
    routing_score_normalization \
    routing_score_scale; do
    append_optional_train_arg "$routing_key"
done

if [ "$CACHE_READY" != "True" ]; then
    echo "Supplementing description cache in $DESCRIPTION_CACHE_DIR"
    echo "Existing cache entries: ${EXISTING_CACHE_ENTRIES:-0}, expected dataset entries: $EXPECTED_CACHE_ENTRIES, max new entries this run: $DESCRIPTION_CACHE_MAX_NEW_ENTRIES"
    "${DEEPSPEED_ENV_PREFIX[@]}" deepspeed "${CACHE_DEEPSPEED_GPU_ARGS[@]}" --master_port "${MASTER_PORT:-9001}" llava/train/train_MOE.py \
        --lora_enable True \
        --lora_r $RANK \
        --lora_alpha $((RANK * 2)) \
        --expert_num $EXPERT \
        --hidarc_protocol "$HIDARC_PROTOCOL" \
        --model_name_or_path $MODEL_NAME \
        --previous_task_model_path $PREVIOUS \
        --version $PROMPT_VERSION \
        --data_path $DATA_PATH \
        --image_folder $IMAGE \
        --vision_tower $VISION_TOWER \
        --text_tower $CLIP_TOWER \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -4 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --cur_task $CUR_TASK \
        --model_max_length $MODEL_MAX_LENGTH \
        --lazy_preprocess True \
        --description_prompt "$DESCRIPTION_PROMPT" \
        --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
        --description_cache_model_source "$DESCRIPTION_CACHE_MODEL_SOURCE" \
        --description_cache_max_new_entries $DESCRIPTION_CACHE_MAX_NEW_ENTRIES \
        --b2_high_layer $B2_HIGH_LAYER \
        --description_max_tokens $DESCRIPTION_MAX_TOKENS \
        --extract_description_cache_only True \
        --spectral_pca_path "$SPECTRAL_PCA_PATH" \
        --spectral_route_channel_dim "$SPECTRAL_ROUTE_CHANNEL_DIM"
fi

"${DEEPSPEED_ENV_PREFIX[@]}" deepspeed "${DEEPSPEED_GPU_ARGS[@]}" --master_port "${MASTER_PORT:-9001}" llava/train/train_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr "$MM_PROJECTOR_LR" \
    --expert_num $EXPERT \
    --hidarc_protocol "$HIDARC_PROTOCOL" \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --text_tower $CLIP_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --cur_task $CUR_TASK \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --description_prompt "$DESCRIPTION_PROMPT" \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --enable_description_cl True \
    --description_max_tokens $DESCRIPTION_MAX_TOKENS \
    --description_focus_weight $DESCRIPTION_FOCUS_WEIGHT \
    --description_focus_alpha $DESCRIPTION_FOCUS_ALPHA \
    --description_energy_weight $DESCRIPTION_ENERGY_WEIGHT \
    --description_energy_margin $DESCRIPTION_ENERGY_MARGIN \
    --b1_low_layer $B1_LOW_LAYER \
    --b1_high_layer $B1_HIGH_LAYER \
    --b2_low_layer $B2_LOW_LAYER \
    --b2_high_layer $B2_HIGH_LAYER \
    --align_band_eta $ALIGN_BAND_ETA \
    --struct_band_eta $STRUCT_BAND_ETA \
    --struct_band_energy_rho $STRUCT_BAND_ENERGY_RHO \
    --loss_band_ema_gamma $LOSS_BAND_EMA_GAMMA \
    --loss_band_position_eps $LOSS_BAND_POSITION_EPS \
    --align_loss_weight $ALIGN_LOSS_WEIGHT \
    --standard_ce_weight $STANDARD_CE_WEIGHT \
    --spectral_pca_path "$SPECTRAL_PCA_PATH" \
    --spectral_route_channel_dim "$SPECTRAL_ROUTE_CHANNEL_DIM" \
    --report_to none \
    $EXTRA_ARGS
