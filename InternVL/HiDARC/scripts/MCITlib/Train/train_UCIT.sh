#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
WORKSPACE_ROOT="$(realpath "$PROJECT_ROOT/../..")"
CONFIG_ROOT="${CONFIG_ROOT:-$WORKSPACE_ROOT/configs}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/internvl.json}"
DATA_ROOT="${DATA_ROOT:-$CONFIG_ROOT/data_configs/UCIT}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/checkpoints/UCIT/InternVL/HiDARC/mainline_$(date +%Y%m%d_%H%M%S)}"
ensure_existing_file "$MODEL_CONFIG" "model config"
validate_model_config_paths "$MODEL_CONFIG"
for data_config in ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k; do
    ensure_existing_file "$DATA_ROOT/${data_config}.json" "dataset config"
    validate_data_config_paths "$DATA_ROOT/${data_config}.json"
done
# Description caches are run/data specific; require an explicit validated root.
CACHE_ROOT="${CACHE_ROOT:?Set CACHE_ROOT to the validated UCIT description-cache root}"
mkdir -p "$CACHE_ROOT"
GPU_NUM="${GPU_NUM:-2}"
RANK="${RANK:-96}"
EXPERT_NUM="${EXPERT_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-5e-5}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DESCRIPTION_FOCUS_WEIGHT="${DESCRIPTION_FOCUS_WEIGHT:-0.05}"
DESCRIPTION_ENERGY_WEIGHT="${DESCRIPTION_ENERGY_WEIGHT:-5e-5}"
DESCRIPTION_ENERGY_MARGIN="${DESCRIPTION_ENERGY_MARGIN:-30.0}"
ALIGN_LOSS_WEIGHT="${ALIGN_LOSS_WEIGHT:-0.005}"
STANDARD_CE_WEIGHT="${STANDARD_CE_WEIGHT:-1.5}"
SPECTRAL_PCA_PATH="${SPECTRAL_PCA_PATH:-$CACHE_ROOT/internvl_spectral_pca_3200_to_512.pt}"
SPECTRAL_ROUTE_CHANNEL_DIM="${SPECTRAL_ROUTE_CHANNEL_DIM:-512}"
B2_HIGH_LAYER="${B2_HIGH_LAYER:-29}"
VISION_TOWER=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG'))['vision_tower'])")
CALIBRATION_DATA_CONFIG="$DATA_ROOT/ImageNet-R.json"
CALIBRATION_DATA_JSON=$(python3 -c "import json; print(json.load(open('$CALIBRATION_DATA_CONFIG'))['train_path'])")
CALIBRATION_IMAGE_FOLDER=$(python3 -c "import json; print(json.load(open('$CALIBRATION_DATA_CONFIG'))['train_folder'])")
if [ ! -f "$SPECTRAL_PCA_PATH" ]; then
    echo "InternVL spectral PCA not found; fitting once at: $SPECTRAL_PCA_PATH"
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/MCITlib/fit_spectral_pca.py" \
        --vision_tower "$VISION_TOWER" \
        --data_json "$CALIBRATION_DATA_JSON" \
        --image_folder "$CALIBRATION_IMAGE_FOLDER" \
        --output_path "$SPECTRAL_PCA_PATH" \
        --output_dim "$SPECTRAL_ROUTE_CHANNEL_DIM"
else
    echo "Reusing InternVL spectral PCA: $SPECTRAL_PCA_PATH"
fi
GLOBAL_BATCH=$((GPU_NUM * BATCH_SIZE * GRAD_ACC))
if [ "$GLOBAL_BATCH" -ne 64 ]; then
    echo "UCIT global batch must remain 64 (got $GLOBAL_BATCH)" >&2
    exit 1
fi
export SPECTRAL_PCA_PATH SPECTRAL_ROUTE_CHANNEL_DIM B2_HIGH_LAYER MODEL_MAX_LENGTH DATALOADER_NUM_WORKERS
export DESCRIPTION_FOCUS_WEIGHT DESCRIPTION_ENERGY_WEIGHT DESCRIPTION_ENERGY_MARGIN ALIGN_LOSS_WEIGHT STANDARD_CE_WEIGHT

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/scripts/zero2.json}"

echo "Description caches are validated/generated one task at a time by Task1.sh/Taskn.sh."

cd "$PROJECT_ROOT"
TRAIN_CONFIG_ROOT="$CONFIG_ROOT/train_configs/HiDARC/InternVL/UCIT/train"
for task_id in 1 2 3 4 5 6; do
    data_name=(ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k)
    cache_name=(task1_imagenet_r task2_arxivqa task3_vizwiz task4_iconqa task5_clevr_math task6_flickr30k)
    export HIDARC_OUTPUT_DIR="$RUN_ROOT/Task${task_id}_internvl_hidarc"
    export DESCRIPTION_CACHE_DIR="$CACHE_ROOT/${cache_name[$((task_id-1))]}"
    if [ "$task_id" -gt 1 ]; then export HIDARC_PREVIOUS_MODEL="$RUN_ROOT/Task$((task_id-1))_internvl_hidarc"; else unset HIDARC_PREVIOUS_MODEL; fi
    if [ "$task_id" -eq 1 ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_ROOT/${data_name[$((task_id-1))]}.json" "$TRAIN_CONFIG_ROOT/task${task_id}.json"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_ROOT/${data_name[$((task_id-1))]}.json" "$TRAIN_CONFIG_ROOT/task${task_id}.json"
    fi
done

echo "InternVL HiDARC UCIT mainline training completed."
