#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
WORKSPACE_ROOT="$(realpath "$PROJECT_ROOT/../..")"
source "$SCRIPT_DIR/../common_path_resolver.sh"

MODEL_CONFIG="${MODEL_CONFIG:-$WORKSPACE_ROOT/configs/model_configs/internvl.json}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/configs/data_configs/UCIT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the InternVL HiDARC run directory.}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT to a new or existing result directory.}"
GPU_NUM="${GPU_NUM:-1}"
CHECKPOINT_STAGES="${CHECKPOINT_STAGES:-1 2 3 4 5 6}"

ensure_existing_file "$MODEL_CONFIG" "InternVL model config"
ensure_existing_dir "$DATA_ROOT" "UCIT data config directory"
ensure_existing_dir "$CHECKPOINT_ROOT" "InternVL HiDARC checkpoint root"
mkdir -p "$RESULT_ROOT/configs"

task_names=(ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k)
eval_scripts=(eval_imagenet.sh eval_arxivqa.sh eval_vizwiz.sh eval_iconqa.sh eval_clevr.sh eval_flickr30k.sh)

write_eval_config() {
    local checkpoint_stage="$1"
    local checkpoint_path="$2"
    local config_path="$3"
    "$PYTHON_BIN" - "$MODEL_CONFIG" "$config_path" "$checkpoint_path" "$RESULT_ROOT" "$GPU_NUM" "$checkpoint_stage" <<'PY'
import json
import sys

model_path, config_path, checkpoint_path, result_root, gpu_num, stage = sys.argv[1:]
with open(model_path, encoding="utf-8") as handle:
    model_config = json.load(handle)
config = {
    "gpu_num": int(gpu_num),
    "stage": f"HiDARC-InternVL-task{stage}",
    "model_path": checkpoint_path,
    "result_path": result_root,
    "text_tower": model_config["clip_tower"],
    "num_task": 6,
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY
}

cd "$PROJECT_ROOT"
for checkpoint_stage in $CHECKPOINT_STAGES; do
    if ! [[ "$checkpoint_stage" =~ ^[1-6]$ ]]; then
        echo "Invalid UCIT checkpoint stage: $checkpoint_stage" >&2
        exit 1
    fi
    checkpoint_path="$CHECKPOINT_ROOT/Task${checkpoint_stage}_internvl_hidarc"
    ensure_existing_dir "$checkpoint_path" "Task${checkpoint_stage} checkpoint"
    eval_config="$RESULT_ROOT/configs/task${checkpoint_stage}.json"
    write_eval_config "$checkpoint_stage" "$checkpoint_path" "$eval_config"

    echo "=== Evaluating InternVL HiDARC Task${checkpoint_stage} checkpoint ==="
    for ((eval_index=0; eval_index<checkpoint_stage; eval_index++)); do
        bash "$SCRIPT_DIR/${eval_scripts[$eval_index]}" \
            "$MODEL_CONFIG" "$DATA_ROOT/${task_names[$eval_index]}.json" "$eval_config"
    done
done
