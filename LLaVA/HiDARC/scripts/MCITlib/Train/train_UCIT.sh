#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="${CONFIG_ROOT:-$MCITLIB_ROOT/configs}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"
DATA_ROOT="${DATA_ROOT:-$CONFIG_ROOT/data_configs/UCIT}"
RUN_ROOT="${RUN_ROOT:-$MCITLIB_ROOT/checkpoints/local/llava_ucit_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-$RUN_ROOT/description_caches}"
TASKS="${TASKS:-1,2,3,4,5,6}"
RUN_EVAL="${RUN_EVAL:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,6}"
export CUDA_VISIBLE_DEVICES

ensure_existing_file "$MODEL_CONFIG" "LLaVA model config"
validate_model_config_paths "$MODEL_CONFIG"
for dataset in ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k; do
    ensure_existing_file "$DATA_ROOT/$dataset.json" "UCIT data config"
    validate_data_config_paths "$DATA_ROOT/$dataset.json"
done

mkdir -p "$RUN_ROOT/configs" "$CACHE_ROOT" "$RUN_ROOT/logs"
cd "$PROJECT_ROOT"

make_train_config() {
    local task_id="$1" output_dir="$2" previous_model="$3" cache_dir="$4" out="$5"
    "$PYTHON_BIN" - "$CONFIG_ROOT/train_configs/HiDARC/LLaVA/UCIT/train/task${task_id}.json" "$out" \
        "$output_dir" "$previous_model" "$cache_dir" "${CUDA_VISIBLE_DEVICES}" <<'PY'
import json, os, sys
source, target, output, previous, cache, visible = sys.argv[1:]
cfg = json.load(open(source, encoding="utf-8"))
collaboration = os.path.normpath(os.path.join(os.path.dirname(source), "..", "collaboration.json"))
if not os.path.isfile(collaboration):
    raise FileNotFoundError(f"Missing collaboration profile: {collaboration}")
cfg.update({
    "protocol": "UCIT",
    "task_count": 6,
    "gpu_num": len([x for x in visible.split(",") if x.strip()]),
    "output_dir": output,
    "description_cache_dir": cache,
})
cfg["collaboration_config"] = collaboration
if previous:
    cfg["previous_model"] = previous
json.dump(cfg, open(target, "w", encoding="utf-8"), indent=2)
PY
}

make_eval_config() {
    local task_id="$1" checkpoint="$2" out="$3"
    "$PYTHON_BIN" - "$CONFIG_ROOT/train_configs/HiDARC/LLaVA/UCIT/eval/task${task_id}.json" "$out" "$checkpoint" "$RUN_ROOT/results" "$MODEL_CONFIG" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json, os, sys
source, target, checkpoint, result_root, model_path, visible = sys.argv[1:]
cfg = json.load(open(source, encoding="utf-8"))
model = json.load(open(model_path, encoding="utf-8"))
collaboration = os.path.normpath(os.path.join(os.path.dirname(source), "..", "collaboration.json"))
if not os.path.isfile(collaboration):
    raise FileNotFoundError(f"Missing collaboration profile: {collaboration}")
cfg.update({
    "gpu_num": len([x for x in visible.split(",") if x.strip()]),
    "model_path": checkpoint,
    "result_path": result_root,
    "text_tower": model["text_tower"],
    "num_task": 6,
    "routing_config_path": collaboration,
})
json.dump(cfg, open(target, "w", encoding="utf-8"), indent=2)
PY
}

run_eval() {
    local stage="$1" checkpoint="$2"
    local eval_cfg="$RUN_ROOT/configs/eval_task${stage}.json"
    make_eval_config "$stage" "$checkpoint" "$eval_cfg"
    local names=(ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k)
    local scripts=(eval_imagenet.sh eval_arxivqa.sh eval_vizwiz.sh eval_iconqa.sh eval_clevr.sh eval_flickr30k.sh)
    local i
    for ((i=0; i<stage; i++)); do
        bash "$PROJECT_ROOT/scripts/MCITlib/Eval_UCIT/${scripts[$i]}" \
            "$MODEL_CONFIG" "$DATA_ROOT/${names[$i]}.json" "$eval_cfg"
    done
}

IFS=',' read -ra SELECTED_TASKS <<< "$TASKS"
names=(ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k)
for task_id in "${SELECTED_TASKS[@]}"; do
    task_id="${task_id//[[:space:]]/}"
    [[ "$task_id" =~ ^[1-6]$ ]] || { echo "Invalid UCIT task: $task_id" >&2; exit 1; }
    output_dir="$RUN_ROOT/Task${task_id}_llava_hidarc"
    previous=""
    if [ "$task_id" -gt 1 ]; then previous="$RUN_ROOT/Task$((task_id-1))_llava_hidarc"; fi
    train_cfg="$RUN_ROOT/configs/train_task${task_id}.json"
    make_train_config "$task_id" "$output_dir" "$previous" "$CACHE_ROOT/task${task_id}_${names[$((task_id-1))]}" "$train_cfg"
    if [ "$task_id" -eq 1 ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_ROOT/${names[$((task_id-1))]}.json" "$train_cfg"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_ROOT/${names[$((task_id-1))]}.json" "$train_cfg"
    fi
    if [ "$RUN_EVAL" = "1" ]; then run_eval "$task_id" "$output_dir"; fi
done
