#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"

ensure_argument_count 3 "$#" "bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh <model_config> <data_config> <eval_config>"

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

ensure_existing_file "$MODEL_CONFIG" "model config"
ensure_existing_file "$DATA_CONFIG" "data config"
ensure_existing_file "$TRAIN_CONFIG" "eval config"

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_first_available() {
    python3 - "$1" "${@:2}" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
keys = sys.argv[2:]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
for key in keys:
    if key in cfg:
        print(cfg[key])
        break
else:
    raise KeyError("/".join(keys))
PY
}

TASK="CLEVR-Math"
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
TEXT_TOWER=$(read_config "$TRAIN_CONFIG" text_tower)
NUM_TASK=$(read_config "$TRAIN_CONFIG" num_task)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_first_available "$DATA_CONFIG" test_folder image_folder)
RESULT_PATH=$(read_config "$TRAIN_CONFIG" result_path)
ROUTING_CONFIG_PATH=$(read_optional_json_field "$TRAIN_CONFIG" routing_config_path)
STAGE1_BAND_SCHEDULE_PATH=$(read_optional_json_field "$TRAIN_CONFIG" stage1_band_schedule_path)
ensure_existing_file "$DATA_PATH" "question file"
ensure_existing_dir "$IMAGE" "image folder"
MODELPATH_RESOLVED=$(resolve_run_scoped_path "$MODELPATH")
if [ "$MODELPATH_RESOLVED" != "$MODELPATH" ]; then
    echo "Resolved model path: $MODELPATH -> $MODELPATH_RESOLVED"
fi
MODELPATH="$MODELPATH_RESOLVED"
if [ ! -d "$MODELPATH" ]; then
    echo "Model checkpoint directory does not exist: $MODELPATH" >&2
    exit 1
fi

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="$RESULT_PATH/$TASK"
mkdir -p "$RESULT_DIR/$STAGE"

PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.eval.CoIN.model_others \
        --model-path "$MODELPATH" \
        --model-base "$MODELBASE" \
        --question-file "$DATA_PATH" \
        --image-folder "$IMAGE" \
        --text-tower "$TEXT_TOWER" \
        --num-task "$NUM_TASK" \
        ${ROUTING_CONFIG_PATH:+--routing-config-path "$ROUTING_CONFIG_PATH"} \
        ${STAGE1_BAND_SCHEDULE_PATH:+--stage1-band-schedule-path "$STAGE1_BAND_SCHEDULE_PATH"} \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    PIDS+=($!)
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "Generation failed for $TASK/$STAGE." >&2
    exit 1
fi

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_file="$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl"
    if [ ! -s "$chunk_file" ]; then
        echo "Missing or empty prediction shard: $chunk_file" >&2
        exit 1
    fi
    cat "$chunk_file" >> "$output_file"
done

if [ ! -s "$output_file" ]; then
    echo "Merged prediction file is empty: $output_file" >&2
    exit 1
fi

python3 -m llava.eval.CoIN.eval_deepseek_r1 \
    --annotation-file "$DATA_PATH" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR/$STAGE"

# /mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python -m llava.eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \
