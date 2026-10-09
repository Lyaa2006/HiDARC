#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
if [ -d "$HARD_PATH/configs" ]; then
    CONFIG_ROOT="$HARD_PATH/configs"
else
    CONFIG_ROOT="$HARD_PATH"
fi

cd "$PROJECT_ROOT"

if [ "$TASK_ID" == "1" ]; then
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task1_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task1.json
    fi
elif [ "$TASK_ID" == "2" ]; then
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task2_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task2.json
    fi
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task2_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task2.json
    fi
elif [ "$TASK_ID" == "3" ]; then
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task3.json
    fi
elif [ "$TASK_ID" == "4" ]; then
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task4.json
    fi
elif [ "$TASK_ID" == "5" ]; then
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5.json
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5.json
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task5.json
    fi
else
    if [ "${UCIT_SMOKE:-0}" = "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
        bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/Flickr30k-smoke.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6_smoke.json
    else
        bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/IconQA.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
        bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $CONFIG_ROOT/model_configs/llava.json $CONFIG_ROOT/data_configs/UCIT/Flickr30k.json $CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/eval/task6.json
    fi
fi
