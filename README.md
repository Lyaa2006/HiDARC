# HiDARC

This repo releases our implementation for the HiDARC.

## Environment setup

```bash
pip install -r requirements.txt
```

## Models

Download the LLaVA and InternVL checkpoints together with their vision/text
towers:

```bash
# LLaVA
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir /your_model_path/LLaVA/llava-v1.5-7b
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir /your_model_path/LLaVA/clip-vit-large-patch14-336

# InternVL
huggingface-cli download OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B \
  --local-dir /your_model_path/InternVL/Internvl-chat-7b
huggingface-cli download OpenGVLab/InternViT-6B-224px \
  --local-dir /your_model_path/InternVL/InternViT-6B-224px
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir /your_model_path/InternVL/clip-vit-large-patch14-336
```

The reference model configuration files are available under
`examples/llava-v1.5-7b/` and `examples/Internvl-chat-7b/`. Replace the model
path placeholders in those files before using them.

## Datasets

UCIT and MLLM-CL are two different benchmarks. UCIT was introduced by
HiDe-LLaVA; its original Hugging Face repository is
[`HaiyangGuo/UCIT`](https://huggingface.co/datasets/HaiyangGuo/UCIT).

Download the original UCIT instructions and released files with:

```bash
huggingface-cli download HaiyangGuo/UCIT \
  --repo-type dataset \
  --local-dir /your_data_path/UCIT
```

Please follow the instructions on the official Hugging Face dataset card to
download and organize the complete image collection for all UCIT tasks.

Download the separate MLLM-CL benchmark with:

```bash
huggingface-cli download MLLM-CL/MLLM-CL \
  --repo-type dataset \
  --local-dir /your_data_path/MLLM-CL
```

The LLaVA and InternVL HiDARC experiments use either the UCIT or the
MLLM-CL task protocols. The downloaded data must be accompanied by the
corresponding local dataset JSON configuration files expected by the
evaluation scripts.

## How to run

Before running a launcher, update the paths in your local configuration files:

- `MODEL_CONFIG`: a model JSON containing `model_name`, `vision_tower`, and
  `text_tower` (and `mm_projector` when required);
- `DATA_ROOT`: the directory containing the UCIT dataset JSON files;
- `RUN_ROOT`: the directory for checkpoints, logs, and description caches;
- `CUDA_VISIBLE_DEVICES`: the GPUs to use.

The HiDARC training settings and collaboration profiles are under
`configs/train_configs/HiDARC/`. Replace any remaining `/your_*` placeholders
in the selected task and evaluation JSON files with your local paths.

Example: train LLaVA on the six UCIT tasks:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_CONFIG=/your_config_path/model_configs/llava.json
export DATA_ROOT=/your_config_path/data_configs/UCIT
export RUN_ROOT=/your_ckpts_path/HiDARC/UCIT/LLaVA

bash LLaVA/HiDARC/scripts/MCITlib/Train/train_UCIT.sh
```

To run selected tasks or disable evaluation, for example:

```bash
TASKS=3,4 RUN_EVAL=0 \
  bash LLaVA/HiDARC/scripts/MCITlib/Train/train_UCIT.sh
```
