# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
import json, deepspeed
import logging
import pathlib, random
import time
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.description_utils import select_expanded_description_tokens
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.hidarc_final import get_fixed_hidarc_config
from llava.mm_utils import tokenizer_image_token

from HiDARC.peft import PeftModel, TaskType, get_peft_model, HiDeMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS=None

local_rank = None
DESCRIPTION_KEY_TERMS = (
    "object",
    "objects",
    "attribute",
    "attributes",
    "shape",
    "shapes",
    "color",
    "colors",
    "texture",
    "textures",
    "scene",
    "text",
    "spatial",
    "relation",
    "relations",
)
DESCRIPTION_CACHE_FORMAT = "expanded_text_v1"


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


def is_dist_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def maybe_init_distributed_for_cache_extraction(training_args):
    if training_args.local_rank in (None, -1):
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available for multi-GPU cache extraction.")
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(training_args.local_rank)
        torch.distributed.init_process_group(backend="nccl")


def get_dist_rank_and_world_size():
    if is_dist_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def count_cached_description_entries(cache_dir):
    if not os.path.isdir(cache_dir):
        return 0
    return sum(1 for name in os.listdir(cache_dir) if name.endswith(".pt"))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_text_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=None)
    hidarc_protocol: Optional[str] = field(default=None)
    # Experiment-specific role/activation profile. Anchor and layer-band
    # settings are intentionally not CLI/configurable.
    use_spectral_image_routing: Optional[bool] = field(default=None)
    use_spectral_role_prototype: Optional[bool] = field(default=None)
    role_reset_on_strategy_change: Optional[bool] = field(default=None)
    use_text_anchor_routing: Optional[bool] = field(default=None)
    eval_use_role_spectral_prototype: Optional[bool] = field(default=None)
    eval_disable_role_image_prototype: Optional[bool] = field(default=None)
    spectral_image_weight: Optional[float] = field(default=None)
    text_weight: Optional[float] = field(default=None)
    history_weight: Optional[float] = field(default=None)
    routing_image_weight: Optional[float] = field(default=None)
    routing_text_weight: Optional[float] = field(default=None)
    routing_history_weight: Optional[float] = field(default=None)
    routing_temperature: Optional[float] = field(default=None)
    routing_min_similarity: Optional[float] = field(default=None)
    routing_prior_momentum: Optional[float] = field(default=None)
    role_top_k: Optional[int] = field(default=None)
    role_birth_threshold: Optional[float] = field(default=None)
    role_assignment_top_k: Optional[int] = field(default=None)
    role_assignment_min_similarity: Optional[float] = field(default=None)
    role_assignment_margin: Optional[float] = field(default=None)
    role_assignment_pair_weight: Optional[float] = field(default=None)
    role_assignment_member_temperature: Optional[float] = field(default=None)
    role_assignment_member_support_mode: Optional[str] = field(default=None)
    role_assignment_member_excess_alpha: Optional[float] = field(default=None)
    role_member_top_k: Optional[int] = field(default=None)
    routing_early_layers: Optional[int] = field(default=None)
    routing_early_mode: Optional[str] = field(default=None)
    routing_early_uniform_mix: Optional[float] = field(default=None)
    routing_early_role_temperature: Optional[float] = field(default=None)
    routing_early_task_temperature: Optional[float] = field(default=None)
    routing_early_role_strength: Optional[float] = field(default=None)
    routing_middle_layers: Optional[int] = field(default=None)
    routing_middle_temperature: Optional[float] = field(default=None)
    routing_middle_role_temperature: Optional[float] = field(default=None)
    routing_middle_role_strength: Optional[float] = field(default=None)
    routing_middle_role_gamma: Optional[float] = field(default=None)
    routing_middle_role_uniform_mix: Optional[float] = field(default=None)
    routing_middle_task_uniform_mix: Optional[float] = field(default=None)
    routing_middle_role_margin_low: Optional[float] = field(default=None)
    routing_middle_role_margin_high: Optional[float] = field(default=None)
    routing_middle_intra_margin_low: Optional[float] = field(default=None)
    routing_middle_intra_margin_high: Optional[float] = field(default=None)
    routing_late_layers: Optional[int] = field(default=None)
    routing_late_role_temperature: Optional[float] = field(default=None)
    routing_late_task_temperature: Optional[float] = field(default=None)
    routing_late_role_strength: Optional[float] = field(default=None)
    routing_late_role_uniform_mix: Optional[float] = field(default=None)
    routing_role_task_floor: Optional[float] = field(default=None)
    routing_score_normalization: Optional[str] = field(default=None)
    routing_score_scale: Optional[float] = field(default=None)
    routing_strategy: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    memory_data_path: str = field(default=None,
                           metadata={"help": "Path to the memory data."})
    description_prompt: str = field(
        default=(
            "Describe the image using visual evidence: objects, attributes, shapes, colors, "
            "textures, scene context, visible text, and spatial relations."
        )
    )
    description_cache_dir: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    enable_description_cl: bool = field(default=False)
    extract_description_cache_only: bool = field(default=False)
    description_cache_model_source: str = field(default="base")
    description_cache_max_new_entries: int = field(default=-1)
    description_max_tokens: int = field(default=32)
    description_focus_weight: float = field(default=0.2)
    description_focus_alpha: float = field(default=0.5)
    description_energy_weight: float = field(default=1e-4)
    description_energy_margin: float = field(default=30.0)
    b1_low_layer: int = field(default=15)
    b1_high_layer: int = field(default=18)
    b2_low_layer: int = field(default=29)
    b2_high_layer: int = field(default=31)
    align_band_eta: float = field(default=0.5)
    struct_band_eta: float = field(default=0.35)
    struct_band_energy_rho: float = field(default=1.0)
    loss_band_ema_gamma: float = field(default=0.9)
    loss_band_position_eps: float = field(default=0.05)
    align_loss_weight: float = field(default=0.01)
    standard_ce_weight: float = field(default=1.0)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        persistent_state_keys = (
            "text_anchors",
            "text_boundary",
            "spectral_image_anchors",
            "spectral_image_boundary",
            "expert_usage_prior",
            "role_spectral_prototypes",
            "role_text_prototypes",
            "role_task_count",
            "role_usage_prior",
            "task_role_membership",
            "active_role_count",
        )
        to_return = {
            k: t
            for k, t in to_return.items()
            if t.requires_grad or any(state_key in k for state_key in persistent_state_keys)
        }
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def build_single_turn_prompt(message: str) -> str:
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_multimodal_instruction_text(
    text: str,
    data_args: DataArguments,
    image_count: int = 1,
) -> str:
    image_prefix = "\n".join([DEFAULT_IMAGE_TOKEN] * max(1, image_count))
    source = [[{"from": "human", "value": f"{image_prefix}\n{text}".strip()}]]
    source = preprocess_multimodal(source, data_args)
    return source[0][0]["value"]


def build_cache_key(prefix: str, index: int) -> str:
    return f"{prefix}_{index:08d}"


def get_description_cache_path(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, f"{cache_key}.pt")


def pad_description_sequences(
    sequences: Sequence[torch.Tensor],
    padding_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if len(sequences) == 0:
        raise ValueError("`sequences` must be non-empty.")
    max_len = max(seq.shape[0] for seq in sequences)
    hidden_size = sequences[0].shape[-1]
    device = sequences[0].device
    dtype = sequences[0].dtype
    padded = torch.full((len(sequences), max_len, hidden_size), padding_value, dtype=dtype, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[idx, :seq_len] = seq
        mask[idx, :seq_len] = True
    return {"states": padded, "mask": mask}


def build_description_key_mask(
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    key_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_token_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    for batch_idx in range(input_ids.shape[0]):
        cur_len = int(attention_mask[batch_idx].long().sum().item())
        for token_idx in range(cur_len):
            token_id = int(input_ids[batch_idx, token_idx].item())
            if token_id < 0 or token_id >= vocab_size:
                continue
            if token_id in special_token_ids:
                continue
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
            except OverflowError:
                continue
            token_text = "".join(ch for ch in token_text if ch.isalnum())
            if not token_text:
                continue
            if any(term in token_text or token_text in term for term in DESCRIPTION_KEY_TERMS):
                key_mask[batch_idx, token_idx] = True
        if not torch.any(key_mask[batch_idx, :cur_len]):
            key_mask[batch_idx, :cur_len] = attention_mask[batch_idx, :cur_len].bool()
    return key_mask


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        for idx, sample in enumerate(list_data_dict):
            sample["_description_cache_key"] = build_cache_key("train", idx)

        if data_args.memory_data_path is not None:
            list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))
            for idx, sample in enumerate(list_memory_data_dict):
                sample["_description_cache_key"] = build_cache_key("memory", idx)

            list_data_dict = list_data_dict + list_memory_data_dict
            
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        image_count = 0
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_count = len(image_file) if isinstance(image_file, list) else 1
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        use_description_data = getattr(self.data_args, "use_description_data", False)
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            if use_description_data:
                description_text = build_multimodal_instruction_text(
                    self.data_args.description_prompt,
                    self.data_args,
                    image_count=image_count,
                )
                description_prompt = build_single_turn_prompt(description_text)
                data_dict["description_input_ids"] = tokenizer_image_token(
                    description_prompt,
                    self.tokenizer,
                    return_tensors='pt',
                )
                data_dict["description_cache_key"] = self.list_data_dict[i]["_description_cache_key"]
                if self.data_args.description_cache_dir is not None:
                    cache_path = get_description_cache_path(
                        self.data_args.description_cache_dir,
                        data_dict["description_cache_key"],
                    )
                    if os.path.exists(cache_path):
                        data_dict["reference_description_states"] = torch.load(cache_path, map_location="cpu")
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def _truncate_preserving_targets(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = self.tokenizer.model_max_length
        if input_ids.shape[0] <= max_length:
            return input_ids, labels

        target_positions = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
        if target_positions.numel() == 0 or target_positions[-1].item() < max_length:
            return input_ids[:max_length], labels[:max_length]

        prefix_len = min(64, max_length - target_positions.numel())
        suffix_len = max_length - prefix_len
        input_ids = torch.cat((input_ids[:prefix_len], input_ids[-suffix_len:]), dim=0)
        labels = torch.cat((labels[:prefix_len], labels[-suffix_len:]), dim=0)
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        truncated = [
            self._truncate_preserving_targets(cur_input_ids, cur_labels)
            for cur_input_ids, cur_labels in zip(input_ids, labels)
        ]
        input_ids, labels = zip(*truncated)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'description_input_ids' in instances[0]:
            description_input_ids = [instance["description_input_ids"] for instance in instances]
            description_input_ids = torch.nn.utils.rnn.pad_sequence(
                description_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            description_input_ids = description_input_ids[:, :self.tokenizer.model_max_length]
            batch["description_input_ids"] = description_input_ids
            batch["description_attention_mask"] = description_input_ids.ne(self.tokenizer.pad_token_id)
            batch["description_key_mask"] = build_description_key_mask(
                description_input_ids,
                self.tokenizer,
                batch["description_attention_mask"],
            )
            batch["description_cache_keys"] = [instance["description_cache_key"] for instance in instances]

        reference_states = [instance.get("reference_description_states") for instance in instances]
        if any(state is not None for state in reference_states):
            prototype_state = next(state for state in reference_states if state is not None)
            filled_reference_states = []
            reference_available = []
            for state in reference_states:
                if state is None:
                    filled_reference_states.append(
                        torch.zeros(
                            (0, prototype_state.shape[-1]),
                            dtype=prototype_state.dtype,
                        )
                    )
                    reference_available.append(False)
                else:
                    filled_reference_states.append(state)
                    reference_available.append(True)
            padded_reference = pad_description_sequences(filled_reference_states)
            batch["reference_description_states"] = padded_reference["states"]
            batch["reference_description_mask"] = padded_reference["mask"]
            batch["reference_description_available"] = torch.tensor(reference_available, dtype=torch.bool)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def resolve_local_checkpoint_dir(checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("Checkpoint path is required.")

    expanded_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if os.path.isdir(expanded_path):
        return expanded_path

    prev_task_path = os.path.join(expanded_path, "prev_task")
    if os.path.isdir(prev_task_path):
        return prev_task_path

    raise FileNotFoundError(
        f"Checkpoint directory does not exist: {checkpoint_path}\n"
        f"Resolved absolute path: {expanded_path}"
    )


def resolve_checkpoint_file(checkpoint_dir, filename, allow_prev_task=False):
    candidates = [os.path.join(checkpoint_dir, filename)]
    if allow_prev_task:
        candidates.append(os.path.join(checkpoint_dir, "prev_task", filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    searched_paths = "\n".join(candidates)
    raise FileNotFoundError(
        f"Could not find required checkpoint file `{filename}` under:\n{searched_paths}"
    )


ANCHOR_ROLE_PROFILE_KEYS = (
    "use_spectral_image_routing",
    "use_spectral_role_prototype",
    "use_text_anchor_routing",
    "spectral_cutoff",
    "spectral_low_bins",
    "spectral_high_bins",
    "text_activation_highpass_exponent",
    "text_activation_magnitude_weight",
    "text_activation_real_weight",
    "text_activation_imag_weight",
    "text_activation_use_fftshift",
    "role_reset_on_strategy_change",
    "role_top_k",
    "role_birth_threshold",
    "role_assignment_top_k",
    "role_assignment_min_similarity",
    "role_assignment_margin",
    "role_assignment_pair_weight",
    "role_assignment_member_temperature",
    "role_assignment_member_support_mode",
    "role_member_top_k",
)


def validate_previous_anchor_role_profile(model, checkpoint_dir):
    config_path = resolve_checkpoint_file(checkpoint_dir, "config.json", allow_prev_task=True)
    with open(config_path, "r") as handle:
        previous_config = json.load(handle)
    previous_profile = previous_config.get("relation_routing_config")
    expected_profile = getattr(model.base_model.model.config, "relation_routing_config", None)
    if not isinstance(previous_profile, dict) or not isinstance(expected_profile, dict):
        raise RuntimeError(
            "Refusing to load a prior task without a complete HiDARC anchor/role profile: "
            f"{config_path}"
        )
    missing = [key for key in ANCHOR_ROLE_PROFILE_KEYS if key not in previous_profile or key not in expected_profile]
    mismatched = [
        key for key in ANCHOR_ROLE_PROFILE_KEYS
        if key not in missing and previous_profile[key] != expected_profile[key]
    ]
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if mismatched:
            details.append(
                "mismatched=" + str({
                    key: (previous_profile[key], expected_profile[key])
                    for key in mismatched
                })
            )
        raise RuntimeError(
            "Refusing to reuse anchors/role bank from a different or legacy HiDARC profile. "
            f"checkpoint={checkpoint_dir}; " + "; ".join(details)
        )


def load_model_from_previous_task(model, previous_task_model_path):
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    # if model.lm_head.weight.shape[0] != token_num:
    #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    checkpoint_dir = resolve_local_checkpoint_dir(previous_task_model_path)
    validate_previous_anchor_role_profile(model, checkpoint_dir)
    non_lora_path = resolve_checkpoint_file(checkpoint_dir, 'non_lora_trainables.bin')
    non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

    model.base_model.model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    filename = resolve_checkpoint_file(checkpoint_dir, WEIGHTS_NAME, allow_prev_task=True)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_result = set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
    print('Model is loaded...')


def move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, list):
            moved_batch[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            moved_batch[key] = value
    return moved_batch


def move_images_to_vision_tower(batch, model):
    images = batch.get("images")
    if images is None:
        return batch

    vision_tower = model.get_vision_tower()
    target_device = vision_tower.device
    target_dtype = vision_tower.dtype

    if isinstance(images, torch.Tensor):
        batch["images"] = images.to(device=target_device, dtype=target_dtype)
    elif isinstance(images, list):
        batch["images"] = [
            image.to(device=target_device, dtype=target_dtype) if isinstance(image, torch.Tensor) else image
            for image in images
        ]
    return batch


def move_model_to_training_device(model, training_args):
    device = training_args.device
    dtype = None
    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16

    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)

    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if dtype is None:
                vision_tower.to(device=device)
            else:
                vision_tower.to(device=device, dtype=dtype)

    if hasattr(model, "get_text_tower"):
        text_tower = model.get_text_tower()
        if text_tower is not None:
            if dtype is None:
                text_tower.to(device=device)
            else:
                text_tower.to(device=device, dtype=dtype)

    if hasattr(model, "get_model") and getattr(model.get_model(), "mm_projector", None) is not None:
        mm_projector = model.get_model().mm_projector
        if dtype is None:
            mm_projector.to(device=device)
        else:
            mm_projector.to(device=device, dtype=dtype)


def extract_description_cache(model, tokenizer, data_args, training_args):
    if data_args.description_cache_dir is None:
        raise ValueError("`description_cache_dir` is required when extracting description cache.")

    os.makedirs(data_args.description_cache_dir, exist_ok=True)
    meta_path = os.path.join(data_args.description_cache_dir, "meta.json")
    existing_total_before = count_cached_description_entries(data_args.description_cache_dir)
    if existing_total_before > 0 and not os.path.exists(meta_path):
        raise ValueError(
            f"Description cache at {data_args.description_cache_dir} has entries but "
            "no metadata. Please use a new cache directory."
        )
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            existing_meta = json.load(f)
        expected_identity = {
            "data_path": data_args.data_path,
            "memory_data_path": data_args.memory_data_path,
            "description_prompt": data_args.description_prompt,
            "description_cache_model_source": training_args.description_cache_model_source,
            "description_cache_format": DESCRIPTION_CACHE_FORMAT,
            "b2_high_layer": training_args.b2_high_layer,
            "description_max_tokens": training_args.description_max_tokens,
        }
        mismatches = [
            key
            for key, value in expected_identity.items()
            if existing_meta.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Description cache at {data_args.description_cache_dir} does not "
                f"match the requested settings ({', '.join(mismatches)}). "
                "Please use a new cache directory."
            )
    rank, world_size = get_dist_rank_and_world_size()
    move_model_to_training_device(model, training_args)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    model.eval()
    wrapped_model = getattr(model, "module", model)
    cache_mode_targets = []
    seen_target_ids = set()
    pending_targets = [wrapped_model]
    while pending_targets:
        target = pending_targets.pop(0)
        if target is None or id(target) in seen_target_ids:
            continue
        seen_target_ids.add(id(target))
        cache_mode_targets.append(target)
        for attr_name in ("get_base_model", "base_model", "model"):
            if attr_name == "get_base_model":
                getter = getattr(target, attr_name, None)
                if callable(getter):
                    try:
                        pending_targets.append(getter())
                    except (AttributeError, TypeError):
                        pass
            else:
                pending_targets.append(getattr(target, attr_name, None))
    previous_cache_modes = [
        (target, hasattr(target, "cache_extraction_mode"), getattr(target, "cache_extraction_mode", None))
        for target in cache_mode_targets
    ]
    for target in cache_mode_targets:
        target.cache_extraction_mode = True
    start_time = time.time()
    cache_manifest = {
        "data_path": data_args.data_path,
        "memory_data_path": data_args.memory_data_path,
        "description_prompt": data_args.description_prompt,
        "description_cache_model_source": training_args.description_cache_model_source,
        "description_cache_format": DESCRIPTION_CACHE_FORMAT,
        "b2_high_layer": training_args.b2_high_layer,
        "description_max_tokens": training_args.description_max_tokens,
        "num_samples": len(train_dataset),
    }
    cached_count = 0
    newly_cached_count = 0
    max_new_entries = int(getattr(training_args, "description_cache_max_new_entries", -1))
    progress_interval = max(250, min(1000, max(1, len(train_dataset) // 20)))
    assigned_indices = range(rank, len(train_dataset), world_size)
    assigned_total = len(assigned_indices)

    rank0_print(
        f"Description cache extraction started with world_size={world_size}; "
        f"rank 0 assigned progress will be reported every ~{progress_interval} cached entries."
    )

    def build_autocast_context():
        if training_args.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if training_args.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    try:
        for local_step, idx in enumerate(assigned_indices, start=1):
            if max_new_entries > 0:
                current_total = count_cached_description_entries(data_args.description_cache_dir)
                if current_total - existing_total_before >= max_new_entries:
                    break
            sample = train_dataset[idx]
            if "description_input_ids" not in sample:
                continue
            cache_path = get_description_cache_path(
                data_args.description_cache_dir,
                sample["description_cache_key"],
            )
            if os.path.exists(cache_path):
                continue
            batch = data_collator([sample])
            batch = move_batch_to_device(batch, training_args.device)
            batch = move_images_to_vision_tower(batch, model)
            with torch.no_grad(), build_autocast_context():
                (
                    _,
                    description_position_ids,
                    description_attention_mask,
                    description_past_key_values,
                    description_inputs_embeds,
                    _,
                    _,
                    expanded_text_mask,
                ) = wrapped_model.prepare_inputs_labels_for_multimodal(
                    batch["description_input_ids"],
                    None,
                    batch["description_attention_mask"],
                    None,
                    None,
                    batch.get("images"),
                    return_token_masks=True,
                )
                outputs = model(
                    input_ids=None,
                    attention_mask=description_attention_mask,
                    position_ids=description_position_ids,
                    past_key_values=description_past_key_values,
                    inputs_embeds=description_inputs_embeds,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[training_args.b2_high_layer]
                description_sequences, _ = select_expanded_description_tokens(
                    hidden_states,
                    expanded_text_mask,
                    batch["description_input_ids"],
                    batch["description_attention_mask"],
                    batch.get("description_key_mask"),
                    training_args.description_max_tokens,
                )
            torch.save(description_sequences[0].cpu(), cache_path)
            cached_count += 1
            newly_cached_count += 1
    finally:
        for target, had_cache_mode, previous_cache_mode in previous_cache_modes:
            if had_cache_mode:
                target.cache_extraction_mode = previous_cache_mode
            else:
                delattr(target, "cache_extraction_mode")

        if rank == 0 and (
            cached_count == 1
            or cached_count % progress_interval == 0
            or local_step == assigned_total
        ):
            elapsed_minutes = (time.time() - start_time) / 60.0
            global_cached_count = count_cached_description_entries(data_args.description_cache_dir)
            progress = (global_cached_count / max(1, len(train_dataset))) * 100.0
            rank0_print(
                f"Description cache progress: cached {global_cached_count}/{len(train_dataset)} "
                f"({progress:.1f}%), rank0_local_step {local_step}/{assigned_total}, "
                f"elapsed {elapsed_minutes:.1f} min"
            )

    if is_dist_initialized():
        torch.distributed.barrier()

    total_cached_count = count_cached_description_entries(data_args.description_cache_dir)
    if rank == 0:
        cache_manifest["cached_entries"] = total_cached_count
        cache_manifest["new_cached_entries_this_run"] = newly_cached_count
        cache_manifest["world_size"] = world_size
        with open(os.path.join(data_args.description_cache_dir, "meta.json"), "w") as f:
            json.dump(cache_manifest, f, indent=2)

        elapsed_minutes = (time.time() - start_time) / 60.0
        rank0_print(
            f"Description cache complete: cached {total_cached_count} entries to "
            f"{data_args.description_cache_dir} in {elapsed_minutes:.1f} min"
        )

    if is_dist_initialized():
        torch.distributed.barrier()

    if total_cached_count == 0:
        raise ValueError("Description cache extraction produced 0 entries.")


def maybe_sync_description_cache_settings(data_args, training_args):
    if not training_args.enable_description_cl or data_args.description_cache_dir is None:
        return

    meta_path = os.path.join(data_args.description_cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r") as f:
        cache_meta = json.load(f)

    cache_format = cache_meta.get("description_cache_format")
    legacy_cache_compatible = (
        cache_format is None and cache_meta.get("description_hidden_layer") is not None
    )
    if cache_format is None and legacy_cache_compatible:
        rank0_print(
            f"Accepting legacy description cache metadata at {meta_path}; "
            f"treating it as {DESCRIPTION_CACHE_FORMAT}."
        )
        cache_format = DESCRIPTION_CACHE_FORMAT

    if cache_format != DESCRIPTION_CACHE_FORMAT:
        raise ValueError(
            f"Description cache at {data_args.description_cache_dir} uses an old "
            "token-selection format. Please regenerate it in a new cache directory."
        )

    cached_max_tokens = cache_meta.get("description_max_tokens")
    if cached_max_tokens is not None and cached_max_tokens != training_args.description_max_tokens:
        rank0_print(
            f"Overriding description_max_tokens from {training_args.description_max_tokens} "
            f"to cached value {cached_max_tokens} based on {meta_path}."
        )
        training_args.description_max_tokens = cached_max_tokens

    cached_b2_high_layer = cache_meta.get("b2_high_layer")
    if cached_b2_high_layer is not None and cached_b2_high_layer != training_args.b2_high_layer:
        rank0_print(
            f"Overriding b2_high_layer from {training_args.b2_high_layer} "
            f"to cached value {cached_b2_high_layer} based on {meta_path}."
        )
        training_args.b2_high_layer = cached_b2_high_layer


def should_load_previous_task_for_cache(training_args):
    source = str(getattr(training_args, "description_cache_model_source", "base")).lower()
    if source not in {"base", "previous"}:
        raise ValueError(
            "`description_cache_model_source` must be either `base` or `previous`, "
            f"got: {training_args.description_cache_model_source}"
        )
    return source == "previous"


def maybe_set_model_task_from_checkpoint(model, checkpoint_dir, fallback_cur_task, fallback_expert_num):
    checkpoint_dir = resolve_local_checkpoint_dir(checkpoint_dir)
    adapter_config_path = resolve_checkpoint_file(checkpoint_dir, "adapter_config.json", allow_prev_task=True)
    checkpoint_cur_task = fallback_cur_task
    checkpoint_expert_num = fallback_expert_num
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    checkpoint_cur_task = int(adapter_config.get("cur_task", checkpoint_cur_task))
    checkpoint_expert_num = int(adapter_config.get("expert_num", checkpoint_expert_num))
    model.set_cur_task(checkpoint_cur_task, checkpoint_expert_num)


def build_model_config_with_local_towers(model_args, training_args):
    config = None
    if model_args.vision_tower is None and model_args.text_tower is None:
        return config

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code='mpt' in model_args.model_name_or_path,
    )
    if 'mpt' in model_args.model_name_or_path and hasattr(config, "attn_config"):
        config.attn_config['attn_impl'] = training_args.mpt_attn_impl

    if model_args.vision_tower is not None:
        config.mm_vision_tower = model_args.vision_tower
        config.vision_tower = model_args.vision_tower
        config.mm_vision_select_layer = model_args.mm_vision_select_layer
        config.mm_vision_select_feature = model_args.mm_vision_select_feature
    if model_args.text_tower is not None:
        config.mm_text_tower = model_args.text_tower
        config.text_tower = model_args.text_tower
    elif model_args.vision_tower is not None:
        config.mm_text_tower = model_args.vision_tower
        config.text_tower = model_args.vision_tower
    if getattr(model_args, "mm_text_select_layer", None) is not None:
        config.mm_text_select_layer = model_args.mm_text_select_layer
    if getattr(model_args, "mm_projector_type", None) is not None:
        config.mm_projector_type = model_args.mm_projector_type
    config.hidarc_protocol = model_args.hidarc_protocol
    routing_config = get_fixed_hidarc_config()
    for key in (
        "use_spectral_image_routing", "use_spectral_role_prototype",
        "role_reset_on_strategy_change", "use_text_anchor_routing",
        "eval_use_role_spectral_prototype", "eval_disable_role_image_prototype",
        "spectral_image_weight", "text_weight", "history_weight",
        "routing_image_weight", "routing_text_weight", "routing_history_weight",
        "routing_temperature", "routing_min_similarity", "routing_prior_momentum",
        "role_top_k", "role_birth_threshold", "role_assignment_top_k",
        "role_assignment_min_similarity", "role_assignment_margin",
        "role_assignment_pair_weight", "role_assignment_member_temperature",
        "role_assignment_member_support_mode", "role_assignment_member_excess_alpha",
        "role_member_top_k", "routing_early_layers", "routing_early_mode",
        "routing_early_uniform_mix", "routing_early_role_temperature",
        "routing_early_task_temperature", "routing_early_role_strength",
        "routing_middle_layers", "routing_middle_temperature",
        "routing_middle_role_temperature", "routing_middle_role_strength",
        "routing_middle_role_gamma", "routing_middle_role_uniform_mix",
        "routing_middle_task_uniform_mix", "routing_middle_role_margin_low",
        "routing_middle_role_margin_high", "routing_middle_intra_margin_low",
        "routing_middle_intra_margin_high", "routing_late_layers",
        "routing_late_role_temperature", "routing_late_task_temperature",
        "routing_late_role_strength", "routing_late_role_uniform_mix",
        "routing_role_task_floor", "routing_score_normalization",
        "routing_score_scale",
    ):
        value = getattr(model_args, key, None)
        if value is not None:
            routing_config[key] = value
    config.relation_routing_config = routing_config
    config.mm_spectral_feature_dim = 768 * (
        int(routing_config.get("spectral_low_bins", 4))
        + int(routing_config.get("spectral_high_bins", 4))
    )
    return config

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.use_description_data = (
        training_args.enable_description_cl or training_args.extract_description_cache_only
    )
    if training_args.enable_description_cl and training_args.gradient_checkpointing:
        rank0_print(
            "Disabling gradient checkpointing because description continual-learning "
            "uses multiple gradient-carrying forwards per step."
        )
        training_args.gradient_checkpointing = False
    local_rank = training_args.local_rank
    if training_args.extract_description_cache_only:
        maybe_init_distributed_for_cache_extraction(training_args)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    pretrained_config = build_model_config_with_local_towers(model_args, training_args)
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = pretrained_config or transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
            )
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=pretrained_config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=pretrained_config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.training = True
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        kwargs = { 
                "task_embedding_dim": model_args.task_embedding_dim,
                "expert_num": model_args.expert_num,
                "cur_task": model_args.cur_task,
            }
        lora_config = HiDeMOELoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM_HiDe,
            **kwargs
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.get_model().initialize_text_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
        text_tower = model.get_text_tower()
        text_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.text_tower,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)
    model.set_cur_task(model_args.cur_task, model_args.expert_num)

    if model_args.previous_task_model_path is not None and (
        not training_args.extract_description_cache_only
        or should_load_previous_task_for_cache(training_args)
    ):
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)

    if training_args.extract_description_cache_only:
        if should_load_previous_task_for_cache(training_args):
            if model_args.previous_task_model_path is None:
                raise ValueError(
                    "`extract_description_cache_only=True` with "
                    "`description_cache_model_source=previous` requires `previous_task_model_path`."
                )
            maybe_set_model_task_from_checkpoint(
                model,
                model_args.previous_task_model_path,
                model_args.cur_task,
                model_args.expert_num,
            )
        else:
            rank0_print("Description cache extraction will use the base model instead of the previous-task checkpoint.")
        extract_description_cache(model, tokenizer, data_args, training_args)
        return

    if training_args.enable_description_cl and data_args.description_cache_dir is None:
        raise ValueError("`enable_description_cl=True` requires `description_cache_dir`.")
    maybe_sync_description_cache_settings(data_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()
    trainer.save_state()
    if hasattr(model, "finalize_current_task_role_memory"):
        model.finalize_current_task_role_memory()

    model.config.use_cache = True

    if training_args.lora_enable:
        model.set_boundary_for_save()
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    remove_dir = training_args.output_dir
    subprocess.run(f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +", shell=True)

if __name__ == "__main__":
    train()
