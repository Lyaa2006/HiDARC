#    Copyright 2023 Haotian Liu
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


import json
import os, sys
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .hidarc_final import FIXED_HIDARC_KEYS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _read_adapter_config(checkpoint_dir):
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return {}
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_effective_num_task(checkpoint_dir, declared_num_task):
    declared_num_task = max(1, int(declared_num_task))
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return declared_num_task

    adapter_config = _read_adapter_config(checkpoint_dir)
    cur_task = adapter_config.get("cur_task")
    if cur_task is not None:
        try:
            return max(1, min(int(cur_task) + 1, declared_num_task))
        except (TypeError, ValueError):
            pass

    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if checkpoint_name.startswith("Task"):
        digits = []
        for ch in checkpoint_name[4:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return max(1, min(int("".join(digits)), declared_num_task))

    return declared_num_task


def _resolve_repo_path(path):
    if path is None:
        return None
    raw_path = os.path.expanduser(path)
    candidates = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.append(os.path.abspath(raw_path))
        candidates.append(os.path.join(PROJECT_ROOT, raw_path))
        candidates.append(os.path.join(WORKSPACE_ROOT, raw_path))

    resolved = None
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate

    return os.path.abspath(candidates[0])


def _load_json_dict(path, label):
    if path is None:
        return None
    resolved = _resolve_repo_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{label} must be a JSON object: {resolved}")
    return data


def _load_local_llava_config(config_dir):
    try:
        return AutoConfig.from_pretrained(config_dir)
    except KeyError as exc:
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.isfile(config_path):
            raise
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if config_dict.get("model_type") == "llava_llama":
            config_dict["model_type"] = "llava"
            return LlavaConfig.from_dict(config_dict)
        raise exc


def _pad_tensor_to_shape(tensor, target_shape):
    padded = torch.zeros(target_shape, dtype=tensor.dtype)
    if tensor.ndim == 1 and len(target_shape) == 1:
        length = min(tensor.shape[0], target_shape[0])
        padded[:length] = tensor[:length]
        return padded
    if tensor.ndim == 2 and len(target_shape) == 2:
        rows = min(tensor.shape[0], target_shape[0])
        cols = min(tensor.shape[1], target_shape[1])
        padded[:rows, :cols] = tensor[:rows, :cols]
        return padded
    return padded


def _coerce_non_lora_state_dict(model, state_dict):
    model_state = model.state_dict()
    coerced = {}
    dropped = []
    routing_state_prefixes = (
        "text_anchors.",
        "text_boundary.",
        "spectral_image_anchors.",
        "spectral_image_boundary.",
        "role_spectral_prototypes.",
        "role_text_prototypes.",
    )
    routing_state_keys = {
        "expert_usage_prior",
        "role_task_count",
        "role_usage_prior",
        "task_role_membership",
        "active_role_count",
    }
    legacy_prefixes = (
        "image_anchors.",
        "image_boundary.",
        "role_image_prototypes.",
    )
    vector_keys = {
        "expert_usage_prior",
        "role_task_count",
        "role_usage_prior",
    }
    matrix_keys = {
        "task_role_membership",
    }
    spectral_prefixes = (
        "spectral_image_anchors.",
        "role_spectral_prototypes.",
    )

    for key, value in state_dict.items():
        if key.startswith("model.") and (
            key[len("model.") :].startswith(routing_state_prefixes)
            or key[len("model.") :] in routing_state_keys
        ):
            remapped_key = key[len("model.") :]
            if remapped_key in model_state:
                key = remapped_key

        if any(key.startswith(prefix) for prefix in legacy_prefixes):
            dropped.append((key, "legacy_state"))
            continue

        target = model_state.get(key)
        if target is None:
            coerced[key] = value
            continue

        if tuple(value.shape) == tuple(target.shape):
            coerced[key] = value
            continue

        short_key = key.split(".")[-1]
        if short_key in vector_keys and value.ndim == 1 and target.ndim == 1:
            coerced[key] = _pad_tensor_to_shape(value, target.shape)
            continue

        if short_key in matrix_keys and value.ndim == 2 and target.ndim == 2:
            coerced[key] = _pad_tensor_to_shape(value, target.shape)
            continue

        if any(key.startswith(prefix) for prefix in spectral_prefixes) and value.ndim == 2 and target.ndim == 2:
            coerced[key] = _pad_tensor_to_shape(value, target.shape)
            continue

        dropped.append((key, f"shape_mismatch:{tuple(value.shape)}->{tuple(target.shape)}"))

    return coerced, dropped


def _restore_routing_state_float32(model, state_dict):
    direct_keys = {
        "expert_usage_prior",
        "role_task_count",
        "role_usage_prior",
        "task_role_membership",
        "active_role_count",
    }
    indexed_prefixes = {
        "text_anchors.": "text_anchors",
        "text_boundary.": "text_boundary",
        "spectral_image_anchors.": "spectral_image_anchors",
        "spectral_image_boundary.": "spectral_image_boundary",
        "role_spectral_prototypes.": "role_spectral_prototypes",
        "role_text_prototypes.": "role_text_prototypes",
    }
    restored = []
    for key, value in state_dict.items():
        if key in direct_keys:
            param = getattr(model, key, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            param.data = value.detach().to(device=param.device, dtype=torch.float32)
            restored.append(key)
            continue
        matched_prefix = None
        for prefix in indexed_prefixes:
            if key.startswith(prefix):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            continue
        index_text = key[len(matched_prefix) :]
        if not index_text.isdigit():
            continue
        bank_name = indexed_prefixes[matched_prefix]
        bank = getattr(model, bank_name, None)
        if bank is None:
            continue
        idx = int(index_text)
        if idx < 0 or idx >= len(bank):
            continue
        param = bank[idx]
        if not isinstance(param, torch.nn.Parameter):
            continue
        param.data = value.detach().to(device=param.device, dtype=torch.float32)
        restored.append(key)
    return restored


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    num_task=10,
    text_tower=None,
    routing_config_path=None,
    **kwargs,
):
    # Older launchers may still pass these names.  They are intentionally
    # ignored: HiDARC routing and stage bands are fixed in the model code.
    kwargs.pop("stage1_band_schedule_path", None)
    kwargs = {"device_map": device_map, **kwargs}
    checkpoint_dir = os.path.abspath(os.path.expanduser(model_path))
    is_local_checkpoint_dir = os.path.isdir(checkpoint_dir)
    has_lora_adapter = is_local_checkpoint_dir and (
        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
        or os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    )
    has_non_lora_weights = is_local_checkpoint_dir and os.path.exists(
        os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    )
    cfg_pretrained = None
    if is_local_checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        cfg_pretrained = _load_local_llava_config(checkpoint_dir)
        if not getattr(cfg_pretrained, "hidarc_protocol", None):
            cfg_pretrained.hidarc_protocol = "UCIT" if int(num_task) == 6 else "MLLM-DCL"
    # Checkpoints created on another host can retain an absolute vision-tower
    # path.  Keep those checkpoints immutable and permit the launcher to
    # supply the equivalent local model directory explicitly.
    vision_tower_override = os.environ.get("HIDESC_VISION_TOWER_OVERRIDE")
    if vision_tower_override:
        if not os.path.isdir(vision_tower_override):
            raise FileNotFoundError(
                "HIDESC_VISION_TOWER_OVERRIDE is not a directory: "
                f"{vision_tower_override}"
            )
        if cfg_pretrained is not None:
            cfg_pretrained.mm_vision_tower = vision_tower_override
            cfg_pretrained.vision_tower = vision_tower_override
    spectral_pca_override = os.environ.get("HIDESC_SPECTRAL_PCA_OVERRIDE")
    if spectral_pca_override:
        if not os.path.isfile(spectral_pca_override):
            raise FileNotFoundError(
                "HIDESC_SPECTRAL_PCA_OVERRIDE is not a file: "
                f"{spectral_pca_override}"
            )
        if cfg_pretrained is not None:
            cfg_pretrained.spectral_pca_path = spectral_pca_override
    text_tower_override = os.environ.get("HIDESC_TEXT_TOWER_OVERRIDE")
    if text_tower_override:
        if not os.path.isdir(text_tower_override):
            raise FileNotFoundError(
                "HIDESC_TEXT_TOWER_OVERRIDE is not a directory: "
                f"{text_tower_override}"
            )
        if cfg_pretrained is not None:
            cfg_pretrained.mm_text_tower = text_tower_override
            cfg_pretrained.text_tower = text_tower_override
    if cfg_pretrained is not None and not hasattr(cfg_pretrained, "mm_text_select_layer"):
        # Older local checkpoints do not serialize this HiDESC-only field.
        # Match the training default (the final CLIP text layer).
        cfg_pretrained.mm_text_select_layer = -1
    architectures = [str(x).lower() for x in getattr(cfg_pretrained, "architectures", [])] if cfg_pretrained is not None else []
    is_llava_model = (
        'llava' in model_name.lower()
        or getattr(cfg_pretrained, "model_type", None) == "llava"
        or any("llava" in arch for arch in architectures)
    )
    is_lora_checkpoint = has_lora_adapter and has_non_lora_weights
    effective_num_task = _infer_effective_num_task(checkpoint_dir, num_task) if is_lora_checkpoint else int(num_task)

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if is_llava_model:
        # Load LLaVA model
        if is_lora_checkpoint and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if is_lora_checkpoint and model_base is not None:
            lora_cfg_pretrained = cfg_pretrained if cfg_pretrained is not None else _load_local_llava_config(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(f"LoRA checkpoint directory does not exist: {model_path}")
            non_lora_path = os.path.join(checkpoint_dir, 'non_lora_trainables.bin')
            if not os.path.exists(non_lora_path):
                raise FileNotFoundError(f"Missing non_lora_trainables.bin in {checkpoint_dir}")
            non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

            adapter_kind = _read_adapter_config(model_path).get("peft_type")
            if adapter_kind == "MOE_LORA_CoIN":
                import HiDARC.peft.tuners.lora as _hidesc_lora
                sys.modules["CoIN.peft.tuners.lora"] = _hidesc_lora
                from CoIN.peft import PeftModel, TaskType, get_peft_model, WEIGHTS_NAME, set_peft_model_state_dict
            else:
                from HiDARC.peft import PeftModel, TaskType, get_peft_model, HiDeMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            # else:
            #     from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            non_lora_trainables, dropped_non_lora = _coerce_non_lora_state_dict(model, non_lora_trainables)
            if dropped_non_lora:
                preview = ", ".join(f"{key}({reason})" for key, reason in dropped_non_lora[:8])
                print(
                    f"Dropped or reshaped incompatible non-LoRA tensors for eval compatibility: "
                    f"{preview}"
                )
            routing_debug_keys = {
                "role_task_count",
                "active_role_count",
                "task_role_membership",
            }
            routing_debug = {
                key: value.detach().float().flatten().tolist()
                for key, value in non_lora_trainables.items()
                if key in routing_debug_keys
            }
            print(f"HiDESC eval non-LoRA routing tensors before load: {routing_debug}")
            load_result = model.load_state_dict(non_lora_trainables, strict=False)
            restored_keys = _restore_routing_state_float32(model, non_lora_trainables)
            print(
                "HiDESC eval non-LoRA load result:",
                f"missing_routing={[key for key in load_result.missing_keys if key in routing_debug_keys]}",
                f"unexpected_routing={[key for key in load_result.unexpected_keys if key in routing_debug_keys]}",
            )
            if restored_keys:
                print(f"HiDESC eval restored routing tensors in float32: {restored_keys}")
            # A checkpoint with a complete injected role bank must keep that bank
            # when the eval strategy is configured below. Otherwise the first
            # forward can rebuild it from anchors and silently change the role
            # partition selected during offline preparation.
            loaded_active_roles = int(
                torch.nan_to_num(
                    model.active_role_count.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).item()
            ) if hasattr(model, "active_role_count") else 0
            loaded_role_counts = (
                torch.nan_to_num(
                    model.role_task_count.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if hasattr(model, "role_task_count")
                else torch.empty(0)
            )
            loaded_membership = (
                torch.nan_to_num(
                    model.task_role_membership.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if hasattr(model, "task_role_membership")
                else torch.empty(0)
            )
            role_bank_is_complete = (
                "active_role_count" in restored_keys
                and "role_task_count" in restored_keys
                and "task_role_membership" in restored_keys
                and loaded_active_roles > 0
                and loaded_active_roles <= loaded_role_counts.shape[0]
                and loaded_active_roles <= loaded_membership.shape[1]
                and bool(
                    (loaded_role_counts[:loaded_active_roles] > 0).all().item()
                )
                and effective_num_task <= loaded_membership.shape[0]
                and bool(
                    (
                        loaded_membership[:effective_num_task, :loaded_active_roles]
                        .abs()
                        .sum(dim=1)
                        > 0
                    )
                    .all()
                    .item()
                )
            )
            if role_bank_is_complete:
                model._role_memory_reset_applied = True
                print(
                    "HiDARC eval preserving loaded role bank:",
                    f"active_role_count={loaded_active_roles}",
                    f"effective_num_task={effective_num_task}",
                )
            print(
                "HiDESC eval routing tensors after load:",
                {
                    key: model.state_dict()[key].detach().float().flatten().tolist()
                    for key in routing_debug_keys
                    if key in model.state_dict()
                },
            )
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = _load_local_llava_config(model_path)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = cfg_pretrained if cfg_pretrained is not None else _load_local_llava_config(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if is_llava_model:
        resolved_text_tower = text_tower or getattr(model.config, "mm_text_tower", None)
        if resolved_text_tower is None:
            raise RuntimeError(
                "HiDESC evaluation requires a CLIP text tower. Pass text_tower "
                "or save mm_text_tower in the model config."
            )
        clip_tokenizer = AutoTokenizer.from_pretrained(
            resolved_text_tower,
            cache_dir=None,
            model_max_length=77,
            padding_side="right",
            use_fast=True,
        )
        if hasattr(model, "set_clip_tokenizer"):
            model.set_clip_tokenizer(clip_tokenizer)
        if hasattr(model, "set_tokenizer"):
            model.set_tokenizer(tokenizer)
        if hasattr(model, "set_eval"):
            model.set_eval(num_task, effective_num_task=effective_num_task)
            print(
                f"Eval task-space: declared_num_task={int(num_task)}, "
                f"effective_num_task={effective_num_task} (from checkpoint cur_task)"
            )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError(
                f"LLaVA checkpoint {checkpoint_dir} was loaded without a vision tower. "
                "Check the saved config for mm_vision_tower/vision_tower."
            )
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        text_tower_model = model.get_text_tower()
        if text_tower_model is None and text_tower is not None:
            raise RuntimeError(
                f"LLaVA checkpoint {checkpoint_dir} was loaded without a text tower, "
                f"but text_tower={text_tower} was requested."
            )
        if text_tower_model is not None:
            if not text_tower_model.is_loaded:
                text_tower_model.load_model()
            text_tower_model.to(device=device, dtype=torch.float16)
        routing_config = _load_json_dict(routing_config_path, "routing config")
        if routing_config:
            relation_owner = model
            for _ in range(3):
                if hasattr(relation_owner, "relation_routing_config"):
                    break
                relation_owner = getattr(relation_owner, "base_model", relation_owner)
                relation_owner = getattr(relation_owner, "model", relation_owner)
            if not hasattr(relation_owner, "relation_routing_config"):
                raise RuntimeError("Loaded HiDARC model has no relation routing state.")
            for key, value in routing_config.items():
                if key not in FIXED_HIDARC_KEYS and key != "stage1_band_schedule_path":
                    relation_owner.relation_routing_config[key] = value
            relation_owner.config.relation_routing_config = dict(
                relation_owner.relation_routing_config
            )
        if hasattr(model, "_get_active_role_count") and hasattr(model, "role_task_count"):
            with torch.no_grad():
                active_roles = int(model._get_active_role_count())
                role_task_count = torch.nan_to_num(
                    model.role_task_count.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                membership = torch.nan_to_num(
                    model.task_role_membership.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                populated_roles = int((role_task_count > 0).sum().item())
                membership_roles = int((membership.abs().sum(dim=0) > 0).sum().item())
                print(
                    "HiDESC eval role bank:",
                    f"active_role_count={active_roles}",
                    f"role_task_count_nonzero={populated_roles}",
                    f"task_role_membership_nonzero={membership_roles}",
                    f"role_task_count={role_task_count.tolist()}",
                )
                if hasattr(model, "role_spectral_prototypes"):
                    spectral_norms = [
                        float(torch.linalg.norm(proto.detach().float()).item())
                        for proto in model.role_spectral_prototypes
                    ]
                    print(f"HiDESC eval spectral role norms={spectral_norms}")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
