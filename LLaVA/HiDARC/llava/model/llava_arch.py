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


from abc import ABC, abstractmethod
import json
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import inspect

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector
from .relation_text_utils import build_text_activation_index

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from HiDARC.peft.tuners import HiDeMOELoraModel
from collections import deque


_FINAL_STAGE1_BAND_SCHEDULE = {
    "early_core": (1, 14), "b1_band": (15, 18), "middle_core": (19, 28),
    "b2_band": (29, 31), "late_core": (32, 32), "b1_core": (16, 17),
    "b2_core": (30, 30),
    "alpha_by_layer": {"15": 0.25, "16": 0.5, "17": 0.5, "18": 0.75, "29": 0.25, "30": 0.5, "31": 0.75},
}


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict = False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        (
            clip_image_features,
            selected_patch_features,
            final_patch_features,
            projected_patch_features,
        ) = self.get_model().get_vision_tower()(images)
        projected_llava_features = self.get_model().mm_projector(selected_patch_features)
        return (
            clip_image_features.to(self.device),
            projected_llava_features.to(self.device),
            final_patch_features.to(self.device),
            projected_patch_features.to(self.device),
        )

    def _get_relation_config(self):
        configured = getattr(self, "relation_routing_config", None)
        if not isinstance(configured, dict):
            raise RuntimeError(
                "HiDARC requires an experiment role/activation profile in the model config."
            )
        return configured
        return getattr(
            self,
            "relation_routing_config",
            {
                "use_spectral_image_routing": True,
                "use_spectral_role_prototype": True,
                "role_reset_on_strategy_change": True,
                "use_text_anchor_routing": True,
                "use_stage1_band_schedule_eval": True,
                "eval_use_role_spectral_prototype": True,
                "eval_disable_role_image_prototype": True,
                "spectral_cutoff": 0.33,
                "spectral_low_bins": 4,
                "spectral_high_bins": 4,
                "spectral_image_weight": 0.5,
                "text_weight": 0.5,
                "history_weight": 0.15,
                "routing_image_weight": 0.5,
                "routing_text_weight": 0.5,
                "routing_history_weight": 0.15,
                "text_activation_ema_decay": 0.8,
                "spectral_image_ema_decay": 0.8,
                "text_activation_highpass_exponent": 0.75,
                "text_activation_magnitude_weight": 0.0,
                "text_activation_real_weight": 0.5,
                "text_activation_imag_weight": 0.5,
                "text_activation_use_fftshift": True,
                "routing_score_normalization": "cosine",
                "routing_score_scale": 2.0,
                "routing_temperature": 0.1,
                "routing_min_similarity": -1.0,
                "routing_prior_momentum": 0.8,
                "role_top_k": 2,
                "role_birth_threshold": 0.80,
                "role_assignment_strategy": "task_affinity_complete_link",
                "role_assignment_score_mode": "member_max",
                "role_assignment_top_k": 2,
                "role_assignment_min_similarity": 0.85,
                "role_assignment_margin": 0.02,
                "role_assignment_pair_weight": 0.50,
                "role_assignment_member_temperature": 0.35,
                "role_assignment_member_support_mode": "blended_excess_mass",
                "role_assignment_member_excess_alpha": 0.35,
                "role_member_top_k": 2,
                "routing_early_layers": 16,
                "routing_early_mode": "task_softmax_within_role",
                "routing_early_uniform_mix": 0.12,
                "routing_early_role_temperature": 0.12,
                "routing_early_task_temperature": 0.18,
                "routing_early_role_strength": 0.20,
                "routing_middle_layers": 13,
                "routing_middle_temperature": 0.10,
                "routing_middle_role_temperature": 0.10,
                "routing_middle_role_strength": 0.12,
                "routing_middle_role_gamma": 1.25,
                "routing_middle_role_uniform_mix": 0.04,
                "routing_middle_task_uniform_mix": 0.02,
                "routing_middle_role_margin_low": 0.12,
                "routing_middle_role_margin_high": 0.35,
                "routing_middle_intra_margin_low": 0.05,
                "routing_middle_intra_margin_high": 0.18,
                "routing_late_layers": 3,
                "routing_late_role_temperature": 0.12,
                "routing_late_task_temperature": 0.05,
                "routing_late_role_strength": 0.06,
                "routing_role_task_floor": 0.05,
                "routing_strategy": "prototype_only_role_constrained_late",
            },
        )

    def _get_stage1_band_schedule(self):
        return _FINAL_STAGE1_BAND_SCHEDULE

    def _get_stage1_band_region(self, layer_idx):
        schedule = self._get_stage1_band_schedule()
        if not schedule:
            return None
        layer_no = int(layer_idx) + 1
        region_ranges = (
            ("early", schedule.get("early_core")),
            ("b1", schedule.get("b1_band")),
            ("middle", schedule.get("middle_core")),
            ("b2", schedule.get("b2_band")),
            ("late", schedule.get("late_core")),
        )
        for region_name, region_range in region_ranges:
            if (
                isinstance(region_range, (list, tuple))
                and len(region_range) == 2
                and int(region_range[0]) <= layer_no <= int(region_range[1])
            ):
                return region_name
        return None

    def _get_stage1_band_core(self, layer_idx):
        schedule = self._get_stage1_band_schedule()
        if not schedule:
            return None
        layer_no = int(layer_idx) + 1
        core_ranges = (
            ("b1_core", schedule.get("b1_core")),
            ("b2_core", schedule.get("b2_core")),
        )
        for core_name, core_range in core_ranges:
            if (
                isinstance(core_range, (list, tuple))
                and len(core_range) == 2
                and int(core_range[0]) <= layer_no <= int(core_range[1])
            ):
                return core_name
        return None

    def _get_stage1_band_alpha(self, layer_idx):
        schedule = self._get_stage1_band_schedule()
        if not schedule:
            return None
        alpha_by_layer = schedule.get("alpha_by_layer")
        if not isinstance(alpha_by_layer, dict):
            return None
        layer_no = str(int(layer_idx) + 1)
        if layer_no not in alpha_by_layer:
            return None
        return float(alpha_by_layer[layer_no])

    def _build_radial_frequency_masks(self, height, width, device):
        config = self._get_relation_config()
        cutoff = float(config.get("spectral_cutoff", 0.33))
        low_bins = int(config.get("spectral_low_bins", 4))
        high_bins = int(config.get("spectral_high_bins", 4))
        if not 0.0 < cutoff < 1.0:
            raise ValueError(f"spectral_cutoff must be in (0, 1), got {cutoff}")
        if low_bins <= 0 or high_bins <= 0:
            raise ValueError(
                "spectral_low_bins and spectral_high_bins must be positive"
            )

        y = torch.arange(height, device=device, dtype=torch.float32) - height // 2
        x = torch.arange(width, device=device, dtype=torch.float32) - width // 2
        y = y / max(height // 2, 1)
        x = x / max(width // 2, 1)
        radius = torch.sqrt(y[:, None].square() + x[None, :].square()) / 2.0**0.5
        radius = radius.clamp(0.0, 1.0)

        low_mask = radius <= cutoff
        high_mask = ~low_mask
        low_bin_ids = torch.clamp(
            torch.floor(radius / cutoff * low_bins).long(),
            min=0,
            max=low_bins - 1,
        )
        high_bin_ids = torch.clamp(
            torch.floor((radius - cutoff) / (1.0 - cutoff) * high_bins).long(),
            min=0,
            max=high_bins - 1,
        )
        low_masks = torch.stack(
            [low_mask & (low_bin_ids == bin_id) for bin_id in range(low_bins)],
            dim=0,
        )
        high_masks = torch.stack(
            [
                high_mask & (high_bin_ids == bin_id)
                for bin_id in range(high_bins)
            ],
            dim=0,
        )
        return low_masks, high_masks, low_mask, high_mask

    def _pool_spectral_bands(self, spectrum, band_masks):
        pooled = []
        for mask in band_masks:
            count = mask.sum().clamp_min(1).to(dtype=spectrum.dtype)
            pooled.append(
                (spectrum * mask[None, :, :, None].to(spectrum.dtype)).sum(
                    dim=(1, 2)
                )
                / count
            )
        return torch.stack(pooled, dim=-1)

    def _extract_image_spectral_descriptor(self, projected_patch_features):
        if projected_patch_features.ndim != 3:
            raise ValueError(
                "projected_patch_features must have shape [B, N, D], got "
                f"{tuple(projected_patch_features.shape)}"
            )
        batch_size, num_patches, feature_dim = projected_patch_features.shape
        grid_size = int(round(num_patches ** 0.5))
        height = grid_size
        width = grid_size
        if num_patches != height * width:
            raise ValueError(
                "Spectral routing requires a square patch grid: "
                f"got N={num_patches}."
            )

        patch_grid = projected_patch_features.reshape(
            batch_size, height, width, feature_dim
        ).float()
        frequency = torch.fft.fft2(patch_grid, dim=(1, 2))
        frequency = torch.fft.fftshift(frequency, dim=(1, 2))
        magnitude = torch.log1p(torch.abs(frequency))
        real_part = frequency.real
        imag_part = frequency.imag
        low_masks, high_masks, low_mask, high_mask = (
            self._build_radial_frequency_masks(
                height,
                width,
                projected_patch_features.device,
            )
        )
        if not torch.equal(low_mask | high_mask, torch.ones_like(low_mask)):
            raise RuntimeError("Low/high spectral masks do not cover all frequencies.")
        if torch.any(low_mask & high_mask):
            raise RuntimeError("Low/high spectral masks overlap.")

        low_magnitude = self._pool_spectral_bands(magnitude, low_masks)
        high_magnitude = self._pool_spectral_bands(magnitude, high_masks)
        low_real = self._pool_spectral_bands(real_part, low_masks)
        high_real = self._pool_spectral_bands(real_part, high_masks)
        low_imag = self._pool_spectral_bands(imag_part, low_masks)
        high_imag = self._pool_spectral_bands(imag_part, high_masks)
        magnitude_descriptor = torch.cat(
            [
                low_magnitude.flatten(1),
                high_magnitude.flatten(1),
            ],
            dim=-1,
        )
        real_descriptor = torch.cat(
            [
                low_real.flatten(1),
                high_real.flatten(1),
            ],
            dim=-1,
        )
        imag_descriptor = torch.cat(
            [
                low_imag.flatten(1),
                high_imag.flatten(1),
            ],
            dim=-1,
        )
        descriptor = (
            0.55 * F.normalize(magnitude_descriptor, dim=-1)
            + 0.25 * F.normalize(real_descriptor, dim=-1)
            + 0.20 * F.normalize(imag_descriptor, dim=-1)
        )
        descriptor = F.normalize(
            torch.nan_to_num(descriptor.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
        return descriptor

    def _normalize_task_score_branch(self, scores):
        scores = torch.nan_to_num(
            scores.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        config = self._get_relation_config()
        normalization = str(
            config.get("routing_score_normalization", "cosine")
        ).lower()
        if normalization in {"zscore", "per_sample_zscore", "per-sample-zscore"}:
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True, unbiased=False)
            return (scores - mean) / std.clamp_min(1e-6)

        # Optional raw-cosine mode for controlled ablations. The default
        # routing configuration uses per-sample z-score calibration.
        scale = max(float(config.get("routing_score_scale", 1.0)), 0.0)
        return scores * scale

    def _spectral_image_bank_available(self, active_experts):
        if not getattr(self, "spectral_image_boundary", None):
            return False
        boundaries = torch.stack(
            [
                self.spectral_image_boundary[idx].detach().float().reshape(())
                for idx in range(active_experts)
            ]
        )
        return bool(torch.isfinite(boundaries).all() and torch.all(boundaries > 0))

    def _anchor_is_trained(self, boundary_bank, task_id, min_count):
        if boundary_bank is None or task_id >= len(boundary_bank):
            return False
        boundary = boundary_bank[task_id].detach().float().reshape(())
        return bool(torch.isfinite(boundary) and float(boundary.item()) > float(min_count))

    def _expert_has_eval_anchors(self, task_id):
        config = self._get_relation_config()
        if bool(config.get("use_text_anchor_routing", True)) and not self._anchor_is_trained(
            getattr(self, "text_boundary", None),
            task_id,
            min_count=1.0,
        ):
            return False
        if bool(config.get("use_spectral_image_routing", True)) and not self._anchor_is_trained(
            getattr(self, "spectral_image_boundary", None),
            task_id,
            min_count=0.0,
        ):
            return False
        return True

    def _get_available_eval_expert_count(self, requested_experts=None):
        if requested_experts is None:
            requested_experts = int(getattr(self, "expert_num", 0))
        requested_experts = min(
            max(int(requested_experts), 0),
            len(getattr(self, "spectral_image_anchors", [])),
            int(getattr(self, "max_task_slots", 0)),
        )
        active_experts = 0
        for task_id in range(requested_experts):
            if not self._expert_has_eval_anchors(task_id):
                break
            active_experts += 1
        return active_experts

    def _get_spectral_image_anchor(self, task_id, device=None):
        anchor = self._safe_normalize(
            self.spectral_image_anchors[task_id].detach()
        )
        return anchor if device is None else anchor.to(device)

    def _extract_text_activation_index(self, text_features):
        config = self._get_relation_config()
        return build_text_activation_index(
            text_features,
            highpass_exponent=float(
                config.get("text_activation_highpass_exponent", 0.75)
            ),
            magnitude_weight=float(
                config.get("text_activation_magnitude_weight", 0.0)
            ),
            real_weight=float(config.get("text_activation_real_weight", 0.5)),
            imag_weight=float(config.get("text_activation_imag_weight", 0.5)),
            use_fftshift=bool(config.get("text_activation_use_fftshift", True)),
        )

    def _update_running_prototype(self, prototype, boundary, features):
        features = torch.nan_to_num(
            features.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        old_count = boundary.detach().float().reshape(())
        new_count = old_count + features.shape[0]
        batch_summary = self._safe_normalize(features.mean(dim=0))
        if float(old_count.item()) <= 0.0:
            updated = batch_summary
        else:
            decay = float(
                self._get_relation_config().get("spectral_image_ema_decay", 0.8)
            )
            decay = max(0.0, min(decay, 0.9999))
            updated = self._safe_normalize(
                decay * prototype.detach().float().reshape(-1)
                + (1.0 - decay) * batch_summary
            )
        prototype.data.copy_(updated.reshape_as(prototype).to(prototype.dtype))
        boundary.data.copy_(new_count.reshape_as(boundary).to(boundary.dtype))

    def _update_running_text_activation_index(self, prototype, boundary, features):
        activation_features = self._extract_text_activation_index(features.detach().float())
        old_count = boundary.detach().float().reshape(())
        new_count = old_count + activation_features.shape[0]
        old_sum = prototype.detach().float().reshape(-1) * old_count
        updated = (old_sum + activation_features.sum(dim=0)) / new_count.clamp_min(1.0)
        prototype.data.copy_(updated.reshape_as(prototype).to(prototype.dtype))
        boundary.data.copy_(new_count.reshape_as(boundary).to(boundary.dtype))
        return activation_features

    def _safe_normalize(self, tensor):
        if tensor.ndim > 1:
            tensor = tensor.squeeze(0)
        tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(tensor, dim=0)

    def _sanitize_relation_memory(self):
        if not torch.isfinite(self.active_role_count.detach()).all():
            self.active_role_count.data.zero_()
        self.expert_usage_prior.data.copy_(
            torch.nan_to_num(self.expert_usage_prior.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        self.role_task_count.data.copy_(
            torch.nan_to_num(self.role_task_count.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        self.role_usage_prior.data.copy_(
            torch.nan_to_num(self.role_usage_prior.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        self.task_role_membership.data.copy_(
            torch.nan_to_num(self.task_role_membership.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        for prototype_bank in (
            self.role_spectral_prototypes,
            self.role_text_prototypes,
        ):
            for prototype in prototype_bank:
                prototype.data.copy_(
                    torch.nan_to_num(prototype.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                )
        for prototype in self.spectral_image_anchors:
            prototype.data.copy_(
                torch.nan_to_num(prototype.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            )
        for prototype in self.text_anchors:
            prototype.data.copy_(
                torch.nan_to_num(prototype.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            )
        for boundary in self.text_boundary:
            boundary.data.copy_(
                torch.nan_to_num(boundary.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            )
        for boundary in self.spectral_image_boundary:
            boundary.data.copy_(
                torch.nan_to_num(boundary.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            )

    def _get_active_role_count(self):
        self._sanitize_relation_memory()
        count = int(self.active_role_count.detach().float().item())
        return min(max(0, count), self._max_supported_role_slots())

    def _set_active_role_count(self, count):
        clamped = min(max(0, int(count)), self._max_supported_role_slots())
        self.active_role_count.data[0] = float(clamped)

    def _reset_role_memory(self):
        for prototype_bank in (
            self.role_spectral_prototypes,
            self.role_text_prototypes,
        ):
            for prototype in prototype_bank:
                prototype.data.zero_()
        self.role_task_count.data.zero_()
        self.role_usage_prior.data.zero_()
        self.task_role_membership.data.zero_()
        self.active_role_count.data.zero_()

    def _max_supported_role_slots(self):
        return min(
            int(getattr(self, "max_role_slots", 0)),
            len(self.role_spectral_prototypes),
            len(self.role_text_prototypes),
            int(self.role_task_count.shape[0]),
            int(self.role_usage_prior.shape[0]),
            int(self.task_role_membership.shape[1]),
        )

    def _infer_active_role_count(self, completed_task_count=None):
        max_supported = self._max_supported_role_slots()
        if max_supported <= 0:
            return 0
        if completed_task_count is None:
            membership = self.task_role_membership.detach()
        else:
            membership = self.task_role_membership[:completed_task_count].detach()
        membership = torch.nan_to_num(membership.float(), nan=0.0, posinf=0.0, neginf=0.0)
        membership_count = int((membership.abs().sum(dim=0) > 0).sum().item())
        prototype_count = int(
            (torch.nan_to_num(self.role_task_count.detach().float(), nan=0.0, posinf=0.0, neginf=0.0) > 0).sum().item()
        )
        return min(max(membership_count, prototype_count), max_supported)

    def _get_task_anchor(self, task_id):
        return self._get_spectral_image_anchor(task_id), self._safe_normalize(
            self.text_anchors[task_id].detach()
        )

    def _get_task_role_anchors(self, task_id):
        spectral_anchor = self._get_spectral_image_anchor(task_id)
        text_anchor = self._safe_normalize(self.text_anchors[task_id].detach())
        return spectral_anchor, text_anchor

    def _compose_relation_logits(self, image_scores, text_scores, history_scores):
        config = self._get_relation_config()
        image_weight = float(
            config.get("spectral_image_weight", config.get("routing_image_weight", 0.5))
        )
        text_weight = float(
            config.get("text_weight", config.get("routing_text_weight", 0.5))
        )
        history_weight = float(
            config.get("history_weight", config.get("routing_history_weight", 0.15))
        )
        if not bool(config.get("use_text_anchor_routing", True)):
            text_weight = 0.0
        return (
            image_weight * image_scores
            + text_weight * text_scores
            + history_weight * history_scores
        )

    def _mask_relation_logits(self, relation_logits):
        min_similarity = float(self._get_relation_config()["routing_min_similarity"])
        if min_similarity <= -1.0:
            return relation_logits
        masked_relation_logits = torch.where(
            relation_logits >= min_similarity,
            relation_logits,
            torch.full_like(relation_logits, float("-inf")),
        )
        if not torch.isfinite(masked_relation_logits).any():
            return relation_logits
        return masked_relation_logits

    def _build_sparse_relation_weights(self, logits, top_k):
        if logits.numel() == 0:
            return logits
        top_k = min(max(int(top_k), 1), logits.numel())
        if top_k < logits.numel():
            top_values, top_indices = torch.topk(logits, k=top_k)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(0, top_indices, top_values)
        else:
            masked_logits = logits
        temperature = max(float(self._get_relation_config()["routing_temperature"]), 1e-6)
        return F.softmax(masked_logits / temperature, dim=0)

    def _normalize_route_weights(self, weights):
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        return weights / weights.sum()

    def _mix_with_uniform(self, weights, mix_ratio):
        if weights.numel() == 0:
            return weights
        mix_ratio = float(max(0.0, min(1.0, mix_ratio)))
        normalized = self._normalize_route_weights(weights)
        if mix_ratio <= 0.0:
            return normalized
        uniform = torch.full_like(normalized, 1.0 / float(normalized.numel()))
        return self._normalize_route_weights(
            (1.0 - mix_ratio) * normalized + mix_ratio * uniform
        )

    def _build_uniform_route_weights(self, active_experts, device):
        if active_experts <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.full(
            (active_experts,),
            1.0 / float(active_experts),
            device=device,
            dtype=torch.float32,
        )

    def _summarize_guide_features(self, guide_features):
        if guide_features.ndim == 1:
            summary = guide_features
        else:
            summary = guide_features.float().mean(dim=0)
        return self._safe_normalize(summary)

    def _should_log_eval_role_activation(self):
        return str(os.environ.get("HIDESC_LOG_EVAL_ROLE_ACTIVATION", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _emit_eval_role_activation_record(self, record):
        payload = json.dumps(record, ensure_ascii=True)
        output_path = os.environ.get("HIDESC_LOG_EVAL_ROLE_ACTIVATION_PATH", "").strip()
        if output_path:
            output_path = os.path.abspath(os.path.expanduser(output_path))
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        else:
            print(payload, file=sys.stderr, flush=True)

    def _build_eval_role_activation_record(
        self,
        active_experts,
        role_scores,
        candidate_experts,
        role_members,
        route_plan,
    ):
        if role_scores.numel() == 0:
            return None

        config = self._get_relation_config()
        stage_specs = [
            (
                "early",
                config.get("routing_early_role_temperature", 0.12),
                config.get("routing_early_role_uniform_mix", 0.12),
            ),
            (
                "middle",
                config.get("routing_middle_role_temperature", 0.10),
                config.get("routing_middle_role_uniform_mix", 0.04),
            ),
            (
                "late",
                config.get("routing_late_role_temperature", 0.12),
                config.get("routing_late_role_uniform_mix", 0.0),
            ),
        ]
        stage_role_weights = {}
        for stage_name, temperature, uniform_mix in stage_specs:
            stage_role_weights[stage_name] = self._build_stage_role_weights(
                role_scores,
                temperature,
                uniform_mix,
            )

        avg_role_activation = torch.stack(
            list(stage_role_weights.values()),
            dim=0,
        ).mean(dim=0)
        membership = torch.nan_to_num(
            self.task_role_membership[:active_experts, : role_scores.numel()]
            .detach()
            .float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        membership = membership / membership.sum(dim=1, keepdim=True).clamp_min(1e-6)
        per_layer_role_activation = []
        per_layer_expert_activation = []
        for expert_weights in route_plan.get("per_layer", []):
            per_layer_expert_activation.append(expert_weights[:active_experts])
            per_layer_role_activation.append(
                torch.matmul(expert_weights[:active_experts], membership)
            )
        if per_layer_role_activation:
            avg_role_activation = torch.stack(
                per_layer_role_activation,
                dim=0,
            ).mean(dim=0)
        if per_layer_expert_activation:
            avg_expert_activation = torch.stack(
                per_layer_expert_activation,
                dim=0,
            ).mean(dim=0)
        else:
            avg_expert_activation = torch.empty(
                0,
                device=role_scores.device,
                dtype=torch.float32,
            )
        record = {
            "task_id": int(getattr(self, "cur_task", -1)),
            "active_experts": int(active_experts),
            "active_roles": int(role_scores.numel()),
            "candidate_experts": [int(idx) for idx in candidate_experts],
            "role_scores": role_scores.detach().float().cpu().tolist(),
            "avg_expert_activation": avg_expert_activation.detach()
            .float()
            .cpu()
            .tolist(),
            "avg_role_activation": avg_role_activation.detach().float().cpu().tolist(),
            "stage_role_activation": {
                name: weights.detach().float().cpu().tolist()
                for name, weights in stage_role_weights.items()
            },
            "per_layer_average_role_activation": avg_role_activation.detach()
            .float()
            .cpu()
            .tolist(),
            "role_member_tasks": role_members,
        }
        if route_plan.get("detail_diagnostics"):
            record["detail_diagnostics"] = route_plan["detail_diagnostics"]
        return record

    def _build_dense_relation_weights(self, logits):
        if logits.numel() == 0:
            return logits
        finite_mask = torch.isfinite(logits)
        if not finite_mask.any():
            return self._build_uniform_route_weights(logits.numel(), logits.device)
        safe_logits = torch.where(
            finite_mask,
            logits,
            torch.full_like(logits, float("-inf")),
        )
        temperature = max(float(self._get_relation_config()["routing_temperature"]), 1e-6)
        weights = F.softmax(safe_logits / temperature, dim=0)
        return self._normalize_route_weights(weights)

    def _build_dense_relation_weights_with_temperature(self, logits, temperature):
        if logits.numel() == 0:
            return logits
        finite_mask = torch.isfinite(logits)
        if not finite_mask.any():
            return self._build_uniform_route_weights(logits.numel(), logits.device)
        safe_logits = torch.where(
            finite_mask,
            logits,
            torch.full_like(logits, float("-inf")),
        )
        weights = F.softmax(safe_logits / max(float(temperature), 1e-6), dim=0)
        return self._normalize_route_weights(weights)

    def _compute_adaptive_blend(self, margin, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            return 1.0 if float(margin) >= upper else 0.0
        blend = (float(margin) - lower) / (upper - lower)
        return float(max(0.0, min(1.0, blend)))

    def _compute_shared_task_scores(
        self,
        image_guide_features,
        text_guide_features,
        active_experts,
    ):
        device = text_guide_features.device
        if active_experts <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        text_bank = torch.stack(
            [self._safe_normalize(self.text_anchors[idx].detach()).to(device) for idx in range(active_experts)],
            dim=0,
        )

        config = self._get_relation_config()
        text_features = F.normalize(text_guide_features.float(), dim=-1)
        text_scores = torch.matmul(text_features, text_bank.T)
        text_scores = self._normalize_task_score_branch(text_scores)

        image_scores = torch.zeros_like(text_scores)
        use_spectral = bool(config.get("use_spectral_image_routing", True))
        if (
            use_spectral
            and image_guide_features is not None
            and image_guide_features.shape[-1]
            == self.spectral_image_anchors[0].shape[-1]
            and self._spectral_image_bank_available(active_experts)
        ):
            spectral_bank = torch.stack(
                [
                    self._get_spectral_image_anchor(idx, device=device)
                    for idx in range(active_experts)
                ],
                dim=0,
            )
            spectral_features = F.normalize(image_guide_features.float(), dim=-1)
            image_scores = torch.matmul(spectral_features, spectral_bank.T)
            image_scores = self._normalize_task_score_branch(image_scores)

        history_scores = torch.nan_to_num(
            self.expert_usage_prior[:active_experts].detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(device)
        history_scores = self._normalize_task_score_branch(
            history_scores.unsqueeze(0).expand(text_scores.shape[0], -1)
        )
        image_weight = float(
            config.get("spectral_image_weight", config.get("routing_image_weight", 0.5))
        )
        text_weight = float(
            config.get("text_weight", config.get("routing_text_weight", 0.5))
        )
        history_weight = float(
            config.get("history_weight", config.get("routing_history_weight", 0.15))
        )
        if not bool(config.get("use_text_anchor_routing", True)):
            text_weight = 0.0
        task_scores = (
            image_weight * image_scores
            + text_weight * text_scores
            + history_weight * history_scores
        )
        task_scores = self._mask_relation_logits(task_scores)
        return task_scores[0] if task_scores.shape[0] == 1 else task_scores

    def _role_member_tasks(self, role_id, active_experts):
        memberships = torch.nan_to_num(
            self.task_role_membership[:active_experts, role_id].detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return [idx for idx in range(active_experts) if float(memberships[idx].item()) > 0.0]

    def _score_tasks(
        self,
        task_indices,
        image_anchor,
        text_anchor,
        device,
    ):
        if len(task_indices) == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        text_bank = torch.stack(
            [self._safe_normalize(self.text_anchors[idx].detach()).to(device) for idx in task_indices],
            dim=0,
        )
        text_scores = torch.matmul(text_bank, text_anchor.to(device))
        text_scores = self._normalize_task_score_branch(text_scores.unsqueeze(0)).squeeze(0)
        spectral_bank = torch.stack(
            [self._get_spectral_image_anchor(idx, device=device) for idx in task_indices],
            dim=0,
        )
        image_scores = torch.matmul(spectral_bank, image_anchor.to(device))
        image_scores = self._normalize_task_score_branch(
            image_scores.unsqueeze(0)
        ).squeeze(0)
        history_scores = torch.nan_to_num(
            self.expert_usage_prior[task_indices].detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(device)
        logits = self._compose_relation_logits(image_scores, text_scores, history_scores)
        return self._mask_relation_logits(logits)

    def _get_role_modal_weights(self):
        config = self._get_relation_config()
        image_weight = float(
            config.get("spectral_image_weight", config.get("routing_image_weight", 0.5))
        )
        text_weight = float(
            config.get("text_weight", config.get("routing_text_weight", 0.5))
        )
        if not bool(config.get("use_text_anchor_routing", True)):
            text_weight = 0.0
        total = max(image_weight + text_weight, 1e-6)
        return image_weight / total, text_weight / total

    def _aggregate_role_member_scores(self, scores, mode=None, top_k=None):
        if scores.numel() == 0:
            return torch.tensor(0.0, device=scores.device, dtype=torch.float32)
        mode = str(mode or "member_max").lower()
        if top_k is None:
            top_k = int(self._get_relation_config().get("role_member_top_k", 2))
        top_k = min(max(int(top_k), 1), scores.numel())
        if mode in {"mean", "member_mean"}:
            return scores.float().mean()
        if mode in {"topk_mean", "member_topk_mean"}:
            return torch.topk(scores.float(), k=top_k).values.mean()
        if mode in {"logsumexp", "member_logsumexp"}:
            return torch.logsumexp(scores.float(), dim=0)
        return scores.float().max()

    def _compute_task_role_pair_score(
        self,
        task_id,
        image_anchor,
        text_anchor,
        device,
    ):
        task_image_anchor = self._get_spectral_image_anchor(task_id, device=device)
        task_text_anchor = self._safe_normalize(
            self.text_anchors[task_id].detach()
        ).to(device)
        image_score = torch.dot(task_image_anchor, image_anchor.to(device))
        text_score = torch.dot(task_text_anchor, text_anchor.to(device))
        image_weight, text_weight = self._get_role_modal_weights()
        return image_weight * image_score + text_weight * text_score

    def _compute_role_pair_compatibility(
        self,
        role_id,
        active_experts,
        image_anchor,
        text_anchor,
        device,
    ):
        member_tasks = self._role_member_tasks(role_id, active_experts)
        if len(member_tasks) == 0:
            return torch.tensor(float("-inf"), device=device, dtype=torch.float32)
        pair_scores = torch.stack(
            [
                self._compute_task_role_pair_score(
                    task_id,
                    image_anchor,
                    text_anchor,
                    device,
                )
                for task_id in member_tasks
            ]
        )
        strategy = "task_affinity_complete_link"
        if strategy in {"average_link", "task_affinity_average_link"}:
            return pair_scores.mean()
        if strategy in {"single_link", "task_affinity_single_link"}:
            return pair_scores.max()
        return pair_scores.min()

    def _calibrate_role_pair_compatibility(self, pair_score):
        if not torch.is_tensor(pair_score):
            pair_score = torch.tensor(
                float(pair_score),
                device=self.device,
                dtype=torch.float32,
            )
        return pair_score.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)

    def _compute_role_assignment_member_support(
        self,
        expert_logits,
        member_tasks,
        active_task_count,
        mode,
        top_k,
        temperature,
    ):
        if expert_logits.numel() == 0 or len(member_tasks) == 0:
            return torch.tensor(
                0.0,
                device=expert_logits.device,
                dtype=torch.float32,
            )
        support_mode = str(
            self._get_relation_config().get(
                "role_assignment_member_support_mode",
                "excess_mass",
            )
        ).lower()
        task_probs = self._build_dense_relation_weights_with_temperature(
            expert_logits,
            temperature,
        )
        member_probs = task_probs[member_tasks]
        base_support = self._aggregate_role_member_scores(
            member_probs,
            mode=mode,
            top_k=top_k,
        ).float().clamp(0.0, 1.0)
        if support_mode in {
            "excess_mass",
            "excess-member-mass",
            "null_calibrated_mass",
            "null-calibrated-mass",
        }:
            member_mass = member_probs.sum().float().clamp(0.0, 1.0)
            active_task_count = max(int(active_task_count), 1)
            null_mass = min(
                float(len(member_tasks)) / float(active_task_count),
                1.0,
            )
            denom = max(1.0 - null_mass, 1e-6)
            excess_support = ((member_mass - null_mass) / denom).clamp(0.0, 1.0)
            if support_mode in {
                "blended_excess_mass",
                "blended-excess-mass",
                "soft_excess_mass",
                "soft-excess-mass",
            }:
                alpha = float(
                    self._get_relation_config().get(
                        "role_assignment_member_excess_alpha",
                        0.35,
                    )
                )
                alpha = max(0.0, min(alpha, 1.0))
                return ((1.0 - alpha) * base_support + alpha * excess_support).clamp(
                    0.0, 1.0
                )
            return excess_support
        return base_support

    def _score_roles(
        self,
        active_experts,
        expert_logits,
        image_anchor,
        text_anchor,
        device,
        return_details=False,
    ):
        self._sanitize_relation_memory()
        active_roles = min(self._get_active_role_count(), self._max_supported_role_slots())
        if active_roles == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            if return_details:
                return empty, empty, empty
            return empty

        role_text_bank = torch.stack(
            [self._safe_normalize(self.role_text_prototypes[idx].detach()).to(device) for idx in range(active_roles)],
            dim=0,
        )
        config = self._get_relation_config()
        if bool(config.get("eval_use_role_spectral_prototype", True)):
            config["use_spectral_role_prototype"] = True
        prototype_scores = torch.zeros(active_roles, device=device, dtype=torch.float32)
        if bool(config.get("use_text_anchor_routing", True)):
            prototype_scores = prototype_scores + float(
                config.get("text_weight", config.get("routing_text_weight", 0.5))
            ) * torch.matmul(role_text_bank, text_anchor.to(device))
        if bool(config.get("use_spectral_role_prototype", False)):
            if not self._spectral_role_bank_available(active_roles):
                raise RuntimeError(
                    "HiDESC eval requires populated role_spectral_prototypes, "
                    "but the spectral role bank is incomplete. Rebuild spectral role memory first."
                )
            role_spectral_bank = torch.stack(
                [
                    self._safe_normalize(
                        self.role_spectral_prototypes[idx].detach()
                    ).to(device)
                    for idx in range(active_roles)
                ],
                dim=0,
            )
            prototype_scores = prototype_scores + float(
                config.get(
                    "spectral_image_weight",
                    config.get("routing_image_weight", 0.5),
                )
            ) * torch.matmul(role_spectral_bank, image_anchor.to(device))

        # Eval role selection can optionally use the current sample's task/expert
        # scores. Prototype-only role selection is too restrictive for a task
        # that shares a role with another expert: once the wrong role wins,
        # late hard-selection makes all experts in the correct role unreachable.
        prototype_weight = float(
            config.get("eval_role_prototype_weight", config.get("routing_role_prototype_weight", 1.0))
        )
        task_weight = float(config.get("eval_role_task_weight", config.get("routing_role_task_weight", 0.0)))
        if task_weight > 0.0 and expert_logits.numel() >= active_experts:
            role_task_scores = torch.zeros_like(prototype_scores)
            score_mode = config.get("role_task_score_mode", "member_max")
            for role_id in range(active_roles):
                member_tasks = self._role_member_tasks(role_id, active_experts)
                if member_tasks:
                    role_task_scores[role_id] = self._aggregate_role_member_scores(
                        expert_logits[member_tasks], mode=score_mode
                    )
            prototype_scores = self._normalize_task_score_branch(
                prototype_scores.unsqueeze(0)
            ).squeeze(0)
            role_task_scores = self._normalize_task_score_branch(
                role_task_scores.unsqueeze(0)
            ).squeeze(0)
            total_weight = max(prototype_weight + task_weight, 1e-6)
            prototype_scores = (
                prototype_weight * prototype_scores
                + task_weight * role_task_scores
            ) / total_weight

        # Dataset-specific evaluation may know which continual-learning task
        # is being evaluated. Use that information as a soft role prior rather
        # than forcing every layer to one expert. This prevents a prototype
        # mismatch from making the target expert unreachable at late stages.
        target_expert = config.get("eval_target_expert", None)
        target_role_bias = float(config.get("eval_target_role_bias", 0.0))
        if target_expert is not None and target_role_bias != 0.0:
            target_expert = int(target_expert)
            if 0 <= target_expert < active_experts:
                target_membership = self.task_role_membership[target_expert, :active_roles]
                target_roles = torch.nonzero(target_membership > 0, as_tuple=False).flatten()
                if target_roles.numel() > 0:
                    prototype_scores[int(target_roles[0].item())] += target_role_bias

        masked_role_scores = self._mask_relation_logits(prototype_scores)
        if return_details:
            return masked_role_scores, prototype_scores, torch.zeros_like(prototype_scores)
        return masked_role_scores

    def _spectral_role_bank_available(self, active_roles):
        if active_roles <= 0:
            return False
        prototypes = torch.stack(
            [
                self.role_spectral_prototypes[idx].detach().float().reshape(-1)
                for idx in range(active_roles)
            ],
            dim=0,
        )
        sizes = torch.nan_to_num(
            self.role_task_count[:active_roles].detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prototype_norms = torch.linalg.norm(prototypes, dim=-1)
        return bool(
            torch.isfinite(prototype_norms).all()
            and torch.isfinite(sizes).all()
            and torch.all(sizes > 0)
            and torch.all(prototype_norms > 0)
        )

    def _build_stage_role_weights(self, role_scores, temperature, uniform_mix=0.0):
        if role_scores.numel() == 0:
            return role_scores
        role_weights = self._build_dense_relation_weights_with_temperature(
            role_scores,
            temperature,
        )
        return self._mix_with_uniform(role_weights, uniform_mix)

    def _build_role_conditioned_task_weights(
        self,
        active_experts,
        task_scores,
        role_scores,
        role_temperature,
        task_temperature,
        role_strength,
        role_uniform_mix=0.0,
    ):
        if role_scores.numel() == 0:
            return self._fallback_expert_weights(task_scores, active_experts)

        config = self._get_relation_config()
        role_weights = self._build_stage_role_weights(
            role_scores,
            role_temperature,
            role_uniform_mix,
        )
        role_support = torch.zeros(
            active_experts,
            device=task_scores.device,
            dtype=torch.float32,
        )
        for role_id, role_weight in enumerate(role_weights):
            member_tasks = self._role_member_tasks(role_id, active_experts)
            if len(member_tasks) == 0:
                continue
            member_prior = role_weight / float(len(member_tasks))
            role_support[member_tasks] += member_prior

        if float(role_support.sum().item()) <= 0.0:
            return self._fallback_expert_weights(task_scores, active_experts)

        role_support = self._normalize_route_weights(role_support)
        role_task_floor = float(
            config.get("routing_role_task_floor", 0.05)
        )
        role_task_floor = max(0.0, min(1.0, role_task_floor))
        if role_task_floor > 0.0:
            role_support = self._normalize_route_weights(
                (1.0 - role_task_floor) * role_support
                + role_task_floor
                * self._build_uniform_route_weights(
                    active_experts,
                    task_scores.device,
                )
            )
        if str(config.get("eval_middle_role_inner_mode", "task_weighted")).lower() == "uniform":
            return role_support
        task_logits = task_scores / max(float(task_temperature), 1e-6)
        task_logits = task_logits + float(role_strength) * torch.log(
            role_support.clamp_min(1e-6)
        )
        return self._normalize_route_weights(F.softmax(task_logits, dim=0))

    def _assign_roles_from_sample(self, active_experts, expert_logits, image_anchor, text_anchor):
        device = image_anchor.device
        active_roles = min(self._get_active_role_count(), self._max_supported_role_slots())
        if active_roles == 0:
            return torch.empty(0, device=image_anchor.device), [], True

        config = self._get_relation_config()
        pair_threshold = float(
            config.get("role_assignment_min_similarity", 0.85)
        )
        birth_threshold = float(config.get("role_birth_threshold", 0.80))
        pair_weight = float(
            max(0.0, min(1.0, config.get("role_assignment_pair_weight", 0.50)))
        )
        assignment_mode = "member_max"
        member_top_k = int(config.get("role_member_top_k", 2))
        member_temperature = float(
            config.get("role_assignment_member_temperature", 0.35)
        )

        candidate_scores = []
        candidate_roles = []
        for role_id in range(active_roles):
            member_tasks = self._role_member_tasks(role_id, active_experts)
            if len(member_tasks) == 0:
                continue
            pair_score = self._compute_role_pair_compatibility(
                role_id,
                active_experts,
                image_anchor,
                text_anchor,
                device,
            )
            pair_score = self._calibrate_role_pair_compatibility(pair_score)
            if float(pair_score.item()) < pair_threshold:
                continue
            member_score = self._compute_role_assignment_member_support(
                expert_logits,
                member_tasks,
                active_experts,
                mode=assignment_mode,
                top_k=member_top_k,
                temperature=member_temperature,
            )
            combined_score = (
                (1.0 - pair_weight) * member_score
                + pair_weight * pair_score
            )
            candidate_roles.append(role_id)
            candidate_scores.append(combined_score)

        if not candidate_scores:
            return torch.empty(0, device=image_anchor.device), [], True

        role_scores = torch.stack(candidate_scores, dim=0)
        top_value, _ = torch.topk(role_scores, k=1)
        best_score = float(top_value[0].item())
        score_margin = float(config.get("role_assignment_margin", 0.02))
        if best_score < birth_threshold:
            return torch.empty(0, device=image_anchor.device), [], True

        if role_scores.numel() > 1 and score_margin > 0.0:
            top2_values = torch.topk(role_scores, k=2).values
            if (
                float((top2_values[0] - top2_values[1]).item()) < score_margin
                and float(top2_values[0].item()) < birth_threshold + score_margin
            ):
                return torch.empty(0, device=image_anchor.device), [], True

        top_m = min(
            max(1, int(config.get("role_assignment_top_k", 2))),
            role_scores.numel(),
        )
        top_values, top_indices = torch.topk(role_scores, k=top_m)
        if top_m == 1:
            membership = torch.ones(1, device=image_anchor.device, dtype=torch.float32)
        else:
            membership = F.softmax(
                top_values / max(float(self._get_relation_config()["routing_temperature"]), 1e-6),
                dim=0,
            )
        selected_roles = [candidate_roles[index] for index in top_indices.tolist()]
        return membership, selected_roles, False

    def _fallback_expert_weights(self, expert_logits, active_experts, sparse_top_k=None):
        if expert_logits.numel() == 0:
            identity = torch.zeros(active_experts, device=self.device, dtype=torch.float32)
            identity[0] = 1.0
            return identity
        if sparse_top_k is None:
            weights = F.softmax(
                expert_logits / max(float(self._get_relation_config()["routing_temperature"]), 1e-6),
                dim=0,
            )
        else:
            weights = self._build_sparse_relation_weights(expert_logits, sparse_top_k)
        return self._normalize_route_weights(weights)

    def _detail_route_confidence(self, task_scores, role_scores):
        """Return an input-conditioned confidence for detail-analysis routing.

        This is deliberately computed from the same task/role affinities already
        used by HiDESC.  It adds no trainable router and does not use task IDs or
        labels.  High confidence makes the adaptive policy sharper; uncertainty
        keeps neighboring stage policies mixed.
        """
        def normalized_confidence(values):
            if values.numel() <= 1:
                return values.new_ones(())
            probs = F.softmax(values.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
            return (1.0 - entropy / np.log(float(values.numel()))).clamp(0.0, 1.0)

        task_conf = normalized_confidence(task_scores)
        role_conf = normalized_confidence(role_scores)
        return (0.5 * task_conf + 0.5 * role_conf).clamp(0.0, 1.0)

    def _detail_stage_position_prior(self, total_layers, device, dtype):
        """Build a smooth positional prior from the existing stage-1 schedule.

        The prior is only a structural reference.  The actual per-layer policy
        remains input-conditioned through the task/role confidence and the
        stage basis vectors.  This lets Full Layer Adaptive use every layer
        without introducing a learned or label-dependent router.
        """
        schedule = self._get_stage1_band_schedule() or {}

        def center(name, fallback):
            value = schedule.get(name)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return 0.5 * (float(value[0]) + float(value[1]))
            return fallback

        early_center = center("early_core", max(1.0, total_layers * 0.25))
        middle_center = center("middle_core", max(1.0, total_layers * 0.70))
        late_center = center("late_core", float(total_layers))
        positions = torch.arange(1, total_layers + 1, device=device, dtype=dtype)

        # These widths are deliberately broad: the schedule supplies the
        # approximate transition locations while the input confidence controls
        # how sharply the policy commits to one stage.
        early_sigma = max(2.0, float(schedule.get("detail_early_sigma", 8.0)))
        middle_sigma = max(2.0, float(schedule.get("detail_middle_sigma", 7.0)))
        late_sigma = max(1.5, float(schedule.get("detail_late_sigma", 4.0)))
        logits = torch.stack(
            [
                -((positions - early_center).abs() / early_sigma),
                -((positions - middle_center).abs() / middle_sigma),
                -((positions - late_center).abs() / late_sigma),
            ],
            dim=-1,
        )
        return F.softmax(logits, dim=-1)

    def _apply_detail_route_policy(
        self,
        per_layer,
        early,
        middle,
        late,
        task_scores,
        role_scores,
        policy,
    ):
        """Apply post-hoc intra-band or full-layer adaptive routing.

        ``intra_band`` preserves the existing stage assignment and adapts only
        the strength of each layer's stage policy.  ``full_layer`` removes the
        hard stage assignment and mixes all three stage basis vectors for every
        layer using an input-conditioned soft stage gate.
        """
        config = self._get_relation_config()
        policy = str(policy).lower()
        if policy not in {"intra_band", "full_layer"}:
            return per_layer, {"policy": "fixed"}

        total_layers = len(per_layer)
        device = task_scores.device
        dtype = task_scores.dtype
        confidence = self._detail_route_confidence(task_scores, role_scores)
        strength = float(config.get("detail_adaptive_strength", 0.35))
        strength = max(0.0, min(1.0, strength))
        temperature = max(float(config.get("detail_stage_temperature", 0.75)), 1e-4)
        task_base = self._normalize_route_weights(task_scores)
        bases = torch.stack([early, middle, late], dim=0)
        position_prior = self._detail_stage_position_prior(total_layers, device, dtype)

        if policy == "full_layer":
            # Confidence-dependent sharpening makes the stage choice genuinely
            # input-conditioned while preserving the existing stage policies.
            adaptive_temperature = temperature / (0.5 + confidence)
            stage_probs = F.softmax(
                position_prior.clamp_min(1e-8).log() / adaptive_temperature,
                dim=-1,
            )
            smoothing = float(config.get("detail_layer_smoothing", 0.10))
            smoothing = max(0.0, min(0.45, smoothing))
            if smoothing > 0.0 and total_layers > 2:
                padded = F.pad(stage_probs.T.unsqueeze(0), (1, 1), mode="replicate")[0].T
                neighbor = 0.5 * (padded[:-2] + padded[2:])
                stage_probs[1:-1] = (
                    (1.0 - smoothing) * stage_probs[1:-1]
                    + smoothing * neighbor[1:-1]
                )
                stage_probs = stage_probs / stage_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            adaptive = [
                self._normalize_route_weights(torch.matmul(stage_probs[layer_idx], bases))
                for layer_idx in range(total_layers)
            ]
            effective = stage_probs.detach().float()
        else:
            # Preserve the existing hard stage/transition schedule.  The
            # layer-specific coefficient controls only how much the current
            # layer trusts its stage basis versus the global task-affinity base.
            adaptive = []
            effective = torch.zeros(total_layers, 3, device=device, dtype=torch.float32)
            for layer_idx, fixed_weights in enumerate(per_layer):
                region = self._get_stage1_band_region(layer_idx)
                if region in {"early", "b1"}:
                    stage_index = 0
                elif region in {"middle", "b2"}:
                    stage_index = 1
                else:
                    stage_index = 2
                local_confidence = position_prior[layer_idx, stage_index]
                alpha = strength * (
                    (1.0 - confidence) * 0.5 + confidence * local_confidence
                )
                alpha = alpha.clamp(0.0, strength)
                adaptive.append(
                    self._normalize_route_weights(
                        (1.0 - alpha) * task_base + alpha * fixed_weights
                    )
                )
                effective[layer_idx, stage_index] = 1.0

        effective_positions = torch.arange(1, total_layers + 1, device=device, dtype=torch.float32)
        early_mass = effective[:, 0]
        middle_mass = effective[:, 1]
        late_mass = effective[:, 2]
        b1 = float((effective_positions * middle_mass).sum().item() / middle_mass.sum().clamp_min(1e-6).item())
        b2 = float((effective_positions * late_mass).sum().item() / late_mass.sum().clamp_min(1e-6).item())
        return adaptive, {
            "policy": policy,
            "confidence": float(confidence.detach().item()),
            "stage_probabilities": effective.detach().float().cpu().tolist(),
            "effective_b1": b1,
            "effective_b2": b2,
        }

    def _scatter_local_weights(self, active_experts, member_tasks, local_weights, device):
        weights = torch.zeros(active_experts, device=device, dtype=torch.float32)
        for local_idx, task_id in enumerate(member_tasks):
            weights[task_id] = local_weights[local_idx]
        return self._normalize_route_weights(weights)

    def _build_progressive_route_plan(self, active_experts, task_scores, image_summary, text_summary):
        device = task_scores.device
        if active_experts <= 1 or task_scores.numel() <= 1:
            identity = torch.zeros(active_experts, device=device, dtype=torch.float32)
            identity[0] = 1.0
            per_layer = [identity for _ in range(len(self.model.layers))]
            return {
                "early_basis": identity,
                "middle_basis": identity,
                "late_basis": identity,
                "per_layer": per_layer,
                "candidate_experts": [0],
            }

        role_scores = self._score_roles(
            active_experts,
            task_scores,
            image_summary,
            text_summary,
            image_summary.device,
        )
        if role_scores.numel() == 0:
            raise RuntimeError(
                "HiDESC routing requires an initialized role prototype bank."
            )

        config = self._get_relation_config()
        early = self._build_role_conditioned_task_weights(
            active_experts,
            task_scores,
            role_scores,
            config.get("routing_early_role_temperature", 0.12),
            config.get("routing_early_task_temperature", 0.18),
            config.get("routing_early_role_strength", 0.20),
            config.get("routing_early_uniform_mix", 0.15),
        )
        middle = self._build_role_conditioned_task_weights(
            active_experts,
            task_scores,
            role_scores,
            config.get("routing_middle_role_temperature", 0.20),
            config.get("routing_middle_temperature", 0.10),
            config.get("routing_middle_role_strength", 0.12),
            config.get("routing_middle_role_uniform_mix", 0.10),
        )
        late = self._build_role_conditioned_task_weights(
            active_experts,
            task_scores,
            role_scores,
            config.get("routing_late_role_temperature", 0.12),
            config.get("routing_late_task_temperature", 0.05),
            config.get("routing_late_role_strength", 0.06),
            config.get("routing_late_role_uniform_mix", 0.0),
        )
        candidate_experts = [int(idx) for idx in torch.nonzero(late > 0, as_tuple=False).flatten().tolist()]
        per_layer = []
        for layer_idx in range(len(self.model.layers)):
            region = self._get_stage1_band_region(layer_idx)
            alpha = self._get_stage1_band_alpha(layer_idx)
            if region == "early":
                layer_weights = early
            elif region == "middle":
                layer_weights = middle
            elif region == "late":
                layer_weights = late
            elif region == "b1":
                alpha = 0.5 if alpha is None else alpha
                layer_weights = self._normalize_route_weights(
                    (1.0 - alpha) * early + alpha * middle
                )
            elif region == "b2":
                alpha = 0.5 if alpha is None else alpha
                layer_weights = self._normalize_route_weights(
                    (1.0 - alpha) * middle + alpha * late
                )
            else:
                stage = self._get_layer_stage(layer_idx, len(self.model.layers))
                layer_weights = {
                    "early": early,
                    "middle": middle,
                    "late": late,
                }[stage]
            per_layer.append(layer_weights)
        detail_policy = config.get("detail_route_policy", "fixed")
        detail_diagnostics = {"policy": "fixed"}
        if str(detail_policy).lower() in {"intra_band", "full_layer"}:
            per_layer, detail_diagnostics = self._apply_detail_route_policy(
                per_layer=per_layer,
                early=early,
                middle=middle,
                late=late,
                task_scores=task_scores,
                role_scores=role_scores,
                policy=detail_policy,
            )
        return {
            "early_basis": early,
            "middle_basis": middle,
            "late_basis": late,
            "per_layer": per_layer,
            "candidate_experts": candidate_experts,
            "role_scores": role_scores,
            "detail_diagnostics": detail_diagnostics,
        }

    def _get_layer_stage(self, layer_idx, total_layers):
        config = self._get_relation_config()
        late_layers = min(max(1, int(config.get("routing_late_layers", 1))), total_layers)
        available_before_late = max(0, total_layers - late_layers)
        early_layers = min(max(0, int(config["routing_early_layers"])), available_before_late)
        middle_layers = min(
            max(0, int(config["routing_middle_layers"])),
            max(0, available_before_late - early_layers),
        )
        if layer_idx < early_layers:
            return "early"
        if layer_idx < early_layers + middle_layers:
            return "middle"
        return "late"

    def _apply_relation_weights_to_experts(self, route_plan):
        proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for layer_idx, layer in enumerate(self.model.layers):
            if "per_layer" in route_plan:
                active_weights = route_plan["per_layer"][layer_idx]
                stage = self._get_stage1_band_region(layer_idx) or self._get_layer_stage(
                    layer_idx,
                    len(self.model.layers),
                )
            else:
                stage = self._get_layer_stage(layer_idx, len(self.model.layers))
                active_weights = route_plan[stage]
            for proj_name in proj_names:
                if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    proj_layer = getattr(layer.self_attn, proj_name)
                else:
                    proj_layer = getattr(layer.mlp, proj_name)
                if hasattr(proj_layer, "expert_weight"):
                    proj_layer.task_fuse_weight = active_weights
                    proj_layer.expert_weight = active_weights
                    proj_layer.progressive_stage = stage

    def _commit_task_to_roles(
        self,
        task_id,
        task_image_anchor,
        task_text_anchor,
        membership,
        candidate_roles,
        role_birth,
    ):
        if task_id < 0 or task_id >= self.max_task_slots:
            return
        if self._get_active_role_count() == 0 or role_birth or len(candidate_roles) == 0:
            role_id = min(self._get_active_role_count(), self.max_role_slots - 1)
            self.role_spectral_prototypes[role_id].data.copy_(
                task_image_anchor.unsqueeze(0).to(
                    self.role_spectral_prototypes[role_id].dtype
                )
            )
            self.role_text_prototypes[role_id].data.copy_(task_text_anchor.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
            self.role_task_count.data[role_id] = max(1.0, float(self.role_task_count[role_id].detach().item()))
            self.role_usage_prior.data[role_id] = max(1.0, float(self.role_usage_prior[role_id].detach().item()))
            self.task_role_membership.data[task_id].zero_()
            self.task_role_membership.data[task_id, role_id] = 1.0
            self._set_active_role_count(max(self._get_active_role_count(), role_id + 1))
        else:
            momentum = float(self._get_relation_config()["routing_prior_momentum"])
            self.task_role_membership.data[task_id].zero_()
            for local_idx, role_id in enumerate(candidate_roles):
                weight = float(membership[local_idx].detach().item())
                if weight <= 0.0:
                    continue
                count = float(self.role_task_count[role_id].detach().item())
                updated_count = count + weight
                updated_spectral = (
                    count
                    * self._safe_normalize(
                        self.role_spectral_prototypes[role_id].detach()
                    )
                    + weight * task_image_anchor
                ) / max(updated_count, 1e-6)
                updated_text = (
                    count * self._safe_normalize(self.role_text_prototypes[role_id].detach())
                    + weight * task_text_anchor
                ) / max(updated_count, 1e-6)
                self.role_spectral_prototypes[role_id].data.copy_(
                    updated_spectral.unsqueeze(0).to(
                        self.role_spectral_prototypes[role_id].dtype
                    )
                )
                self.role_text_prototypes[role_id].data.copy_(updated_text.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
                self.role_task_count.data[role_id] = updated_count
                self.role_usage_prior.data[role_id] = momentum * self.role_usage_prior[role_id].detach().float() + (1.0 - momentum) * weight
                self.task_role_membership.data[task_id, role_id] = weight
        self.expert_usage_prior.data[task_id] = max(1.0, float(self.expert_usage_prior[task_id].detach().item()))

    def ensure_role_bank_initialized(self, completed_task_count):
        rebuild_debug = not getattr(self, "_hidesc_rebuild_debug_logged", False)
        rebuild_before = {
            "active_role_count": self._get_active_role_count(),
            "role_task_count_nonzero": int(
                (torch.nan_to_num(self.role_task_count.detach().float()) > 0).sum().item()
            ),
            "membership_nonzero": int(
                (torch.nan_to_num(self.task_role_membership.detach().float()) != 0).sum().item()
            ),
        }

        def log_rebuild(stage):
            if not rebuild_debug:
                return
            image_norms = [
                float(torch.nan_to_num(x.detach().float()).norm().item())
                for x in self.spectral_image_anchors[:completed_task_count]
            ]
            text_norms = [
                float(torch.nan_to_num(x.detach().float()).norm().item())
                for x in self.text_anchors[:completed_task_count]
            ]
            print(
                "HiDESC eval role-bank rebuild:",
                f"stage={stage}",
                f"completed_task_count={completed_task_count}",
                f"before={rebuild_before}",
                f"after_active_role_count={self._get_active_role_count()}",
                f"after_role_task_count_nonzero={int((torch.nan_to_num(self.role_task_count.detach().float()) > 0).sum().item())}",
                f"after_membership_nonzero={int((torch.nan_to_num(self.task_role_membership.detach().float()) != 0).sum().item())}",
                f"image_anchor_norms={image_norms}",
                f"text_anchor_norms={text_norms}",
            )
            self._hidesc_rebuild_debug_logged = True

        if (
            bool(self._get_relation_config().get("role_reset_on_strategy_change", False))
            and not getattr(self, "_role_memory_reset_applied", False)
        ):
            self._reset_role_memory()
            self._role_memory_reset_applied = True
        self._sanitize_relation_memory()
        completed_task_count = min(int(completed_task_count), self.max_task_slots)
        if completed_task_count <= 0:
            log_rebuild("skip_no_completed_tasks")
            return
        current_active_roles = self._get_active_role_count()
        inferred_active_roles = self._infer_active_role_count(completed_task_count)
        if current_active_roles > 0:
            if inferred_active_roles > 0:
                self._set_active_role_count(max(current_active_roles, inferred_active_roles))
                log_rebuild("reuse_existing_role_bank")
                return
            # Older checkpoints may carry a stale active_role_count without any task-role membership.
            self._set_active_role_count(0)
        existing_membership = self.task_role_membership[:completed_task_count].detach().abs().sum().item()
        if existing_membership > 0:
            self._set_active_role_count(inferred_active_roles)
            log_rebuild("reuse_existing_membership")
            return

        for task_id in range(completed_task_count):
            (
                task_image_anchor,
                task_text_anchor,
            ) = self._get_task_role_anchors(task_id)
            if task_id == 0 and self._get_active_role_count() == 0:
                self._commit_task_to_roles(
                    task_id,
                    task_image_anchor,
                    task_text_anchor,
                    None,
                    [],
                    True,
                )
                continue
            expert_logits = self._score_tasks(
                list(range(task_id)),
                task_image_anchor,
                task_text_anchor,
                task_image_anchor.device,
            )
            membership, candidate_roles, role_birth = self._assign_roles_from_sample(
                task_id,
                expert_logits,
                task_image_anchor,
                task_text_anchor,
            )
            self._commit_task_to_roles(
                task_id,
                task_image_anchor,
                task_text_anchor,
                membership,
                candidate_roles,
                role_birth,
            )
        log_rebuild("rebuilt_from_task_anchors")

    def _finalize_current_task_role_memory_impl(self):
        task_id = min(max(int(self.cur_task), 0), self.max_task_slots - 1)
        if task_id > 0:
            self.ensure_role_bank_initialized(task_id)
        (
            task_image_anchor,
            task_text_anchor,
        ) = self._get_task_role_anchors(task_id)
        if task_id == 0 and self._get_active_role_count() == 0:
            self._commit_task_to_roles(
                task_id,
                task_image_anchor,
                task_text_anchor,
                None,
                [],
                True,
            )
            return
        expert_logits = self._score_tasks(
            list(range(task_id)),
            task_image_anchor,
            task_text_anchor,
            task_image_anchor.device,
        )
        membership, candidate_roles, role_birth = self._assign_roles_from_sample(
            task_id,
            expert_logits,
            task_image_anchor,
            task_text_anchor,
        )
        self._commit_task_to_roles(
            task_id,
            task_image_anchor,
            task_text_anchor,
            membership,
            candidate_roles,
            role_birth,
        )

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        return_token_masks=False,
    ):
        
        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            if return_token_masks:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            (
                clip_image_features,
                image_features,
                final_patch_features,
                projected_patch_features,
            ) = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            clip_image_features = torch.split(
                clip_image_features, split_sizes, dim=0
            )
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
            clip_image_features = torch.stack(
                [x.float().mean(dim=0) for x in clip_image_features], dim=0
            )
            image_spectral_features = self._extract_image_spectral_descriptor(
                projected_patch_features
            )
            image_spectral_features = torch.stack(
                [
                    x.float().mean(dim=0)
                    for x in torch.split(image_spectral_features, split_sizes, dim=0)
                ],
                dim=0,
            )
        else:
            (
                clip_image_features,
                image_features,
                final_patch_features,
                projected_patch_features,
            ) = self.encode_images(images)
            image_spectral_features = self._extract_image_spectral_descriptor(
                projected_patch_features
            )

        text_tower = self.get_text_tower()

        # with torch.no_grad():
        #     # image_guide_features: bs, 4096
        #     image_guide_features = image_features[:,0]
        
        input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,input_ids.cpu().detach().numpy(),self.tokenizer.pad_token_id)
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

        clip_text_inputs = self.clip_tokenizer(
                decoded_clip_inputs,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

        # text_guide_features: bs, 768
        text_guide_features = text_tower(clip_text_inputs)
        text_activation_features = self._extract_text_activation_index(
            text_guide_features
        )

        if self.training or getattr(self, "cache_extraction_mode", False):
            if not getattr(self, "disable_anchor_update", False):
                task_id = self.cur_task

                self._update_running_text_activation_index(
                    self.text_anchors[task_id],
                    self.text_boundary[task_id],
                    text_guide_features,
                )
                self._update_running_prototype(
                    self.spectral_image_anchors[task_id],
                    self.spectral_image_boundary[task_id],
                    image_spectral_features,
                )
        else:
            active_experts = self._get_available_eval_expert_count(
                requested_experts=int(self.expert_num)
            )
            if active_experts <= 0:
                raise RuntimeError(
                    "No trained experts with the required eval anchors are available. "
                    "Check text/spectral anchor boundaries in the loaded checkpoint."
                )
            self.ensure_role_bank_initialized(active_experts)
            image_summary = self._summarize_guide_features(image_spectral_features)
            text_summary = self._summarize_guide_features(text_activation_features)
            task_scores = self._compute_shared_task_scores(
                image_guide_features=image_spectral_features,
                text_guide_features=text_activation_features,
                active_experts=active_experts,
            )
            if task_scores.ndim != 1:
                raise RuntimeError(
                    "HiDESC relation routing currently supports batch_size=1 "
                    f"because expert weights are shared across the batch; got "
                    f"task_scores shape {tuple(task_scores.shape)}."
                )
            route_plan = self._build_progressive_route_plan(
                active_experts,
                task_scores,
                image_summary,
                text_summary,
            )
            self._apply_relation_weights_to_experts(route_plan)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_image_token_masks = []
        new_text_token_masks = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_image_token_masks.append(
                    torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
                )
                new_text_token_masks.append(
                    torch.ones(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
                )
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_image_mask_parts = []
            cur_text_mask_parts = []

            for i in range(num_images + 1):
                cur_text_embeds = cur_input_embeds_no_im[i]
                cur_new_input_embeds.append(cur_text_embeds)
                cur_new_labels.append(cur_labels_noim[i])
                cur_image_mask_parts.append(
                    torch.zeros(cur_text_embeds.shape[0], dtype=torch.bool, device=cur_text_embeds.device)
                )
                cur_text_mask_parts.append(
                    torch.ones(cur_text_embeds.shape[0], dtype=torch.bool, device=cur_text_embeds.device)
                )
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_image_mask_parts.append(
                        torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device)
                    )
                    cur_text_mask_parts.append(
                        torch.zeros(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device)
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_image_token_mask = torch.cat(cur_image_mask_parts)
            cur_text_token_mask = torch.cat(cur_text_mask_parts)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_image_token_masks.append(cur_image_token_mask)
            new_text_token_masks.append(cur_text_token_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_image_token_masks = [x[:tokenizer_model_max_length] for x in new_image_token_masks]
            new_text_token_masks = [x[:tokenizer_model_max_length] for x in new_text_token_masks]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_image_token_masks_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
        new_text_token_masks_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_image_token_mask, cur_text_token_mask) in enumerate(
            zip(new_input_embeds, new_labels, new_image_token_masks, new_text_token_masks)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_image_token_masks_padded[i, -cur_len:] = cur_image_token_mask
                    new_text_token_masks_padded[i, -cur_len:] = cur_text_token_mask
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_image_token_masks_padded[i, :cur_len] = cur_image_token_mask
                    new_text_token_masks_padded[i, :cur_len] = cur_text_token_mask
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if return_token_masks:
            return (
                None,
                position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                new_image_token_masks_padded,
                new_text_token_masks_padded,
            )

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
