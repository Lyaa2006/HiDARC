import os
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from torch.utils.data import Sampler
from llava.train.description_utils import select_expanded_description_tokens

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import Dict, List, Optional

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    @contextmanager
    def _temporary_anchor_update(self, model, enabled: bool):
        wrapped = getattr(model, "module", model)
        had_attr = hasattr(wrapped, "disable_anchor_update")
        prev_value = getattr(wrapped, "disable_anchor_update", None)
        try:
            wrapped.disable_anchor_update = not bool(enabled)
            yield
        finally:
            if had_attr:
                wrapped.disable_anchor_update = prev_value
            else:
                try:
                    delattr(wrapped, "disable_anchor_update")
                except AttributeError:
                    pass

    def _unwrap_model(self, model):
        return getattr(model, "module", model)

    def _get_transformer_layers(self, model):
        wrapped = self._unwrap_model(model)
        candidates = [
            wrapped,
            getattr(wrapped, "model", None),
            getattr(getattr(wrapped, "base_model", None), "model", None),
            getattr(getattr(getattr(wrapped, "base_model", None), "model", None), "model", None),
        ]
        if hasattr(wrapped, "get_model"):
            candidates.append(wrapped.get_model())
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "layers"):
                return candidate.layers
            inner_model = getattr(candidate, "model", None)
            if inner_model is not None and hasattr(inner_model, "layers"):
                return inner_model.layers
        raise ValueError("Could not locate decoder layers for HiDESC loss bands.")

    def _get_num_decoder_layers(self, model):
        return len(self._get_transformer_layers(model))

    def _resolve_loss_band(self, model, low_layer, high_layer, name):
        num_layers = self._get_num_decoder_layers(model)
        low_layer = int(low_layer)
        high_layer = int(high_layer)
        if low_layer > high_layer:
            raise ValueError(
                f"{name} is invalid: low layer {low_layer} exceeds high layer {high_layer}."
            )
        if low_layer < 1 or high_layer > num_layers:
            raise ValueError(
                f"{name} must stay within decoder layers [1, {num_layers}], "
                f"got [{low_layer}, {high_layer}]."
            )
        return list(range(low_layer, high_layer + 1))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _pad_description_sequences(self, sequences, dtype=None):
        max_len = max(seq.shape[0] for seq in sequences)
        hidden_size = sequences[0].shape[-1]
        device = sequences[0].device
        if dtype is None:
            dtype = sequences[0].dtype
        padded = torch.zeros((len(sequences), max_len, hidden_size), dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
        for idx, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded[idx, :seq_len] = seq.to(dtype=dtype)
            mask[idx, :seq_len] = True
        return padded, mask

    def _pad_description_key_masks(self, masks, max_len, device):
        if masks is None:
            return None
        padded = torch.zeros((len(masks), max_len), dtype=torch.bool, device=device)
        for idx, mask in enumerate(masks):
            seq_len = min(mask.shape[0], max_len)
            padded[idx, :seq_len] = mask[:seq_len].to(device=device, dtype=torch.bool)
        return padded

    def _masked_mean_pool(self, hidden_states, mask):
        if hidden_states.shape[1] != mask.shape[1]:
            shared_seq_len = min(hidden_states.shape[1], mask.shape[1])
            padding_side = getattr(self._unwrap_model(self.model).config, "tokenizer_padding_side", "right")
            if padding_side == "left":
                hidden_states = hidden_states[:, -shared_seq_len:]
                mask = mask[:, -shared_seq_len:]
            else:
                hidden_states = hidden_states[:, :shared_seq_len]
                mask = mask[:, :shared_seq_len]
        mask = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _prepare_answer_multimodal_inputs(self, model, inputs):
        wrapped = self._unwrap_model(model)
        return wrapped.prepare_inputs_labels_for_multimodal(
            inputs["input_ids"],
            inputs.get("position_ids"),
            inputs["attention_mask"],
            None,
            inputs["labels"],
            inputs.get("images"),
            return_token_masks=True,
        )

    def _prepare_description_multimodal_inputs(self, model, inputs):
        wrapped = self._unwrap_model(model)
        return wrapped.prepare_inputs_labels_for_multimodal(
            inputs["description_input_ids"],
            None,
            inputs["description_attention_mask"],
            None,
            None,
            inputs.get("images"),
            return_token_masks=True,
        )

    def _compute_align_loss_for_hidden(self, hidden_states, image_token_mask, text_token_mask):
        zero = hidden_states.new_zeros(())
        if image_token_mask is None or text_token_mask is None:
            return zero, zero, zero

        shared_seq_len = min(hidden_states.shape[1], image_token_mask.shape[1], text_token_mask.shape[1])
        padding_side = getattr(self._unwrap_model(self.model).config, "tokenizer_padding_side", "right")
        if padding_side == "left":
            hidden_states = hidden_states[:, -shared_seq_len:]
            image_token_mask = image_token_mask[:, -shared_seq_len:]
            text_token_mask = text_token_mask[:, -shared_seq_len:]
        else:
            hidden_states = hidden_states[:, :shared_seq_len]
            image_token_mask = image_token_mask[:, :shared_seq_len]
            text_token_mask = text_token_mask[:, :shared_seq_len]

        image_token_mask = image_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        valid_samples = image_token_mask.any(dim=1) & text_token_mask.any(dim=1)
        if not torch.any(valid_samples):
            return zero, zero, zero

        pooled_image = self._masked_mean_pool(hidden_states.float(), image_token_mask)
        pooled_text = self._masked_mean_pool(hidden_states.float(), text_token_mask)
        pooled_image = F.normalize(pooled_image[valid_samples], dim=-1)
        pooled_text = F.normalize(pooled_text[valid_samples], dim=-1)
        cosine = (pooled_image * pooled_text).sum(dim=-1).clamp(-1.0, 1.0)
        align_loss = (1.0 - cosine).mean()
        return align_loss, valid_samples.float().sum(), cosine.mean()

    def _get_band_ema_state(self, attr_name):
        state = getattr(self, attr_name, None)
        if state is None:
            state = {}
            setattr(self, attr_name, state)
        return state

    def _compute_band_weights(self, losses, band_layers, ema_attr_name, eta):
        if not losses:
            raise ValueError("Band weight computation requires at least one layer loss.")
        ema_state = self._get_band_ema_state(ema_attr_name)
        gamma = float(getattr(self.args, "loss_band_ema_gamma", 0.9))
        position_eps = float(getattr(self.args, "loss_band_position_eps", 0.05))
        difficulties = torch.stack([loss.detach().float() for loss in losses])
        ema_values = []
        for layer, difficulty in zip(band_layers, difficulties):
            current_value = float(difficulty.item())
            previous_value = ema_state.get(layer, current_value)
            updated_value = gamma * previous_value + (1.0 - gamma) * current_value
            ema_state[layer] = updated_value
            ema_values.append(updated_value)
        ema_tensor = difficulties.new_tensor(ema_values)
        normalized = (ema_tensor - ema_tensor.mean()) / ema_tensor.std(unbiased=False).clamp_min(1e-6)
        if len(band_layers) == 1:
            prior = difficulties.new_ones((1,))
        else:
            ramp = torch.linspace(0.0, 1.0, steps=len(band_layers), device=difficulties.device)
            prior = position_eps + (1.0 - position_eps) * ramp
        logits = torch.log(prior.clamp_min(1e-6)) + float(eta) * normalized
        return F.softmax(logits, dim=0)

    def _compute_align_band_loss(self, hidden_states, image_token_mask, text_token_mask, band_layers):
        losses = []
        cosines = []
        align_valid_samples = None
        for layer in band_layers:
            layer_loss, valid_samples, cosine = self._compute_align_loss_for_hidden(
                hidden_states[layer],
                image_token_mask,
                text_token_mask,
            )
            losses.append(layer_loss)
            cosines.append(cosine)
            if align_valid_samples is None:
                align_valid_samples = valid_samples
        weights = self._compute_band_weights(
            losses,
            band_layers,
            "_align_band_ema_state",
            getattr(self.args, "align_band_eta", 0.5),
        )
        weighted_loss = sum(weight * loss for weight, loss in zip(weights, losses))
        weighted_cosine = sum(weight * cosine for weight, cosine in zip(weights, cosines))
        return weighted_loss, align_valid_samples, weighted_cosine, weights, losses

    def _extract_description_states(self, model, inputs, band_layers):
        with self._temporary_anchor_update(model, enabled=False):
            (
                _,
                description_position_ids,
                description_attention_mask,
                description_past_key_values,
                description_inputs_embeds,
                _,
                _,
                expanded_text_mask,
            ) = self._prepare_description_multimodal_inputs(model, inputs)
            description_outputs = model(
                input_ids=None,
                attention_mask=description_attention_mask,
                position_ids=description_position_ids,
                past_key_values=description_past_key_values,
                inputs_embeds=description_inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        description_sequences = {}
        description_key_masks = {}
        description_key_mask = inputs.get("description_key_mask")
        selected_key_masks = None
        for layer in band_layers:
            layer_hidden = description_outputs.hidden_states[layer]
            layer_sequences, layer_key_masks = select_expanded_description_tokens(
                layer_hidden,
                expanded_text_mask,
                inputs["description_input_ids"],
                inputs["description_attention_mask"],
                description_key_mask,
                int(self.args.description_max_tokens),
            )
            description_sequences[layer] = layer_sequences
            description_key_masks[layer] = layer_key_masks
            if selected_key_masks is None:
                selected_key_masks = layer_key_masks
        return description_sequences, selected_key_masks

    def _build_effective_key_mask(self, valid_mask, key_mask):
        if key_mask is None:
            return valid_mask
        effective_key_mask = valid_mask & key_mask
        has_key = effective_key_mask.any(dim=1, keepdim=True)
        return torch.where(has_key, effective_key_mask, valid_mask)

    def _rms_normalize_hidden(self, hidden_states, eps=1e-6):
        rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
        return hidden_states.float() / rms

    def _compute_focus_loss(self, current_hidden, reference_hidden, valid_mask, key_mask):
        normalized_current = self._rms_normalize_hidden(current_hidden)
        normalized_reference = self._rms_normalize_hidden(reference_hidden.detach())
        token_change = (normalized_current - normalized_reference).norm(dim=-1) * valid_mask.float()

        effective_key_mask = self._build_effective_key_mask(valid_mask, key_mask)
        non_key_mask = valid_mask & (~effective_key_mask)

        total_change = token_change.sum(dim=1)
        active_samples = total_change > 1e-6
        change_distribution = token_change / total_change.clamp_min(1e-6).unsqueeze(-1)

        key_denom = effective_key_mask.sum(dim=1).float()
        non_key_denom = non_key_mask.sum(dim=1).float()
        key_mass = (change_distribution * effective_key_mask.float()).sum(dim=1)
        key_density = key_mass / key_denom.clamp_min(1.0)
        non_key_density = (
            (change_distribution * non_key_mask.float()).sum(dim=1)
            / non_key_denom.clamp_min(1.0)
        )
        focus_per_sample = -torch.log(key_mass.clamp_min(1e-6))
        focus_per_sample = torch.where(
            active_samples,
            focus_per_sample,
            torch.zeros_like(focus_per_sample),
        )
        valid_focus_samples = active_samples
        focus_loss = (
            focus_per_sample[valid_focus_samples].mean()
            if torch.any(valid_focus_samples)
            else token_change.new_zeros(())
        )
        focus_ratio = non_key_density / key_density.clamp_min(1e-6)
        return (
            focus_loss,
            key_mass[active_samples].mean() if torch.any(active_samples) else token_change.new_zeros(()),
            key_density[active_samples].mean() if torch.any(active_samples) else token_change.new_zeros(()),
            non_key_density[active_samples].mean() if torch.any(active_samples) else token_change.new_zeros(()),
            focus_ratio[active_samples].mean() if torch.any(active_samples) else token_change.new_zeros(()),
        )

    def _compute_energy_loss(self, current_hidden, reference_hidden, valid_mask):
        normalized_current = self._rms_normalize_hidden(current_hidden)
        normalized_reference = self._rms_normalize_hidden(reference_hidden.detach())
        token_change = (normalized_current - normalized_reference).norm(dim=-1) * valid_mask.float()
        valid_counts = valid_mask.sum(dim=1)
        active_samples = valid_counts > 0
        mean_change = token_change.sum(dim=1) / valid_counts.clamp_min(1).float()
        energy_margin = float(getattr(self.args, "description_energy_margin", 1.0))
        per_sample_loss = F.relu(mean_change - energy_margin).pow(2)
        if not torch.any(active_samples):
            zero = token_change.new_zeros(())
            return zero, zero
        return per_sample_loss[active_samples].mean(), mean_change[active_samples].mean()

    def _compute_struct_band_loss(
        self,
        description_states_by_layer,
        description_key_masks,
        reference_description_states,
        reference_description_mask,
        reference_available,
        focus_weight,
        energy_weight,
    ):
        band_layers = sorted(description_states_by_layer.keys())
        first_layer = band_layers[0]
        current_description_states, current_description_mask = self._pad_description_sequences(
            description_states_by_layer[first_layer],
            dtype=reference_description_states.dtype,
        )
        reference_description_states = reference_description_states.to(current_description_states.device)
        reference_description_mask = reference_description_mask.to(current_description_states.device)
        shared_seq_len = min(current_description_states.shape[1], reference_description_states.shape[1])
        current_description_mask = current_description_mask[:, :shared_seq_len]
        reference_description_states = reference_description_states[:, :shared_seq_len]
        reference_description_mask = reference_description_mask[:, :shared_seq_len]
        valid_mask = current_description_mask & reference_description_mask
        base_valid_token_count = valid_mask.float().sum()
        description_key_mask = self._pad_description_key_masks(
            description_key_masks,
            shared_seq_len,
            current_description_states.device,
        )
        valid_mask = valid_mask & reference_available[:, None].to(valid_mask.device)
        effective_key_mask = self._build_effective_key_mask(valid_mask, description_key_mask)
        valid_token_count = valid_mask.float().sum()

        per_layer_focus_losses = []
        per_layer_energy_losses = []
        per_layer_key_mass = []
        per_layer_key_density = []
        per_layer_non_key_density = []
        per_layer_focus_ratio = []
        per_layer_mean_change = []

        for layer in band_layers:
            current_states, _ = self._pad_description_sequences(
                description_states_by_layer[layer],
                dtype=reference_description_states.dtype,
            )
            current_states = current_states[:, :shared_seq_len]
            (
                focus_loss,
                key_mass,
                key_density,
                non_key_density,
                focus_ratio,
            ) = self._compute_focus_loss(
                current_states,
                reference_description_states,
                valid_mask,
                effective_key_mask,
            )
            energy_loss, mean_change = self._compute_energy_loss(
                current_states,
                reference_description_states,
                valid_mask,
            )
            per_layer_focus_losses.append(focus_loss)
            per_layer_energy_losses.append(energy_loss)
            per_layer_key_mass.append(key_mass)
            per_layer_key_density.append(key_density)
            per_layer_non_key_density.append(non_key_density)
            per_layer_focus_ratio.append(focus_ratio)
            per_layer_mean_change.append(mean_change)

        difficulty_rho = float(getattr(self.args, "struct_band_energy_rho", 1.0))
        difficulties = [
            focus_loss + difficulty_rho * energy_loss
            for focus_loss, energy_loss in zip(per_layer_focus_losses, per_layer_energy_losses)
        ]
        weights = self._compute_band_weights(
            difficulties,
            band_layers,
            "_struct_band_ema_state",
            getattr(self.args, "struct_band_eta", 0.35),
        )
        struct_loss = sum(
            weight * (focus_weight * focus_loss + energy_weight * energy_loss)
            for weight, focus_loss, energy_loss in zip(weights, per_layer_focus_losses, per_layer_energy_losses)
        )
        focus_loss = sum(weight * loss for weight, loss in zip(weights, per_layer_focus_losses))
        energy_loss = sum(weight * loss for weight, loss in zip(weights, per_layer_energy_losses))
        return {
            "struct_loss": struct_loss,
            "focus_loss": focus_loss,
            "energy_loss": energy_loss,
            "valid_token_count": valid_token_count,
            "base_valid_token_count": base_valid_token_count,
            "shared_seq_len": shared_seq_len,
            "key_mass": sum(weight * value for weight, value in zip(weights, per_layer_key_mass)),
            "key_density": sum(weight * value for weight, value in zip(weights, per_layer_key_density)),
            "non_key_density": sum(weight * value for weight, value in zip(weights, per_layer_non_key_density)),
            "focus_ratio": sum(weight * value for weight, value in zip(weights, per_layer_focus_ratio)),
            "mean_change": sum(weight * value for weight, value in zip(weights, per_layer_mean_change)),
            "weights": weights,
            "band_layers": band_layers,
            "per_layer_focus_losses": per_layer_focus_losses,
            "per_layer_energy_losses": per_layer_energy_losses,
        }

    def _maybe_log_description_losses(
        self,
        total_loss,
        standard_loss,
        description_focus_loss,
        description_energy_loss,
        struct_loss,
        align_loss,
        valid_token_count,
        base_valid_token_count,
        shared_seq_len,
        reference_sample_count,
        batch_sample_count,
        key_mass,
        key_density,
        non_key_density,
        focus_ratio,
        mean_change,
        align_valid_samples,
        align_cosine,
    ):
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        global_step = int(getattr(self.state, "global_step", 0))
        if global_step % logging_steps != 0:
            return
        if getattr(self, "_last_description_loss_log_step", None) == global_step:
            return
        self._last_description_loss_log_step = global_step

        focus_weight = float(getattr(self.args, "description_focus_weight", 0.0))
        energy_weight = float(getattr(self.args, "description_energy_weight", 0.0))
        align_weight = float(getattr(self.args, "align_loss_weight", 0.0))
        ce_weight = float(getattr(self.args, "standard_ce_weight", 1.0))
        base_valid = base_valid_token_count.detach().float().clamp_min(1.0)
        used_valid = valid_token_count.detach().float()

        self.log({
            "loss/total": total_loss.detach().float().item(),
            "loss/ce": standard_loss.detach().float().item(),
            "loss/standard_ce": standard_loss.detach().float().item(),
            "loss/focus": description_focus_loss.detach().float().item(),
            "loss/description_focus": description_focus_loss.detach().float().item(),
            "loss/energy": description_energy_loss.detach().float().item(),
            "loss/description_energy": description_energy_loss.detach().float().item(),
            "loss/struct": struct_loss.detach().float().item(),
            "loss/align": align_loss.detach().float().item(),
            "loss_weighted/standard_ce": (standard_loss.detach().float() * ce_weight).item(),
            "loss_weighted/description_focus": (description_focus_loss.detach().float() * focus_weight).item(),
            "loss_weighted/description_energy": (description_energy_loss.detach().float() * energy_weight).item(),
            "loss_weighted/struct": struct_loss.detach().float().item(),
            "loss_weighted/align": (align_loss.detach().float() * align_weight).item(),
            "description/align_token_fraction": (used_valid / base_valid).item(),
            "description/align_tokens": used_valid.item(),
            "description/base_valid_tokens": base_valid_token_count.detach().float().item(),
            "description/key_mass": key_mass.detach().float().item(),
            "description/key_density": key_density.detach().float().item(),
            "description/non_key_density": non_key_density.detach().float().item(),
            "description/focus_ratio": focus_ratio.detach().float().item(),
            "description/mean_change": mean_change.detach().float().item(),
            "description/shared_seq_len": float(shared_seq_len),
            "description/reference_samples": float(reference_sample_count),
            "description/reference_sample_fraction": float(reference_sample_count) / max(float(batch_sample_count), 1.0),
            "align/valid_samples": align_valid_samples.detach().float().item(),
            "align/cosine_mean": align_cosine.detach().float().item(),
            "config/description_focus_weight": focus_weight,
            "config/description_focus_alpha": float(
                getattr(self.args, "description_focus_alpha", 0.5)
            ),
            "config/description_energy_weight": energy_weight,
            "config/align_loss_weight": align_weight,
            "config/b1_low_layer": int(getattr(self.args, "b1_low_layer", 15)),
            "config/b1_high_layer": int(getattr(self.args, "b1_high_layer", 18)),
            "config/b2_low_layer": int(getattr(self.args, "b2_low_layer", 29)),
            "config/b2_high_layer": int(getattr(self.args, "b2_high_layer", 31)),
            "config/standard_ce_weight": ce_weight,
        })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.args, "enable_description_cl", False) or "description_input_ids" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        b1_layers = self._resolve_loss_band(
            model,
            getattr(self.args, "b1_low_layer", 15),
            getattr(self.args, "b1_high_layer", 18),
            "B1 align band",
        )
        b2_layers = self._resolve_loss_band(
            model,
            getattr(self.args, "b2_low_layer", 29),
            getattr(self.args, "b2_high_layer", 31),
            "B2 struct band",
        )
        focus_weight = float(getattr(self.args, "description_focus_weight", 0.0))
        energy_weight = float(getattr(self.args, "description_energy_weight", 0.0))
        align_weight = float(getattr(self.args, "align_loss_weight", 0.0))
        require_standard_hidden_states = align_weight > 0.0
        (
            standard_input_ids,
            standard_position_ids,
            standard_attention_mask,
            standard_past_key_values,
            standard_inputs_embeds,
            standard_labels,
            image_token_mask,
            text_token_mask,
        ) = self._prepare_answer_multimodal_inputs(model, inputs)
        standard_outputs = model(
            input_ids=standard_input_ids,
            attention_mask=standard_attention_mask,
            position_ids=standard_position_ids,
            past_key_values=standard_past_key_values,
            inputs_embeds=standard_inputs_embeds,
            labels=standard_labels,
            return_dict=True,
            output_hidden_states=require_standard_hidden_states,
            use_cache=False,
        )
        if require_standard_hidden_states:
            (
                boundary_align_loss,
                align_valid_samples,
                align_cosine,
                align_band_weights,
                per_layer_align_losses,
            ) = self._compute_align_band_loss(
                standard_outputs.hidden_states,
                image_token_mask,
                text_token_mask,
                b1_layers,
            )
        else:
            boundary_align_loss = standard_outputs.loss.new_zeros(())
            align_valid_samples = standard_outputs.loss.new_zeros(())
            align_cosine = standard_outputs.loss.new_zeros(())
            align_band_weights = None
            per_layer_align_losses = None
        standard_loss = standard_outputs.loss

        description_states, description_key_masks = self._extract_description_states(model, inputs, b2_layers)
        has_reference_states = (
            "reference_description_states" in inputs and "reference_description_mask" in inputs
        )
        reference_sample_count = 0
        batch_sample_count = len(next(iter(description_states.values())))
        if has_reference_states:
            reference_available = inputs.get("reference_description_available")
            if reference_available is None:
                reference_available = torch.ones(
                    batch_sample_count,
                    dtype=torch.bool,
                    device=standard_loss.device,
                )
            else:
                reference_available = reference_available.to(device=standard_loss.device, dtype=torch.bool)
            reference_sample_count = int(reference_available.long().sum().item())
            struct_metrics = self._compute_struct_band_loss(
                description_states_by_layer=description_states,
                description_key_masks=description_key_masks,
                reference_description_states=inputs["reference_description_states"],
                reference_description_mask=inputs["reference_description_mask"],
                reference_available=reference_available,
                focus_weight=focus_weight,
                energy_weight=energy_weight,
            )
            description_focus_loss = struct_metrics["focus_loss"]
            description_energy_loss = struct_metrics["energy_loss"]
            valid_token_count = struct_metrics["valid_token_count"]
            base_valid_token_count = struct_metrics["base_valid_token_count"]
            shared_seq_len = struct_metrics["shared_seq_len"]
            key_mass = struct_metrics["key_mass"]
            key_density = struct_metrics["key_density"]
            non_key_density = struct_metrics["non_key_density"]
            focus_ratio = struct_metrics["focus_ratio"]
            mean_change = struct_metrics["mean_change"]
            struct_loss = struct_metrics["struct_loss"]
            struct_band_weights = struct_metrics["weights"]
            per_layer_focus_losses = struct_metrics["per_layer_focus_losses"]
            per_layer_energy_losses = struct_metrics["per_layer_energy_losses"]
        else:
            description_focus_loss = standard_loss.new_zeros(())
            description_energy_loss = standard_loss.new_zeros(())
            valid_token_count = standard_loss.new_zeros(())
            base_valid_token_count = standard_loss.new_zeros(())
            shared_seq_len = 0
            key_mass = standard_loss.new_zeros(())
            key_density = standard_loss.new_zeros(())
            non_key_density = standard_loss.new_zeros(())
            focus_ratio = standard_loss.new_zeros(())
            mean_change = standard_loss.new_zeros(())
            struct_loss = standard_loss.new_zeros(())
            struct_band_weights = None
            per_layer_focus_losses = None
            per_layer_energy_losses = None
        total_loss = (
            self.args.standard_ce_weight * standard_loss
            + struct_loss
            + align_weight * boundary_align_loss
        )

        self._maybe_log_description_losses(
            total_loss=total_loss,
            standard_loss=standard_loss,
            description_focus_loss=description_focus_loss,
            description_energy_loss=description_energy_loss,
            struct_loss=struct_loss,
            align_loss=boundary_align_loss,
            valid_token_count=valid_token_count,
            base_valid_token_count=base_valid_token_count,
            shared_seq_len=shared_seq_len,
            reference_sample_count=reference_sample_count,
            batch_sample_count=batch_sample_count,
            key_mass=key_mass,
            key_density=key_density,
            non_key_density=non_key_density,
            focus_ratio=focus_ratio,
            mean_change=mean_change,
            align_valid_samples=align_valid_samples,
            align_cosine=align_cosine,
        )
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        global_step = int(getattr(self.state, "global_step", 0))
        if global_step % logging_steps == 0:
            if align_band_weights is not None:
                self.log({
                    f"align/layer_{layer}_weight": weight.detach().float().item()
                    for layer, weight in zip(b1_layers, align_band_weights)
                })
                self.log({
                    f"align/layer_{layer}_loss": loss.detach().float().item()
                    for layer, loss in zip(b1_layers, per_layer_align_losses)
                })
            if struct_band_weights is not None:
                self.log({
                    f"struct/layer_{layer}_weight": weight.detach().float().item()
                    for layer, weight in zip(b2_layers, struct_band_weights)
                })
                self.log({
                    f"struct/layer_{layer}_focus": loss.detach().float().item()
                    for layer, loss in zip(b2_layers, per_layer_focus_losses)
                })
                self.log({
                    f"struct/layer_{layer}_energy": loss.detach().float().item()
                    for layer, loss in zip(b2_layers, per_layer_energy_losses)
                })

        if return_outputs:
            return total_loss, {
                "standard_outputs": standard_outputs,
                "loss_ce": standard_loss.detach(),
                "loss_focus": description_focus_loss.detach(),
                "loss_energy": description_energy_loss.detach(),
                "loss_struct": struct_loss.detach(),
                "loss_align": boundary_align_loss.detach(),
            }
        return total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
