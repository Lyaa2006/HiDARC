from typing import List, Optional, Tuple

import torch

from llava.constants import IMAGE_TOKEN_INDEX


def select_expanded_description_tokens(
    hidden_states: torch.Tensor,
    expanded_text_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    key_mask: Optional[torch.Tensor],
    max_tokens: int,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Select text states after the image sentinel has expanded to image patches."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if key_mask is not None:
        key_mask = key_mask.bool()

    selected_states = []
    selected_key_masks = [] if key_mask is not None else None
    for batch_idx in range(hidden_states.shape[0]):
        raw_valid = attention_mask[batch_idx]
        raw_ids = input_ids[batch_idx][raw_valid]
        raw_key_mask = key_mask[batch_idx][raw_valid] if key_mask is not None else None
        raw_text_key_mask = (
            raw_key_mask[raw_ids != IMAGE_TOKEN_INDEX]
            if raw_key_mask is not None
            else None
        )

        text_positions = torch.nonzero(
            expanded_text_mask[batch_idx].bool(), as_tuple=False
        ).flatten()
        if raw_text_key_mask is not None:
            if text_positions.numel() > raw_text_key_mask.numel():
                raise ValueError(
                    "Expanded description text mask contains more tokens than "
                    "the raw description input."
                )
            aligned_key_mask = raw_text_key_mask[: text_positions.numel()]
            expanded_key_mask = torch.zeros(
                expanded_text_mask.shape[1],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            expanded_key_mask[text_positions] = aligned_key_mask.to(
                device=hidden_states.device
            )
        else:
            expanded_key_mask = None

        selected_positions = text_positions[-max_tokens:]
        selected_states.append(hidden_states[batch_idx, selected_positions])
        if selected_key_masks is not None:
            selected_key_masks.append(expanded_key_mask[selected_positions])

    return selected_states, selected_key_masks
