import torch
import torch.nn.functional as F


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    features = torch.nan_to_num(
        features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    norms = torch.linalg.norm(features, dim=-1, keepdim=True)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    normalized = features / safe_norms
    return torch.where(norms > 0, normalized, torch.zeros_like(normalized))


def build_text_activation_index(
    text_features: torch.Tensor,
    highpass_exponent: float = 0.75,
    magnitude_weight: float = 0.0,
    real_weight: float = 0.5,
    imag_weight: float = 0.5,
    use_fftshift: bool = True,
) -> torch.Tensor:
    if text_features.ndim != 2:
        raise ValueError(
            "text_features must have shape [B, D], got "
            f"{tuple(text_features.shape)}"
        )
    text_features = torch.nan_to_num(
        text_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    centered_features = text_features - text_features.mean(dim=-1, keepdim=True)
    frequency = torch.fft.fft(centered_features, dim=-1)
    if use_fftshift:
        frequency = torch.fft.fftshift(frequency, dim=-1)
    frequency_weights = torch.fft.fftfreq(
        text_features.shape[-1],
        d=1.0,
        device=text_features.device,
    ).abs()
    frequency_weights = frequency_weights / frequency_weights.max().clamp_min(1e-6)
    frequency_weights = frequency_weights.pow(max(float(highpass_exponent), 0.0))
    weighted_frequency = frequency * frequency_weights.unsqueeze(0)
    magnitude = torch.log1p(torch.abs(weighted_frequency))
    real_part = weighted_frequency.real
    imag_part = weighted_frequency.imag
    weight_total = max(
        float(magnitude_weight) + float(real_weight) + float(imag_weight),
        1e-6,
    )
    descriptor = float(magnitude_weight) / weight_total * _row_normalize(magnitude)
    descriptor = descriptor + float(real_weight) / weight_total * _row_normalize(real_part)
    descriptor = descriptor + float(imag_weight) / weight_total * _row_normalize(imag_part)
    return _row_normalize(descriptor)


def build_task_anchor_bank(
    task_vectors: torch.Tensor,
    *,
    remove_global_mean: bool = True,
    contrast_weight: float = 0.0,
    hard_negative_top_k: int = 1,
    preserve_mean_weight: float = 0.0,
) -> torch.Tensor:
    if task_vectors.ndim != 2:
        raise ValueError(
            "task_vectors must have shape [T, D], got "
            f"{tuple(task_vectors.shape)}"
        )
    raw_bank = _row_normalize(task_vectors)
    if raw_bank.shape[0] == 0:
        return raw_bank

    centered_bank = raw_bank
    if remove_global_mean and raw_bank.shape[0] > 1:
        global_mean = raw_bank.mean(dim=0, keepdim=True)
        centered_bank = _row_normalize(raw_bank - global_mean)
        fallback_mask = centered_bank.abs().sum(dim=-1, keepdim=True).eq(0.0)
        centered_bank = torch.where(fallback_mask, raw_bank, centered_bank)

    refined_bank = centered_bank
    contrast_weight = max(float(contrast_weight), 0.0)
    if contrast_weight > 0.0 and raw_bank.shape[0] > 1:
        top_k = min(max(int(hard_negative_top_k), 1), raw_bank.shape[0] - 1)
        similarity = torch.matmul(centered_bank, centered_bank.T)
        similarity.fill_diagonal_(float("-inf"))
        negative_scores, negative_indices = torch.topk(
            similarity,
            k=top_k,
            dim=-1,
        )
        negative_weights = F.softmax(negative_scores, dim=-1)
        negative_bank = centered_bank[negative_indices]
        negative_mix = (
            negative_weights.unsqueeze(-1) * negative_bank
        ).sum(dim=1)
        refined_bank = _row_normalize(
            centered_bank - contrast_weight * negative_mix
        )

    preserve_mean_weight = max(0.0, min(float(preserve_mean_weight), 1.0))
    if preserve_mean_weight > 0.0:
        refined_bank = _row_normalize(
            (1.0 - preserve_mean_weight) * refined_bank
            + preserve_mean_weight * raw_bank
        )
    return refined_bank
