"""Fixed PCA projection used by the HiDESC spectral routing branch.

The projection is fitted offline and is deliberately not trainable.  Keeping
the basis fixed makes task anchors, role prototypes, and evaluation use one
stable descriptor space across all continual-learning stages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class FixedPCARouteProjector(nn.Module):
    """Apply a saved PCA basis to the last dimension of a tensor."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        explained_variance: Optional[torch.Tensor] = None,
        explained_variance_ratio: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        components = torch.as_tensor(components, dtype=torch.float32)
        if components.ndim != 2:
            raise ValueError(
                f"PCA components must have shape [output_dim, input_dim], got {tuple(components.shape)}"
            )
        if components.shape[1] != mean.numel():
            raise ValueError(
                "PCA mean/components dimension mismatch: "
                f"mean={mean.numel()}, components={tuple(components.shape)}"
            )
        if components.shape[0] <= 0:
            raise ValueError("PCA output dimension must be positive")

        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("components", components, persistent=True)
        variance = (
            torch.as_tensor(explained_variance, dtype=torch.float32).flatten()
            if explained_variance is not None
            else torch.empty(0, dtype=torch.float32)
        )
        ratio = (
            torch.as_tensor(explained_variance_ratio, dtype=torch.float32).flatten()
            if explained_variance_ratio is not None
            else torch.empty(0, dtype=torch.float32)
        )
        if variance.numel() not in (0, components.shape[0]):
            raise ValueError("explained_variance must match PCA output dimension")
        if ratio.numel() not in (0, components.shape[0]):
            raise ValueError("explained_variance_ratio must match PCA output dimension")
        self.register_buffer("explained_variance", variance, persistent=True)
        self.register_buffer("explained_variance_ratio", ratio, persistent=True)
        self.metadata = dict(metadata or {})

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 1:
            raise ValueError("PCA input must have at least one dimension")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"PCA input width mismatch: expected {self.input_dim}, got {features.shape[-1]}"
            )
        input_dtype = features.dtype
        projected = (features.float() - self.mean.float()) @ self.components.float().T
        # The downstream FFT immediately promotes to float32.  Keeping the
        # caller's dtype here avoids an unnecessary activation-size increase.
        return projected.to(input_dtype) if input_dtype.is_floating_point else projected

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        expected_input_dim: Optional[int] = None,
        expected_output_dim: Optional[int] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "FixedPCARouteProjector":
        if not path:
            raise ValueError("A PCA file path is required")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing spectral PCA file: {path}")
        payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, dict):
            raise TypeError(f"Spectral PCA file must contain a dict: {path}")
        projector = cls(
            payload.get("mean"),
            payload.get("components"),
            payload.get("explained_variance"),
            payload.get("explained_variance_ratio"),
            payload.get("metadata"),
        )
        if expected_input_dim is not None and projector.input_dim != int(expected_input_dim):
            raise ValueError(
                f"Spectral PCA input width mismatch: file={projector.input_dim}, "
                f"vision={int(expected_input_dim)}"
            )
        if expected_output_dim is not None and projector.output_dim != int(expected_output_dim):
            raise ValueError(
                f"Spectral PCA output width mismatch: file={projector.output_dim}, "
                f"configured={int(expected_output_dim)}"
            )
        return projector

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "explained_variance": self.explained_variance.detach().cpu(),
            "explained_variance_ratio": self.explained_variance_ratio.detach().cpu(),
            "metadata": {**self.metadata, **(metadata or {})},
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def fit(
        cls,
        samples: torch.Tensor,
        output_dim: int,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        niter: int = 2,
    ) -> "FixedPCARouteProjector":
        if samples.ndim != 2:
            raise ValueError(f"PCA samples must have shape [samples, channels], got {tuple(samples.shape)}")
        samples = samples.float()
        n_samples, input_dim = samples.shape
        output_dim = int(output_dim)
        if output_dim <= 0 or output_dim > min(n_samples, input_dim):
            raise ValueError(
                f"PCA output_dim must be in [1, min(samples, input_dim)] = "
                f"[1, {min(n_samples, input_dim)}], got {output_dim}"
            )
        mean = samples.mean(dim=0)
        centered = samples - mean
        _, singular_values, components_t = torch.pca_lowrank(
            centered, q=output_dim, center=False, niter=int(niter)
        )
        components = components_t.T.contiguous()
        explained_variance = singular_values.square() / max(n_samples - 1, 1)
        total_variance = centered.square().sum() / max(n_samples - 1, 1)
        explained_ratio = explained_variance / total_variance.clamp_min(torch.finfo(torch.float32).eps)
        return cls(mean, components, explained_variance, explained_ratio, metadata)


def load_spectral_pca(
    path: str,
    *,
    expected_input_dim: Optional[int] = None,
    expected_output_dim: Optional[int] = None,
) -> FixedPCARouteProjector:
    """Small functional wrapper used by model-building code and tests."""
    return FixedPCARouteProjector.from_file(
        path,
        expected_input_dim=expected_input_dim,
        expected_output_dim=expected_output_dim,
    )
