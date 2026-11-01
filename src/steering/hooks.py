"""Shared residual-stream steering hook."""

from __future__ import annotations

from typing import Any

import torch


def steering_hook(alpha: float, vector: torch.Tensor):
    """Return a hook that adds ``alpha * vector`` at every residual position."""
    if vector.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got {tuple(vector.shape)}")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        addition = alpha * vector.to(device=residual.device, dtype=residual.dtype)
        if residual.shape[-1] != addition.shape[0]:
            raise ValueError(
                f"vector width {addition.shape[0]} does not match residual width "
                f"{residual.shape[-1]}"
            )
        steered = residual + addition
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    return hook


def add_steering_hook(module: Any, alpha: float, vector: torch.Tensor):
    """Register the shared hook and return its removable handle."""
    return module.register_forward_hook(steering_hook(alpha, vector))
