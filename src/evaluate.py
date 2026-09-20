"""Masked forecast error used for validation in this repo."""

import torch


def rmse(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Average three lead-day RMSEs in Celsius, pooling examples and ocean cells.

    predictions and targets have shape (samples, 3, H, W), in Celsius.
    mask is (H, W), with True cells included; no mask means all cells count.
    The result is one scalar tensor, and lower is better.

    Take each day's square root before averaging days. This differs from a
    single RMSE over all days, and from averaging scores of individual batches.
    """
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.ndim != 4 or predictions.shape[1] != 3:
        raise ValueError("Expected shape (examples, 3, latitude, longitude)")
    if any(size == 0 for size in predictions.shape):
        raise ValueError("Cannot score empty predictions")
    squared = (predictions - targets) ** 2
    if mask is not None:
        if tuple(mask.shape) != tuple(predictions.shape[-2:]):
            raise ValueError("mask must have shape (latitude, longitude)")
        mask = mask.to(device=predictions.device, dtype=torch.bool)
        if not mask.any():
            raise ValueError("mask must contain at least one ocean cell")
        day_mse = squared[:, :, mask].mean(dim=(0, 2))
    else:
        day_mse = squared.mean(dim=(0, 2, 3))
    if not torch.isfinite(day_mse).all():
        raise ValueError("Scored ocean cells contain non-finite errors")
    return torch.sqrt(day_mse).mean()
