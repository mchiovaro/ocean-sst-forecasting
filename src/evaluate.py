"""Official competition metric."""
import torch

def rmse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of the three lead-day spatial RMSE values."""
    if predictions.shape != targets.shape: raise ValueError("predictions and targets must have the same shape")
    if predictions.ndim != 4 or predictions.shape[1] != 3: raise ValueError("Expected shape (examples, 3, latitude, longitude)")
    squared = (predictions - targets) ** 2
    if mask is not None:
        if tuple(mask.shape) != tuple(predictions.shape[-2:]): raise ValueError("mask must have shape (latitude, longitude)")
        day_mse = squared[:, :, mask.bool()].mean(dim=(0, 2))
    else:
        day_mse = squared.mean(dim=(0, 2, 3))
    return torch.sqrt(day_mse).mean()
