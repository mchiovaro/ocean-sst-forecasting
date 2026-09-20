"""Models used in the competition."""
from __future__ import annotations
import torch
from torch import nn

class PersistenceModel(nn.Module):
    """Repeat the most recently observed SST map for each forecast day."""
    def __init__(self, forecast_days: int = 3):
        super().__init__(); self.forecast_days = forecast_days
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4: raise ValueError("Expected x with shape (batch, time, latitude, longitude)")
        return x[:, -1:, :, :].repeat(1, self.forecast_days, 1, 1)

class SmallCNN(nn.Module):
    """Compact spatial model that predicts corrections to persistence."""
    def __init__(self, hidden_channels=32, forecast_days=3, input_mean=0.0, input_std=1.0):
        super().__init__()
        if input_std <= 0: raise ValueError("input_std must be positive")
        self.forecast_days, self.hidden_channels = forecast_days, hidden_channels
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        self.network = nn.Sequential(
            nn.Conv2d(14, hidden_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_channels, forecast_days, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 14: raise ValueError("Expected x with shape (batch, 14, latitude, longitude)")
        correction = self.network((x - self.input_mean) / self.input_std) * self.input_std
        persistence = x[:, -1:, :, :].repeat(1, self.forecast_days, 1, 1)
        return persistence + correction

def build_model(model_type: str, *, hidden_channels=32, input_mean=0.0, input_std=1.0) -> nn.Module:
    if model_type == "persistence": return PersistenceModel()
    if model_type == "small_cnn":
        return SmallCNN(hidden_channels=hidden_channels, input_mean=input_mean, input_std=input_std)
    raise ValueError(f"Unknown model type: {model_type}")
