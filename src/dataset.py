"""PyTorch Dataset for prepared SST arrays."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class SSTDataset(Dataset):
    def __init__(self, path: str | Path, require_targets: bool = True):
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"Data file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "X" not in data: raise ValueError(f"{path} does not contain X")
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.float32) if "y" in data else None
            self.mask = np.asarray(data["mask"], dtype=bool) if "mask" in data else np.ones(self.X.shape[-2:], bool)
        if self.X.ndim != 4 or self.X.shape[1] != 14: raise ValueError("X must have shape (examples, 14, latitude, longitude)")
        if not np.isfinite(self.X).all(): raise ValueError(f"{path} contains non-finite values in X")
        if self.mask.shape != self.X.shape[-2:]: raise ValueError("mask must have shape (latitude, longitude)")
        if require_targets and self.y is None: raise ValueError(f"Targets y are required but missing from {path}")
        if self.y is not None:
            expected = (len(self.X), 3, self.X.shape[2], self.X.shape[3])
            if self.y.shape != expected: raise ValueError(f"y must have shape {expected}; got {self.y.shape}")
            if not np.isfinite(self.y).all(): raise ValueError(f"{path} contains non-finite values in y")
    def __len__(self): return len(self.X)
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        return x if self.y is None else (x, torch.from_numpy(self.y[index]))
    def input_statistics(self):
        valid = self.X[:, :, self.mask]
        mean, std = float(valid.mean()), float(valid.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0: raise ValueError("Could not calculate valid input statistics")
        return mean, std
