"""PyTorch Dataset for prepared SST arrays."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SSTDataset(Dataset):
    """Load finite Celsius maps; samples include targets whenever they are present.

    ``require_targets=False`` permits unlabeled files but does not discard labels.
    Metadata is optional for small experiments; competition files require it.
    """

    def __init__(self, path: str | Path, require_targets: bool = True):
        """Read one NPZ file into memory and check its arrays.

        Inputs are (samples, 14, rows, columns), and targets, when present,
        are (samples, 3, rows, columns). Temperatures stay in Celsius here.
        Missing files raise FileNotFoundError; invalid arrays raise ValueError.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "X" not in data:
                raise ValueError(f"{path} does not contain X")
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.float32) if "y" in data else None
            self.metadata = {
                key: data[key] for key in ("dates", "lat", "lon") if key in data
            }
            if "mask" in data and not np.isin(data["mask"], [0, 1]).all():
                raise ValueError("mask must contain only boolean or 0/1 values")
            self.mask = (
                np.asarray(data["mask"], dtype=bool)
                if "mask" in data
                else np.ones(self.X.shape[-2:], bool)
            )
        if self.X.ndim != 4 or self.X.shape[1] != 14:
            raise ValueError("X must have shape (examples, 14, latitude, longitude)")
        if any(size == 0 for size in self.X.shape):
            raise ValueError("X must contain examples and a nonempty spatial grid")
        if not np.isfinite(self.X).all():
            raise ValueError(f"{path} contains non-finite values in X")
        if self.mask.shape != self.X.shape[-2:]:
            raise ValueError("mask must have shape (latitude, longitude)")
        if not self.mask.any():
            raise ValueError("mask must contain at least one ocean cell")
        for key, size in (
            ("dates", len(self.X)),
            ("lat", self.X.shape[2]),
            ("lon", self.X.shape[3]),
        ):
            if key in self.metadata and self.metadata[key].shape != (size,):
                raise ValueError(f"{key} must have shape ({size},)")
        for key in ("lat", "lon"):
            if key in self.metadata and not np.isfinite(self.metadata[key]).all():
                raise ValueError(f"{key} contains non-finite coordinates")
        if require_targets and self.y is None:
            raise ValueError(f"Targets y are required but missing from {path}")
        if self.y is not None:
            expected = (len(self.X), 3, self.X.shape[2], self.X.shape[3])
            if self.y.shape != expected:
                raise ValueError(f"y must have shape {expected}; got {self.y.shape}")
            if not np.isfinite(self.y).all():
                raise ValueError(f"{path} contains non-finite values in y")

    def __len__(self):
        """Count samples, each containing a full 14-day input window."""
        return len(self.X)

    def __getitem__(self, index):
        """Return x (14, H, W), plus y (3, H, W) when available, as tensors.

        H and W are the grid's row and column counts. The DataLoader adds
        a batch dimension when it groups these samples together.
        """
        x = torch.from_numpy(self.X[index])
        return x if self.y is None else (x, torch.from_numpy(self.y[index]))

    def input_statistics(self):
        """Return the ocean input mean and standard deviation in Celsius.

        Both are Python floats calculated across all samples and input days
        in this dataset. Call this on training data so validation and test
        data do not influence the normalization. A zero spread is rejected.
        """
        valid = self.X[:, :, self.mask]
        mean, std = float(valid.mean()), float(valid.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Could not calculate valid input statistics")
        return mean, std
