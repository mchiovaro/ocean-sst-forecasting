"""Check the three provided files before running a model."""
from pathlib import Path

import numpy as np


def check(path: Path, targets_required: bool) -> tuple[tuple[int, ...], int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Extract the data ZIP into data/.")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "dates", "lat", "lon", "mask"}
        if targets_required:
            required.add("y")
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if not targets_required and "y" in data.files:
            raise ValueError("test_inputs.npz should not contain targets")
        X = data["X"]
        mask = data["mask"].astype(bool)
        if X.ndim != 4 or X.shape[1] != 14 or mask.shape != X.shape[-2:]:
            raise ValueError(f"Unexpected shapes in {path}: X={X.shape}, mask={mask.shape}")
        return X.shape, int(mask.sum())


def main():
    results = [
        ("train.npz", True),
        ("validation.npz", True),
        ("test_inputs.npz", False),
    ]
    for filename, targets_required in results:
        shape, ocean_cells = check(Path("data") / filename, targets_required)
        print(f"PASS  {filename:16s} X={shape}  scored ocean cells={ocean_cells}")
    print("Data check complete. You are ready to run the persistence baseline.")


if __name__ == "__main__":
    main()
