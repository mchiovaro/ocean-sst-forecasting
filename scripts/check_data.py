"""Check the three provided files before running a model."""

from pathlib import Path
import sys

# Support the documented direct invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SSTDataset

import numpy as np


def check(path: Path, targets_required: bool) -> tuple[tuple[int, ...], int]:
    """Check a supplied NPZ file and return (input shape, ocean-cell count).

    Require dates, coordinates, and a mask as well as valid temperature arrays.
    targets_required=True requires y; False rejects y for the test input file.
    """
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
    dataset = SSTDataset(path, require_targets=targets_required)
    return dataset.X.shape, int(dataset.mask.sum())


def main():
    """Check all three files in data/ and their shared coordinates and mask."""
    results = [
        ("train.npz", True),
        ("validation.npz", True),
        ("test_inputs.npz", False),
    ]
    reference = None
    for filename, targets_required in results:
        shape, ocean_cells = check(Path("data") / filename, targets_required)
        with np.load(Path("data") / filename, allow_pickle=False) as data:
            grid = tuple(data[key] for key in ("lat", "lon", "mask"))
        if reference is not None and any(
            not np.array_equal(a, b) for a, b in zip(reference, grid)
        ):
            raise ValueError(f"{filename} does not share the training grid and mask")
        reference = grid
        print(f"PASS  {filename:16s} X={shape}  scored ocean cells={ocean_cells}")
    print("Data check complete. You are ready to run the persistence baseline.")


if __name__ == "__main__":
    main()
