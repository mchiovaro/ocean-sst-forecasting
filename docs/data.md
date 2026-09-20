# Data notes

The files in `data/` are already prepared for modeling. An `.npz` file is a bundle of named NumPy arrays, so one file holds the temperature maps and the information needed to place them on a grid.

## What is in each file?

A sample (also called an example) is one 14-day input window and, when available, its next 3 daily maps. Each map covers the same Florida Keys area. Temperatures are in degrees Celsius.

| Field | Shape | What it holds |
| --- | --- | --- |
| `X` | `(N, 14, H, W)` | The 14 input maps for each sample, with the most recent map last |
| `y` | `(N, 3, H, W)` | The next 3 maps we want to predict, in forecast order |
| `dates` | `(N,)` | One date label per sample |
| `lat` | `(H,)` | Latitude coordinate for each grid row |
| `lon` | `(W,)` | Longitude coordinate for each grid column |
| `mask` | `(H, W)` | `True` for ocean cells included in the score; `False` for excluded cells |

Here, `N` is the number of samples, `H` is the number of latitude rows, and `W` is the number of longitude columns. The supplied files have a 12 × 14 grid with 143 ocean cells.

| File | Samples | Use |
| --- | ---: | --- |
| `train.npz` | 8,764 | Fit the CNN and calculate its input normalization |
| `validation.npz` | 728 | Compare predictions with known answers and choose the best checkpoint |
| `test_inputs.npz` | 729 | Make submission predictions; this file has no `y` |

The data-generation code is not included here, so the exact meaning of `dates` still needs confirming: it could label an input-window boundary or a forecast day. The current code carries these labels through to the prediction file; it does not use them to calculate temperatures or submission IDs.

## Reading the array indices

Python starts counting at zero. A colon means “take everything along this axis,” and `-1` means the last item.

```python
import numpy as np

with np.load("data/train.npz", allow_pickle=False) as data:
    X = data["X"]
    y = data["y"]
    mask = data["mask"]

print(X.shape)       # (8764, 14, 12, 14)
last_map = X[0, -1, :, :]  # first sample, most recent input day
next_map = y[0, 0, :, :]   # first sample, first forecast day
last_ocean_values = last_map[mask]  # just the cells included in scoring
```

`X[0, -1, 2, 5]` is one temperature: sample 0, last input day, row 2, column 5. Its location is `lat[2]`, `lon[5]`. Row and column numbers are array positions, not latitude and longitude values. Keep the original coordinate order when saving predictions.

## Masks and normalization

The mask is shared by every sample and day. The CNN still receives the whole rectangular grid, but its loss and validation score use only ocean cells. All stored temperatures, including excluded cells, must be finite because the model reads the full maps.

Normalization means subtracting the training input mean and dividing by the training input standard deviation. Both numbers come from training ocean cells only. The CNN uses those same numbers for validation and test predictions, then converts its correction back to Celsius. Validation and test data do not get their own normalization statistics.

`SSTDataset` loads arrays as PyTorch tensors when samples are requested. A tensor is the array type PyTorch uses for model calculations. `dataset[0]` returns `(x, y)` when targets are present, or just `x` when they are missing. `require_targets=False` allows missing targets; it does not remove targets from a labeled file.

For small experiments, the loader allows missing metadata and treats a missing mask as all ocean. The supplied competition files are expected to include every metadata field and the real mask; `scripts/check_data.py` checks that requirement.

## Prediction files and CSVs

`src.predict` writes another `.npz` file with `predictions` shaped `(N, 3, H, W)`, the original `dates`, `lat`, `lon`, and `mask`, plus the model name. These predictions are already in Celsius.

The submission script turns that array into columns named `Id` and `sst_c`. For example, `0000_lead_1_row_02_col_05` refers to sample 0, forecast day 1, row 2, column 5. Forecast days in IDs start at 1; sample, row, and column indices start at 0.

The CSV includes land rows too. The mask controls scoring, not which rows get written. For the supplied test file, that gives `729 × 3 × 12 × 14 = 367,416` rows, plus the header. A competition sample submission is not included in this repo, so compare its IDs and columns before uploading.

[Back to the README](../README.md) · [Workflow notes](walkthrough.md)
