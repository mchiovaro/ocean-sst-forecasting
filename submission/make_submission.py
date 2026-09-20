"""Convert prediction arrays to the Kaggle submission format."""

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.predictions, allow_pickle=False) as data:
        predictions = np.asarray(data["predictions"])
        dates, lat, lon = data["dates"], data["lat"], data["lon"]

    expected = (len(dates), 3, len(lat), len(lon))
    if predictions.shape != expected:
        raise ValueError(f"Expected predictions with shape {expected}; got {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Id", "sst_c"])
        for i in range(len(dates)):
            for lead in range(3):
                for row in range(len(lat)):
                    for col in range(len(lon)):
                        row_id = f"{i:04d}_lead_{lead + 1}_row_{row:02d}_col_{col:02d}"
                        writer.writerow([row_id, float(predictions[i, lead, row, col])])

    print(f"Saved submission CSV to {output}")


if __name__ == "__main__":
    main()
