"""Generate three-day predictions for test_inputs.npz."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import SSTDataset
from src.model import build_model


def main():
    """Read --input and write Celsius forecasts and metadata to --output.

    Predictions have shape (samples, 3, H, W) in the original sample order.
    Use persistence by default, or restore a trained model with --checkpoint.
    Targets may be present in the input file, but are not used for prediction.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/test_inputs.npz")
    p.add_argument("--output", default="outputs/test_predictions.npz")
    p.add_argument("--model", default="persistence")
    p.add_argument("--checkpoint")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    device = torch.device(device_name)
    dataset = SSTDataset(args.input, require_targets=False)
    missing = {"dates", "lat", "lon"} - dataset.metadata.keys()
    if missing:
        raise ValueError(f"Prediction input is missing metadata: {sorted(missing)}")
    # Iterate over inputs only, even when forecasting a labeled validation file.
    loader = DataLoader(dataset.X, batch_size=args.batch_size, shuffle=False)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model_type = checkpoint["model_type"]
        model_cfg = checkpoint.get("model_config", {})
        model = build_model(
            model_type,
            hidden_channels=int(model_cfg.get("hidden_channels", 32)),
            input_mean=float(checkpoint["input_mean"]),
            input_std=float(checkpoint["input_std"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_type = args.model
        if model_type != "persistence":
            raise ValueError("A trained model requires --checkpoint")
        model = build_model(model_type)
    model = model.to(device).eval()
    batches = []
    with torch.no_grad():
        for x in loader:
            batches.append(model(x.to(device)).cpu().numpy())
    predictions = np.concatenate(batches, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    dates, lat, lon = (dataset.metadata[key] for key in ("dates", "lat", "lon"))
    mask = dataset.mask
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predictions=predictions,
        dates=dates,
        lat=lat,
        lon=lon,
        mask=mask,
        model=np.asarray(model_type),
    )
    print(f"Saved {predictions.shape[0]} forecasts from {model_type} to {output}")


if __name__ == "__main__":
    main()
