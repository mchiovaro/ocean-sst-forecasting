"""Generate three-day predictions for test_inputs.npz."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import SSTDataset
from src.model import build_model

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default="data/test_inputs.npz"); p.add_argument("--output", default="outputs/test_predictions.npz")
    p.add_argument("--model", default="persistence"); p.add_argument("--checkpoint"); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"]); args = p.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    device = torch.device(device_name); dataset = SSTDataset(args.input, require_targets=False); loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); model_type = checkpoint["model_type"]; model_cfg = checkpoint.get("model_config", {})
        model = build_model(model_type, hidden_channels=int(model_cfg.get("hidden_channels", 32)), input_mean=float(checkpoint["input_mean"]), input_std=float(checkpoint["input_std"]))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_type = args.model
        if model_type != "persistence": raise ValueError("A trained model requires --checkpoint")
        model = build_model(model_type)
    model = model.to(device).eval(); batches = []
    with torch.no_grad():
        for x in loader: batches.append(model(x.to(device)).cpu().numpy())
    predictions = np.concatenate(batches, axis=0)
    with np.load(args.input, allow_pickle=False) as source:
        dates, lat, lon = source["dates"], source["lat"], source["lon"]
        mask = source["mask"] if "mask" in source else np.ones((len(lat), len(lon)), bool)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predictions=predictions, dates=dates, lat=lat, lon=lon, mask=mask, model=np.asarray(model_type))
    print(f"Saved {predictions.shape[0]} forecasts from {model_type} to {output}")

if __name__ == "__main__": main()
