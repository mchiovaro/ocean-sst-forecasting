"""Train or evaluate one competition model from a YAML configuration."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader
from src.dataset import SSTDataset
from src.evaluate import rmse
from src.model import build_model

def choose_device(requested):
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    return torch.device(requested)

def evaluate(model, loader, mask, device):
    model.eval(); predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu()); targets.append(y)
    return float(rmse(torch.cat(predictions), torch.cat(targets), mask))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/baseline.yaml"); args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream: config = yaml.safe_load(stream)
    seed = int(config.get("seed", 444)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_cfg = config["model"]; model_type = model_cfg["type"]
    batch_size = int(config.get("batch_size", 64)); device = choose_device(config.get("device", "auto"))
    validation = SSTDataset(config["data"]["validation_file"])
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    mask = torch.from_numpy(validation.mask)
    output_dir = Path(config.get("output_dir", f"outputs/{model_type}")); output_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "persistence":
        model = build_model("persistence").to(device)
        score = evaluate(model, validation_loader, mask, device)
        metrics = {"model": model_type, "validation_mean_rmse_c": score}
    else:
        training = SSTDataset(config["data"]["train_file"])
        if not np.array_equal(training.mask, validation.mask): raise ValueError("Training and validation masks do not match")
        mean, std = training.input_statistics()
        model = build_model(model_type, hidden_channels=int(model_cfg.get("hidden_channels", 32)), input_mean=mean, input_std=std).to(device)
        training_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=int(config.get("num_workers", 0)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 0.0)))
        epochs = int(config.get("epochs", 10)); history = []; best_score = float("inf"); checkpoint_path = output_dir / "best_model.pt"
        for epoch in range(1, epochs + 1):
            model.train(); total_squared_error = 0.0; total_values = 0; mask_device = mask.to(device)
            for x, y in training_loader:
                x, y = x.to(device), y.to(device); optimizer.zero_grad(); prediction = model(x)
                error = (prediction[:, :, mask_device] - y[:, :, mask_device]) ** 2
                loss = error.mean(); loss.backward(); optimizer.step()
                total_squared_error += float(error.detach().sum().cpu()); total_values += error.numel()
            train_rmse = (total_squared_error / total_values) ** 0.5
            validation_score = evaluate(model, validation_loader, mask, device)
            history.append({"epoch": epoch, "training_rmse_c": train_rmse, "validation_mean_rmse_c": validation_score})
            print(f"Epoch {epoch:02d}: training RMSE {train_rmse:.4f} C; validation mean RMSE {validation_score:.4f} C")
            if validation_score < best_score:
                best_score = validation_score
                torch.save({"model_type": model_type, "model_config": model_cfg, "input_mean": mean, "input_std": std, "model_state_dict": model.state_dict(), "validation_mean_rmse_c": best_score}, checkpoint_path)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        score = best_score
        metrics = {"model": model_type, "device": str(device), "epochs": epochs, "input_mean_c": mean, "input_std_c": std, "best_validation_mean_rmse_c": score, "checkpoint": str(checkpoint_path)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Validation mean RMSE: {score:.4f} C"); print(f"Saved results to {output_dir}")

if __name__ == "__main__": main()
