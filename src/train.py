"""Train or evaluate one competition model from a YAML configuration."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.dataset import SSTDataset
from src.evaluate import rmse
from src.model import build_model


def choose_device(requested):
    """Return a torch device; auto picks CUDA if available, otherwise CPU.

    An explicit CUDA request fails if no CUDA GPU is available.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    return torch.device(requested)


def evaluate(model, loader, mask, device):
    """Return one validation RMSE float in Celsius, without updating weights.

    loader supplies batches of (x, y); mask is the shared (H, W) ocean mask.
    Collect all predictions before scoring so batch size does not change
    the metric. This puts the model in evaluation mode.
    """
    # Switch layers to evaluation behavior; this does not itself turn off gradients.
    model.eval()
    predictions, targets = [], []
    # We are only scoring here, so do not build the gradient calculations.
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu())
            targets.append(y)
    # cat joins the batches into one set before calculating the score.
    return float(rmse(torch.cat(predictions), torch.cat(targets), mask))


def train(config, output_dir, tracker=None):
    """Run the experiment; tracker is MLflow when enabled, otherwise None."""
    # Reset the random starting point for every run. get() uses the fallback
    # value if a setting was left out of the YAML.
    seed = int(config.get("seed", 444))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    batch_size = int(config.get("batch_size", 64))
    device = choose_device(config.get("device", "auto"))
    validation = SSTDataset(config["data"]["validation_file"])
    # The loader groups samples into batches. Validation keeps its original order.
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    mask = torch.from_numpy(validation.mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tracker:
        # Log the actual defaults too, so missing YAML fields are not ambiguous.
        params = {
            "model": model_type,
            "seed": seed,
            "batch_size": batch_size,
            "device": str(device),
        }
        if model_type != "persistence":
            params.update(
                hidden_channels=int(model_cfg.get("hidden_channels", 32)),
                epochs=int(config.get("epochs", 10)),
                learning_rate=float(config.get("learning_rate", 1e-3)),
                weight_decay=float(config.get("weight_decay", 0.0)),
                num_workers=int(config.get("num_workers", 0)),
            )
        tracker.log_params(params)
    # Keep the settings with the results, even if we edit the original YAML later.
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    # Persistence has no weights to learn, so it goes straight to evaluation.
    if model_type == "persistence":
        model = build_model("persistence").to(device)
        score = evaluate(model, validation_loader, mask, device)
        metrics = {"model": model_type, "validation_mean_rmse_c": score}
    else:
        training = SSTDataset(config["data"]["train_file"])
        if not np.array_equal(training.mask, validation.mask):
            raise ValueError("Training and validation masks do not match")
        for key in ("lat", "lon"):
            if key in training.metadata and key in validation.metadata:
                if not np.array_equal(training.metadata[key], validation.metadata[key]):
                    raise ValueError(
                        f"Training and validation {key} coordinates do not match"
                    )
        # Calculate normalization from training inputs only, never validation.
        mean, std = training.input_statistics()
        model = build_model(
            model_type,
            hidden_channels=int(model_cfg.get("hidden_channels", 32)),
            input_mean=mean,
            input_std=std,
        ).to(device)
        if tracker:
            tracker.log_params({"input_mean_c": mean, "input_std_c": std})
        # Shuffle whole samples, not the 14 days inside each sample.
        training_loader = DataLoader(
            training,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
        )
        # Adam updates the model's trainable weights using the gradients below.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        epochs = int(config.get("epochs", 10))
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history = []
        # Start at infinity so the first finite validation score becomes the best.
        best_score = float("inf")
        checkpoint_path = output_dir / "best_model.pt"
        for epoch in range(1, epochs + 1):
            # Switch back from evaluation mode; this line does not train by itself.
            model.train()
            total_squared_error = 0.0
            total_values = 0
            mask_device = mask.to(device)
            for x, y in training_loader:
                x, y = x.to(device), y.to(device)
                # Clear gradients from the previous batch; PyTorch adds to them otherwise.
                optimizer.zero_grad()
                # Calling model(x) runs its forward() method.
                prediction = model(x)
                # Select ocean cells across every sample and forecast day.
                error = (prediction[:, :, mask_device] - y[:, :, mask_device]) ** 2
                # Optimize pooled ocean MSE; select checkpoints by mean lead-day RMSE.
                loss = error.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite")
                # Work out how each weight contributed to the loss.
                loss.backward()
                # Apply one weight update using those gradients.
                optimizer.step()
                # detach() keeps bookkeeping out of the gradient graph; numel()
                # counts values so the smaller final batch gets the right weight.
                total_squared_error += float(error.detach().sum().cpu())
                total_values += error.numel()
            train_rmse = (total_squared_error / total_values) ** 0.5
            validation_score = evaluate(model, validation_loader, mask, device)
            history.append(
                {
                    "epoch": epoch,
                    "training_rmse_c": train_rmse,
                    "validation_mean_rmse_c": validation_score,
                }
            )
            print(
                f"Epoch {epoch:02d}: training RMSE {train_rmse:.4f} C; validation mean RMSE {validation_score:.4f} C"
            )
            if tracker:
                # Separate names because training and validation use different RMSEs.
                tracker.log_metrics(
                    {
                        "training_rmse_c": train_rmse,
                        "validation_mean_rmse_c": validation_score,
                    },
                    step=epoch,
                )
            # Save only when validation improves, even if training loss keeps falling.
            if validation_score < best_score:
                best_score = validation_score
                best_epoch = epoch
                # state_dict holds weights and buffers (including normalization).
                # Optimizer state is not saved, so this checkpoint is for prediction.
                torch.save(
                    {
                        "epoch": epoch,
                        "model_type": model_type,
                        "model_config": model_cfg,
                        "input_mean": mean,
                        "input_std": std,
                        "model_state_dict": model.state_dict(),
                        "validation_mean_rmse_c": best_score,
                    },
                    checkpoint_path,
                )
            # Save after each epoch so an interrupted run keeps its completed scores.
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
        score = best_score
        metrics = {
            "model": model_type,
            "device": str(device),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "input_mean_c": mean,
            "input_std_c": std,
            "best_validation_mean_rmse_c": score,
            "checkpoint": str(checkpoint_path),
        }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if tracker:
        # One shared best-score column makes persistence and learned runs comparable.
        tracker.log_metric("best_validation_mean_rmse_c", score)
        tracker.log_param("parameter_count", sum(p.numel() for p in model.parameters()))
        if model_type == "persistence":
            tracker.log_metric("validation_mean_rmse_c", score, step=0)
        else:
            tracker.log_metric("best_epoch", best_epoch)
    print(f"Validation mean RMSE: {score:.4f} C")
    print(f"Saved results to {output_dir}")
    return score  # Optuna uses the best validation error to compare trials


def main():
    """Read a YAML config and optionally track this run with --track."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--track", action="store_true", help="Save this run in local MLflow"
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if args.track:
        from src.tracking import tracked_run

        with tracked_run(config, args.config) as (tracker, output_dir):
            train(config, output_dir, tracker)
    else:
        output_dir = Path(
            config.get("output_dir", f"outputs/{config['model']['type']}")
        )
        train(config, output_dir)


if __name__ == "__main__":
    main()
