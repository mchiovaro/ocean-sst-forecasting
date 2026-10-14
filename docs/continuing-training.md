# Picking up training from a checkpoint

Sometimes we want to give a model a few more epochs without starting over. The important part is loading what it already learned before making any new weight updates.

There is one catch in our current code: `src/train.py` always starts a new model. Running the command again, even with the same output folder, does not continue training. `--checkpoint` belongs to `src.predict`, and that command only makes predictions. There is no training `--resume` option yet.

These notes explain the pieces we would add to the training code to do this.

## What we have saved so far

Our `best_model.pt` contains:

- The model type and model settings.
- Learned weights and buffers in `model_state_dict`.
- The training input mean and standard deviation.
- The saved epoch and its validation score.

A state dictionary is just a collection of named tensors. It includes the learned weights and our normalization buffers, but it does not contain the Python code defining the model.

`best_model.pt` means the best validation score, not necessarily the last epoch. If a run finishes at epoch 10 but epoch 9 was best, this file takes us back to epoch 9. `history.json` can describe later epochs without containing their weights.

Note: Our current checkpoints do not save Adam's state (if that was the optimizer we were using). We can continue learning from the saved weights with a fresh optimizer, but that is a new continuation experiment. It will not reproduce the updates we would have made by leaving the original run going for additional epochs.

## 1. Read the saved run

Start with a learned model checkpoint and the `config.yaml` for it. For a tracked run, use its actual run folder instead of the path below.

```python
from pathlib import Path
import torch
import yaml

from src.model import build_model
from src.train import choose_device

# load the config
checkpoint_path = Path("outputs/transformer/best_model.pt")
config = yaml.safe_load(
    (checkpoint_path.parent / "config.yaml").read_text()
)

# load the checkpoint
device = choose_device(config.get("device", "auto"))
checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=True
)

print("Starting from epoch:", checkpoint["epoch"])
print("Saved validation RMSE:", checkpoint["validation_mean_rmse_c"])
```

I use the saved config because the original file in `configs/` might have changed since the run (it shouldn't because we should be leaving them alone, but sometimes we make changes on accident).

## 2. Rebuild the model, then load its weights

This is the same idea used in `src.predict`. First create the layers with the saved settings, then put the saved values into those layers.

```python
model_type = checkpoint["model_type"]
model_cfg = checkpoint["model_config"]
extra_settings = {}

if model_type == "tcn":
    extra_settings = {
        "kernel_size": model_cfg["kernel_size"],
        "dilations": model_cfg["dilations"],
        "dropout": model_cfg["dropout"],
    }
if model_type == "transformer":
    extra_settings = {
        "nhead": model_cfg["nhead"],
        "num_layers": model_cfg["num_layers"],
        "dropout": model_cfg["dropout"],
    }

model = build_model(
    model_type,
    hidden_channels=int(model_cfg.get("hidden_channels", 32)),
    input_mean=float(checkpoint["input_mean"]),
    input_std=float(checkpoint["input_std"]),
    **extra_settings,
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
```

`**extra_settings` passes the dictionary entries as named arguments. CNN, RNN, and LSTM checkpoints use an empty dictionary here.

Keep the original model architecture. For example, a TCN checkpoint with two convolutions per block cannot be loaded into our one-convolution version. Even settings that do not change weight shapes (such as dilation or attention head count) still need to match. Keep the corresponding code version with your experiment notes.

Use the saved normalization throughout the continuation.

## 3. Set up the optimizer

For our existing checkpoints, create a fresh Adam optimizer:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config.get("learning_rate", 1e-3)),
    weight_decay=float(config.get("weight_decay", 0.0)),
)
```

The model remembers its learned weights. Adam starts its running estimates of past gradients over again. That can change the first few updates after loading.

I would first keep the saved learning rate. Trying a smaller rate is another potential experiment.

## 4. Continue the existing training loop

In your continuation code, use the dataset loading, mask checks, and DataLoaders already in `src/train.py`, but you when you copy the fresh-model setup from `src/train.py`, replace it's fresh-model setup with the loading steps above.

Make the epoch counting explicit:

```python
additional_epochs = 5
start_epoch = checkpoint["epoch"] + 1
end_epoch = start_epoch + additional_epochs

for epoch in range(start_epoch, end_epoch):
    model.train()
    for x, y in training_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(x)
        error = (prediction[:, :, mask_device] - y[:, :, mask_device]) ** 2
        loss = error.mean()
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite")
        loss.backward()
        optimizer.step()
```

Here, `training_loader` is the existing training DataLoader and `mask_device` is the ocean mask moved to the same device as the model. Keep the existing metric accumulation, validation, and checkpoint-saving code around this loop. This excerpt only shows the weight updates.

If the loaded checkpoint is epoch 9, five additional epochs means epochs 10–14. It does not mean a five-epoch total run.

Call `model.train()` at the beginning of every training epoch. Our validation function switches to `model.eval()`, which turns dropout off. The next training epoch needs to turn it back on. Do not wrap training in `torch.no_grad()`.

## 5. Keep the original result and compare fairly

Use a new output folder, such as `outputs/transformer_continued`. Record the source checkpoint path, its epoch, the number of added epochs, and that the optimizer was restarted. With MLflow, use a new run and log that same information; reopening the dashboard does not resume anything.

Before updating weights, evaluate the restored model on the same validation set. Its score should agree with the saved score within numerical precision.

Initialize `best_score` and `best_epoch` from the loaded checkpoint instead of infinity and zero. Also copy the source checkpoint into the new folder as `best_model.pt` before training. That way, if none of the new epochs improves validation, the new folder still contains the best model you have. Save the continued run's config and history there too, with the actual epoch numbers.

When saving another checkpoint, retain the loaded `model_type`, `model_config`, `input_mean`, and `input_std`. The continuation should not accidentally write different architecture settings or normalization beside the weights.

Note: More training is not guaranteed to help. Compare validation scores, and keep test targets out of this decision.

## For a fuller restart next time (restoring optimizer state)

For future runs, we can extend the checkpoint dictionary in `src/train.py` with:

```python
"optimizer_state_dict": optimizer.state_dict(),
```

That line belongs inside the dictionary passed to `torch.save`. When loading a checkpoint made with this addition, create the model and optimizer first, then restore the optimizer:

```python
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

This restores Adam's running estimates and saved optimizer settings, including its learning rate. It is only available if the earlier run actually saved it. Our existing checkpoint files cannot recover that missing history. PyTorch's [saving and loading guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) describes this model-plus-optimizer pattern.

I would keep two files: `best_model.pt` for the best validation result and `last_checkpoint.pt` saved after every completed epoch for continuing an interrupted run. The latest checkpoint should include the current model and optimizer state, epoch, best score and best epoch so far, config, normalization, and history. Keep the best model file alongside it, as the latest weights might be worse.

Restoring model and optimizer state still does not promise an identical continuation. Dropout and shuffled batches use random numbers. To reproduce the same next steps, also save and restore Python, NumPy, and PyTorch random states (including CUDA states when used), plus any separate DataLoader generator state. Restore these after constructing the model and optimizer, before starting the next epoch.

If you later add a learning-rate scheduler, save and restore those states too.

[Back to the README](../README.md) · [Experiment notes](experiments.md)
