# Experiment notes

The models are working through the same training and prediction code now. The thing getting harder to follow is which settings produced which result.

I check the saved settings before comparing scores. A folder name can stay the same even after I change the config and run it again.

## What we save now

New runs save a copy of the loaded config as `config.yaml` in the output folder. Learned models also save the best epoch in `metrics.json` and in the checkpoint. `history.json` is updated after each epoch, so completed scores survive an interrupted training run.

This does not prevent overwrites: use a different `output_dir` for a new experiment. Older output folders may contain runs made with different settings; check their saved metrics and configs. Old output folders do not automatically gain config snapshots.

The test files currently work locally, but `.gitignore` excludes new files under `tests/`. Already tracked files stay tracked. If we want the RNN, LSTM, and workflow checks to come along when someone clones the repo, those tests need to be included in Git. I need to keep this in mind when sharing the repo: a fresh clone may have fewer tests than my local folder.

## Track a run with MLflow

Tracking in this way is optional. You can use your own system, just make sure your team agrees on the method. The usual training commands still work without MLflow. Install the optional requirements in the activated Python 3.12 environment:

```bash
python -m pip install -r requirements-tracking.txt
```

This includes the base requirements so NumPy and PyTorch stay at the versions
used by this project. The tracking dependency is pinned to MLflow 3.16.1.

Add `--track` to any training command:

```bash
python -m src.train --config configs/baseline.yaml --track
python -m src.train --config configs/lstm.yaml --track
```

Each command gets a new run ID. Results go into `output_dir/<run-id>/`, for example
`outputs/lstm/<run-id>/best_model.pt`. The command prints the actual path.
Without `--track`, results still go directly into the config's output directory.

Tracking uses a local database at `outputs/mlflow/mlflow.db` and saved artifacts
under `outputs/mlflow/artifacts/`. An artifact is just a saved file attached to a
run. These paths are already covered by the `outputs/` Git ignore rule.
No server needs to be running while training.

To open the dashboard, run this from the repository root in another terminal
with the same environment activated:

```bash
.venv/bin/python -m mlflow server --backend-store-uri sqlite:///outputs/mlflow/mlflow.db --host 127.0.0.1 --port 5001
```

Then open http://127.0.0.1:5001 and select the **ocean-sst** experiment.
Choose runs to compare their parameters and learning curves. Press Ctrl+C in
the server terminal when finished; closing the server does not delete runs.
See the [MLflow local tracking docs](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)
for the database and UI setup.

## What gets logged

- Model, seed, batch size, device, and the training settings actually used, including defaults.
- Training and validation RMSE at each epoch, under separate names because their calculations differ.
- `best_validation_mean_rmse_c` for every model, including persistence. Use this column to compare best scores; `validation_mean_rmse_c` shows the latest epoch for learned models.
- Best epoch for learned models, parameter count, normalization, and elapsed run time (after initial tracking setup).
- Python and package versions, Git commit and dirty status, and SHA-256 checksums of the data files used. Dirty status tells me there are uncommitted changes; it does not save those changes. I commit the code when I want to keep a reproducible experiment.
- Requested config plus the completed run's config, metrics, history, and checkpoint files.

`src/tracking.py` handles the local database, run context, and metadata. The logging
calls inside `src/train.py` show exactly when scores are recorded. We use explicit
logging rather than autologging, keeping the normal PyTorch training loop visible.

If training raises an error, MLflow marks the run as failed. Completed epochs are
still visible in its metrics, and any files written so far remain in the run's
output folder. The final artifact copy happens only after a successful run.
A hard process kill may leave a run marked as running.

For prediction, use the checkpoint path printed for that run:

```bash
python -m src.predict --input data/test_inputs.npz --checkpoint outputs/lstm/<run-id>/best_model.pt --output outputs/lstm_test_predictions.npz
```

Replace `<run-id>` with the actual ID before running the command. Existing runs
are not imported automatically, and tracking does not resume training or change
the forecast model. Keep both the database and artifact folder when backing up
tracking results; artifact locations are absolute local paths.

## Other options

| Tool | What it adds | How I would use it here |
| --- | --- | --- |
| [TensorBoard](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) | Plots metrics logged from PyTorch with `SummaryWriter` | A smaller first step if all you want is learning curves |
| [MLflow](https://mlflow.org/docs/latest/ml/tracking/) | Tracks run settings, metrics, and artifacts | Best fit for comparing these experiments and keeping their files together |
| [Optuna](https://optuna.org/) | Searches hyperparameter settings automatically | Available through `src.tune`; see [tuning notes](tuning.md) |

## Keeping comparisons useful

Use the same training and validation splits and compare several seeds before deciding one model is better. The RNN and LSTM have different numbers of weights even with the same hidden size, so also record parameter count and runtime (MLFlow does this for you!). Keep the test targets out of model selection.

The RNN/LSTM builders currently assume the fixed 12 × 14 grid. Input normalization is calculated from training ocean cells only, and scoring pools examples before averaging the three lead-day RMSEs. Those choices should stay consistent between experiments.

The source data-generation code and official scoring specification are not here. We can check the arrays and our implemented score, but cannot fully establish how the source windows were split or certify the competition's metric from this repo alone.

[Back to the README](../README.md)
