# Tuning notes

Optuna chooses settings to try. Each trial trains a fresh model and returns its
best validation RMSE. Lower is better. MLflow is optional and records the runs
so we can compare their learning curves in the dashboard.

## Run a search

From the repo root, with the environment activated:

```bash
python -m pip install -r requirements-tuning.txt
python -m src.tune --config configs/lstm.yaml --trials 10 --epochs 10 --study-dir outputs/tuning/lstm --track
```

This means 10 training runs, each with 10 epochs. It can take roughly ten times
as long as a single 10-epoch run, plus tracking and data-loading overhead.
Leave off `--track` to use Optuna without MLflow logging. Normal training still
works without the tuning dependencies.

We currently search two settings:

| Setting | Values tried |
| --- | --- |
| Learning rate | Between 0.0001 and 0.01, sampled on a logarithmic scale |
| Hidden size (`hidden_channels` in YAML) | 16, 32, or 64 |

The learning rate and hidden size in the source config are replaced for each
trial. `--epochs` overrides the YAML epoch count. Everything else, including the
model, split, seed, batch size, and weight decay, stays fixed. The same command
works for `rnn` or `small_cnn`; give each search a different study folder.
For the CNN, hidden size means the number of convolution channels.

Optuna's TPE sampler starts with five random trials, then uses completed scores
to guide later suggestions. Pruning is disabled for now, so every successful
trial gets its full epoch budget. See [Optuna's study documentation](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.study.Study.html).

## What gets saved

Under the study folder:

- `study.db`: the persistent Optuna study, including parameters and scores.
- `trial_0000/`, etc.: each trial's config, history, metrics, and checkpoint.
  With `--track`, these files are one level deeper, inside the MLflow run ID.
- `best_trial.json`: the best completed trial's score, parameters, checkpoint path,
  and MLflow run ID when tracking is enabled.
- `best_config.yaml`: a runnable config using the winning settings, with a separate
  `best_retrain` output folder. Running it trains from scratch.

The two best-result files update after each completed trial. Predict directly
from the checkpoint path in `best_trial.json`, or retrain the winning config:

```bash
python -m src.train --config outputs/tuning/lstm/best_config.yaml --track
```

Use longer training and several seeds to check promising settings. The lowest
search score is a promising result to check, not proof that the model is always better. Keep test
inputs out of parameter selection.

## Resume a search

Run the same command with the same study folder. `--trials 10` means **10 more
trials**, not a total limit of 10. Trial numbers keep increasing, so earlier
trial output is preserved. Run only one tuning process per study at a time.

The script rejects changes to the fixed config, epoch budget, or data contents
within a study. Use a new folder when changing those. It does not check source
code changes, so also use a new folder after changing the model or scoring code.

The study resumes its history, not a partially trained model. An exception stops
the search and records a failed trial. A hard kill can leave a running trial
record. Completed results remain available. The sampler is recreated with the
same seed on restart; suggestions can repeat and a resumed search need not
match the exact sequence of one uninterrupted search.

A short search can favor settings that learn quickly. I would check the promising
settings again with the same longer epoch budget, rather than assume the best
10-epoch trial will also be the best at 50 epochs. See [comparing models](comparing-models.md).

## Where the code lives

`src/train.py` now returns the best score after writing results. `src/tune.py`
contains the search space and calls that same training function for each trial.
There is no separate tuning version of the model or training loop.

In MLflow's **ocean-sst** experiment, trial runs have `optuna.study_dir` and
`optuna.trial` tags. Compare `best_validation_mean_rmse_c`, then inspect the
whole validation curve before choosing a model.

[Back to the README](../README.md) · [Tracking notes](experiments.md)
