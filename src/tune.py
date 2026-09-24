"""Try learning rates and hidden sizes, saving each experiment separately."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from src.train import train


def tune(config, study_dir, trials=10, epochs=10, track=False):
    """Add trials to a local study and return it. Run one search process at a time."""
    # Only load Optuna when tuning, so normal training does not need it.
    import optuna

    if trials < 1 or epochs < 1:
        raise ValueError("trials and epochs must be positive")
    if config["model"]["type"] not in ("rnn", "lstm", "small_cnn"):
        raise ValueError("Choose a learned model; persistence has nothing to tune")
    
    # Copy the nested dictionaries too. Each trial can change its own settings
    # without changing the original config or another trial.
    base = deepcopy(config)
    # The command line epoch total takes priority over the YAML value.
    base["epochs"] = epochs
    # resolve() turns this into a full path; mkdir also creates missing parents.
    study_dir = Path(study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)

    # A resumed study must still be comparing the same experiment and data.
    signature = deepcopy(base)
    # Leave out the output location and the two settings we are searching.
    # pop(..., None) means it is okay if the setting is not in the config.
    signature.pop("output_dir", None)
    signature.pop("learning_rate", None)
    signature["model"].pop("hidden_channels", None)
    # A hash is a fingerprint of the file contents. Same filename with changed
    # temperatures should count as different data when we resume a search.
    hashes = {}
    for key in ("train_file", "validation_file"):
        # rb reads the file as bytes; we are not loading arrays into the model here.
        with open(base["data"][key], "rb") as stream:
            hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    signature = {
        "config": signature,
        "data_sha256": hashes,
        "search": {"learning_rate": [0.0001, 0.01], "hidden_channels": [16, 32, 64]},
    }
    # JSON round-trip gives the same representation before and after SQLite storage.
    signature = json.loads(json.dumps(signature))
    # A study is the whole search; a trial is one set of settings and its run.
    # SQLite keeps the study in a local file so closing Python does not lose it.
    study = optuna.create_study(
        study_name="sst-search",
        storage=f"sqlite:///{study_dir / 'study.db'}",
        direction="minimize",  # smaller validation RMSE is better
        load_if_exists=True,  # reuse this study if its database already exists
        # Try five random choices first, then use earlier results to guide choices.
        sampler=optuna.samplers.TPESampler(
            seed=int(base.get("seed", 444)), n_startup_trials=5
        ),
        pruner=optuna.pruners.NopPruner(),  # do not stop weak trials early yet
    )
    # user_attrs are extra notes stored with the study. Our experiment note
    # lets us check that old and new trials are still comparable.
    previous = study.user_attrs.get("experiment")
    if previous is not None and previous != signature:
        raise ValueError(
            "Settings or data changed. Use a new --study-dir for this experiment."
        )
    study.set_user_attr("experiment", signature)

    # Optuna calls this function once per trial. The number we return is the
    # score it uses to decide which settings did best.
    def objective(trial):
        # Keep everything else fixed so we can see what these two settings do.
        trial_config = deepcopy(base)
        # 1e-4 = 0.0001 and 1e-2 = 0.01. log=True spreads choices across
        # orders of magnitude instead of evenly across the raw number line.
        trial_config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        # categorical means pick one item from this list. For RNN/LSTM this
        # sets hidden_size; for the CNN it sets the convolution channel count.
        trial_config["model"]["hidden_channels"] = trial.suggest_categorical(
            "hidden_channels", [16, 32, 64]
        )
        # Trial numbers start at zero. :04d pads them: trial_0000, trial_0001, etc.
        folder = study_dir / f"trial_{trial.number:04d}"
        trial_config["output_dir"] = str(folder)
        if track:
            from src.tracking import tracked_run

            # tracked_run adds a unique MLflow ID to the trial folder. The folder
            # returned here is the actual location of this trial's saved model.
            with tracked_run(
                trial_config, f"{study_dir.name}-trial-{trial.number}.yaml"
            ) as (tracker, folder):
                # These labels connect the MLflow dashboard run to its Optuna trial.
                tracker.set_tags(
                    {"optuna.study_dir": str(study_dir), "optuna.trial": trial.number}
                )
                trial.set_user_attr("mlflow_run_id", tracker.active_run().info.run_id)
                trial.set_user_attr("output_dir", str(folder))
                # Same training function as a manual run: fresh model, full epoch
                # budget, and the best validation score returned to Optuna.
                return train(trial_config, folder, tracker)
        # No MLflow requested? Still save all the normal training files.
        trial.set_user_attr("output_dir", str(folder))
        return train(trial_config, folder)

    # This is a callback: Optuna calls it after a trial ends. It supplies both
    # arguments, but we use the whole study to find the best trial so far.
    def save_best(study, trial):
        # Refresh after each completed trial, not just after the whole search.
        if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            return
        # best may be an earlier trial, not the one that just finished.
        best = study.best_trial
        folder = Path(best.user_attrs["output_dir"])
        best_config = yaml.safe_load((folder / "config.yaml").read_text())
        # Retraining should not overwrite the winning trial's checkpoint.
        # This file contains settings, not weights; training it starts over.
        best_config["output_dir"] = str(study_dir / "best_retrain")
        (study_dir / "best_config.yaml").write_text(
            yaml.safe_dump(best_config, sort_keys=False)
        )
        # Keep a small result summary, including the existing checkpoint path
        # so we can make predictions without retraining the winner.
        (study_dir / "best_trial.json").write_text(
            json.dumps(
                {
                    "trial": best.number,
                    "validation_mean_rmse_c": best.value,
                    "parameters": best.params,
                    "checkpoint": str(folder / "best_model.pt"),
                    "mlflow_run_id": best.user_attrs.get("mlflow_run_id"),
                },
                indent=2,
            )
            + "\n"
        )

    # n_trials means additional trials on each invocation, including when resuming.
    # Trials run one at a time. gc_after_trial asks Python to clean up unused
    # objects between runs; callbacks refresh our best-result files.
    study.optimize(
        objective, n_trials=trials, callbacks=[save_best], gc_after_trial=True
    )
    print(
        f"Best trial: {study.best_trial.number}; validation RMSE: {study.best_value:.4f} C"
    )
    print(f"Best settings and checkpoint path: {study_dir}")
    # Return the study object too, which is useful from Python or in tests.
    return study


def main():
    # argparse reads options like --trials 10 from the terminal command.
    # __doc__ is the short description at the top of this file.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lstm.yaml")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--study-dir", default="outputs/tuning/lstm")
    # A switch: False unless --track appears in the command.
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        # Turn the YAML text into the dictionary that train() already understands.
        config = yaml.safe_load(stream)
    tune(config, args.study_dir, args.trials, args.epochs, args.track)


# Only run the command-line entry point when this file is executed, not imported.
if __name__ == "__main__":
    main()
