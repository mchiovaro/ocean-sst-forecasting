"""Optional local MLflow setup, kept here so the training loop stays readable."""

from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import time


# This lets us use "with tracked_run(...)" to wrap the lifetime of a run.
@contextmanager
def tracked_run(config, config_path):
    """Give one training command its own MLflow run and output folder.

    Import MLflow only when tracking is requested. Everything is stored locally
    under outputs/mlflow; no tracking server needs to be running during training.
    """
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "Install tracking first: python -m pip install -r requirements-tracking.txt"
        ) from error

    storage = Path("outputs/mlflow").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    # Point MLflow at our local SQLite database, not a remote service.
    mlflow.set_tracking_uri(f"sqlite:///{storage / 'mlflow.db'}")
    # An MLflow experiment groups runs in the dashboard. All models go in
    # ocean-sst so persistence, CNN, RNN, and LSTM can be compared together.
    experiment = mlflow.get_experiment_by_name("ocean-sst")
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            "ocean-sst", artifact_location=(storage / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    # The name is for people; the unique ID keeps repeated runs from overwriting.
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=Path(config_path).stem
    ) as run:
        output_dir = Path(
            config.get("output_dir", f"outputs/{config['model']['type']}")
        )
        output_dir = output_dir / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tags(
            {"config_path": str(config_path), "output_dir": str(output_dir)}
        )
        # Artifacts are saved files attached to a run. Save the requested settings
        # and package versions so we can look them up after editing the code.
        mlflow.log_dict(config, "requested_config.json")
        mlflow.log_dict(
            {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("torch", "numpy", "PyYAML", "mlflow")
                },
            },
            "environment.json",
        )
        # A commit alone does not describe code we have edited but not committed.
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            # Nonempty status means there are uncommitted changes. This records
            # that fact, not a backup of those changes. Ignored files are excluded.
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            mlflow.set_tags({"git_commit": commit, "git_dirty": bool(dirty)})
        except (OSError, subprocess.CalledProcessError):
            mlflow.set_tag("git_commit", "unavailable")

        # Hash the files we actually use, without copying the data into MLflow.
        data_keys = ["validation_file"]
        if config["model"]["type"] != "persistence":
            data_keys.append("train_file")
        for key in data_keys:
            path = Path(config["data"][key])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            mlflow.log_params({f"data.{key}": str(path), f"data.{key}.sha256": digest})

        print(f"MLflow run: {run.info.run_id}")
        print(f"Run folder: {output_dir}")
        # Start the timer after metadata setup; it measures the training block.
        started = time.perf_counter()
        # Exceptions escape this block so MLflow marks the run as failed.
        # Hand these two objects to the caller and pause here while it trains.
        # On success, execution comes back here to save runtime and output files.
        yield mlflow, output_dir
        mlflow.log_metric("runtime_seconds", time.perf_counter() - started)
        # Copy the run folder into MLflow artifacts. The original files stay put.
        mlflow.log_artifacts(str(output_dir), artifact_path="results")
