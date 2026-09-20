# EGR 444 Competition

Predict the next 3 daily sea-surface temperature (SST) maps from the previous 14 maps. The prepared data cover a fixed box around the Florida Keys and store SST in degrees Celsius.

Start with the persistence baseline. Later, use the included small CNN as a working example before building temporal models. A learned model is not automatically better; compare every result with persistence.

Note: Must be using Python 3.12.

## Data files

The competition data:

- `data/train.npz`: `X`, `y`, `dates`, `lat`, `lon`, and `mask`
- `data/validation.npz`: the same fields
- `data/test_inputs.npz`: `X`, `dates`, `lat`, `lon`, and `mask`, with no targets

`X` has shape `(examples, 14, latitude, longitude)`. `y` has shape `(examples, 3, latitude, longitude)`. `mask` identifies ocean cells. Do not commit the data files to Git.

Download the provided data ZIP, extract it, and place the three `.npz` files directly inside this repository's `data/` folder.

## Choose a working environment

Colab or a local computer is sufficient for the baseline. Unity is optional and should be used only when an experiment needs more time, memory, or GPU access.

For Colab, open `notebooks/competition_quickstart.ipynb`, set `REPO_DIR`, and run the cells in order. Keep durable files in Google Drive because the Colab runtime is temporary.

For a local environment (substitute `python` for `python3` or `py` based on your machine):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Required setup check

```bash
python3 scripts/check_data.py # checks to make sure the three data files are properly formatted and in the correct location
python3 -m unittest discover -s tests # quick unit tests
python3 -m src.train --config configs/baseline.yaml # run baseline model
```

The baseline repeats the most recent input map for all three forecast days. It writes its score to `outputs/baseline/metrics.json`.

If these commands run without an error, your repository, data files, and Python environment are set up correctly. You're ready to make predicitons and generate a submission file for Kaggle!

Generate the baseline submission files:

```bash
python -m src.predict --input data/test_inputs.npz --output outputs/test_predictions.npz
python submission/make_submission.py --predictions outputs/test_predictions.npz --output outputs/submission.csv
```

## First learned model

The small CNN treats the 14 input days as channels and predicts a correction to persistence. It is intentionally compact for the small spatial grid. It provides a complete training, validation, checkpoint, and prediction example.

```bash
python -m src.train --config configs/small_cnn.yaml
python -m src.predict \
  --input data/test_inputs.npz \
  --checkpoint outputs/small_cnn/best_model.pt \
  --output outputs/small_cnn_test_predictions.npz
python submission/make_submission.py \
  --predictions outputs/small_cnn_test_predictions.npz \
  --output outputs/small_cnn_submission.csv
```

The training script calculates normalization statistics from training inputs only, reports validation RMSE after each epoch, and saves the best checkpoint. The official metric ignores masked land cells.

## Project map

- `src/dataset.py`: validates and loads prepared arrays
- `src/model.py`: persistence and small CNN models
- `src/evaluate.py`: official masked RMSE
- `src/train.py`: baseline evaluation and learned-model training
- `src/predict.py`: persistence or checkpoint-based test predictions
- `configs/`: named experiment settings
- `notebooks/competition_quickstart.ipynb`: Colab setup path
- `submission/make_submission.py`: prediction-to-CSV conversion
- `scripts/train_unity.sh`: optional Slurm example

## Team workflow

Keep code, configuration, tests, and documentation in Git. Keep data, environments, checkpoints, predictions, and generated outputs out of Git. Use branches and pull requests for normal competition changes.

Develop and debug with a small run first. Change one main idea at a time and record the Git commit, configuration, validation score, runtime, and conclusion for each experiment.
