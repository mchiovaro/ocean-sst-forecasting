# Ocean SST Forecasting

Predict the next 3 daily sea-surface temperature (SST) maps from the previous 14 maps. The prepared data cover a fixed box around the Florida Keys and store SST in degrees Celsius.

I start with persistence, then compare a small CNN, RNN, and LSTM. The point is to see what each model adds. A more complicated model is not automatically better.

## Start here

The idea is to give the model 14 consecutive daily temperature maps and ask for the next 3. For example, maps from days 1–14 go in, and predictions for days 15–17 come out. A map is a grid of temperatures over the same area, so each prediction has a temperature for every grid cell.

A few terms used throughout the code:

- **Sample / example:** one 14-day input window and its 3-day forecast window.
- **Inputs (`X`):** the maps the model gets to see.
- **Targets (`y`):** the observed maps we compare its predictions with.
- **Training set:** examples used to fit a model's weights, or the numbers it learns.
- **Validation set:** examples with known targets used to compare models and choose which saved model to keep. These targets do not update the weights.
- **Test set:** inputs with targets withheld, used to make the submission.

My starting point is persistence: copy the most recent map for all three forecast days. That gives me something simple to compare everything else against. A training run finishing successfully means the code worked; a lower validation error than persistence means it did better on this validation set.

More detailed notes:

- [Data notes](docs/data.md): what the arrays contain, how to read their indices, and how masks and submission IDs work.
- [Workflow notes](docs/walkthrough.md): how the files connect, what each command saves, and how to read the scores.
- [Comparing models](docs/comparing-models.md): what makes a comparison useful and what the current results can tell us.

Uses Python 3.12. Run commands from the repository root (the folder containing this README).

## Data files

The data:

- `data/train.npz`: `X`, `y`, `dates`, `lat`, `lon`, and `mask`
- `data/validation.npz`: the same fields
- `data/test_inputs.npz`: `X`, `dates`, `lat`, `lon`, and `mask`, with no targets

`X` has shape `(examples, 14, latitude, longitude)`. `y` has shape `(examples, 3, latitude, longitude)`. `mask` identifies ocean cells.

Place the three `.npz` files directly inside this repository's `data/` folder.

## Choose a working environment

Colab or a local computer is sufficient for the baseline and smaller models. HPC is optional.

For Colab, use a Python 3.12 runtime to match the pinned dependencies. Open `notebooks/competition_quickstart.ipynb`, set `REPO_DIR`, and run the cells in order.

For a local environment on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. After activation, use `python` for the commands below.

## Check the setup

```bash
python scripts/check_data.py # check the data files
python -m unittest discover -s tests # quick unit tests
python -m src.train --config configs/baseline.yaml # run baseline model
```

The baseline repeats the most recent input map for all three forecast days. It writes its score to `outputs/baseline/metrics.json`.

These checks tell me the basic local workflow runs. They do not tell me whether a learned model beats persistence.

Generate the baseline submission files:

```bash
python -m src.predict --input data/test_inputs.npz --output outputs/test_predictions.npz
python submission/make_submission.py --predictions outputs/test_predictions.npz --output outputs/submission.csv
```

## First learned model

The small CNN treats the 14 input days as channels and predicts a correction to persistence. I keep it small because the grid is small. This is the first example of fitting weights, checking validation, and saving a model.

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

The training script calculates normalization statistics from training inputs only, reports validation RMSE after each epoch, and saves the best checkpoint. The repository metric ignores masked land cells: it pools squared errors across examples and ocean cells separately for each lead day, takes three square roots, then averages them. Training minimizes pooled ocean-cell MSE; the logged training RMSE therefore differs from the validation metric. Scores are in degrees Celsius (lower is better).

Prediction without `--checkpoint` always uses persistence. A checkpoint supplies the model architecture, weights, and training normalization. `--device auto` uses CUDA when available and otherwise CPU.

Submission rows preserve input example order, then lead day (1–3), latitude index, and longitude index.

## First RNN run

The RNN reads the 14 days in order, using one flattened map per day. Like the CNN, it predicts corrections to the last observed map. The current model uses our fixed 12 × 14 grid.

```bash
python -m src.train --config configs/rnn.yaml
```

For a quick setup check, use one epoch in a copied config with its own output folder. `hidden_channels` is the config name we already use; for the RNN it sets `hidden_size`, the number of values in the learned summary. The included RNN, LSTM, and CNN configs use the same training settings to give us a starting comparison.

Results go into `outputs/rnn/`: `history.json` records each epoch, `metrics.json` reports the best validation score, and `best_model.pt` holds the model for prediction. Compare validation RMSE with `outputs/baseline/metrics.json`; lower is better. Check `epochs` in the config before running; the included learned-model configs currently use 10.

Each training command starts a new model from scratch; it does not continue a previous checkpoint. Copy the config and change `output_dir` for a separate experiment, or use `--track` to get a unique run folder.

## LSTM runs

`python -m src.train --config configs/lstm.yaml` trains the LSTM and saves to `outputs/lstm/`. For a quick setup check, copy the config, set `epochs` to 1, and choose a different output folder.

The LSTM carries both a hidden state and a cell state through the 14 days. Like the RNN, it uses the final hidden state to predict three corrections to persistence. It uses the same fixed grid and `hidden_channels` config setting.

New training runs also save `config.yaml` beside their results, and learned models record the best epoch in `metrics.json`. History is saved after each epoch. Reusing an output folder still overwrites results, so use a new folder for each experiment.

## Optional experiment tracking

Add `--track` to a training command to save settings, learning curves, and checkpoints in local MLflow. Each tracked run gets a separate output subfolder.

```bash
python -m pip install -r requirements-tracking.txt
python -m src.train --config configs/lstm.yaml --track
.venv/bin/python -m mlflow server --backend-store-uri sqlite:///outputs/mlflow/mlflow.db --host 127.0.0.1 --port 5001
```

The server command above uses the macOS/Linux environment path. On Windows, use `.venv\Scripts\python.exe -m mlflow server` with the same arguments.

Open http://127.0.0.1:5001 and select **ocean-sst**. [Experiment tracking notes](docs/experiments.md) explain what gets saved and how to compare runs.

## Project map

- `src/dataset.py`: validates and loads prepared arrays
- `src/model.py`: models
- `src/evaluate.py`: masked RMSE used to compare models
- `src/train.py`: baseline evaluation and learned-model training
- `src/predict.py`: persistence or checkpoint-based test predictions
- `src/tracking.py`: optional MLflow run records
- `src/tune.py`: Optuna search over learning rate and hidden size
- `configs/`: experiment settings
- `notebooks/competition_quickstart.ipynb`: Colab setup path
- `submission/make_submission.py`: prediction-to-CSV conversion
- `scripts/check_data.py`: checks the supplied files and shared grid
- `tests/`: small checks for loading, scoring, models, and the submission workflow
- `docs/`: data and workflow notes

## Automatic tuning with Optuna

Optuna tries learning rates and hidden sizes using the same training loop. Start with a small search:

```bash
python -m pip install -r requirements-tuning.txt
python -m src.tune --config configs/lstm.yaml --trials 10 --epochs 10 --study-dir outputs/tuning/lstm --track
```

Each trial gets its own output folder and, with `--track`, an MLflow run. The study saves `best_config.yaml` and `best_trial.json`. Repeating the command adds 10 more trials to the saved study. [Tuning notes](docs/tuning.md) explain the search settings, results, and resuming.
