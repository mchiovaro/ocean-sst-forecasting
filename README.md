# Ocean SST Forecasting

Predict the next 3 daily sea-surface temperature (SST) maps from the previous 14 maps. The prepared data cover a fixed box around the Florida Keys and store SST in degrees Celsius.

Start with the persistence baseline. Later, use the included small CNN as a working example before building temporal models. A learned model is not automatically better; compare every result with persistence.

## Start here

The idea is to give the model 14 consecutive daily temperature maps and ask for the next 3. For example, maps from days 1–14 go in, and predictions for days 15–17 come out. A map is a grid of temperatures over the same area, so each prediction has a temperature for every grid cell.

A few terms used throughout the code:

- **Sample / example:** one 14-day input window and its 3-day forecast window.
- **Inputs (`X`):** the maps the model gets to see.
- **Targets (`y`):** the observed maps we compare its predictions with.
- **Training set:** examples used to fit the CNN's weights, or learned settings.
- **Validation set:** examples with known targets used to compare models and choose which saved model to keep. These targets do not update the weights.
- **Test set:** inputs with targets withheld, used to make the submission.

My starting point is persistence: copy the most recent map for all three forecast days. That gives me something simple to compare everything else against. A CNN run finishing successfully means the code worked; a lower validation error than persistence means it improved the forecast.

More detailed notes:

- [Data notes](docs/data.md): what the arrays contain, how to read their indices, and how masks and submission IDs work.
- [Workflow notes](docs/walkthrough.md): how the files connect, what each command saves, and how to read the scores.

Uses Python 3.12. Run commands from the repository root (the folder containing this README).

## Data files

The data:

- `data/train.npz`: `X`, `y`, `dates`, `lat`, `lon`, and `mask`
- `data/validation.npz`: the same fields
- `data/test_inputs.npz`: `X`, `dates`, `lat`, `lon`, and `mask`, with no targets

`X` has shape `(examples, 14, latitude, longitude)`. `y` has shape `(examples, 3, latitude, longitude)`. `mask` identifies ocean cells. 

Place the three `.npz` files directly inside this repository's `data/` folder.

## Choose a working environment

Colab or a local computer is sufficient for the baseline. HPC is optional.

For Colab, use a Python 3.12 runtime to match the pinned dependencies. Open `notebooks/competition_quickstart.ipynb`, set `REPO_DIR`, and run the cells in order.

For a local environment (substitute `python` for `python3` or `py` based on your machine):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required setup check

```bash
python3 scripts/check_data.py # checks to make sure the three data files are properly formatted and in the correct location
python3 -m unittest discover -s tests # quick unit tests
python3 -m src.train --config configs/baseline.yaml # run baseline model
```

The baseline repeats the most recent input map for all three forecast days. It writes its score to `outputs/baseline/metrics.json`.

If these commands run without an error, the repository, data files, and Python environment are set up correctly.

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

The training script calculates normalization statistics from training inputs only, reports validation RMSE after each epoch, and saves the best checkpoint. The repository metric ignores masked land cells: it pools squared errors across examples and ocean cells separately for each lead day, takes three square roots, then averages them. Training minimizes pooled ocean-cell MSE; the logged training RMSE therefore differs from the validation metric. Scores are in degrees Celsius (lower is better).

Prediction without `--checkpoint` always uses persistence. A checkpoint supplies the CNN architecture, weights, and training normalization. `--device auto` uses CUDA when available and otherwise CPU.

Submission rows preserve input example order, then lead day (1–3), latitude index, and longitude index.

## Project map

- `src/dataset.py`: validates and loads prepared arrays
- `src/model.py`: models
- `src/evaluate.py`: masked RMSE used to compare models
- `src/train.py`: baseline evaluation and learned-model training
- `src/predict.py`: persistence or checkpoint-based test predictions
- `configs/`: named experiment settings
- `notebooks/competition_quickstart.ipynb`: Colab setup path
- `submission/make_submission.py`: prediction-to-CSV conversion

- `scripts/check_data.py`: checks the supplied files and shared grid
- `tests/`: small checks for loading, scoring, models, and the submission workflow
- `docs/`: data and workflow notes
