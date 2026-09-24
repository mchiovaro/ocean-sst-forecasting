# Workflow notes

The basic path through the repo is:

```text
data/*.npz → dataset.py → model.py → predicted temperature maps
                                      ├─ evaluate.py → validation score
                                      └─ predict.py → prediction file → make_submission.py → CSV
```

`train.py` connects the dataset, model, and evaluation code. The YAML files in `configs/` tell it which model and settings to use. Run the commands below from the repository root with the Python environment activated.

## Start with persistence

```bash
python scripts/check_data.py
python -m unittest discover -s tests
python -m src.train --config configs/baseline.yaml
```

The data check looks at required fields, array shapes, finite values, and whether the files share the same grid and mask. The tests check the code using small examples, including a short CNN run and CSV conversion.

Despite the command name, persistence does not train anything. For one sample, it takes the last of the 14 input maps and copies it three times. The assumption is simply that tomorrow and the following two days will look like the most recent observation.

`train.py` compares those copies with validation targets and writes `outputs/baseline/metrics.json`. The `validation_mean_rmse_c` value is the baseline to beat. There is no persistence checkpoint because there are no learned weights to save.

## What the CNN does

```bash
python -m src.train --config configs/small_cnn.yaml
```

The CNN (convolutional neural network) looks at nearby grid cells to learn spatial patterns. It treats the 14 input days as 14 channels, or stacked input maps, and produces three correction maps. It uses all 14 days at once rather than stepping through them one at a time.

For each sample, the calculation is:

```text
14 input maps → normalize → CNN → 3 corrections in Celsius
last input map + each correction → 3 forecast maps in Celsius
```

A batch is a group of samples processed together. During training, the model predicts a batch, measures its errors against `y`, and updates its weights (the numbers it learns). One epoch is one pass through the training samples. After each epoch, the code checks the model on validation samples without updating the weights.

The run writes these files under `outputs/small_cnn/`:

| File | What I use it for |
| --- | --- |
| `config.yaml` | Settings loaded for this run |
| `history.json` | Training and validation scores for each epoch |
| `metrics.json` | Best validation score, normalization statistics, and checkpoint path |
| `best_model.pt` | Saved model weights, architecture settings, and normalization needed for prediction |

A checkpoint is a saved model snapshot. This one is kept whenever validation improves, so it may come from an earlier epoch than the last one. It supports prediction; it does not save the optimizer state needed to resume the exact training run.

Reusing the same output directory overwrites results. For a separate experiment, copy the config and give it a different `output_dir` so the scores and model stay together.

## Where the RNN and LSTM fit

Both use the same training function as the CNN. The RNN reads one flattened map
at a time and updates its hidden state, a learned summary of the sequence.
The LSTM also has a cell state, with learned gates that control what information
is kept or changed. Neither state is carried from one sample to the next.

Both turn the final hidden state into three correction maps and add them to the
last observed map. Flattening keeps the temperature values and their order, but
does not explicitly show the model which cells are neighbors. The builders use
our fixed 12 × 14 grid.

```bash
python -m src.train --config configs/rnn.yaml
python -m src.train --config configs/lstm.yaml
```

The same checkpoint and prediction steps apply to these models. Add `--track`
to training for a separate MLflow run folder. [Tracking notes](experiments.md)
explain that part; [tuning notes](tuning.md) cover trying settings with Optuna.

## Reading the scores

RMSE means root mean squared error. For each forecast day, the code squares the temperature errors, averages them across all validation samples and ocean cells, and takes the square root. It then averages the three daily RMSEs. Lower is better, and the units are degrees Celsius.

For example, daily RMSEs of 0.2, 0.3, and 0.4 °C give a reported validation score of 0.3 °C. That is an aggregate error measure, not a promise that every prediction is within 0.3 °C.

Training minimizes MSE (mean squared error) across all three days together. The logged training RMSE takes one square root of that pooled error. It also combines batches seen while the weights were changing. That makes it different from the validation score, which checks one fixed model and averages the three daily RMSEs.

Use validation scores to compare models. Finishing a run tells me the code worked; beating persistence tells me the model did better on this validation set. Repeatedly choosing settings using validation also means that set is part of model development, so it is useful to keep the test set separate.

## Make predictions, then a submission

For persistence:

```bash
python -m src.predict --input data/test_inputs.npz --output outputs/test_predictions.npz
python submission/make_submission.py --predictions outputs/test_predictions.npz --output outputs/submission.csv
```

For the trained CNN:

```bash
python -m src.predict \
  --input data/test_inputs.npz \
  --checkpoint outputs/small_cnn/best_model.pt \
  --output outputs/small_cnn_test_predictions.npz
python submission/make_submission.py \
  --predictions outputs/small_cnn_test_predictions.npz \
  --output outputs/small_cnn_submission.csv
```

Training a learned model does not change the prediction command's default: without `--checkpoint`, it still uses persistence. With a checkpoint, it loads that model's settings, normalization, and weights. Prediction preserves sample order and does not need targets.

The `.npz` keeps maps in their original array shape, which is useful for analysis or plotting. The CSV lists one temperature per row for submission. The conversion script does not run a model or change its predictions. See [data notes](data.md) for fields, shapes, and row IDs.

The commands use the configured output paths relative to the working directory. Downloaded data and generated outputs are excluded from Git by `.gitignore`; code, configs, and these notes belong in the repo.

[Comparing models](comparing-models.md) · [Back to the README](../README.md)
