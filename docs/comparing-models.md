# Notes on comparing models

We want to know whether a change helps forecast temperatures, not just whether
it makes the training loss smaller. These are the choices behind the current code.

## Keep the comparison consistent

We start with persistence. Then we use the same data split and validation metric
for the CNN, RNN, and LSTM. Training normalization comes from training inputs
only. Reusing that normalization on validation and test data avoids letting
those sets influence the preprocessing. This follows the usual separation of
[training, validation, and test data](https://scikit-learn.org/stable/modules/cross_validation.html).

Training minimizes ocean-cell MSE, a simple regression loss. Validation averages
three lead-day RMSEs. They are related but not identical, so I use the validation
score to choose checkpoints and compare models. Saving the best checkpoint is not
early stopping; this code still finishes every requested epoch.

A small network and a fixed learning rate are reasonable starting choices.
They are not guaranteed to be the best settings for SST prediction. I would add a scheduler,
gradient clipping, or a bigger model when the learning curves give me a reason.
For example, clipping can help if recurrent-model gradients become unstable.
It is not needed just because the model is an RNN.

## Time order matters

These are daily forecasts, so we will train on earlier dates and evaluate on
later ones. Randomly dividing overlapping windows can put closely related
examples on both sides of the split (data leakage!). If I build new splits, 
I need to check the actual input and target date ranges around each boundary. 
Training must not use targets from beyond the evaluation forecast's information 
cutoff.

Shuffling training *samples* in the DataLoader is okay. Each
sample still contains its 14 days in order, and the recurrent state starts fresh
for each sample. We are not shuffling days *within* a forecast window.

## Remember: A best score is still just *one* result

Optuna selects settings using validation, so that validation set is part of
model development. Keep the test set for the final check, and do not repeatedly change settings
based on test or leaderboard results.

Before calling one model better, I would repeat the promising settings with a
few seeds and look at the spread of scores, not just the lowest one. I would also
check errors separately for each forecast day and across seasons. The current
script saves one overall validation score, so those extra seasonality-breakdowns 
are not yet implemented.

The same hidden size does not mean the same model size: an LSTM has more weights
than a basic RNN. MLflow records parameter count and runtime so we can consider
cost alongside error. The runtime includes validation and logging inside the
training block, not just the weight updates.

## Repeating a run

The seed sets the starting point for random choices. It helps repeat a run in
the same setup, but it does not guarantee identical results across GPUs, CPUs,
or package versions. We do not force deterministic GPU algorithms here.
[PyTorch explains these limits](https://docs.pytorch.org/docs/stable/notes/randomness.html).

An Optuna study keeps completed trials in SQLite, but this script recreates the
sampler when restarted. The [Optuna FAQ](https://optuna.readthedocs.io/en/v4.9.0/faq.html) 
explains study persistence and reproducibility. Tracked runs also record
package versions, file hashes, and whether Git had uncommitted changes. 

[Back to the README](../README.md)
