# TCN notes

## 1. Keep the convolution causal

`CausalConv1d` takes `(batch, channels, time)`. Causal means an output at day t
can read day t and earlier days, but cannot read later positions.

A kernel of size 3 with dilation 1 reads three consecutive positions. Dilation 2
spaces them two positions apart; dilation 4 spaces them four positions apart.
Padding only the left side with `dilation * (kernel_size - 1)` zeros gives the
filter zeros to read before the first day and keeps the output length unchanged.
The `Conv1d` layer itself uses no padding.

## 2. Add a temporal block

Each `TemporalBlock` has one causal convolution, with ReLU and dropout after
each one. Dropout randomly zeros some activations during training and switches
off for evaluation and prediction.

The shortcut carries the block input around those layers. If the channel counts
match, `Identity` passes it through; otherwise, a 1x1 convolution changes the
channel count at each time position. We add the paths, then apply ReLU.

## 3. Preparing our data

With the default 32 hidden channels:

```text
(batch, 14, 12, 14)    temperature maps in Celsius
(batch, 14, 168)       normalize and flatten each map
(batch, 168, 14)       transpose so Conv1d reads along time (IMPORTANT)
(batch, 32, 14)        three temporal blocks, dilations 1, 2, 4
(batch, 32)            take the final time position
(batch, 504)           linear readout: 3 days * 168 cells
(batch, 3, 12, 14)     reshape (if using correction method, multiply by input_std)
```

With two convolutions per block, the theoretical receptive field is
`1 + 2 * (3 - 1) * (1 + 2 + 4) = 29` positions. The last position can therefore
use the complete 14-day input.

## 4. Train and check it

```bash
python -m unittest discover -s tests -p test_tcn.py
python -m src.train --config configs/tcn.yaml
python -m src.predict --input data/test_inputs.npz --checkpoint outputs/tcn/best_model.pt --output outputs/tcn_predictions.npz
python submission/make_submission.py --predictions outputs/tcn_predictions.npz --output outputs/tcn_submission.csv
```

The config uses the same 10-epoch training settings as the other learned models.
Add `--track` to training for MLflow; then use the checkpoint in the printed run
folder.

I set `hidden_channels`, `kernel_size`, `dilations`, and `dropout` in the
`model` section of `configs/tcn.yaml`. Each dilation adds one temporal block.
These settings are saved in the checkpoint so prediction rebuilds the same model.
TCN training configs and saved checkpoints must include `kernel_size`,
`dilations`, and `dropout`. Prediction will raise an error if an older TCN checkpoint
is missing these settings. The grid stays fixed at 12 x 14.

If you change the kernel size or dilations, the receptive field changes too:
`1 + 2 * (kernel_size - 1) * sum(dilations)`.

[Back to the README](../README.md)
