"""Models used in the competition."""

from __future__ import annotations
import torch
from torch import nn


class PersistenceModel(nn.Module):
    """Repeat the most recently observed SST map for each forecast day."""

    def __init__(self, forecast_days: int = 3):
        super().__init__()
        self.forecast_days = forecast_days

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy the last input map into (batch, forecast_days, H, W).

        x has shape (batch, input_days, H, W). Values stay in Celsius.
        """
        if x.ndim != 4:
            raise ValueError("Expected x with shape (batch, time, latitude, longitude)")
        return x[:, -1:, :, :].repeat(1, self.forecast_days, 1, 1)


class SmallCNN(nn.Module):
    """Use 14 input days as channels and learn a correction for each forecast day.

    hidden_channels sets the width of the two intermediate convolution layers.
    input_mean and input_std come from training ocean cells and are saved as
    model buffers, so prediction uses the same normalization as training.
    """

    def __init__(
        self, hidden_channels=32, forecast_days=3, input_mean=0.0, input_std=1.0
    ):
        super().__init__()
        if input_std <= 0:
            raise ValueError("input_std must be positive")
        self.forecast_days, self.hidden_channels = forecast_days, hidden_channels
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        self.network = nn.Sequential(
            nn.Conv2d(14, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, forecast_days, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Turn (batch, 14, H, W) inputs into forecast maps in Celsius.

        Output shape is (batch, forecast_days, H, W), with three days by default.
        Normalization and conversion back to Celsius happen inside the model.
        """
        if x.ndim != 4 or x.shape[1] != 14:
            raise ValueError("Expected x with shape (batch, 14, latitude, longitude)")
        # Buffers travel with the checkpoint. Rescale the residual to Celsius
        # before adding it to the last observed map.
        correction = (
            self.network((x - self.input_mean) / self.input_std) * self.input_std
        )
        persistence = x[:, -1:, :, :].repeat(1, self.forecast_days, 1, 1)
        return persistence + correction


class RNNModel(nn.Module):
    """Read 14 daily maps in order and predict corrections to persistence.

    grid_shape is (latitude rows, longitude columns). Flatten each map
    so we can use a basic RNN. The values stay in order, but the RNN is not
    explicitly told which cells are neighbors.
    hidden_size is the number of values in its learned summary.
    This version uses one recurrent layer and predicts all days at once.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        hidden_size: int = 32,  # num values in learned summary
        forecast_days: int = 3,
        input_mean: float = 0.0,
        input_std: float = 1.0,
    ):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.forecast_days = forecast_days
        self.hidden_size = hidden_size
        # register_buffer ensures that these values are saved in the model
        # state_dict but are not trainable parameters.
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))

        cells_per_map = self.grid_shape[0] * self.grid_shape[1]
        # ends up being 12 x 14 = 168
        # each sequence step is one day; its features are all the grid cells.
        # batch_first=True means the input shape is (batch, time, features)
        # which is true for us.
        self.rnn = nn.RNN(
            input_size=cells_per_map, hidden_size=hidden_size, batch_first=True
        )
        # convert the summary into corrections for all forecast days
        # with our defaults: hidden_size summary values become 504 values (3 maps of 168 cells)
        self.readout = nn.Linear(hidden_size, forecast_days * cells_per_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, 14, H, W) inputs to (batch, forecast_days, H, W), in Celsius."""

        # normalize and flatten the maps
        normalized = (x - self.input_mean) / self.input_std
        sequence = normalized.flatten(start_dim=2)  # (batch, 14, H * W))

        # read all 14 days; the final state summarizes the sequence after the last day
        # no state is passed in: each sample starts with a fresh zero state.
        _, final_hidden = self.rnn(sequence)
        summary = final_hidden[-1]  # last layer's summary is (batch, hidden_size)

        # make a correction for each forecast day, then reshape to (batch, forecast_days, H, W)
        correction = self.readout(summary).reshape(
            x.shape[0], self.forecast_days, *self.grid_shape
        )
        # convert back to Celsius before adding to the last observed map
        correction = correction * self.input_std
        return x[:, -1:, :, :] + correction


class LSTMModel(nn.Module):
    """Read daily maps with an LSTM.

    The LSTM carries a hidden state and a cell state through the sequence.
    Learned gates control what information to keep, add, and expose.
    grid_shape is (rows, columns); hidden_size sets the size of both states.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        hidden_size: int = 32,
        forecast_days: int = 3,
        input_mean: float = 0.0,
        input_std: float = 1.0,
    ):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.forecast_days = forecast_days
        self.hidden_size = hidden_size
        # Same training normalization as the other models, saved with the weights.
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))

        cells_per_map = self.grid_shape[0] * self.grid_shape[1]
        # Still one flattened map per day: (batch, 14, 168) for our data.
        self.lstm = nn.LSTM(
            input_size=cells_per_map, hidden_size=hidden_size, batch_first=True
        )
        # convert to corrections
        self.readout = nn.Linear(hidden_size, forecast_days * cells_per_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, 14, H, W) inputs to (batch, forecast_days, H, W), in Celsius."""

        # normalize and flatten the maps to (batch, 14, H * W)
        normalized = (x - self.input_mean) / self.input_std
        sequence = normalized.flatten(start_dim=2)

        # The LSTM returns final hidden and cell states. Their initial values
        # are zero here because we do not pass in a starting state.
        # The cell state helps carry information between days inside the LSTM;
        # we only use the hidden state to make a forecast.
        _, (final_hidden, final_cell) = self.lstm(sequence)
        summary = final_hidden[-1]  # last layer, after reading all 14 days
        correction = self.readout(summary).reshape(
            x.shape[0], self.forecast_days, *self.grid_shape
        )
        # Corrections are differences, so scale them without adding the mean.
        correction = correction * self.input_std
        return x[:, -1:, :, :] + correction


def build_model(
    model_type: str, *, hidden_channels=32, input_mean=0.0, input_std=1.0
) -> nn.Module:
    """Build persistence, small_cnn, rnn, or lstm from the config settings.

    All predict three days. Learned weights must be trained or loaded separately;
    creating the model alone does not restore a checkpoint.
    """
    if model_type == "persistence":
        return PersistenceModel()
    if model_type == "small_cnn":
        return SmallCNN(
            hidden_channels=hidden_channels, input_mean=input_mean, input_std=input_std
        )
    if model_type == "rnn":
        # Use our fixed data grid; hidden_channels sets the RNN summary size too.
        return RNNModel(
            grid_shape=(12, 14),
            hidden_size=hidden_channels,
            input_mean=input_mean,
            input_std=input_std,
        )
    if model_type == "lstm":
        return LSTMModel(
            grid_shape=(12, 14),
            hidden_size=hidden_channels,
            input_mean=input_mean,
            input_std=input_std,
        )
    raise ValueError(f"Unknown model type: {model_type}")
