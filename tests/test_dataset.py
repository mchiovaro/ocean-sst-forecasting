import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import SSTDataset
from src.evaluate import rmse
from src.model import PersistenceModel, SmallCNN


class StarterTests(unittest.TestCase):
    def test_dataset_model_and_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            X = np.arange(2 * 14 * 2 * 3, dtype=np.float32).reshape(2, 14, 2, 3)
            y = np.repeat(X[:, -1:, :, :], 3, axis=1)
            np.savez(path, X=X, y=y)

            dataset = SSTDataset(path)
            x0, y0 = dataset[0]
            prediction = PersistenceModel()(x0.unsqueeze(0))

            self.assertEqual(tuple(x0.shape), (14, 2, 3))
            self.assertEqual(tuple(prediction.shape), (1, 3, 2, 3))
            self.assertAlmostEqual(float(rmse(prediction, y0.unsqueeze(0))), 0.0)

    def test_masked_metric_and_small_cnn(self):
        X = torch.randn(4, 14, 3, 4)
        prediction = SmallCNN(hidden_channels=8, input_mean=25.0, input_std=3.0)(X)
        self.assertEqual(tuple(prediction.shape), (4, 3, 3, 4))
        target = prediction.clone()
        target[:, :, 0, 0] += 100.0
        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[0, 0] = False
        self.assertAlmostEqual(float(rmse(prediction, target, mask).detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
