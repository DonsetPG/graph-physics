import unittest
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning as L
import shutil
import tempfile
import os
import numpy as np
from graphphysics.training.lightning_module import LightningModule
from tests.mock import (
    MOCK_H5_META_SAVE_PATH,
    MOCK_H5_SAVE_PATH,
)
import pyvista as pv

with patch("graphphysics.training.parse_parameters.get_model") as mock_get_model, patch(
    "graphphysics.training.parse_parameters.get_simulator"
) as mock_get_simulator, patch("graphphysics.utils.loss.L2Loss") as MockL2Loss, patch(
    "graphphysics.utils.scheduler.CosineWarmupScheduler"
) as MockCosineWarmupScheduler:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class MockDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            x = torch.randn(10, 8)
            x = torch.abs(x)
            edge_index = torch.randint(0, 10, (2, 20))
            edge_attr = torch.randn(20, 4)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.y = torch.randn(10, 3)
            data.traj_index = 0
            return data

    class TestLitLightningModule(unittest.TestCase):
        def setUp(self):
            self.parameters = {
                "transformations": {
                    "preprocessing": {
                        "noise": 0.1,
                        "noise_index_start": 0,
                        "noise_index_end": 3,
                    },
                    "world_pos_parameters": {
                        "use": True,
                        "world_pos_index_start": 3,
                        "world_pos_index_end": 6,
                    },
                },
                "index": {
                    "node_type_index": 7,
                    "feature_index_start": 0,
                    "feature_index_end": 6,
                    "output_index_start": 0,
                    "output_index_end": 3,
                },
                "model": {
                    "type": "epd",
                    "message_passing_num": 5,
                    "node_input_size": 6,
                    "edge_input_size": 4,
                    "output_size": 3,
                    "hidden_size": 128,
                },
                "dataset": {
                    "extension": "h5",
                    "h5_path": MOCK_H5_SAVE_PATH,
                    "meta_path": MOCK_H5_META_SAVE_PATH,
                    "khop": 2,
                },
            }
            self.learning_rate = 0.001
            self.num_steps = 100
            self.warmup = 10
            self.only_processor = False

            # Mock the processor and simulator models
            self.mock_processor = MagicMock()
            self.mock_simulator = MagicMock()
            self.mock_simulator.node_type_index = 0
            # Mock the model's forward method
            self.mock_simulator.return_value = (
                torch.randn(10, 3),
                torch.randn(10, 3),
                None,
            )

            mock_get_model.return_value = self.mock_processor
            mock_get_simulator.return_value = self.mock_simulator

            self.trajectory_length = 599

            self.model = LightningModule(
                parameters=self.parameters,
                learning_rate=self.learning_rate,
                num_steps=self.num_steps,
                warmup=self.warmup,
                only_processor=self.only_processor,
                trajectory_length=self.trajectory_length,
            )

            self.dataset = MockDataset()
            self.dataloader = DataLoader(self.dataset, batch_size=2)

        def test_forward(self):
            batch = next(iter(self.dataloader))
            output = self.model.forward(batch.to(device))
            self.assertIsNotNone(output)

        def test_training_step(self):
            batch = next(iter(self.dataloader))
            loss = self.model.training_step(batch.to(device))
            self.assertIsNotNone(loss)
            self.assertTrue(isinstance(loss, torch.Tensor))

        def test_configure_optimizers(self):
            optimizers = self.model.configure_optimizers()
            self.assertIn("optimizer", optimizers)
            self.assertIn("lr_scheduler", optimizers)
            self.assertIsNotNone(optimizers["optimizer"])
            self.assertIsNotNone(optimizers["lr_scheduler"])

        def test_full_training_loop(self):
            trainer = L.Trainer(fast_dev_run=True)
            trainer.fit(self.model, train_dataloaders=self.dataloader)

        def test_validation_step(self):
            self.dataloader = DataLoader(self.dataset, batch_size=1)
            batch = next(iter(self.dataloader))

            # Run validation step
            self.model.eval()
            self.model.validation_step(batch.to(device), batch_idx=0)

            # Check that val_step_outputs and val_step_targets have been updated
            self.assertEqual(len(self.model.val_step_outputs), 1)
            self.assertEqual(len(self.model.val_step_targets), 1)
            self.assertEqual(self.model.val_step_outputs[0].shape, (10, 3))
            self.assertEqual(self.model.val_step_targets[0].shape, (10, 3))

            # Check that last_val_prediction is set
            self.assertIsNotNone(self.model.last_val_prediction)
            self.assertEqual(self.model.last_val_prediction.shape, (10, 3))

        def test_on_validation_epoch_end(self):
            # Simulate multiple validation steps
            num_steps = 3
            batch_size = 5
            output_dim = 2
            self.model.eval()

            # Simulate trajectory_to_save with sample graphs
            num_graphs = 3
            for i in range(num_graphs):
                # Create a simple graph
                pos = torch.tensor(
                    [[0.0 + i, 0.0], [1.0 + i, 0.0], [1.0 + i, 1.0], [0.0 + i, 1.0]],
                    dtype=torch.float,
                )
                edge_index = torch.tensor(
                    [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
                )
                x = torch.tensor(
                    [
                        [i * 10 + 1, i * 10 + 1],
                        [i * 10 + 2, i * 10 + 2],
                        [i * 10 + 3, i * 10 + 3],
                        [i * 10 + 4, i * 10 + 4],
                    ],
                    dtype=torch.float,
                )
                graph = Data(pos=pos, edge_index=edge_index, x=x)
                self.model.trajectory_to_save.append(graph)

            # Simulate val_step_outputs and val_step_targets
            for i in range(num_steps):
                predicted_outputs = torch.randn(batch_size, output_dim)
                targets = torch.randn(batch_size, output_dim)
                self.model.val_step_outputs.append(predicted_outputs)
                self.model.val_step_targets.append(targets)

            # Run on_validation_epoch_end
            with patch.object(self.model, "log") as mock_log:
                self.model.on_validation_epoch_end()

                # Check that RMSE is computed and logged
                mock_log.assert_called_with(
                    "val_all_rollout_rmse",
                    unittest.mock.ANY,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            # Check that val_step_outputs and val_step_targets are cleared
            self.assertEqual(len(self.model.val_step_outputs), 0)
            self.assertEqual(len(self.model.val_step_targets), 0)
            self.assertEqual(self.model.current_val_trajectory, 0)
            self.assertIsNone(self.model.last_val_prediction)

        def test_validation_step_resets_trajectory(self):
            # Create mock batches
            self.dataloader = DataLoader(self.dataset, batch_size=1)
            batch = next(iter(self.dataloader))
            self.model.eval()

            # Run validation steps with batch_idx increasing
            self.model.validation_step(batch.to(device), batch_idx=0)
            first_prediction = self.model.last_val_prediction.clone()
            assert self.model.current_val_trajectory == 0
            batch.traj_index = 1
            self.model.validation_step(
                batch, batch_idx=self.trajectory_length
            )  # Should reset trajectory
            assert self.model.current_val_trajectory == 1

    if __name__ == "__main__":
        unittest.main()
