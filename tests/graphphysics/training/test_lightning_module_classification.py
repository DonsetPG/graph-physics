import unittest
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning as L
from graphphysics.training.lightning_module_classification import LightningModuleClassification
from graphphysics.dataset.dataset_classification import GraphClassificationDataset

from tests.mock import (
    MOCK_CLASSIFICATION_META_SAVE_PATH,
    MOCK_CLASSIFICATION_SAVE_PATH,   
)


device = "cuda" if torch.cuda.is_available() else "cpu"

class MockDataset(GraphClassificationDataset):
    def __init__(self):
        self.file_paths = ["mock_path_1.xdmf", "mock_path_2.xdmf"]
        self.labels = [0, 1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing = None
        self.masking_ratio = None

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        x = torch.randn(10, 8)
        x = torch.abs(x)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 4)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.randint(0, 2, ()).float()
        return data



class TestLightningModuleClassification(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "transformations": {
                "preprocessing": {
                    "noise": [10.0],
                    "noise_index_start": [0],
                    "noise_index_end": [1],
                    "masking": 0.1
                },
                "world_pos_parameters": {
                    "use": False,
                    "world_pos_index_start": 0,
                    "world_pos_index_end": 3
                }
            },
            "index": {
                "feature_index_start": 0,
                "feature_index_end": 2,
                "output_index_start": 0,
                "output_index_end": 1,
                "node_type_index": 1
            },
            "model": {
                "type": "classification",
                "message_passing_num": 10,
                "hidden_size": 128,
                "node_input_size": 2,
                "output_size": 1,
                "edge_input_size": 4,
                "num_heads": 4
            },
            "dataset": {
                "extension": "xdmf",
                "obj_folder": MOCK_CLASSIFICATION_SAVE_PATH, 
                "meta_path": MOCK_CLASSIFICATION_META_SAVE_PATH,
                "khop": 1
            },
            "training": {
                "batch_size": 1
            }
    }
        self.learning_rate = 0.001
        self.num_steps = 100
        self.warmup = 10

        self.model = LightningModuleClassification(
            parameters=self.parameters,
            learning_rate=self.learning_rate,
            num_steps=self.num_steps,
            warmup=self.warmup,
        )

        self.dataset = MockDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=1)

    def test_forward(self):
        batch = next(iter(self.dataloader))
        output = self.model.forward(batch.to(device))
        self.assertIsNotNone(output)

    def test_training_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.training_step(batch.to(device))
        self.assertIsNotNone(loss)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_validation_step(self):
        self.dataloader = DataLoader(self.dataset, batch_size=1)
        batch = next(iter(self.dataloader))

        # Run validation step
        self.model.eval()
        self.model.validation_step(batch.to(device), batch_idx=0)

        # Check that val_step_outputs and val_step_targets have been updated
        self.assertEqual(len(self.model.val_step_outputs), 1)
        self.assertEqual(len(self.model.val_step_targets), 1)
        self.assertEqual(self.model.val_step_outputs[0].shape, (1,))
        self.assertEqual(self.model.val_step_targets[0].shape, (1,))

    def test_on_validation_epoch_end(self):
        # Simulate multiple validation steps
        num_steps = 3
        batch_size = 5
        output_dim = 1
        self.model.eval()

        # Simulate val_step_outputs and val_step_targets
        for i in range(num_steps):
            predicted_outputs = torch.randint(0, 2, (batch_size, output_dim)).float()
            targets = torch.randint(0, 2, (batch_size, output_dim)).float()
            self.model.val_step_outputs.append(predicted_outputs)
            self.model.val_step_targets.append(targets)

        # Run on_validation_epoch_end
        with patch.object(self.model, "log") as mock_log:
            self.model.on_validation_epoch_end()

            # Check that confusion matrix and F1 score are computed and logged
            mock_log.assert_any_call("val_conf_matrix", unittest.mock.ANY)
            mock_log.assert_any_call("val_f1_score", unittest.mock.ANY)

        # Check that val_step_outputs and val_step_targets are cleared
        self.assertEqual(len(self.model.val_step_outputs), 0)
        self.assertEqual(len(self.model.val_step_targets), 0)

    def test_configure_optimizers(self):
        optimizers = self.model.configure_optimizers()
        self.assertIn("optimizer", optimizers)
        self.assertIn("lr_scheduler", optimizers)
        self.assertIsNotNone(optimizers["optimizer"])
        self.assertIsNotNone(optimizers["lr_scheduler"])

    def test_full_training_loop(self):
        trainer = L.Trainer(fast_dev_run=True)
        trainer.fit(self.model, train_dataloaders=self.dataloader)

if __name__ == "__main__":
    unittest.main()
