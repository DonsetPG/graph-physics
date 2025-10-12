import unittest

from copy import deepcopy
import torch
from unittest.mock import MagicMock, patch

from graphphysics.training.parse_parameters import (
    get_preprocessing,
    get_model,
    get_simulator,
    get_dataset,
    get_loss,
    get_gradient_method,
)
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.loss import L2Loss, MultiLoss, DivergenceL2Loss, CosineLoss
from graphphysics.models.layers import set_use_silu_activation, use_silu_activation

from tests.mock import (
    MOCK_H5_META_SAVE_PATH,
    MOCK_H5_SAVE_PATH,
)
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)

# Mock imports from 'graphphysics' package
with patch(
    "graphphysics.training.parse_parameters.build_preprocessing"
) as mock_build_preprocessing, patch(
    "graphphysics.models.processors.EncodeProcessDecode"
) as MockEncodeProcessDecode, patch(
    "graphphysics.models.processors.EncodeTransformDecode"
) as MockEncodeTransformDecode, patch(
    "graphphysics.training.parse_parameters.EncodeTransformDecode",
    new=MockEncodeTransformDecode,
), patch(
    "graphphysics.training.parse_parameters.EncodeProcessDecode",
    new=MockEncodeProcessDecode,
), patch(
    "graphphysics.training.parse_parameters.TransolverProcessor"
) as MockTransolverProcessor, patch(
    "graphphysics.training.parse_parameters.Simulator"
) as MockSimulator, patch(
    "graphphysics.dataset.h5_dataset.H5Dataset"
) as MockH5Dataset, patch(
    "graphphysics.dataset.xdmf_dataset.XDMFDataset"
) as MockXDMFDataset:

    class TestConfigurationHelpers(unittest.TestCase):
        def setUp(self):
            self.device = torch.device("cpu")
            set_use_silu_activation(False)
            self.addCleanup(lambda: set_use_silu_activation(False))
            self.param = {
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
                    "output_index_start": 6,
                    "output_index_end": 9,
                },
                "model": {
                    "type": "epd",
                    "message_passing_num": 5,
                    "node_input_size": 6,
                    "edge_input_size": 4,
                    "output_size": 3,
                    "hidden_size": 128,
                    "use_silu_activation": False,
                    "use_rope_embeddings": False,
                    "use_gated_attention": False,
                    "use_gated_mlp": False,
                },
                "dataset": {
                    "extension": "h5",
                    "h5_path": MOCK_H5_SAVE_PATH,
                    "meta_path": MOCK_H5_META_SAVE_PATH,
                    "khop": 2,
                },
                "loss": {
                    "type": ["l2loss", "gradientl2loss", "divergencel2loss"],
                    "weights": [0.5, 0.5, 0.5],
                    "gradient_method": "finite_diff",
                },
            }
            self.param["training"] = {
                "use_spatial_mtp": False,
                "use_temporal_block": False,
            }

        def test_get_preprocessing(self):
            preprocessing_function = get_preprocessing(self.param, self.device)

        def test_get_model_epd(self):
            MockEncodeProcessDecode.reset_mock()
            model = get_model(self.param)

        def test_get_model_epd_with_gated_mlp(self):
            MockEncodeProcessDecode.reset_mock()
            self.param["model"]["use_gated_mlp"] = True
            get_model(self.param)

        def test_get_model_epd_with_rope_and_gate(self):
            MockEncodeProcessDecode.reset_mock()
            self.param["model"]["use_rope_embeddings"] = True
            self.param["model"]["use_gated_attention"] = True
            self.param["model"]["rope_pos_dimension"] = 3
            get_model(self.param)

        def test_get_model_transformer(self):
            MockEncodeTransformDecode.reset_mock()
            self.param["model"]["type"] = "transformer"
            self.param["model"]["num_heads"] = 8
            self.param["model"]["use_silu_activation"] = True
            self.param["model"]["use_rope_embeddings"] = True
            self.param["model"]["use_gated_attention"] = True
            self.param["training"]["use_temporal_block"] = True
            model = get_model(self.param)
            self.assertTrue(use_silu_activation())

        def test_get_model_invalid(self):
            self.param["model"]["type"] = "invalid_model_type"
            with self.assertRaises(ValueError) as context:
                get_model(self.param)
            self.assertIn(
                "Model type 'invalid_model_type' not supported", str(context.exception)
            )

        def test_get_simulator(self):
            model = get_model(self.param)
            simulator = get_simulator(self.param, model, self.device)

            expected_node_input_size = (
                self.param["model"]["node_input_size"] + NodeType.SIZE
            )
            self.assertEqual(simulator.node_input_size, expected_node_input_size)
            self.assertEqual(
                simulator.edge_input_size, self.param["model"]["edge_input_size"]
            )
            self.assertEqual(simulator.output_size, self.param["model"]["output_size"])
            self.assertEqual(simulator.model, model)

        def test_get_dataset_h5(self):
            preprocessing = MagicMock()
            dataset = get_dataset(self.param, preprocessing)

            self.assertEqual(dataset.h5_path, self.param["dataset"]["h5_path"])
            self.assertEqual(dataset.meta_path, self.param["dataset"]["meta_path"])
            self.assertEqual(dataset.preprocessing, preprocessing)
            self.assertEqual(dataset.khop, self.param["dataset"]["khop"])
            self.assertTrue(dataset.add_edge_features, True)

        def test_get_dataset_xdmf(self):
            self.param["dataset"]["extension"] = "xdmf"
            self.param["dataset"]["meta_path"] = MOCK_H5_META10_SAVE_PATH
            self.param["dataset"]["xdmf_folder"] = MOCK_XDMF_FOLDER
            dataset = get_dataset(self.param, preprocessing=MagicMock())

            self.assertEqual(dataset.xdmf_folder, self.param["dataset"]["xdmf_folder"])
            self.assertEqual(dataset.meta_path, self.param["dataset"]["meta_path"])

        def test_get_dataset_invalid(self):
            self.param["dataset"]["extension"] = "invalid_extension"
            with self.assertRaises(ValueError) as context:
                get_dataset(self.param, preprocessing=MagicMock())
            self.assertIn(
                "Dataset extension 'invalid_extension' not supported",
                str(context.exception),
            )

        def test_get_gradient_method(self):
            param_wo_loss = deepcopy(self.param)
            del param_wo_loss["loss"]

            gradient_method = get_gradient_method(param=self.param)
            self.assertEqual(gradient_method, "finite_diff")

            gradient_method_wo_loss = get_gradient_method(param=param_wo_loss)
            self.assertIsNone(gradient_method_wo_loss)

        def test_get_loss(self):
            multi_loss, loss_name = get_loss(param=self.param)
            self.assertIsInstance(multi_loss, MultiLoss)
            self.assertEqual(len(loss_name), len(self.param["loss"]["type"]))

            self.param["loss"]["type"] = ["divergencel2loss"]
            single_loss, loss_name = get_loss(param=self.param)
            self.assertIsInstance(single_loss, DivergenceL2Loss)
            self.assertEqual(loss_name, "DIVERGENCEL2LOSS")

            self.param["loss"]["type"] = ["cosinel2loss"]
            cosine_loss, loss_name = get_loss(param=self.param)
            self.assertIsInstance(cosine_loss, CosineLoss)
            self.assertEqual(loss_name, "COSINEL2LOSS")

            # Assert that with no loss parameters, default loss is L2Loss
            param_wo_loss = deepcopy(self.param)
            del param_wo_loss["loss"]

            l2_loss, loss_name = get_loss(param=param_wo_loss)
            self.assertIsInstance(l2_loss, L2Loss)
            self.assertEqual(loss_name, "L2LOSS")

    if __name__ == "__main__":
        unittest.main()
