import torch
from torch_geometric.data import Data
from torch.nn.modules.loss import _Loss

from graphphysics.utils.loss import MultiLoss, VolumeLoss
from graphphysics.utils.volume import compute_tetrahedra_volumes, compute_volume


def _unit_tetrahedron():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tetra = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    return positions, tetra


def test_compute_volume_single_tetrahedron():
    positions, tetra = _unit_tetrahedron()

    volumes = compute_tetrahedra_volumes(positions, tetra)
    total_volume = compute_volume(positions, tetra)

    print(f"Tetrahedra volumes: {volumes.cpu().numpy().tolist()}")
    print(f"Total volume: {total_volume.item()}")

    expected = torch.tensor([1.0 / 6.0], dtype=torch.float32)
    assert torch.allclose(volumes, expected, atol=1e-6)
    assert torch.isclose(total_volume, expected.sum(), atol=1e-6)


def test_volume_loss_matches_absolute_difference():
    volume_loss = VolumeLoss()
    volume = torch.tensor(0.83)
    volume_target = torch.tensor(0.75)

    loss_value = volume_loss(
        target=None,
        network_output=None,
        node_type=None,
        masks=None,
        volume=volume,
        volume_target=volume_target,
    )

    print(
        f"VolumeLoss -> volume: {volume.item()}, "
        f"target: {volume_target.item()}, loss: {loss_value.item()}"
    )

    expected = torch.abs(volume - volume_target)
    assert torch.isclose(loss_value, expected, atol=1e-6)


class _DummyL2Loss(_Loss):
    def forward(self, target, network_output, **_):
        return torch.mean((network_output - target) ** 2)


def test_multiloss_with_volume_loss_has_weighted_components():
    positions, tetra = _unit_tetrahedron()
    offset = torch.tensor(
        [
            [0.10, -0.05, 0.02],
            [0.05, 0.02, -0.03],
            [-0.04, 0.01, 0.06],
            [0.02, 0.03, -0.01],
        ],
        dtype=torch.float32,
    )

    target_physical = positions
    network_output_physical = positions + offset

    volume_target = compute_volume(target_physical, tetra)
    volume = compute_volume(network_output_physical, tetra)

    target = torch.zeros((positions.shape[0], 3), dtype=torch.float32)
    network_output = target + 0.2

    multi_loss = MultiLoss([_DummyL2Loss(), VolumeLoss()], weights=[0.9, 0.1])
    total_loss, components = multi_loss(
        graph=Data(),
        network_output_physical=network_output_physical,
        target_physical=target_physical,
        network_output=network_output,
        target=target,
        node_type=None,
        masks=None,
        gradient_method=None,
        return_all_losses=True,
        volume=volume,
        volume_target=volume_target,
    )

    weighted_l2 = 0.9 * torch.mean((network_output - target) ** 2)
    weighted_volume = 0.1 * torch.abs(volume - volume_target)
    expected_total = weighted_l2 + weighted_volume

    print(
        f"MultiLoss components (weighted): "
        f"L2={components[0].item()}, Volume={components[1].item()}"
    )
    print(f"MultiLoss total: {total_loss.item()}")

    assert torch.isclose(components[0], weighted_l2, atol=1e-6)
    assert torch.isclose(components[1], weighted_volume, atol=1e-6)
    assert torch.isclose(total_loss, expected_total, atol=1e-6)
