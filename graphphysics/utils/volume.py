import torch
from torch import Tensor


def compute_tetrahedra_volumes(positions: Tensor, tetra: Tensor) -> Tensor:
    """
    Compute the volumes of tetrahedral elements.

    Args:
        positions (Tensor): Node positions of shape (N, 3).
        tetra (Tensor): Tetra connectivity of shape (4, M).

    Returns:
        Tensor: Element-wise volumes of shape (M,).
    """
    v0 = positions[tetra[0]]
    v1 = positions[tetra[1]]
    v2 = positions[tetra[2]]
    v3 = positions[tetra[3]]

    a = v0 - v3
    b = v1 - v3
    c = v2 - v3

    triple = torch.einsum("ij,ij->i", a, torch.cross(b, c, dim=1))
    return torch.abs(triple) / 6.0


def compute_volume(positions: Tensor, tetra: Tensor) -> Tensor:
    """
    Compute the total volume of the tetrahedral mesh.
    """
    return compute_tetrahedra_volumes(positions, tetra).sum()
