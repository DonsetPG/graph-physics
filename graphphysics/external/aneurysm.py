import torch
from torch_geometric.data import Data
import pyvista as pv

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.meshio_mesh import convert_to_meshio_vtu


device = "cuda" if torch.cuda.is_available() else "cpu"


def aneurysm_node_type(graph: Data) -> torch.Tensor:
    tawss = graph.x[:, 0]

    node_type = torch.ones(tawss.shape, device=device) * NodeType.WALL_BOUNDARY

    inflow_mask = torch.logical_and(graph.pos[:, 1] <= 0.001, graph.pos[:, 0] <= 0)

    outflow_mask = torch.logical_and(graph.pos[:, 1] <= 0.001, graph.pos[:, 0] >= 0)

    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    return node_type.to(device)


def build_features(graph: Data) -> Data:
    node_type = aneurysm_node_type(graph)

    mesh = convert_to_meshio_vtu(graph)
    pv_mesh = pv.from_meshio(mesh)
    surf = pv_mesh.extract_surface(pass_pointid=True)
    surf_0 = surf.compute_normals(cell_normals=False, point_normals=True)
    normals = torch.from_numpy(surf_0.point_normals).float().to(graph.x.device)

    graph.x = graph.x[:, :80] # les inflow_moy
    graph.x = torch.cat(
    (
        graph.x,
        graph.pos,
        normals,
        node_type.unsqueeze(1),
    ),
    dim=1,
)
    return graph
