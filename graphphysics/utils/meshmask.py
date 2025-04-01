import torch
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_edges(edge_index: torch.Tensor, node_index: torch.Tensor):
    """Filters edges based on the given node indices.

    Args:
        edge_index (Tensor): The edge indices.
        node_index (Tensor): The node indices to filter.

    Returns:
        Tuple[Tensor, Tensor]: Filtered edge indices and attributes.
    """
    num_nodes = maybe_num_nodes(edge_index, None)
    node_index = node_index.to(device)
    cluster_index = torch.arange(node_index.size(0), device=node_index.device)

    mask = node_index.new_full((num_nodes,), -1).to(device)
    mask[node_index] = cluster_index

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0), mask


def build_masked_graph(
    masked_graph: Data,
    selected_indexes: torch.Tensor,
):
    """
    Masks a PyTorch Geometric Data object based on selected indices.

    This function creates a masked version of the input graph by selecting only the nodes and edges
    specified by the provided indices. It updates the node features, positions, edge attributes, and
    edge indices accordingly.

    Parameters:
        - masked_graph: A PyTorch Geometric Data object to be masked.
        - selected_indexes: A tensor containing the indices of the nodes to be selected.

    Returns:
        A masked PyTorch Geometric Data object based on the selected indices.
    """
    masked_e_index, _ = filter_edges(masked_graph.edge_index, selected_indexes)

    masked_graph.edge_index = masked_e_index
    masked_graph.x = masked_graph.x[selected_indexes]
    masked_graph.pos = masked_graph.pos[selected_indexes]

    return masked_graph


def reconstruct_graph(
    graph: Data,
    latent_masked_graph: Data,
    selected_indexes: torch.Tensor,
    node_mask_token: torch.nn.Parameter,
) -> Data:
    """
    Given a graph and it's masked version, we assign a feature vector for each node based on:
      - it's computed value inside of the `latent_masked_graph` if this node was not masked
      - a [MASK] token otherwise

    The [MASK] token should be initialized and trained inside of the Masked Decoder as the
    following attribute: `self.node_mask_token = torch.nn.Parameter(torch.zeros(embedding_dim))`

    Parameters:
        - graph: A PyTorch Geometric Data object to be reconstructed.
        - latent_masked_graph: A Masked PyTorch Geometric Data object to be fetch features from.
        - selected_indexes: A tensor containing the indices of the nodes to be selected.
        - node_mask_token: torch.nn.Parameter to be used as a [MASK] token

    Returns:
        A PyTorch Geometric Data object.
    """
    n, f = graph.x.shape
    features = torch.zeros(n, f).to(device)
    features += node_mask_token.expand(n, -1)
    features[selected_indexes] = latent_masked_graph.x

    latent_graph = graph.clone()
    latent_graph.x = features

    return latent_graph
