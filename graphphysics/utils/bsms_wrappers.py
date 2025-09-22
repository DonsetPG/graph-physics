import os
import pickle
from typing import Optional

from enum import Enum
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
import torch

# Try importing sparse_dot_mkl
try:
    from sparse_dot_mkl import dot_product_mkl as spdot

    print("Using MKL-accelerated sparse dot product")
except ImportError:
    print("MKL not available, falling back to scipy.sparse.dot")

    def spdot(A, B):
        # Fallback using scipy
        return A.dot(B)


_INF = 1 + 1e10


def sparse_square(adj_mat):
    if torch.cuda.is_available():
        # Ensure COO format
        adj_mat = adj_mat.tocoo()

        # Convert SciPy COO -> PyTorch sparse tensor
        indices = torch.tensor(
            np.vstack((adj_mat.row, adj_mat.col)), dtype=torch.long, device="cuda"
        )
        values = torch.tensor(adj_mat.data, dtype=torch.float32, device="cuda")
        A = torch.sparse_coo_tensor(indices, values, adj_mat.shape, device="cuda")

        # Sparse × Sparse multiply (GPU)
        C = torch.sparse.mm(A, A)

        # Convert back to SciPy COO
        C = C.coalesce()  # remove duplicates
        indices = C.indices().cpu().numpy()
        values = C.values().cpu().numpy()
        return coo_matrix((values, (indices[0], indices[1])), shape=C.shape)
    else:
        # CPU fallback (SciPy)
        return adj_mat @ adj_mat


class GraphType(Enum):
    """
    Enumeration to define the types of graph representations.
    """

    FLAT_EDGE = 1
    ADJ_LIST = 2
    ADJ_MAT = 3


class Graph:
    def __init__(self, g, g_type, num_nodes):
        """
        Initialize the Graph object.

        Parameters:
        g (np.ndarray, list, or scipy.sparse.coo_matrix): The graph data in the specified format.
        g_type (GraphType): The type of the input graph representation.
        num_nodes (int): The number of nodes in the graph.
        """
        self.num_nodes = num_nodes

        if g_type == GraphType.FLAT_EDGE:
            self.flat_edges = g
        elif g_type == GraphType.ADJ_LIST:
            self.flat_edges = self.adj_list_to_flat_edge(g)
        elif g_type == GraphType.ADJ_MAT:
            self.flat_edges = self.adj_mat_to_flat_edge(g)
        else:
            raise ValueError(f"Unknown graph type: {g_type}")

        self.adj_list = self.get_adj_list()
        self.clusters = self.find_clusters()

    def get_flat_edge(self):
        """
        Get the flat edge representation of the graph.

        Returns:
        np.ndarray: The flat edge representation of the graph.
        """
        return self.flat_edges

    def get_adj_list(self):
        """
        Get the adjacency list representation of the graph.

        Returns:
        list: The adjacency list representation of the graph.
        """
        return self.flat_edge_to_adj_list(self.flat_edges, self.num_nodes)

    def get_sparse_adj_mat(self):
        """
        Get the sparse adjacency matrix representation of the graph.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix representation of the graph.
        """
        return self.flat_edge_to_adj_mat(self.flat_edges, self.num_nodes)

    def bfs_dist(self, seed):
        """
        Perform a Breadth-First Search (BFS) to find the shortest distance from the seed to all other nodes.

        Parameters:
        seed (int or list): The starting node(s) for BFS.

        Returns:
        np.ndarray: The shortest distance from the seed to all other nodes.
        """
        _INF = 1 + 1e10
        res = np.ones(self.num_nodes) * _INF
        visited = [False for _ in range(self.num_nodes)]
        if isinstance(seed, list):
            for s in seed:
                res[s] = 0
                visited[s] = True
            frontier = seed
        else:
            res[seed] = 0
            visited[seed] = True
            frontier = [seed]

        depth = 0
        track = [frontier]
        while frontier:
            this_level = frontier
            depth += 1
            frontier = []
            while this_level:
                f = this_level.pop(0)
                for n in self.adj_list[f]:
                    if not visited[n]:
                        visited[n] = True
                        frontier.append(n)
                        res[n] = depth
            track.append(frontier)

        return res

    def find_clusters(self):
        """
        Find connected clusters in the graph using BFS.

        Returns:
        list: A list of clusters, each cluster is a list of node indices.
        """
        _INF = 1 + 1e10
        remaining_nodes = list(range(self.num_nodes))
        clusters = []
        while remaining_nodes:
            if len(remaining_nodes) > 1:
                seed = remaining_nodes[0]
                dist = self.bfs_dist(seed)
                tmp = []
                new_remaining = []
                for n in remaining_nodes:
                    if dist[n] != _INF:
                        tmp.append(n)
                    else:
                        new_remaining.append(n)
                clusters.append(tmp)
                remaining_nodes = new_remaining
            else:
                clusters.append([remaining_nodes[0]])
                break

        return clusters

    @staticmethod
    def flat_edge_to_adj_mat(edge_list, n):
        """
        Convert a flat edge list to a sparse adjacency matrix.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix.
        """
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n)
        )
        return adj_mat

    @staticmethod
    def flat_edge_to_adj_list(edge_list, n):
        """
        Convert a flat edge list to an adjacency list.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        list: The adjacency list.
        """
        adj_list = [[] for _ in range(n)]
        for i in range(len(edge_list[0])):
            adj_list[edge_list[0, i]].append(edge_list[1, i])
        return adj_list

    @staticmethod
    def adj_list_to_flat_edge(adj_list):
        """
        Convert an adjacency list to a flat edge list.

        Parameters:
        adj_list (list): The adjacency list.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        edge_list = []
        for i in range(len(adj_list)):
            for n in adj_list[i]:
                edge_list.append([i, n])
        return np.array(edge_list).transpose()

    @staticmethod
    def adj_mat_to_flat_edge(adj_mat):
        """
        Convert a sparse adjacency matrix to a flat edge list.

        Parameters:
        adj_mat (np.ndarray or scipy.sparse.spmatrix): The sparse adjacency matrix.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        if isinstance(adj_mat, np.ndarray):
            s, r = np.where(adj_mat.astype(bool))
        elif isinstance(adj_mat, scipy.sparse.coo_matrix):
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        elif isinstance(adj_mat, scipy.sparse.csr_matrix):
            adj_mat = scipy.sparse.coo_matrix(adj_mat)
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        else:
            raise ValueError(
                "Unsupported adjacency matrix type in adj_mat_to_flat_edge."
            )
        return np.array([s, r])


class BistrideMultiLayerGraph:
    def __init__(self, flat_edge, num_layers, num_nodes, pos_mesh):
        """
        Initialize the BistrideMultiLayerGraph object.

        Parameters:
        flat_edge (np.ndarray): The flat edge list of shape [2, num_edges].
        num_layers (int): The number of layers to generate.
        num_nodes (int): The number of nodes in the graph.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.pos_mesh = pos_mesh

        # Initialize the first layer graph
        self.m_gs = [Graph(flat_edge, GraphType.FLAT_EDGE, num_nodes)]
        self.m_flat_es = [self.m_gs[0].get_flat_edge()]
        self.m_ids = []

        self.generate_multi_layer_graphs()

    def generate_multi_layer_graphs(self):
        """
        Generate multiple layers of graphs with pooling.
        """
        g_l = self.m_gs[0]
        pos_l = self.pos_mesh

        index_to_keep = list(range(self.num_nodes))
        for layer in range(self.num_layers):
            n_l = self.num_nodes if layer == 0 else len(index_to_keep)
            index_to_keep, g_l = self.bstride_selection(g_l, pos_l, n_l)
            index_to_keep = np.array(index_to_keep)
            pos_l = pos_l[index_to_keep]
            self.m_gs.append(g_l)
            self.m_flat_es.append(g_l.get_flat_edge())
            self.m_ids.append(index_to_keep)

    def get_multi_layer_graphs(self):
        """
        Get the multi-layer graph structures.

        Returns:
        tuple: A tuple containing three lists:
            - m_gs (list): List of graph wrappers for each layer.
            - m_flat_es (list): List of flat edges for each layer.
            - m_ids (list): List of node indices to be pooled at each layer.
        """
        return self.m_gs, self.m_flat_es, self.m_ids

    @staticmethod
    def bstride_selection(g, pos_mesh, n):
        """
        Perform bstride selection to pool nodes and edges.

        Parameters:
        g (Graph): The graph wrapper object.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        n (int): The number of nodes.

        Returns:
        tuple: A tuple containing:
            - combined_idx_kept (list): List of node indices to be pooled.
            - new_g (Graph): The new graph wrapper object after pooling.
        """
        combined_idx_kept = set()
        adj_mat = g.get_sparse_adj_mat()
        adj_mat.setdiag(1)
        clusters = g.clusters

        seeds = BistrideMultiLayerGraph.nearest_center_seed(pos_mesh, clusters)

        for seed, c in zip(seeds, clusters):
            even, odd = set(), set()
            dist_from_central_node = g.bfs_dist(seed)

            for i, dist in enumerate(dist_from_central_node):
                if dist % 2 == 0 and dist != _INF:
                    even.add(i)
                elif dist % 2 == 1 and dist != _INF:
                    odd.add(i)

            if len(even) <= len(odd) or not odd:
                index_kept, _ = even, odd
            else:
                index_kept, _ = odd, even

            combined_idx_kept = combined_idx_kept.union(index_kept)

        combined_idx_kept = list(combined_idx_kept)
        combined_idx_kept.sort()
        adj_mat = adj_mat.tocsr().astype(float)
        adj_mat = sparse_square(adj_mat)
        adj_mat.setdiag(0)
        new_g = BistrideMultiLayerGraph.pool_edge(adj_mat, n, combined_idx_kept)

        return combined_idx_kept, new_g

    @staticmethod
    def nearest_center_seed(pos_mesh, clusters):
        """
        Find the nearest center seed for each cluster.

        Parameters:
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        clusters (list): List of clusters, each cluster is a list of node indices.

        Returns:
        list: List of seeds per cluster.
        """
        seeds = []
        for c in clusters:
            center = np.mean(pos_mesh[c], axis=0)
            delta_to_center = pos_mesh[c] - center[None, :]
            dist_to_center = np.linalg.norm(delta_to_center, 2, axis=-1)
            min_node = c[np.argmin(dist_to_center)]
            seeds.append(min_node)

        return seeds

    @staticmethod
    def pool_edge(adj_mat, num_nodes, idx):
        """
        Pool the edges based on the provided node indices.

        Parameters:
        adj_mat (scipy.sparse.csr_matrix): The adjacency matrix in CSR format.
        num_nodes (int): The number of nodes in the input graph.
        idx (list): List of node indices to be kept.

        Returns:
        Graph: The new graph wrapper object after pooling.
        """
        flat_e = Graph.adj_mat_to_flat_edge(adj_mat)
        idx = np.array(idx, dtype=np.int64)
        idx_new_valid = np.arange(len(idx)).astype(np.int64)
        idx_new_all = -1 * np.ones(num_nodes).astype(np.int64)
        idx_new_all[idx] = idx_new_valid
        new_flat_e = -1 * np.ones_like(flat_e).astype(np.int64)
        new_flat_e[0] = idx_new_all[flat_e[0]]
        new_flat_e[1] = idx_new_all[flat_e[1]]
        both_valid = np.logical_and(new_flat_e[0] >= 0, new_flat_e[1] >= 0)
        e_idx = np.where(both_valid)[0]
        new_flat_e = new_flat_e[:, e_idx]
        new_g = Graph(new_flat_e, GraphType.FLAT_EDGE, len(idx))

        return new_g


def to_flat_edge(mesh, mesh_type):
    if mesh_type == "triangle":
        return triangles_to_edges(mesh)
    elif mesh_type == "tetra":
        return tetras_to_edges(mesh)
    elif mesh_type == "flat":
        return mesh
    else:
        raise ValueError(f"Unsupported mesh type {mesh_type} in to_flat_edge.")


def triangles_to_edges(cells):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    t_cell = torch.tensor(cells)
    edge_index = torch.cat(
        (
            t_cell[:, :2],
            t_cell[:, 1:3],
            torch.cat((t_cell[:, 2].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1),
        ),
        0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def tetras_to_edges(cells):
    """Computes mesh edges from tetrahedrons."""
    # collect edges from triangles
    t_cell = torch.tensor(cells)
    edge_index = torch.cat(
        (
            t_cell[:, :2],
            t_cell[:, 1:3],
            t_cell[:, 2:4],
            torch.cat((t_cell[:, 3].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1),
            torch.cat((t_cell[:, 0].unsqueeze(1), t_cell[:, 2].unsqueeze(1)), -1),
            torch.cat((t_cell[:, 1].unsqueeze(1), t_cell[:, 3].unsqueeze(1)), -1),
        ),
        0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def calculate_multi_mesh(
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    mesh_type: str,
    message_passing_num: int,
    mm_layer_dir: str,
    mesh_id: Optional[str] = None,
):
    """
    Calculate the multi-layer mesh.

    Parameters
    ----------
    mesh_pos : np.ndarray
        The positions of the mesh nodes.
    """
    mmfile = os.path.join(mm_layer_dir, f"mmesh_layer_{mesh_id}.dat")
    if not os.path.isfile(mmfile):
        edge_i = to_flat_edge(cells, mesh_type)
        num_nodes = mesh_pos.shape[0]
        m_graph = BistrideMultiLayerGraph(
            edge_i, message_passing_num, num_nodes, mesh_pos
        )
        _, m_gs, m_ids = m_graph.get_multi_layer_graphs()
        m_gs = [torch.tensor(g, dtype=torch.long) for g in m_gs]
        m_ids = [torch.tensor(ids, dtype=torch.long) for ids in m_ids]
        m_mesh = {"m_gs": m_gs, "m_ids": m_ids}
        with open(mmfile, "wb") as f:
            pickle.dump(m_mesh, f)
    else:
        with open(mmfile, "rb") as f:
            m_mesh = pickle.load(f)
        m_gs, m_ids = m_mesh["m_gs"], m_mesh["m_ids"]


def load_multi_mesh(mm_layer_dir: str, mesh_id: str, device: torch.device = "cpu"):
    mmfile = os.path.join(mm_layer_dir, f"mmesh_layer_{mesh_id}.dat")
    if not os.path.isfile(mmfile):
        raise FileNotFoundError(f"Multi-mesh file {mmfile} not found.")
    with open(mmfile, "rb") as f:
        m_mesh = pickle.load(f)
    m_gs, m_ids = m_mesh["m_gs"], m_mesh["m_ids"]
    m_gs = [g.to(device) for g in m_gs]
    m_ids = [ids.to(device) for ids in m_ids]
    return m_gs, m_ids
