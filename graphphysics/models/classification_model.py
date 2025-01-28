import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import (
    MLP,
    MessagePassing,
    PointNetConv,
    PointTransformerConv,
    fps,
    global_max_pool,
    global_mean_pool,
    knn,
    knn_graph,
    radius,
)
from torch_geometric.utils import scatter

from graphphysics.models.layers import GraphNetBlock, build_mlp


# The PointNet classification model and layer with message passing
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(
        self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNetClassifier(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 3,
        hidden_layers: int = 2,
        hidden_size: int = 64,
        output_size: int = 2,
    ):
        super().__init__()

        self.conv1 = PointNetLayer(node_input_size, hidden_size)

        self.processer_list = nn.ModuleList(
            [PointNetLayer(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )

        self.classifier = Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph=Batch) -> Tensor:

        h = graph.x
        pos = graph.pos
        edge_index = graph.edge_index
        batch = graph.batch

        # Perform two-layers of message passing:
        h = self.conv1(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        for layer in self.processer_list:
            h = layer(h=h, pos=pos, edge_index=edge_index)
            h = h.relu()

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        h = self.classifier(h)  # [num_examples, output_channels]

        # Classifier:
        return self.softmax(h)


# The PointNet++ classification model and layer
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, num_neighbors=16):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.num_neighbors = num_neighbors

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ClassificationPointNetP2(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 3,
        dim_model: list = [[64, 128, 128], [128, 128, 256], [256, 512, 1024]],
        output_size: int = 2,
        num_neighbors: int = 16,
    ):
        super().__init__()

        self.sa_modules = nn.ModuleList()
        for i in range(len(dim_model)):
            if i == 0:
                self.sa_modules.append(
                    SAModule(
                        0.5,
                        0.2,
                        MLP([3 + node_input_size] + dim_model[i]),
                        num_neighbors,
                    )
                )
            elif i == len(dim_model) - 1:
                self.sa_modules.append(
                    GlobalSAModule(MLP([dim_model[i - 1][-1] + 3] + dim_model[i]))
                )
            else:
                self.sa_modules.append(
                    SAModule(
                        0.25,
                        0.4,
                        MLP([dim_model[i - 1][-1] + 3] + dim_model[i]),
                        num_neighbors,
                    )
                )

        self.mlp = MLP([1024, 512, 256, output_size], dropout=0.5, norm=None)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)

        for sa in self.sa_modules:
            sa0_out = sa(*sa0_out)

        x, pos, batch = sa0_out

        x = self.mlp(x)

        return self.softmax(x)

        # return self.mlp(x).log_softmax(dim=-1)


# The PointTransformer classification model and layer
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP(
            [out_channels, 64, out_channels], norm=None, plain_last=False
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class ClassificationPointTransformer(torch.nn.Module):
    def __init__(self, in_channels, dim_model, out_channels=2, num_neighbors=16):
        super().__init__()
        self.num_neighbors = num_neighbors

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0], out_channels=dim_model[0]
        )
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    k=self.num_neighbors,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, data: Batch) -> Tensor:

        x = data.x
        pos = data.pos
        batch = data.batch

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.num_neighbors, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.num_neighbors, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return self.softmax(out)


# A Classification model with message passing
class ClassificationModel(nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int = 1,
        hidden_size: int = 128,
    ):
        super(ClassificationModel, self).__init__()
        self.hidden_size = hidden_size

        self.nodes_encoder = build_mlp(
            in_size=node_input_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
        )

        self.edges_encoder = build_mlp(
            in_size=edge_input_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
        )

        self.processer_list = nn.ModuleList(
            [GraphNetBlock(hidden_size=hidden_size) for _ in range(message_passing_num)]
        )

        self.decode_module = Decoder_1(hidden_size, hidden_size)

    def forward(self, graph: Batch) -> torch.Tensor:
        edge_index = graph.edge_index
        x = self.nodes_encoder(graph.x)
        edge_attr = self.edges_encoder(graph.edge_attr)
        for block in self.processer_list:
            x, edge_attr = block(x, edge_index, edge_attr)

        x_decoded = self.decode_module(x, graph.batch)
        return x_decoded


class Decoder_1(nn.Module):
    def __init__(
        self, in_size, hidden_size, out_size=1, nb_of_layers=4, layer_norm=True
    ):
        super().__init__()
        self.mlp = build_mlp(in_size, hidden_size, out_size, nb_of_layers, layer_norm)

    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return torch.sigmoid(x)
