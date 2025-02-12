import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ReLU
from torch_geometric.data import Batch
from torch_geometric.nn import (
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

from graphphysics.models.layers import build_mlp


# The PointNet classification model and layer with message passing
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = build_mlp(
            in_size=in_channels + 3,
            hidden_size=out_channels,
            out_size=out_channels,
            nb_of_layers=2,
            layer_norm=False,
        )

    def forward(
        self,
        graph: Batch,
    ) -> Tensor:
        # Start propagating messages.
        x = graph.x
        edge_index = graph.edge_index
        pos = graph.pos
        return self.propagate(edge_index, h=x, pos=pos)

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
        node_input_size: int = 8,
        hidden_layers: int = 2,
        hidden_size: int = 64,
        output_size: int = 2,
        **kwargs
    ):
        super().__init__()

        self.processer_list = nn.ModuleList(
            [PointNetLayer(node_input_size, hidden_size), ReLU()]
        )

        for _ in range(hidden_layers):
            self.processer_list.append(PointNetLayer(hidden_size, hidden_size))
            self.processer_list.append(ReLU())

        self.classifier = Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph=Batch) -> Tensor:

        for layer in self.processer_list:
            graph.x = layer(graph)

        # Global Pooling:
        x = global_max_pool(graph.x, graph.batch)  # [num_examples, hidden_channels]

        x = self.classifier(x)  # [num_examples, output_channels]

        # Classifier:
        return self.softmax(x)


# The PointNet++ classification model and layer
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, number_of_connections=16):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.number_of_connections = number_of_connections

    def _radius(self, idx, pos, batch):
        return radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.number_of_connections,
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = self._radius(idx, pos, batch)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, pos, batch):
        x = self.net(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ClassificationPointNetP2(torch.nn.Module):
    def __init__(
        self,
        node_input_size: int = 8,
        dim_model: list = [
            [64, 128, 128],
            [128, 128, 256],
            [256, 512, 1024],
            [1024, 512],
        ],
        output_size: int = 2,
        number_of_connections: int = 16,
        **kwargs
    ):
        super().__init__()

        self.sa_modules = nn.ModuleList()
        # Initialize the first SAModule
        self.sa_modules.append(
            SAModule(
                0.5,
                0.2,
                build_mlp(
                    3 + node_input_size,
                    dim_model[0][0],
                    dim_model[0][-1],
                    len(dim_model[0]),
                ),
                number_of_connections,
            )
        )

        # Add the intermediate SAModules
        for i in range(1, len(dim_model) - 2):
            self.sa_modules.append(
                SAModule(
                    0.25,
                    0.4,
                    build_mlp(
                        dim_model[i - 1][-1] + 3,
                        dim_model[i][0],
                        dim_model[i][-1],
                        len(dim_model[i]),
                    ),
                    number_of_connections,
                )
            )

        # Add the final GlobalSAModule
        self.sa_modules.append(
            GlobalSAModule(
                build_mlp(
                    dim_model[-3][-1] + 3,
                    dim_model[-2][0],
                    dim_model[-2][-1],
                    len(dim_model[-1]),
                )
            )
        )

        self.mlp = build_mlp(
            dim_model[-1][0],
            dim_model[-1][1],
            output_size,
            2,
            dropout=0.5,
            layer_norm=False,
        )

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
    def __init__(self, in_channels, hidden_size, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = build_mlp(
            in_size=in_channels,
            hidden_size=hidden_size,
            out_size=out_channels,
            nb_of_layers=2,
            layer_norm=False,
            plain_last=False,
        )

        self.attn_nn = build_mlp(
            in_size=out_channels,
            hidden_size=hidden_size,
            out_size=out_channels,
            nb_of_layers=2,
            layer_norm=False,
            plain_last=False,
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

    def __init__(self, node_input_size, output_size, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = build_mlp(
            node_input_size, output_size, nb_of_layers=1, layer_norm=False
        )

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
    def __init__(
        self,
        node_input_size: int = 3,
        dim_model: list = [32, 64, 128, 256, 512, 64],
        hidden_size: int = 64,
        output_size: int = 2,
        number_of_connections: int = 16,
        **kwargs
    ):
        super().__init__()
        self.number_of_connections = number_of_connections

        # dummy feature is created if there is none given
        node_input_size = max(node_input_size, 1)

        # first block
        self.mlp_input = build_mlp(
            node_input_size, dim_model[0], nb_of_layers=1, plain_last=False
        )

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0], hidden_size=hidden_size, out_channels=dim_model[0]
        )
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 2):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    k=self.number_of_connections,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

        # class score computation
        self.mlp_output = build_mlp(
            dim_model[-2], dim_model[-1], output_size, 2, layer_norm=False
        )

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
        edge_index = knn_graph(pos, k=self.number_of_connections, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.number_of_connections, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return self.softmax(out)
