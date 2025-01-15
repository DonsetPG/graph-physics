
import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphphysics.models.layers import GraphNetBlock, Transformer, build_mlp


class ClassificationModel(nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size : int = 1,
        hidden_size: int=128,
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

        self.decode_module = decoder_1(hidden_size, hidden_size)

        # self.lin = nn.Linear(hidden_size, 1)

    def forward(self, graph: Data)-> float:
        edge_index = graph.edge_index

        x = self.nodes_encoder(graph.x)
        edge_attr = self.edges_encoder(graph.edge_attr)

        for block in self.processer_list:
            x, edge_attr = block(x, edge_index, edge_attr)

        x_decoded = self.decode_module(x)

        return x_decoded.squeeze()
        # x = latent_graph.x
        # x = torch.mean(x, dim=0)
        # x = self.lin(x)
        # return x.squeeze()



class decoder_1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=1, nb_of_layers=4, layer_norm=False):
        super().__init__()
        self.mlp = build_mlp(in_size, hidden_size, out_size, nb_of_layers, layer_norm)
    
    def forward(self, x: torch.Tensor) -> float:
        x = torch.mean(x, dim=0)
        x = self.mlp(x)
        return x


class decoder_2(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=1, nb_of_layers=4, layer_norm=False):
        super().__init__()
        self.mlp = build_mlp(in_size*hidden_size, hidden_size, out_size, nb_of_layers, layer_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x)
        x = self.mlp(x)
        return x






    