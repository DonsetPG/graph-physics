import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphphysics.models.layers import GraphNetBlock, Transformer, build_mlp


class ClassificationModel(nn.Module):
   def __init__(
       self,
       message_passing_num,
       node_input_size,
       edge_input_size,
       output_size,
       hidden_size=128,
   ):
       super(ClassificationModel, self).__init__()
       self.hidden_size = hidden_size
       self.encoder = Encoder(
           edge_input_size=edge_input_size,
           node_input_size=node_input_size,
           hidden_size=hidden_size,
       )

       self.processer_list = [
           GraphNetBlock(hidden_size=hidden_size, use_gated_mlp=False)
           for _ in range(message_passing_num)
       ]
       self.processer_list = nn.ModuleList(self.processer_list)

       self.lin = nn.Linear(hidden_size, 1)
       #decode1 : self.decode_module = decoder_1
       #self.decode_module = build_mlp ou créer des fonctions comme build_cnn ou build_decisions_tree

   def forward(self, graph):
       latent_graph = self.encoder(graph)

       for model in self.processer_list:
           latent_graph = model(latent_graph)

       x = latent_graph.x

       x = torch.mean(x, dim=0)
       x = self.lin(x)

       return x.squeeze()


def decoder_1(
        graph : Data,
        in_size: int,
        hidden_size: int,
        out_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = False,
)  -> torch.Tensor:
    x = graph.x
    x = torch.mean(x, dim=0)
    mlp  = build_mlp(in_size, hidden_size, out_size, nb_of_layers, layer_norm)
    x = mlp(x)
    return x 

def decoder_2(
        graph : Data,
        in_size: int,
        hidden_size: int,
        out_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = False,
)  -> torch.Tensor:
    x = graph.x
    x = nn.flatten(x)
    mlp  = build_mlp(in_size**2, hidden_size, out_size, nb_of_layers, layer_norm)
    x = mlp(x)
    return x
    ### Faire une technique qui ne fait pas de mean avant. (peux aussi faire le flatten juste avant le self.lin)

    



    