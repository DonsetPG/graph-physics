import json
import os
import random

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset

from graphphysics.dataset.preprocessing import Random3DRotate
from graphphysics.utils.torch_graph import get_masked_indexes, meshdata_to_graph


class GraphClassificationDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        meta_path: str,
        switch_to_val: bool = False,
        number_of_sample: int = 1024,
        number_of_connections: int = 6,
    ):

        if switch_to_val:
            root_folder = root_folder.replace("training", "validation")

        self.root_folder = root_folder
        self.classes = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.file_paths = []
        self.labels = []
        for cls in self.classes:
            cls_path = os.path.join(root_folder, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".xdmf"):
                    self.file_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

        with open(meta_path, "r") as fp:
            self.meta = json.loads(fp.read())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.number_of_sample = number_of_sample
        self.number_of_connections = number_of_connections


    def __len__(self):
        return len(self.file_paths)
    
    def get_processing(self):
        return T.Compose(
            [
                T.NormalizeScale(),
                T.RandomFlip(axis=random.randint(0, 2)),
                T.RandomJitter(0.002),
                Random3DRotate(),
                T.SamplePoints(
                    num=self.number_of_sample,
                    remove_faces=False,
                    include_normals=True,
                ),
                T.KNNGraph(k=self.number_of_connections),
            ]
        )
        
    def set_label(self, label):
        if label == 1:
            return torch.tensor([0, 1], dtype=torch.float32)

        else:
            return torch.tensor([1, 0], dtype=torch.float32)


    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        label = self.labels[idx]

        mesh_io = meshio.read(file_path)
        points, cells = mesh_io.points, mesh_io.cells

        t = meshio.Mesh(points, cells)
        if "triangle" in t.cells_dict:
            faces = t.cells_dict["triangle"]
        else:
            face = torch.tensor(t.cells_dict["tetra"].T, dtype=torch.long)
            faces = torch.cat(
                [
                    face[0:3],
                    face[1:4],
                    torch.stack([face[2], face[3], face[0]], dim=0),
                    torch.stack([face[3], face[0], face[1]], dim=0),
                ],
                dim=1,
            )
            faces = faces.T.numpy()

        graph = meshdata_to_graph(
            points=np.array(points).astype(np.float32),
            cells=np.array(faces).astype(np.int32),
            point_data=None,
        )
        graph = graph.to(self.device)

        classification_processing = self.get_processing()
        graph = classification_processing(graph)

        graph.x = graph.normal

        graph.y = self.set_label(label)

        return graph
