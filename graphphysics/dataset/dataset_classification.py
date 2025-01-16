import json
import os
import random

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset

from graphphysics.utils.torch_graph import get_masked_indexes, meshdata_to_graph


class GraphClassificationDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        meta_path: str,
        preprocessing=None,
        masking_ratio=None,
        switch_to_val: bool = False,
    ):
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
        self.preprocessing = preprocessing
        self.masking_ratio = masking_ratio

    def __len__(self):
        return len(self.file_paths)

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
 
        classification_processing = T.Compose(
            [T.RandomFlip(axis=random.randint(0, 2)), T.RandomJitter(translate=0.002)]
        )
 
        graph = classification_processing(graph)
 
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)

        if self.masking_ratio is not None:
            selected_indices = get_masked_indexes(graph, self.masking_ratio)
        else :
            selected_indices = None

        graph.y = label 

        if selected_indices is not None: 
            return graph, selected_indices
        else:
            return graph