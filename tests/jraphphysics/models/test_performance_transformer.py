from torch.utils import data
import unittest
import jax.numpy as jnp
import jax.random as random
from jraphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import MOCK_XDMF_FOLDER, MOCK_H5_META10_SAVE_PATH, MOCK_XDMF_TARGETS
from jax.experimental import sparse as jsparse

import flax.nnx as nnx
from jraphphysics.utils.jax_graph import batched_meshdata_to_graph

from jraphphysics.models.layers import (
    Transformer,
)

ITEMS_TO_FETCH = 20
BATCH_SIZE = 4
NUM_WORKERS = 2
PREFETCH_FACTOR = 1
PERSISTENT_WORKERS = True


def process_items_xdmfdataloader_jraph_no_adjacency():
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        targets=MOCK_XDMF_TARGETS,
    )
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    indx = 0
    key = random.key(0)
    rngs = nnx.Rngs(key)

    transformer = Transformer(input_dim=3, output_dim=3, num_heads=1, rngs=rngs)
    for batch_g in dataloader:
        batched_jraph = batched_meshdata_to_graph(
            points=batch_g["points"].numpy(),
            cells=batch_g["cells"].numpy(),
            point_data=batch_g["point_data"],
        )
        transformer(batched_jraph.nodes["features"])
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break


def build_adj(graph):
    num_nodes = graph.n_node.sum()

    senders = graph.senders
    receivers = graph.receivers
    indices = jnp.stack([senders, receivers], axis=-1)

    return jsparse.BCOO(
        (
            jnp.ones_like(senders, dtype=jnp.float32),  # data
            indices,  # indices
        ),
        shape=(num_nodes, num_nodes),
    )


def process_items_xdmfdataloader_jraph_adjacency():
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        targets=MOCK_XDMF_TARGETS,
    )
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    indx = 0
    key = random.key(0)
    rngs = nnx.Rngs(key)
    transformer = Transformer(input_dim=3, output_dim=3, num_heads=1, rngs=rngs)
    for batch_g in dataloader:
        batched_jraph = batched_meshdata_to_graph(
            points=batch_g["points"].numpy(),
            cells=batch_g["cells"].numpy(),
            point_data=batch_g["point_data"],
        )
        adj = build_adj(batched_jraph)
        transformer(batched_jraph.nodes["features"], adj)
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break


def test_process_items_xdmfdataloader_jraph_no_adjacency(benchmark):
    benchmark(process_items_xdmfdataloader_jraph_no_adjacency)


def test_process_items_xdmfdataloader_jraph_adjacency(benchmark):
    benchmark(process_items_xdmfdataloader_jraph_adjacency)
