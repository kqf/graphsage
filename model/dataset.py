import torch
import pathlib

import numpy as np
import pandas as pd

from functools import partial


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, edge_list, labels):
        self.features = features
        self.edge_list = edge_list
        self.labels = labels

    def __getitem__(self, idx):
        return idx, self.labels[idx]

    def __len__(self):
        return self.features.shape[0]


def choice(seq, size):
    return np.random.choice(size, min(len(seq), size)).tolist()


def sample_edges(nodes, edge_list, size):
    mask = np.isin(edge_list[:, 0], nodes)
    candidates = pd.DataFrame(edge_list[mask], columns=["source", "target"])
    sampled = candidates.groupby("source").agg(partial(choice, size=size))
    edges_sublist = sampled.explode("target").reset_index()

    un = edges_sublist["target"].unique()
    new_nodes = np.unique(np.concatenate([nodes, un]))
    return new_nodes, edges_sublist


class GraphLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, sizes=[10], **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)
        self.sizes = sizes

    def collate_fn(self, batch):
        reverse_layers = []
        batch = np.array(batch)

        nodes = batch[:, 0]
        for size in self.sizes:
            nodes, edges = sample_edges(nodes, self.dataset.edge_list, size)
            reverse_layers.append([nodes, edges])

        layers = reverse_layers[::-1]
        return self.dataset.features, batch, layers


def sampling_iterator(dataset, **kwargs):
    return GraphLoader(dataset, **kwargs)


def load_cora(path="data/cora"):
    path = pathlib.Path(path)

    features = pd.read_table(path / "cora.content", header=None)
    features.rename(columns={
        features.columns[0]: "node",
        features.columns[-1]: "target"},
        inplace=True,
    )

    raw = features.drop(columns=["node", "target"])
    x = raw / raw.mean(axis=1).values[:, None]

    y = features["target"].astype("category").cat.codes.values

    iton = features["node"].to_list()
    ntoi = {n: i for i, n in enumerate(iton)}

    df = pd.read_table(path / "cora.cites", names=["source", "target"])
    df["source"] = df["source"].map(ntoi)
    df["target"] = df["target"].map(ntoi)

    edge_list = df.values
    return GraphDataset(x, edge_list, y)
