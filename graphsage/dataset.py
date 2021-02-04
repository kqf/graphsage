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


def amap(seq, mapping):
    return np.array(list(map(mapping.get, seq)))


def to_batch(features, all_nodes, layers):
    tensors = []
    for nodes, edges in layers:
        tnodes = torch.tensor(nodes, dtype=torch.int64)
        tedges = torch.tensor(edges.values, dtype=torch.int64)
        tensors.append([tnodes, tedges])

    batch = {}
    batch["features"] = torch.tensor(features, dtype=torch.float32)
    batch["nodes"] = torch.tensor(all_nodes, dtype=torch.int64)
    batch["layers"] = tensors
    return batch


def sample_edges(nodes, edge_list, size):
    mask = edge_list["source"].isin(nodes)
    sampled = edge_list[mask].groupby("source").agg(partial(choice, size=size))
    edges_sublist = sampled.explode("target").reset_index()

    un = edges_sublist["target"].unique()
    new_nodes = np.unique(np.concatenate([nodes, un]))
    return new_nodes, edges_sublist


def sample_nodes(nodes, edge_list):
    mask = edge_list["source"].isin(nodes)
    sampled = edge_list[mask].groupby("source").agg(partial(choice, size=1))
    return sampled.explode("target").values


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

        y = batch[:, 1]
        all_nodes, batch, layers = self.to_local(batch[:, 0], layers)

        x = self.dataset.features.iloc[all_nodes].values
        return to_batch(x, batch, layers), y

    def to_local(self, batch, layers):
        # Calculate unique indices
        uniq = set()
        uniq.update(batch.reshape(-1))
        for _, edges in layers:
            uniq.update(edges.values.reshape(-1))

        # The local mapping
        node2index = {v: i for i, v in enumerate(uniq)}

        # New datastructure to local mapping
        local_layers = []
        for nodes, edges in layers:
            lnodes = amap(nodes, node2index)
            edges["source"] = edges["source"].map(node2index)
            edges["target"] = edges["target"].map(node2index)
            local_layers.append([lnodes, edges])

        return np.unique(list(uniq)), amap(batch, node2index), local_layers


class NegativeGraphLoader(GraphLoader):
    def collate_fn(self, batch):
        batch = np.array(batch)
        negatives = np.random.randint(0, len(self.dataset), batch.shape)
        new_batch = np.concatenate([batch, negatives])
        return super().collate_fn(new_batch)


def sampling_iterator(dataset, negatives=False, **kwargs):
    if negatives:
        return NegativeGraphLoader(dataset, **kwargs)
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

    edge_list = df
    return GraphDataset(x, edge_list, y)
