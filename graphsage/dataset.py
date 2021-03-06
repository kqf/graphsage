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


class IndexMapper(dict):
    def append(self, x):
        if x in self:
            return
        self[x] = len(self)

    def map(self, seq):
        return np.array(list(map(self.get, seq)))


def choice(seq, size):
    return np.random.choice(size, min(len(seq), size)).tolist()


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
    # mask = edge_list["source"].isin(nodes)
    # sampled = edge_list[mask].groupby("source").agg(list)
    # sampled["target"] = sampled["target"].apply(random.choice).values
    positives = np.random.randint(0, max(edge_list.max()), nodes.shape)
    return positives


class GraphLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, sizes=[10, 10], **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)
        self.sizes = sizes

    def collate_fn(self, batch):
        batch, y = zip(*batch)
        return self.collate_batch(batch), torch.tensor(y).long()

    def collate_batch(self, batch):
        reverse_layers = []

        batch = np.array(batch)

        nodes = batch
        for size in self.sizes:
            nodes, edges = sample_edges(nodes, self.dataset.edge_list, size)
            reverse_layers.append([nodes, edges])

        layers = reverse_layers[::-1]

        all_nodes, batch, layers = self.to_local(batch, layers)

        x = self.dataset.features.iloc[all_nodes].values
        return to_batch(x, batch, layers)

    def to_local(self, batch, layers):
        # Calculate unique indices
        node2index = IndexMapper()
        for idx in batch.reshape(-1):
            node2index.append(idx)

        for nodes, _ in layers:
            for node in nodes:
                node2index.append(node)

        # New datastructure to local mapping
        local_layers = []
        for nodes, edges in layers:
            lnodes = node2index.map(nodes)
            edges["source"] = edges["source"].map(node2index)
            edges["target"] = edges["target"].map(node2index)
            local_layers.append([lnodes, edges])

        all_nodes = np.array(list(node2index.keys()))
        return all_nodes, node2index.map(batch), local_layers


class NegativeGraphLoader(GraphLoader):
    def collate_batch(self, batch):
        batch = np.array(batch)
        positives = sample_nodes(batch, self.dataset.edge_list)
        negatives = np.random.randint(0, len(self.dataset), batch.shape)
        full = np.concatenate([batch, negatives, positives])
        return super().collate_batch(full)


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
