import torch
import pandas as pd
import pathlib


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, edge_list, labels):
        self.features = features
        self.edge_list = edge_list
        self.labels = labels

    def __getitem__(self, idx):
        return self.features, self.edge_list, idx, self.labels[idx]

    def __len__(self):
        return self.features.shape[0]


def sample(batch):
    return batch


def sampling_iterator(dataset, **kwargs):
    return torch.utils.data.DataLoader(dataset, collate_fn=sample, **kwargs)


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
