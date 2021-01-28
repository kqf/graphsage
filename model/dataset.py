import pandas as pd
import pathlib


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
    return x, edge_list, y
