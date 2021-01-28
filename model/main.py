import pandas as pd


def main():
    features = pd.read_table("data/cora/cora.content", header=None)
    features.rename(columns={
        features.columns[0]: "node",
        features.columns[-1]: "target"},
        inplace=True,
    )

    x = features.drop(columns=["node", "target"]).values
    y = features["target"].astype("category").cat.codes.values

    iton = features["node"].to_list()
    ntoi = {n: i for i, n in enumerate(iton)}

    df = pd.read_table("data/cora/cora.cites", names=["source", "target"])
    df["source"] = df["source"].map(ntoi)
    df["target"] = df["target"].map(ntoi)

    edge_list = df.values
    return x, edge_list, y


if __name__ == '__main__':
    main()
