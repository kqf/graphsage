import pandas as pd


def main():
    features = pd.read_table("data/cora/cora.content", header=None)
    features.rename(columns={
        features.columns[0]: "node",
        features.columns[-1]: "target"},
        inplace=True,
    )

    iton = features["node"].to_list()
    ntoi = {n: i for i, n in enumerate(iton)}
    print(ntoi)

    df = pd.read_table("data/cora/cora.cites", names=["source", "target"])
    df["source"] = df["source"].map(ntoi)
    df["target"] = df["target"].map(ntoi)
    print(df)


if __name__ == '__main__':
    main()
