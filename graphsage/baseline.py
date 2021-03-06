import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


class CooRecommender:
    def __init__(self, k=3):
        self.k = k
        self._vec = CountVectorizer(lowercase=False, tokenizer=lambda x: x)

    def fit(self, X):
        self._vec.fit(X)
        occurrence = self._vec.transform(X)
        cooccurrence = occurrence.T @ occurrence
        cooccurrence.setdiag(0)

        self.coo = cooccurrence

        vcb = self._vec.vocabulary_
        self._itos = [k for k, v in sorted(vcb.items(), key=lambda x: x[0])]

        return self

    def predict(self, X):
        occ = self._vec.transform(X).todense()

        scores = occ @ self.coo

        # take into account only unobserved
        scores[occ > 0] = 0
        print(scores)

        preds = (-scores).argpartition(self.k, -1)[..., :self.k]
        return np.take(self._itos, preds)


def build_model(k=3):
    return CooRecommender(k=k)


def main():
    df = pd.DataFrame({
        "batch_in": [[1, 2], [2, 3], [3, 4]]
    })
    print(df.head())
    model = build_model()
    model.fit(df["batch_in"])

    print(df["batch_in"].str[:1])
    print(model.predict(df["batch_in"].str[:1]))


if __name__ == '__main__':
    main()
