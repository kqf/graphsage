import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


class CooRecommender:
    def __init__(self, k=5):
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

        preds = (-scores).argpartition(self.k, -1)[..., :self.k]
        return np.take(self._itos, preds)


# Make sure our API is consistent with other modules
def build_model(k=5):
    return CooRecommender(k=k)
