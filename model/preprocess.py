import torch
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Example, Dataset, Field, BucketIterator


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        with warnings.catch_warnings(record=True):
            fields = [(name, field) for (name, field) in self.fields
                      if name in X]
            proc = [X[col].apply(f.preprocess) for col, f in fields]
            examples = [Example.fromlist(f, fields) for f in zip(*proc)]
            return Dataset(examples, fields)


def build_preprocessor(min_freq=5):
    with warnings.catch_warnings(record=True):
        text_field = Field(
            tokenize=None,
            init_token=None,
            pad_token="<unk>",
            unk_token="<unk>",
            eos_token=None,
            batch_first=True,
            # pad_first=True,
        )
        fields = [
            ('observed', text_field),
            ('gold', text_field),
        ]
        return TextPreprocessor(fields, min_freq=min_freq)


class SequenceIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings(record=True):
            super().__init__(*args, **kwargs)

    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                target = torch.empty(0)
                if 'gold' in batch.fields:
                    target = batch.gold.view(-1)
                yield batch.observed, target
