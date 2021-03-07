import pytest
import numpy as np
import pandas as pd

from graphsage.dataset import GraphLoader, NegativeGraphLoader


def test_iterator(data):
    batches = GraphLoader(data, batch_size=64, drop_last=True)
    for batch, y in batches:
        assert len(batch["nodes"]) == 64
        assert "layers" in batch
        assert "features" in batch


def test_negative_sampling(data):

    batches = NegativeGraphLoader(
        data,
        batch_size=64,
        drop_last=True
    )

    for batch, y in batches:
        assert len(batch["nodes"]) == 64 * 3
        assert "layers" in batch
        assert "features" in batch


@pytest.fixture
def batch_layers():
    anchors = np.array([1, 2, 3, 4])
    layers = [
        (
            anchors,
            pd.DataFrame({
                "source": [1, 2, 3, 4],
                "target": [2, 3, 4, 5],
            })
        )
    ]
    layers.append(
        (
            np.array([1, 2, 3, 4, 5]),
            pd.DataFrame({
                "source": [2, 3, 4, 5],
                "target": [3, 4, 5, 6],
            })
        )
    )
    return anchors, layers


def test_local_mapping(batch_layers):
    batch, layers = batch_layers
