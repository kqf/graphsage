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
            np.array([1, 2, 3, 4, 5, 6]),
            pd.DataFrame({
                "source": [2, 3, 4, 5],
                "target": [3, 4, 5, 6],
            })
        )
    )
    return anchors, layers[::-1]


def test_local_mapping(batch_layers):
    batch, layers = batch_layers
    all_nodes, lbatch, llayers = GraphLoader.to_local(batch, layers)

    # Global mapping
    np.testing.assert_equal(all_nodes, np.array([1, 2, 3, 4, 5, 6]))

    # The local mapping
    np.testing.assert_equal(lbatch, np.array([0, 1, 2, 3]))
