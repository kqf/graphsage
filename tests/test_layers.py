import torch
import pytest
from graphsage.layers import SAGEConv


@pytest.fixture
def data(size=64):
    edge_list = torch.randint(0, size, (2, size))

    batch = torch.unique(edge_list[0])
    return torch.randn(1000, 100), batch, edge_list


def test_layer(data):
    model = SAGEConv()

    features, batch, edge_list = data

    expected_shape = (len(batch), features.shape[-1])
    assert model(features, batch, edge_list).shape == expected_shape
