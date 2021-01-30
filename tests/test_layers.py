import torch
import pytest
from model.layers import SAGEConv


@pytest.fixture
def data(size=64):
    edge_list = torch.randint(0, 64, (64, 2))
    return torch.randn(64, 100), edge_list.T


def test_layer(data):
    model = SAGEConv()

    features, edge_list = data
    assert model(features, edge_list).shape == features.shape
