import pytest
import torch

from graphsage.losses import TripletLoss


@pytest.fixture
def vectors(batch_size=64, size=100):
    return torch.rand((batch_size, size))


@pytest.mark.parametrize("reduction, result", [
    ("mean", 0.5),
    ("sum", 32.0),
])
def test_loss(vectors, result, reduction):
    loss = TripletLoss(margin=0.5, reduction=reduction)
    assert loss(vectors, vectors, vectors) == result
