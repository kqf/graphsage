import pytest
import torch

from graphsage.losses import TripletLoss


@pytest.fixture
def vectors(batch_size=64, size=100):
    return torch.rand((batch_size, size))


def test_loss(vectors):
    loss = TripletLoss(margin=0.5)
    assert loss(vectors, vectors, vectors) == 0.5
