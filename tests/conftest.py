import pytest
from graphsage.dataset import load_cora


@pytest.fixture
def data():
    return load_cora()
