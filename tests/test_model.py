import pytest
from model.model import build_model
from model.dataset import load_cora


@pytest.mark.xfail("Length consistency")
def test_integrates():
    data = load_cora()
    model = build_model(max_epochs=2)
    model.fit(data)
