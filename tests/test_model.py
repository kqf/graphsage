from model.model import build_model
from model.dataset import load_cora


def test_integrates():
    data = load_cora()
    model = build_model(max_epochs=2)
    model.fit(data)
