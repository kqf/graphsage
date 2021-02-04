from graphsage.model import build_model


def test_integrates(data):
    model = build_model(max_epochs=2)
    model.fit(data)
