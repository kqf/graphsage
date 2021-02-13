import pytest
from graphsage.model import build_model


@pytest.mark.parametrize("build", [
    build_model
])
def test_integrates(build, data):
    model = build(max_epochs=2)
    model.fit(data)
