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
