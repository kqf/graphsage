from graphsage.dataset import sampling_iterator


def test_iterator(data):
    batches = sampling_iterator(data, batch_size=64, drop_last=True)
    for batch, y in batches:
        assert len(batch["nodes"]) == 64
        assert "layers" in batch
        assert "features" in batch


def test_negative_sampling(data):

    batches = sampling_iterator(
        data,
        negatives=True,
        batch_size=64,
        drop_last=True
    )

    for batch, y in batches:
        assert len(batch["nodes"]) == 64 * 2
        assert "layers" in batch
        assert "features" in batch
