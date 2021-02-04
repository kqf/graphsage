from graphsage.dataset import load_cora, sampling_iterator


def test_dataset():
    dataset = load_cora()

    for example in dataset:
        pass

    batches = sampling_iterator(dataset, batch_size=64, drop_last=True)
    for batch, y in batches:
        assert len(batch["nodes"]) == 64
        assert "layers" in batch
        assert "features" in batch


def test_negative_sampling():
    dataset = load_cora()

    batches = sampling_iterator(
        dataset,
        negatives=True,
        batch_size=64,
        drop_last=True
    )

    for batch, y in batches:
        assert len(batch["nodes"]) == 64 * 2
        assert "layers" in batch
        assert "features" in batch
