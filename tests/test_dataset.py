from model.dataset import load_cora, sampling_iterator


def test_dataset():
    dataset = load_cora()

    for example in dataset:
        pass

    batches = sampling_iterator(dataset, batch_size=64, drop_last=True)
    for batch, y in batches:
        assert len(batch["nodes"]) == 64
        assert "layers" in batch
        assert "features" in batch
