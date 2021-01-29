from model.dataset import load_cora, sampling_iterator


def test_dataset():
    dataset = load_cora()

    for example in dataset:
        pass

    batches = sampling_iterator(dataset, batch_size=64, drop_last=True)
    for features, batch, layers in batches:
        assert len(batch) == 64
