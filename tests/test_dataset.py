from model.dataset import load_cora


def test_dataset():
    dataset = load_cora()

    for example in dataset:
        pass
