from model.dataset import load_cora


def test_iterates():
    dataset = load_cora()

    for example in dataset:
        pass
