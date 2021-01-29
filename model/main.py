from model.dataset import load_cora, sampling_iterator
from model.layers import GraphSAGE


def main():
    dataset = load_cora()

    model = GraphSAGE()

    batches = sampling_iterator(dataset, batch_size=64)
    for batch in batches:
        model.forward(*batches)


if __name__ == '__main__':
    main()
