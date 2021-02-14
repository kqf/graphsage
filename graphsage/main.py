from graphsage.dataset import load_cora
from graphsage.model import build_model


def main():
    data = load_cora()
    model = build_model(
        max_epochs=20
    )
    model.fit(data)


if __name__ == '__main__':
    main()
