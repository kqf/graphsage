from model.model import build_model
from model.dataset import load_cora


def main():
    data = load_cora()
    model = build_model(max_epochs=20)
    model.fit(data)


if __name__ == '__main__':
    main()
