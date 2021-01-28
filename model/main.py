from model.dataset import load_cora


def main():
    x, edge_list, y = load_cora()
    print(x, edge_list, y)


if __name__ == '__main__':
    main()
