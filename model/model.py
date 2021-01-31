import torch
import skorch
from skorch.toy import MLPModule


from model.dataset import GraphLoader


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    model = skorch.NeuralNetClassifier(
        MLPModule,
        criterion=torch.nn.CrossEntropyLoss,
        batch_size=32,
        max_epochs=max_epochs,
        # optimizer__momentum=0.9,
        iterator_train=GraphLoader,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid=GraphLoader,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=4,
        train_split=train_split,
        callbacks=[
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Initializer("*", init),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
