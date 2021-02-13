import torch
import skorch

from skorch.dataset import get_len
from skorch.dataset import unpack_data
from skorch.dataset import uses_placeholder_y

from graphsage.layers import GraphSAGE
from graphsage.dataset import NegativeGraphLoader
from graphsage.losses import TripletLoss


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_uniform_(w)


class GraphNet(skorch.NeuralNet):
    def run_single_epoch(
            self, dataset, training, prefix, step_fn, **fit_params):
        is_placeholder_y = uses_placeholder_y(dataset)

        batch_count = 0
        for data in self.get_iterator(dataset, training=training):
            Xi, yi = unpack_data(data)
            yi_res = yi if not is_placeholder_y else None
            self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            self.history.record_batch(prefix + "_batch_size",
                                      get_len(Xi["nodes"]))
            self.notify("on_batch_end", X=Xi, y=yi_res,
                        training=training, **step)
            batch_count += 1
        self.history.record(prefix + "_batch_count", batch_count)


class UnsupervisedGraphNet(GraphNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        anchor, positive, negative = y_pred.split(y_pred.shape[0] // 3, dim=0)
        return self.criterion_(anchor, positive, negative)


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    model = UnsupervisedGraphNet(
        GraphSAGE,
        module__input_dim=1433,
        criterion=TripletLoss,
        batch_size=256,
        max_epochs=max_epochs,
        # optimizer__momentum=0.9,
        iterator_train=NegativeGraphLoader,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid=NegativeGraphLoader,
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
