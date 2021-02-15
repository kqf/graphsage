import torch
from torch_scatter import scatter


class SAGEConv(torch.nn.Module):

    def __init__(
        self,
        input_dim=100,
        hidden_dims=100,
        output_dim=100,
        dropout=0.5,
    ):
        super().__init__()
        self.fcs = torch.nn.Linear(2 * input_dim, output_dim)
        self.bns = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, features, batch, edge_index):
        sources, targets = edge_index

        size = features.shape[0]

        aggregated = scatter(features[targets], sources, dim=0, dim_size=size)

        out = torch.cat((features[batch], aggregated[batch]), dim=1)
        out = self.fcs(out)
        out = self.relu(out)
        out = self.bns(out)
        out = self.dropout(out)
        out = out.div(out.norm(dim=1, keepdim=True) + 1e-6)
        return out


class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        input_dim=100,
        hidden_dims=[100],
        output_dim=100,
        dropout=0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        sizes = [input_dim] + hidden_dims + [output_dim]

        self.layers = torch.nn.ModuleList([
            SAGEConv(fin, fout, dropout=dropout)
            for fin, fout in zip(sizes[:-1], sizes[1:])
        ])
        self._fc = torch.nn.Identity()

    def forward(self, features, nodes, layers):
        out = features
        for layer, (_nodes, edge_index) in zip(self.layers, layers):
            out = layer(out, _nodes.T, edge_index.T)
        return self._fc(out[nodes])


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logits = torch.nn.Linear(self.output_dim, n_classes)

    def forward(self, features, batch, edge_index):
        x = super().forward(features, batch, edge_index)
        return self._logits(x)
