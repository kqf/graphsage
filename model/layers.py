import torch
from torch_scatter import scatter


class SAGEConv(torch.nn.Module):

    def __init__(
        self,
        input_dim=100,
        hidden_dims=100,
        output_dim=100,
        dropout=0.5,
        num_samples=25,
    ):
        super().__init__()
        self.fcs = torch.nn.Linear(2 * input_dim, output_dim)
        self.bns = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, features, edge_index):
        sources, targets = edge_index

        aggregated = scatter(features[targets], sources, dim=0)
        out = torch.cat((features, aggregated), dim=1)
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
        num_samples=25,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        sizes = [input_dim] + hidden_dims + [output_dim]

        self.layers = torch.nn.ModuleList([
            SAGEConv(fin, fout) for fin, fout in zip(sizes[:-1], sizes[1:])
        ])

    def forward(self, features, edges):
        out = features
        for layer, edge_index in zip(self.layers, edges):
            out = layer(features, edge_index)
        return out
