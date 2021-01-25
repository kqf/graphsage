import torch
import numpy as np


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

    def forward(self, features, nodes, mapping, rows):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of node features.
        """

        aggregated = self.agg(features, nodes, mapping, rows, self.num_samples)
        current = np.array([mapping[v] for v in nodes], dtype=np.int64)
        out = torch.cat((features[current, :], aggregated), dim=1)
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
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dims : list of ints
            Dimension of hidden layers. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Dropout rate. Default: 0.5.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        sizes = [input_dim] + hidden_dims + [output_dim]

        self.layers = torch.nn.ModuleList([
            SAGEConv(fin, fout) for fin, fout in zip(sizes[:-1], sizes[1:])
        ])

    def forward(self, features, node_layers, mappings, rows):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of node features.
        """

        out = features
        for k, layer in range(self.num_layers):
            nodes = node_layers[k + 1]
            mapping = mappings[k]

            initial = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
            cur_rows = rows[initial]

            out = layer(out, nodes, mapping, cur_rows, self.num_samples)
        return out
