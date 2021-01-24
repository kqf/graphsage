import torch
import numpy as np


class MeanAggregator(torch.nn.Module):

    def __init__(self, input_dim=None, output_dim=None):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computationn graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """  # noqa
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [
            np.array([mapping[v] for v in row], dtype=np.int64)
            for row in rows
        ]

        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(
                row) < num_samples) for row in mapped_rows]

        n = _len(nodes)
        out = torch.zeros(n, self.output_dim)

        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])
        return out

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)

class SAGEConv(torch.nn.Module):

    def __init__(
        self,
        input_dim=100,
        hidden_dims=100,
        output_dim=100,
        dropout=0.5,
        num_samples=25,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Dropout rate. Default: 0.5.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        """
        super(GraphSAGE, self).__init__()
        self.agg = MeanAggregator(input_dim, input_dim)
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
        super(GraphSAGE, self).__init__()

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