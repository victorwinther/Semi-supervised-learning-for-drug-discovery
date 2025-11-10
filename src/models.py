import torch
import torch.nn.functional as F

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, GINConv

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x

class GINNet(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 128,
        num_layers: int = 5,
        dropout: float = 0.2,
        out_channels: int = 1,  # 1 target by default (scalar regression)
    ):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # Initial linear projection of input node features
        self.input_lin = nn.Linear(num_node_features, hidden_channels)

        # Build GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # MLP readout head after pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Project input features
        x = self.input_lin(x)

        # 2. Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Global pooling (graph-level embedding)
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Final MLP head
        x = self.mlp(x)  # [batch_size, out_channels]

        # For scalar regression, this will be shape [batch_size, 1]
        return x

from torch_geometric.nn import SchNet
class SchNetRegressor(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        readout: str = "add",
        dropout: float = 0.0,
        add_mlp_head: bool = False,
        mlp_hidden: int = 256,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.add_mlp_head = add_mlp_head
        if add_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(out_channels, mlp_hidden),
                nn.SiLU(),
                nn.Dropout(dropout) if self.dropout is not None else nn.Identity(),
                nn.Linear(mlp_hidden, 1),
            )
        else:
            self.head = None

    def forward(self, data):
        # Expects data.z, data.pos, data.batch
        out = self.schnet(data.z, data.pos, data.batch)
        if out.dim() == 1:
            out = out.unsqueeze(-1)  # [B] -> [B,1]
        if self.dropout is not None:
            out = self.dropout(out)
        if self.head is not None:
            out = self.head(out)
        return out
