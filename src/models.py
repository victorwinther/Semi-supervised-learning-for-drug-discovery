import torch
import torch.nn.functional as F

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, GINConv

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch_geometric.nn.models import DimeNetPlusPlus


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
    
    
class DimeNetPPModel(nn.Module):
    def __init__(self, config: Optional[DimeNetPPConfig] = None):
        super().__init__()
        if config is None:
            config = DimeNetPPConfig()
        self.config = config
        if config.pretrained:
            if config.target is None:
                raise ValueError("target must be specified for pre‑trained models")
            model, *_ = DimeNetPlusPlus.from_qm9_pretrained(
                root="./pretrained_models", target=config.target
            )
            self.dimenet = model
        else:
            self.dimenet = DimeNetPlusPlus(
                hidden_channels=config.hidden_channels,
                out_channels=config.out_channels,
                num_blocks=config.num_blocks,
                int_emb_size=config.int_emb_size,
                basis_emb_size=config.basis_emb_size,
                out_emb_channels=config.out_emb_channels,
                num_spherical=config.num_spherical,
                num_radial=config.num_radial,
                cutoff=config.cutoff,
                max_num_neighbors=config.max_num_neighbors,
                num_before_skip=config.num_before_skip,
                num_after_skip=config.num_after_skip,
                num_output_layers=config.num_output_layers,
                act=config.act,
                output_initializer=config.output_initializer,
            )
        self.linear = nn.Linear(config.out_channels, 1) if config.out_channels != 1 else None

    def forward(self, data):
        # data must contain `z` (atomic numbers) and `pos` (coordinates)
        out = self.dimenet(data.z, data.pos, data.batch)
        return self.linear(out) if self.linear is not None else out

    @classmethod
    def from_pretrained(cls, target: int, **kwargs):
        config = DimeNetPPConfig(pretrained=True, target=target, **kwargs)
        return cls(config)