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
    
import torch
import torch.nn as nn
from torch_geometric.nn.models import DimeNetPlusPlus


class DimeNetPPModel(nn.Module):
    """
    Thin wrapper around torch_geometric.nn.models.DimeNetPlusPlus
    that matches your existing GCN/GIN API (forward(data) -> [B, out_channels]).

    Assumes QM9-like graphs with:
      - data.z   : [num_nodes] (atomic numbers, dtype long)
      - data.pos : [num_nodes, 3] (3D coordinates, float)
      - data.batch : [num_nodes] (graph indices)
    """

    def __init__(
        self,
        # Kept for compatibility with other models / config:
        num_node_features: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 64,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,   # currently unused in this wrapper
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "swish",
        output_initializer: str = "zeros",  # accepted but NOT forwarded
        pretrained: bool = False,           # accepted but not used here
        target: int = 2,                    # for multi-target setups if needed
    ):
        super().__init__()

        self.target = target

        act_module = self._build_activation(act)

        # IMPORTANT: do NOT pass output_init here, your PyG version doesn't support it
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act_module,     # this is supported in your version
        )

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "swish":
            # Swish ~ SiLU, what DimeNet++ uses by default
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, data):
        # DimeNet++ uses z, pos, batch (not x/edge_index)
        out = self.model(data.z, data.pos, data.batch)
        # out shape: [batch_size, out_channels] (1 by default)

        # If you ever set out_channels > 1 and want just one property:
        if out.dim() == 2 and out.size(-1) > 1 and self.target is not None:
            out = out[:, self.target].unsqueeze(-1)

        return out
