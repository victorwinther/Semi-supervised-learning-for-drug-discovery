import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, GINConv
from dataclasses import dataclass
from typing import Optional
from torch_geometric.nn import SchNet, DimeNetPlusPlus
from typing import Optional


class GCN(torch.nn.Module):
    def __init__(
        self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.2
    ):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Last GCN layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP head for regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Add residual connection (skip connection) after first layer
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Global pooling (combine mean and max pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_add_pool(x, batch)
        x = x_mean + x_max

        # Apply MLP head
        x = self.mlp(x)

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
        max_num_neighbors: int = 32,  # currently unused in this wrapper
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "swish",
        output_initializer: str = "zeros",  # accepted but NOT forwarded
        pretrained: bool = False,  # accepted but not used here
        target: int = 2,  # for multi-target setups if needed
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
            act=act_module,  # this is supported in your version
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
        self.add_mlp_head = add_mlp_head
        if add_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(out_channels, mlp_hidden),
                nn.SiLU(),
                nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity(),
                nn.Linear(mlp_hidden, 1),
            )
        else:
            self.head = None

    def forward(self, data):
        # Expects data.z, data.pos, data.batch
        out = self.schnet(data.z, data.pos, data.batch)
        if out.dim() == 1:
            out = out.unsqueeze(-1)  # [B] -> [B,1]
        if self.head is not None:
            out = self.head(out)
        return out
