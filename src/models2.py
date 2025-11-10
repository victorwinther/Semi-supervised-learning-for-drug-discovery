"""
Suggested advanced model for the `gnn_intro` project.

This module defines a PyTorch module built on top of the
`torch_geometric.nn.models.DimeNetPlusPlus` architecture.  DimeNet++ is
specifically designed for molecular property prediction and has been
demonstrated to outperform simple Graph Convolutional Networks (GCNs) on
quantum chemistry benchmarks such as QM9.  Compared to the two‐layer GCN
currently used in the project, DimeNet++ leverages directional
message passing and incorporates bond angles and distances, which makes
it better suited for predicting orbital energies (e.g. the HOMO energy
target used when `target=2` in the dataset configuration).

References
----------
* Buterez et al., "Modelling local and general quantum mechanical properties
  with attention‑based pooling" (2023) show that replacing simple readout
  functions with more expressive pooling, combined with architectures like
  SchNet or DimeNet++, yields significantly lower mean absolute errors on
  HOMO energy prediction for the QM9 dataset.  In their experiments the
  DimeNet++ baseline achieves a test MAE of about 23.7 meV on QM9 and
  21.6 meV when using attention‑based pooling, whereas a vanilla two‑layer
  GCN yields much higher error【513215758202187†L688-L733】.
* The same study reports that mean pooling reduces MAE for HOMO prediction
  by 9.75% for SchNet and 2.26% for DimeNet++, and that attention‑based
  pooling (ABP) further reduces the error—21.67% for SchNet and 9.89% for
  DimeNet++ on QM9【513215758202187†L721-L733】.

The class defined here wraps `DimeNetPlusPlus` with a final linear
layer to predict a single scalar property.  It exposes a `from_pretrained`
class method that can optionally download a model pre‑trained on QM9 from
PyTorch Geometric.  If you choose to use pre‑trained weights, pass the
appropriate `target` index corresponding to your property (e.g. `target=2`
for HOMO energy) and set `pretrained=True` when constructing the model.

Example
-------
.. code-block:: python

    from torch_geometric.datasets import QM9
    from suggested_model import DimeNetPPModel

    # Load your data and select a target property
    dataset = QM9(root="data")
    data = dataset[0]

    # Instantiate the model
    model = DimeNetPPModel(hidden_channels=128, num_blocks=6, pretrained=False)
    out = model(data)  # returns a tensor with shape [1]

    # If you want to load pre‑trained weights
    model_pretrained = DimeNetPPModel.from_pretrained(target=2)
    out_pretrained = model_pretrained(data)

This module requires PyTorch Geometric to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

try:
    from torch_geometric.nn.models import DimeNetPlusPlus
    from torch_geometric.datasets import QM9
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "DimeNetPPModel requires torch_geometric to be installed."
    ) from e


@dataclass
class DimeNetPPConfig:
    """Configuration for the DimeNetPPModel.

    Parameters
    ----------
    hidden_channels : int
        Hidden embedding size used inside the DimeNet++ interaction blocks.
    out_channels : int
        Number of output channels (should be 1 for scalar regression).
    num_blocks : int
        Number of interaction blocks in DimeNet++.
    int_emb_size : int
        Size of the intermediate embedding in each block.
    basis_emb_size : int
        Size of the basis embedding.
    out_emb_channels : int
        Size of the embedding in the output blocks.
    num_spherical : int
        Number of spherical harmonics used for the angular basis functions.
    num_radial : int
        Number of radial basis functions.
    cutoff : float
        Cutoff distance for interatomic interactions.
    max_num_neighbors : int
        Maximum number of neighbors within the cutoff distance.
    num_before_skip : int
        Number of residual layers before the skip connection.
    num_after_skip : int
        Number of residual layers after the skip connection.
    num_output_layers : int
        Number of linear layers for the output blocks.
    act : str
        Activation function used inside the network.
    output_initializer : str
        Initialization method for the output layer.
    pretrained : bool
        If true, load weights pre‑trained on the QM9 dataset.
    target : Optional[int]
        Index of the property to predict when loading a pre‑trained model.
    """

    hidden_channels: int = 128
    out_channels: int = 1
    num_blocks: int = 4
    int_emb_size: int = 64
    basis_emb_size: int = 8
    out_emb_channels: int = 64
    num_spherical: int = 7
    num_radial: int = 6
    cutoff: float = 5.0
    max_num_neighbors: int = 32
    num_before_skip: int = 1
    num_after_skip: int = 2
    num_output_layers: int = 3
    act: str = "swish"
    output_initializer: str = "zeros"
    pretrained: bool = False
    target: Optional[int] = None


class DimeNetPPModel(nn.Module):
    """DimeNet++ model wrapper for molecular property prediction.

    This module wraps the `torch_geometric.nn.models.DimeNetPlusPlus` model and
    adds a final linear layer if necessary.  It accepts a `Data` object from
    PyTorch Geometric and returns a one‑dimensional tensor corresponding to
    the predicted property for each graph in the batch.

    To integrate seamlessly with configuration frameworks like Hydra, the
    constructor exposes individual keyword arguments corresponding to the
    hyper‑parameters of DimeNet++.  These default to the same values as
    `DimeNetPPConfig`.  A `num_node_features` argument is also accepted for
    compatibility with other models (e.g. GCN and GIN) but is ignored since
    DimeNet++ uses atomic numbers and positions directly.

    Parameters
    ----------
    num_node_features : int, optional
        Number of input node features.  Included for API consistency but
        unused by this model.
    hidden_channels : int
        Hidden embedding size used inside the DimeNet++ interaction blocks.
    out_channels : int
        Number of output channels (should be 1 for scalar regression).
    num_blocks : int
        Number of interaction blocks in DimeNet++.
    int_emb_size : int
        Size of the intermediate embedding in each block.
    basis_emb_size : int
        Size of the basis embedding.
    out_emb_channels : int
        Size of the embedding in the output blocks.
    num_spherical : int
        Number of spherical harmonics used for the angular basis functions.
    num_radial : int
        Number of radial basis functions.
    cutoff : float
        Cutoff distance for interatomic interactions.
    max_num_neighbors : int
        Maximum number of neighbors within the cutoff distance.
    num_before_skip : int
        Number of residual layers before the skip connection.
    num_after_skip : int
        Number of residual layers after the skip connection.
    num_output_layers : int
        Number of linear layers for the output blocks.
    act : str
        Activation function used inside the network.
    output_initializer : str
        Initialization method for the output layer.
    pretrained : bool
        If true, load weights pre‑trained on the QM9 dataset.
    target : Optional[int]
        Index of the property to predict when loading a pre‑trained model.
    """

    def __init__(
        self,
        num_node_features: Optional[int] = None,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 64,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "swish",
        output_initializer: str = "zeros",
        pretrained: bool = False,
        target: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Pack parameters into a config for convenience
        self.config = DimeNetPPConfig(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
            pretrained=pretrained,
            target=target,
        )

        if pretrained:
            # Load a pre‑trained DimeNet++ model on QM9.  The from_qm9_pretrained
            # class method returns the model along with training/validation/test
            # splits, which are ignored here.
            if target is None:
                raise ValueError(
                    "`target` must be specified when using pretrained=True.")
            model, *_ = DimeNetPlusPlus.from_qm9_pretrained(
                root="./pretrained_models", target=target
            )
            self.dimenet = model
        else:
            # Construct a fresh DimeNet++ model
            self.dimenet = DimeNetPlusPlus(
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                int_emb_size=int_emb_size,
                basis_emb_size=basis_emb_size,
                out_emb_channels=out_emb_channels,
                num_spherical=num_spherical,
                num_radial=num_radial,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                num_output_layers=num_output_layers,
                act=act,
                output_initializer=output_initializer,
            )

        # If the underlying DimeNet++ already produces a scalar output, no
        # additional linear layer is required.  However, to remain flexible
        # (e.g. when out_channels > 1), we include an optional linear layer.
        self.linear = None
        if out_channels != 1:
            self.linear = nn.Linear(out_channels, 1)

    def forward(self, data: "torch_geometric.data.Data") -> torch.Tensor:
        """Compute the property prediction for a batch of molecules.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Batch of molecular graphs.  The Data object must contain the
            following attributes:

            * ``z`` – a tensor of atomic numbers for each atom in the batch.
            * ``pos`` – a tensor of atomic coordinates of shape [num_atoms, 3].
            * ``batch`` – a tensor assigning each atom to a particular graph.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size, 1] containing the predicted
            property for each graph in the batch.
        """
        # Ensure the necessary attributes are present
        if not hasattr(data, "z") or not hasattr(data, "pos"):
            raise ValueError(
                "Data object must contain `z` (atomic numbers) and `pos` (coordinates)."
            )

        # The DimeNet++ model returns a tensor with shape [batch_size, out_channels]
        out = self.dimenet(data.z, data.pos, data.batch)
        # If a linear layer is defined, apply it to reduce dimensions
        if self.linear is not None:
            out = self.linear(out)
        return out

    @classmethod
    def from_pretrained(cls, target: int, **kwargs) -> "DimeNetPPModel":
        """Instantiate a DimeNetPPModel with pre‑trained QM9 weights.

        Parameters
        ----------
        target : int
            Index of the QM9 property to predict.  For example, ``target=2``
            corresponds to the HOMO energy (highest occupied molecular orbital).
        **kwargs
            Additional keyword arguments corresponding to the constructor of
            :class:`DimeNetPPModel`.  These will override the defaults but will
            be ignored for architecture parameters because the pre‑trained
            weights fix the model structure.

        Returns
        -------
        DimeNetPPModel
            A model instance with pre‑trained weights for the specified
            property.
        """
        # When using pre‑trained weights, we enforce `pretrained=True` and
        # propagate the target.  Other kwargs (e.g. hidden_channels) are
        # ignored because the architecture is determined by the checkpoint.
        return cls(pretrained=True, target=target, **kwargs)


__all__ = ["DimeNetPPModel", "DimeNetPPConfig"]