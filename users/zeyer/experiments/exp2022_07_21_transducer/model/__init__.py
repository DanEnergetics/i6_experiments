"""
Model logic
"""

from typing import Protocol, TypeVar, List, Dict
import dataclasses
from sisyphus import tk
from i6_core.returnn.training import Checkpoint
from returnn_common import nn


ModelT = TypeVar("ModelT", bound=nn.Module)


class ModelDef(Protocol[ModelT]):
    """
    Creates the model, per epoch
    """
    def __call__(self, *, epoch: int, target_dim: nn.Dim) -> ModelT:
        raise NotImplementedError


class TrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """
    def __call__(self, *,
                 model: ModelT,
                 data: nn.Data, data_spatial_dim: nn.Dim,
                 targets: nn.Data, targets_spatial_dim: nn.Dim
                 ):
        raise NotImplementedError


class FramewiseTrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """
    def __call__(self, *,
                 model: ModelT,
                 data: nn.Data, data_spatial_dim: nn.Dim,
                 align_targets: nn.Data, align_targets_spatial_dim: nn.Dim
                 ):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpoint:
    """
    Model
    """
    definition: ModelDef
    checkpoint: Checkpoint


@dataclasses.dataclass(frozen=True)
class Alignment:
    """Alignment, for one specific dataset"""
    hdf_files: List[tk.Path]


@dataclasses.dataclass(frozen=True)
class AlignmentCollection:
    """Alignment for multiple datasets"""
    alignments: Dict[str, Alignment]
