import torch
from dataclasses import dataclass
from typing import Callable
from torch import nn
from functools import partial

from i6_models.primitives.specaugment import specaugment_v1_by_length

from .specaugment import SpecAugmentV1ByLengthConfig

@dataclass
class FeedForwardConfig:
    input_dim: int
    hidden_dim: int
    dropout: float
    window_size: int = 1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu

@dataclass
class ModelConfig:
    feature_dim: int
    output_dim: int
    num_layers: int = 6
    hidden_dim: int = 2048
    window_size: int = 15
    dropout: float = 0.1
    spec_aug_cfg: SpecAugmentV1ByLengthConfig = None
    hidden_activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu
    out_activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.log_softmax

class FeedForward(nn.Module):
    """
    Feedforward layer module
    """
    def __init__(self, cfg: FeedForwardConfig):
        super().__init__()

        if cfg.window_size == 1:
            self.linear_ff = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim, bias=True)
        else:
            self.conv = nn.Conv1d(
                in_channels=cfg.input_dim,
                out_channels=cfg.hidden_dim,
                kernel_size=cfg.window_size,
                padding="same",
                bias=True,
            )
            self.linear_ff = lambda x: self.conv(x.transpose(1, 2)).transpose(1, 2)
        self.activation = cfg.activation
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = self.dropout(tensor)  # [B,T,F]
        return tensor

class Model(nn.Module):
    """
    Feedforward network module
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.layers = nn.ModuleList()
        hidden_dims = [cfg.hidden_dim] * cfg.num_layers
        log_sfmx_func = partial(cfg.out_activation, dim=-1)
        activations = [cfg.hidden_activation] * cfg.num_layers + [log_sfmx_func]
        for in_dim, out_dim, act_func, wds in zip(
            [cfg.feature_dim] + hidden_dims,
            hidden_dims + [cfg.output_dim],
            activations,
            [cfg.window_size] + [1] * cfg.num_layers
        ):
            self.layers.append(FeedForward(FeedForwardConfig(
                input_dim=in_dim,
                hidden_dim=out_dim,
                dropout=cfg.dropout,
                activation=act_func,
                window_size=wds
            )))
        

        if cfg.spec_aug_cfg is not None:
            self.spec_aug_func = partial(
                specaugment_v1_by_length,
                time_min_num_masks=cfg.spec_aug_cfg.time_min_num_masks,
                time_max_mask_per_n_frames=cfg.spec_aug_cfg.time_max_mask_per_n_frames,
                time_mask_max_size=cfg.spec_aug_cfg.time_mask_max_size,
                freq_min_num_masks=cfg.spec_aug_cfg.freq_min_num_masks,
                freq_max_num_masks=cfg.spec_aug_cfg.freq_max_num_masks,
                freq_mask_max_size=cfg.spec_aug_cfg.freq_mask_max_size,
            )
        else:
            self.spec_aug_func = lambda x: x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: torch.no_grad()?
        if self.training:
            x = self.spec_aug_func(x)
        for layer in self.layers:
            x = layer(x)
        return x

def get_model(**kwargs):
    return Model(model_cfg)

def forward_step(*, model: Model, extern_data, **_kwargs):
    import returnn.frontend as rf
    data = extern_data["data"]
    out = model(data.raw_tensor)
    rf.get_run_ctx().expected_outputs["classes"].dims[1].dyn_size_ext.raw_tensor = data.dims[1].dyn_size_ext.raw_tensor
    rf.get_run_ctx().mark_as_output(tensor=out, name="classes")
