import torch
import numpy as np
import returnn.frontend as rf

import nativeops
# import librasr as rasr
from typing import Tuple
from functools import reduce

from dataclasses import dataclass

from .config import TrainStepConfig

cuda_device="cuda"

class FastBaumWelchLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, am_scores, fsa, seq_lens):
        num_states, edge_tensor, weight_tensor, start_end_states = fsa

        grad, loss = nativeops.fbw(
            am_scores, edge_tensor, weight_tensor,
            start_end_states, seq_lens, int(num_states),
            nativeops.DebugOptions()
        )
        ctx.save_for_backward(grad)
        return loss
    
    @staticmethod
    def backward(ctx, grad_loss):
        # negative log prob -> prob
        grad = ctx.saved_tensors[0].neg().exp()
        return grad, None, None

class TorchFsaBuilder:
    def __init__(self, config_path: str, tdp_scale: float=1.0):
        import librasr as rasr
        config = rasr.Configuration()
        config.set_from_file(config_path)
        self.builder = rasr.AllophoneStateFsaBuilder(config)
        self.tdp_scale = tdp_scale
    
    def build(self, seq_tag):
        raw_fsa = self.builder.build_by_segment_name(seq_tag)
        return raw_fsa
    
    def __call__(self, *args, **kwargs):
        return self.build_batch(*args, **kwargs)

    def build_batch(self, seq_tags):
        def concat_fsas(a: Tuple, b: Tuple):
            edges = torch.from_numpy(np.int32(b[2])).reshape((3, b[1]))
            return (
                a[0] + [b[0]], # num states
                a[1] + [b[1]], # num edges
                torch.hstack([a[2], edges]), # edges
                torch.cat([a[3], torch.from_numpy(b[3])]), # weights
            )
        fsas = map(self.builder.build_by_segment_name, seq_tags)
        num_states, num_edges, all_edges, all_weights = reduce(
            concat_fsas, fsas, ([], [], torch.empty((3, 0), dtype=torch.int32), torch.empty((0,)))
        )
        num_edges = torch.tensor(num_edges, dtype=torch.int32)
        num_states = torch.tensor(num_states, dtype=torch.int32)

        cum_num_states = torch.cumsum(num_states, dim=0, dtype=torch.int32)
        state_offsets = torch.cat([torch.zeros((1,), dtype=torch.int32), cum_num_states[:-1]])
        start_end_states = torch.vstack([state_offsets, cum_num_states - 1])

        edge_seq_idxs = torch.repeat_interleave(num_edges)
        all_edges[:2,:] += torch.repeat_interleave(state_offsets, num_edges)
        all_edges = torch.vstack([all_edges, edge_seq_idxs])

        if self.tdp_scale != 1.0:
            all_weights *= self.tdp_scale

        return (
            cum_num_states[-1],
            all_edges.to(cuda_device),
            all_weights.to(cuda_device),
            start_end_states.to(cuda_device)
        )

class TrainStep:
    def __init__(self, cfg: TrainStepConfig):
        self.fsa_builder = TorchFsaBuilder(cfg.config_path, cfg.tdp_scale)
        self.am_scale = cfg.am_scale

    def __call__(self, *, model: torch.nn.Module, extern_data, **_kwargs):
        data = extern_data["data"]
        seq_tags = extern_data["seq_tag"]
        logits = model(data.raw_tensor)
        logits_len = data.dims[1].dyn_size_ext.raw_tensor.to(device="cuda")

        # transpose and negative log space
        logits = (
            logits
            .transpose(0, 1)
            .contiguous()
            .mul(-self.am_scale)
        )

        target_fsa = self.fsa_builder.build_batch(seq_tags.raw_tensor)
        ml_loss = FastBaumWelchLoss.apply(logits, target_fsa, logits_len)
        rf.get_run_ctx().mark_as_loss(name="hmm-log-likelihood", loss=ml_loss)
