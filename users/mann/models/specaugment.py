from i6_models.config import ModelConfiguration
from dataclasses import dataclass

@dataclass
class SpecAugmentV1ByLengthConfig(ModelConfiguration):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int
