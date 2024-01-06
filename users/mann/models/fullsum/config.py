from dataclasses import dataclass

@dataclass
class TrainStepConfig:
    config_path: str
    am_scale: float = 1.0
    tdp_scale: float = 1.0
