from dataclasses import dataclass


@dataclass
class PipelineConfig:
    image_size: int = 224
    noise_std: float = 0.05
