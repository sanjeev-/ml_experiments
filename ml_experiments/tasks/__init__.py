"""Task-specific experiment implementations."""

from .text_to_image import TextToImageConfig, TextToImageExperiment
from .matting import MattingConfig, MattingExperiment
from .segmentation import SegmentationConfig, SegmentationExperiment
from .synthetic_data import SyntheticDataConfig, SyntheticDataExperiment
from .vae import VAEConfig, VAEExperiment
from .text_encoder import TextEncoderConfig, TextEncoderExperiment

__all__ = [
    "TextToImageConfig",
    "TextToImageExperiment",
    "MattingConfig",
    "MattingExperiment",
    "SegmentationConfig",
    "SegmentationExperiment",
    "SyntheticDataConfig",
    "SyntheticDataExperiment",
    "VAEConfig",
    "VAEExperiment",
    "TextEncoderConfig",
    "TextEncoderExperiment",
]
