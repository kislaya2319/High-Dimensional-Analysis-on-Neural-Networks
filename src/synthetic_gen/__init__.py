"""
Spectral denoising and synthetic dataset generation modules.
"""

from .spectral_denoiser import (
    SpectralActivationDenoiser,
    attach_spectral_hooks,
    generate_synthetic_dataset,
    denoise_activation_tensor,
    SyntheticDataSample,
)

__all__ = [
    "SpectralActivationDenoiser",
    "attach_spectral_hooks",
    "generate_synthetic_dataset",
    "denoise_activation_tensor",
    "SyntheticDataSample",
]
