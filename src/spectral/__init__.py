"""
Spectral analysis and Random Matrix Theory (RMT) tools.
"""

from .rmt_analysis import (
    compute_correlation_matrix,
    compute_eigenvalues,
    fit_marchenko_pastur,
    compute_spectral_metrics,
    filter_noise_eigenvalues,
    marchenko_pastur_pdf,
    SpectralAnalysisResult,
)

__all__ = [
    "compute_correlation_matrix",
    "compute_eigenvalues",
    "fit_marchenko_pastur",
    "compute_spectral_metrics",
    "filter_noise_eigenvalues",
    "marchenko_pastur_pdf",
    "SpectralAnalysisResult",
]
