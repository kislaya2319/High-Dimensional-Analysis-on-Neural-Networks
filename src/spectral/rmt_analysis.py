"""
Random Matrix Theory (RMT) and High-Dimensional Spectral Analysis Core.

Provides analytical tools to compute Empirical Spectral Densities (ESD),
fit Marchenko-Pastur bulk noise distributions, calculate effective rank,
and filter noise eigenvalues from neural network weight matrices.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch


@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results of a weight matrix."""
    matrix_name: str
    shape: Tuple[int, int]
    aspect_ratio_Q: float
    eigenvalues: np.ndarray
    sigma_sq: float
    lambda_minus: float
    lambda_plus: float
    stable_rank: float
    effective_rank: float
    spectral_entropy: float
    condition_number: float
    spectral_norm: float
    signal_eigenvalue_count: int
    signal_energy_ratio: float
    noise_energy_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_name": self.matrix_name,
            "shape": list(self.shape),
            "aspect_ratio_Q": float(self.aspect_ratio_Q),
            "sigma_sq": float(self.sigma_sq),
            "lambda_minus": float(self.lambda_minus),
            "lambda_plus": float(self.lambda_plus),
            "stable_rank": float(self.stable_rank),
            "effective_rank": float(self.effective_rank),
            "spectral_entropy": float(self.spectral_entropy),
            "condition_number": float(self.condition_number),
            "spectral_norm": float(self.spectral_norm),
            "signal_eigenvalue_count": int(self.signal_eigenvalue_count),
            "signal_energy_ratio": float(self.signal_energy_ratio),
            "noise_energy_ratio": float(self.noise_energy_ratio),
        }


def to_numpy(tensor_or_array: Any) -> np.ndarray:
    """Helper to convert PyTorch tensor or array-like object to numpy ndarray."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().to(torch.float32).numpy()
    return np.asarray(tensor_or_array, dtype=np.float32)


def compute_correlation_matrix(
    W: np.ndarray,
    center: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Computes the empirical correlation/sample covariance matrix C = (1/N) * W * W^T.
    
    Args:
        W: 2D weight matrix with shape (M, N). We orient such that M <= N.
        center: If True, subtract the column mean.
        normalize: If True, scale by standard deviation per column.
        
    Returns:
        C: Empirical correlation matrix of shape (min(M, N), min(M, N)).
    """
    W = to_numpy(W)
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {W.shape}")

    # Ensure shape is (M, N) with M <= N
    if W.shape[0] > W.shape[1]:
        W = W.T

    M, N = W.shape

    if center:
        W = W - np.mean(W, axis=1, keepdims=True)

    if normalize:
        std = np.std(W, axis=1, keepdims=True)
        std[std == 0] = 1.0
        W = W / std

    # Empirical correlation matrix: C = (1/N) W W^T
    C = (W @ W.T) / float(N)
    return C


def compute_eigenvalues(
    W: np.ndarray,
    center: bool = True,
    normalize: bool = False
) -> np.ndarray:
    """
    Computes sorted eigenvalues of empirical correlation matrix C = (1/N) W W^T.
    
    Returns:
        Sorted eigenvalues in descending order: lambda_1 >= lambda_2 >= ... >= lambda_M >= 0.
    """
    C = compute_correlation_matrix(W, center=center, normalize=normalize)
    # Hermite eigenvalue solver for symmetric matrix
    eigenvalues = np.linalg.eigvalsh(C)
    # Clip negative values due to numerical precision
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues


def marchenko_pastur_pdf(
    x: np.ndarray,
    Q: float,
    sigma_sq: float = 1.0
) -> np.ndarray:
    """
    Theoretical Marchenko-Pastur probability density function rho_MP(lambda).
    
    rho_MP(lambda) = (Q / (2 * pi * sigma^2 * lambda)) * sqrt((lambda_+ - lambda)(lambda - lambda_-))
    for lambda_- <= lambda <= lambda_+, and 0 otherwise.
    
    Args:
        x: Points at which to evaluate density.
        Q: Aspect ratio N / M (Q >= 1).
        sigma_sq: Variance parameter.
    """
    lambda_minus = sigma_sq * (1.0 - np.sqrt(1.0 / Q)) ** 2
    lambda_plus = sigma_sq * (1.0 + np.sqrt(1.0 / Q)) ** 2

    pdf = np.zeros_like(x, dtype=np.float32)
    mask = (x >= lambda_minus) & (x <= lambda_plus)

    if np.any(mask):
        valid_x = x[mask]
        num = np.sqrt((lambda_plus - valid_x) * (valid_x - lambda_minus))
        denom = (2.0 * np.pi * sigma_sq * np.maximum(valid_x, 1e-12)) / Q
        pdf[mask] = num / denom

    return pdf


def fit_marchenko_pastur(
    eigenvalues: np.ndarray,
    Q: float,
    sigma_sq: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Fits Marchenko-Pastur distribution to empirical eigenvalues.
    
    If sigma_sq is None, estimates sigma^2 by matching the median of empirical
    eigenvalues in the noise bulk or using empirical trace / variance.
    
    Returns:
        (sigma_sq, lambda_minus, lambda_plus)
    """
    if Q < 1.0:
        Q = 1.0 / Q

    if sigma_sq is None:
        # Robust estimation of noise variance using median of eigenvalues
        # Since top eigenvalues may be outlier spikes, median is resistant to outliers.
        med = float(np.median(eigenvalues))
        # For MP with aspect ratio Q, the median is approximately sigma^2 * (1 - Q^(-1/2)/3)^2 or similar;
        # A simple robust estimate is sigma^2 = mean(eigenvalues) or median-calibrated
        sigma_sq = max(med, 1e-8)

    lambda_minus = float(sigma_sq * (1.0 - np.sqrt(1.0 / Q)) ** 2)
    lambda_plus = float(sigma_sq * (1.0 + np.sqrt(1.0 / Q)) ** 2)

    return sigma_sq, lambda_minus, lambda_plus


def compute_spectral_metrics(
    eigenvalues: np.ndarray,
    lambda_plus: float
) -> Dict[str, float]:
    """
    Computes key spectral metrics from eigenvalues:
    - Stable Rank: ||W||_F^2 / ||W||_2^2 = sum(lambda_i) / lambda_max
    - Effective Rank: exp(Spectral Entropy)
    - Spectral Entropy: -sum(p_i * ln(p_i)) where p_i = lambda_i / sum(lambda)
    - Condition Number: lambda_max / (lambda_min + eps)
    - Signal Energy Ratio: sum(lambda_i > lambda_+) / sum(lambda_i)
    - Noise Energy Ratio: sum(lambda_i <= lambda_+) / sum(lambda_i)
    """
    total_energy = float(np.sum(eigenvalues))
    if total_energy <= 1e-12:
        return {
            "stable_rank": 1.0,
            "effective_rank": 1.0,
            "spectral_entropy": 0.0,
            "condition_number": 1.0,
            "spectral_norm": 0.0,
            "signal_eigenvalue_count": 0,
            "signal_energy_ratio": 0.0,
            "noise_energy_ratio": 1.0,
        }

    lambda_max = float(eigenvalues[0])
    lambda_min = float(eigenvalues[-1])

    # Stable rank: sum(lambda_i) / lambda_max
    stable_rank = total_energy / max(lambda_max, 1e-12)

    # Probabilities for spectral entropy
    p = eigenvalues / total_energy
    p_nz = p[p > 1e-12]
    spectral_entropy = float(-np.sum(p_nz * np.log(p_nz)))
    effective_rank = float(np.exp(spectral_entropy))

    # Condition number
    condition_number = float(lambda_max / max(lambda_min, 1e-12))

    # Signal vs Noise decomposition
    signal_mask = eigenvalues > lambda_plus
    signal_count = int(np.sum(signal_mask))
    signal_energy = float(np.sum(eigenvalues[signal_mask]))
    signal_ratio = signal_energy / total_energy
    noise_ratio = 1.0 - signal_ratio

    return {
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "spectral_entropy": spectral_entropy,
        "condition_number": condition_number,
        "spectral_norm": np.sqrt(lambda_max),
        "signal_eigenvalue_count": signal_count,
        "signal_energy_ratio": signal_ratio,
        "noise_energy_ratio": noise_ratio,
    }


def analyze_matrix_spectrum(
    W: np.ndarray,
    name: str = "matrix"
) -> SpectralAnalysisResult:
    """
    Performs complete end-to-end RMT spectral analysis on a 2D weight matrix.
    """
    W = to_numpy(W)
    if W.ndim != 2:
        raise ValueError(f"Weight matrix must be 2D, got shape {W.shape}")

    M, N = W.shape
    Q = float(max(M, N)) / float(min(M, N))

    eigenvalues = compute_eigenvalues(W, center=True, normalize=False)
    sigma_sq, lambda_minus, lambda_plus = fit_marchenko_pastur(eigenvalues, Q)
    metrics = compute_spectral_metrics(eigenvalues, lambda_plus)

    return SpectralAnalysisResult(
        matrix_name=name,
        shape=(M, N),
        aspect_ratio_Q=Q,
        eigenvalues=eigenvalues,
        sigma_sq=sigma_sq,
        lambda_minus=lambda_minus,
        lambda_plus=lambda_plus,
        stable_rank=metrics["stable_rank"],
        effective_rank=metrics["effective_rank"],
        spectral_entropy=metrics["spectral_entropy"],
        condition_number=metrics["condition_number"],
        spectral_norm=metrics["spectral_norm"],
        signal_eigenvalue_count=metrics["signal_eigenvalue_count"],
        signal_energy_ratio=metrics["signal_energy_ratio"],
        noise_energy_ratio=metrics["noise_energy_ratio"],
    )


def filter_noise_eigenvalues(
    W: np.ndarray,
    lambda_plus_threshold: Optional[float] = None,
    keep_signal_only: bool = True
) -> np.ndarray:
    """
    Filters Marchenko-Pastur bulk noise from a weight matrix using SVD truncation.
    
    W = U S V^T
    Singular values s_i = sqrt(N * lambda_i).
    We set singular values below the MP noise threshold to 0.
    
    Args:
        W: 2D weight matrix (M, N).
        lambda_plus_threshold: Noise cutoff. If None, computed automatically via RMT.
        keep_signal_only: If True, set noise singular values to zero.
        
    Returns:
        W_denoised: Reconstructed clean low-rank matrix.
    """
    W_np = to_numpy(W)
    is_transposed = False
    if W_np.shape[0] > W_np.shape[1]:
        W_np = W_np.T
        is_transposed = True

    M, N = W_np.shape
    Q = float(N) / float(M)

    # Perform full or economy SVD
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    
    # Eigenvalues of (1/N) W W^T are (S^2) / N
    eigenvalues = (S ** 2) / float(N)

    if lambda_plus_threshold is None:
        _, _, lambda_plus_threshold = fit_marchenko_pastur(eigenvalues, Q)

    # Singular value threshold: s_cutoff = sqrt(N * lambda_plus)
    s_cutoff = np.sqrt(float(N) * lambda_plus_threshold)

    S_denoised = S.copy()
    if keep_signal_only:
        S_denoised[S < s_cutoff] = 0.0

    W_denoised = U @ np.diag(S_denoised) @ Vt

    if is_transposed:
        W_denoised = W_denoised.T

    return W_denoised
