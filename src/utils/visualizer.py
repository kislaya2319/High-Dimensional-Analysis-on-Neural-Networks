"""
Visualization & Plotting Suite for High-Dimensional Spectral Analysis.

Generates publication-quality figures:
- Empirical Spectral Density (ESD) histograms with theoretical Marchenko-Pastur fits.
- Log-log eigenvalue tail distributions (power-law analysis).
- Layer-wise stable rank, effective rank, and signal-to-noise ratio profiles.
- Cross-layer representational similarity heatmaps.
"""

from typing import List, Optional, Any
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from ..spectral.rmt_analysis import marchenko_pastur_pdf


def plot_esd_and_mp(
    eigenvalues: np.ndarray,
    lambda_minus: float,
    lambda_plus: float,
    sigma_sq: float,
    Q: float,
    title: str = "Empirical Spectral Density vs Marchenko-Pastur",
    save_path: Optional[str] = None
) -> None:
    """
    Plots the empirical eigenvalue histogram overlaid with the theoretical MP density curve.
    """
    plt.figure(figsize=(9, 5.5), dpi=150)
    
    # Plot empirical histogram
    counts, bins, _ = plt.hist(
        eigenvalues,
        bins=60,
        density=True,
        alpha=0.6,
        color="#2b5c8f",
        edgecolor="#1a365d",
        label="Empirical Spectral Density (ESD)"
    )

    # Plot theoretical Marchenko-Pastur PDF curve
    x = np.linspace(max(0.0, lambda_minus * 0.8), lambda_plus * 1.2, 500)
    pdf = marchenko_pastur_pdf(x, Q=Q, sigma_sq=sigma_sq)
    plt.plot(x, pdf, color="#d9534f", linewidth=2.5, label=f"Marchenko-Pastur Fit ($\\sigma^2={sigma_sq:.3f}$)")

    # Mark spectral edges
    plt.axvline(lambda_plus, color="#d9534f", linestyle="--", alpha=0.8, label=f"Bulk Edge $\\lambda_+={lambda_plus:.3f}$")
    if lambda_minus > 0:
        plt.axvline(lambda_minus, color="#d9534f", linestyle=":", alpha=0.6, label=f"Bulk Edge $\\lambda_-={lambda_minus:.3f}$")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Eigenvalue ($\\lambda$)", fontsize=11)
    plt.ylabel("Density $\\rho(\\lambda)$", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_scree_and_powerlaw(
    eigenvalues: np.ndarray,
    title: str = "Eigenvalue Scree & Heavy Tail Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a 2-panel figure: (1) Linear Scree Plot, (2) Log-Log Tail Distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Panel 1: Scree Plot (Eigenvalue magnitude vs Index)
    indices = np.arange(1, len(eigenvalues) + 1)
    ax1.plot(indices, eigenvalues, marker="o", markersize=3, color="#2e7d32", linewidth=1.5)
    ax1.set_title("Eigenvalue Scree Plot", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rank Index $i$", fontsize=10)
    ax1.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Log-Log plot for power-law detection
    valid_mask = eigenvalues > 1e-10
    if np.any(valid_mask):
        valid_evals = eigenvalues[valid_mask]
        valid_idx = indices[valid_mask]
        ax2.loglog(valid_idx, valid_evals, marker="s", markersize=2.5, color="#c2185b", linewidth=1.5, label="Log-Log $\\lambda_i$")
        ax2.set_title("Log-Log Spectrum (Heavy Tail Analysis)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Rank Index $\\log(i)$", fontsize=10)
        ax2.set_ylabel("Eigenvalue $\\log(\\lambda_i)$", fontsize=10)
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_layer_metrics_profile(
    layer_scores: List[Any],
    save_path: Optional[str] = None
) -> None:
    """
    Plots layer-wise variation of Stable Rank, Effective Rank, and Signal/Noise Energy Ratio.
    """
    layers = [s.layer_idx for s in layer_scores]
    stable_ranks = [s.mean_stable_rank for s in layer_scores]
    effective_ranks = [s.mean_effective_rank for s in layer_scores]
    signal_ratios = [s.mean_signal_energy_ratio * 100 for s in layer_scores]
    noise_ratios = [s.mean_noise_energy_ratio * 100 for s in layer_scores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Panel 1: Stable Rank & Effective Rank across layers
    ax1.plot(layers, stable_ranks, marker="o", color="#1976d2", label="Stable Rank", linewidth=2)
    ax1.plot(layers, effective_ranks, marker="^", color="#f57c00", label="Effective Rank", linewidth=2)
    ax1.set_title("Layer-wise Dimensionality (Stable & Effective Rank)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Transformer Layer Index", fontsize=10)
    ax1.set_ylabel("Rank Value", fontsize=10)
    ax1.set_xticks(layers)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Stacked Bar of Signal vs Noise Energy
    bar_width = 0.55
    ax2.bar(layers, signal_ratios, bar_width, label="Signal Energy (%)", color="#388e3c", alpha=0.85)
    ax2.bar(layers, noise_ratios, bar_width, bottom=signal_ratios, label="Noise Energy (%)", color="#d32f2f", alpha=0.7)
    ax2.set_title("Signal vs Marchenko-Pastur Noise Energy Ratio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Transformer Layer Index", fontsize=10)
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_xticks(layers)
    ax2.legend(frameon=True, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_layer_similarity_heatmap(
    similarity_matrix: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a heatmap of cross-layer similarity.
    """
    num_layers = similarity_matrix.shape[0]
    plt.figure(figsize=(7, 6), dpi=150)
    im = plt.imshow(similarity_matrix, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine Similarity")
    
    plt.title("Cross-Layer Representation Similarity Heatmap", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Transformer Layer Index", fontsize=10)
    plt.ylabel("Transformer Layer Index", fontsize=10)
    plt.xticks(np.arange(num_layers))
    plt.yticks(np.arange(num_layers))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
