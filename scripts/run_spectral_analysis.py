#!/usr/bin/env python3
"""
CLI Script: High-Dimensional Spectral Analysis on GPT-2 Weight Matrices.

Usage:
    python scripts/run_spectral_analysis.py --model_name gpt2 --output_dir results/spectral_analysis --plot
"""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gpt2_extractor import load_gpt2_model_and_tokenizer, extract_layer_weights
from src.spectral.rmt_analysis import analyze_matrix_spectrum
from src.pruning.layer_reduction import compute_layer_importance_scores, compute_layer_similarity_matrix
from src.utils.visualizer import (
    plot_esd_and_mp,
    plot_scree_and_powerlaw,
    plot_layer_metrics_profile,
    plot_layer_similarity_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description="Run High-Dimensional Spectral Analysis on GPT-2")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Hugging Face model identifier (e.g. gpt2, gpt2-medium)")
    parser.add_argument("--output_dir", type=str, default="results/spectral_analysis", help="Directory to save metrics and plots")
    parser.add_argument("--plot", action="store_true", help="Generate publication-quality diagnostic plots")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f" High-Dimensional Spectral Analysis on {args.model_name}")
    print(f"=======================================================\n")

    print(f"[1/4] Loading model: {args.model_name} on {args.device}...")
    model, tokenizer = load_gpt2_model_and_tokenizer(args.model_name, device=args.device)
    num_layers = len(model.transformer.h)
    print(f"      Model loaded successfully with {num_layers} transformer blocks.\n")

    print("[2/4] Performing layer-wise Random Matrix Theory (RMT) spectral decomposition...")
    layer_scores = compute_layer_importance_scores(model)

    # Print summary table to stdout
    print(f"\n{'Layer':<6} | {'Stable Rank':<12} | {'Effective Rank':<15} | {'Signal Energy %':<16} | {'Noise Energy %':<15}")
    print("-" * 75)
    for score in layer_scores:
        print(
            f"{score.layer_idx:<6} | "
            f"{score.mean_stable_rank:<12.2f} | "
            f"{score.mean_effective_rank:<15.2f} | "
            f"{score.mean_signal_energy_ratio * 100:<15.2f}% | "
            f"{score.mean_noise_energy_ratio * 100:<14.2f}%"
        )
    print("-" * 75 + "\n")

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "spectral_metrics.json")
    results_json = {
        "model_name": args.model_name,
        "num_layers": num_layers,
        "layers": [s.to_dict() for s in layer_scores]
    }
    with open(metrics_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[3/4] Saved spectral metrics to: {metrics_path}")

    # Compute similarity matrix
    sim_matrix = compute_layer_similarity_matrix(model)

    # Generate plots if requested
    if args.plot:
        print("[4/4] Generating spectral diagnostic plots...")
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot 1: Layer-wise metric profile
        plot_layer_metrics_profile(
            layer_scores,
            save_path=os.path.join(plots_dir, "layer_metrics_profile.png")
        )

        # Plot 2: Cross-layer similarity heatmap
        plot_layer_similarity_heatmap(
            sim_matrix,
            save_path=os.path.join(plots_dir, "layer_similarity_heatmap.png")
        )

        # Plot 3: ESD & Scree plots for selected layers (first, middle, last)
        sample_indices = [0, num_layers // 2, num_layers - 1]
        for l_idx in sample_indices:
            weights = extract_layer_weights(model, l_idx)
            # Analyze attention projection
            attn_res = analyze_matrix_spectrum(weights.attn_proj.numpy(), name=f"layer_{l_idx}_attn_proj")
            plot_esd_and_mp(
                eigenvalues=attn_res.eigenvalues,
                lambda_minus=attn_res.lambda_minus,
                lambda_plus=attn_res.lambda_plus,
                sigma_sq=attn_res.sigma_sq,
                Q=attn_res.aspect_ratio_Q,
                title=f"Layer {l_idx} Attn Proj - ESD vs Marchenko-Pastur",
                save_path=os.path.join(plots_dir, f"esd_layer_{l_idx}_attn_proj.png")
            )
            plot_scree_and_powerlaw(
                eigenvalues=attn_res.eigenvalues,
                title=f"Layer {l_idx} Attn Proj - Scree & Power-Law Tail",
                save_path=os.path.join(plots_dir, f"scree_layer_{l_idx}_attn_proj.png")
            )

        print(f"      Saved all figures to: {plots_dir}")

    print("\n Spectral analysis complete!\n")


if __name__ == "__main__":
    main()
