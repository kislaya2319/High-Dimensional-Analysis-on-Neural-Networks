#!/usr/bin/env python3
"""
CLI Script: RMT-Guided Layer Pruning & Complexity Reduction in GPT-2.

Usage:
    python scripts/run_layer_pruning.py --model_name gpt2 --prune_layers 3 --metric min_signal_energy --output_dir results/pruning
"""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gpt2_extractor import load_gpt2_model_and_tokenizer
from src.pruning.layer_reduction import (
    prune_model_and_benchmark,
    apply_layerwise_svd_noise_filter,
    evaluate_perplexity,
)


DEFAULT_EVAL_TEXTS = [
    "High-dimensional statistical analysis reveals deep regularities in the empirical spectral distribution of deep neural networks.",
    "Random Matrix Theory offers universal tools such as the Marchenko-Pastur distribution to identify noise eigenvalues in overparameterized models.",
    "Layer pruning reduces computational latency and memory consumption while retaining essential semantic representations across transformer blocks.",
    "The GPT-2 architecture is composed of stacked decoder-only multi-head self-attention and feed-forward neural networks.",
]


def main():
    parser = argparse.ArgumentParser(description="Reduce GPT-2 Layer Complexity via Spectral Pruning")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Hugging Face model name")
    parser.add_argument("--prune_layers", type=int, default=3, help="Number of transformer layers to prune")
    parser.add_argument(
        "--strategy",
        type=str,
        default="min_signal_energy",
        choices=["min_signal_energy", "max_noise_ratio", "min_effective_rank", "uniform_middle"],
        help="Spectral criteria for selecting candidate layers to prune"
    )
    parser.add_argument("--truncate_noise", action="store_true", help="Apply SVD truncation to strip MP noise from remaining layers")
    parser.add_argument("--eval_text", type=str, default=None, help="Custom text passage for perplexity benchmark")
    parser.add_argument("--output_dir", type=str, default="results/pruning_results", help="Directory to save pruned model config & benchmark")
    parser.add_argument("--save_model", action="store_true", help="Save the pruned PyTorch model weights to disk")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f" Layer Complexity Reduction & Pruning on {args.model_name}")
    print(f"=======================================================\n")

    print(f"[1/4] Loading model: {args.model_name} on {args.device}...")
    model, tokenizer = load_gpt2_model_and_tokenizer(args.model_name, device=args.device)

    eval_texts = DEFAULT_EVAL_TEXTS
    if args.eval_text:
        eval_texts = [args.eval_text]

    print(f"[2/4] Evaluating and pruning {args.prune_layers} layers using strategy: '{args.strategy}'...")
    results = prune_model_and_benchmark(
        model=model,
        tokenizer=tokenizer,
        num_layers_to_prune=args.prune_layers,
        eval_texts=eval_texts,
        strategy=args.strategy,
        device=args.device,
    )

    pruned_model = results["pruned_model"]

    if args.truncate_noise:
        print("[*] Applying Marchenko-Pastur SVD noise truncation to all remaining layers...")
        pruned_model = apply_layerwise_svd_noise_filter(pruned_model)
        trunc_eval = evaluate_perplexity(pruned_model, tokenizer, eval_texts, device=args.device)
        results["svd_truncated_perplexity"] = trunc_eval["perplexity"]

    # Print summary benchmark
    print("\n" + "=" * 55)
    print("           LAYER PRUNING BENCHMARK REPORT")
    print("=" * 55)
    print(f" Original Transformer Blocks : {results['original_layers']}")
    print(f" Pruned Transformer Blocks   : {results['pruned_layers']}")
    print(f" Removed Layer Indices       : {results['pruned_indices']}")
    print(f" Pruning Strategy            : {results['strategy']}")
    print("-" * 55)
    print(f" Original Parameter Count    : {results['original_params']['total_parameters']:,} ({results['original_params']['size_mb']:.2f} MB)")
    print(f" Pruned Parameter Count      : {results['pruned_params']['total_parameters']:,} ({results['pruned_params']['size_mb']:.2f} MB)")
    print(f" Parameter Reduction         : {results['param_reduction_pct']:.2f}%")
    print("-" * 55)
    print(f" Baseline Perplexity (PPL)   : {results['original_perplexity']:.3f}")
    print(f" Pruned Model Perplexity     : {results['pruned_perplexity']:.3f}")
    print(f" Perplexity Degradation      : +{results['perplexity_delta']:.3f}")
    if args.truncate_noise:
        print(f" SVD Truncated Perplexity    : {results.get('svd_truncated_perplexity', 0.0):.3f}")
    print("=" * 55 + "\n")

    # Export report to JSON (excluding torch model object)
    export_data = {k: v for k, v in results.items() if k != "pruned_model"}
    report_file = os.path.join(args.output_dir, "pruning_report.json")
    with open(report_file, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"[3/4] Exported pruning benchmark report to: {report_file}")

    if args.save_model:
        model_save_dir = os.path.join(args.output_dir, "pruned_model_weights")
        pruned_model.save_pretrained(model_save_dir)
        if tokenizer:
            tokenizer.save_pretrained(model_save_dir)
        print(f"[4/4] Saved pruned model weights to: {model_save_dir}")
    else:
        print(f"[4/4] Pruned model ready (pass --save_model to write checkpoints to disk).")

    print("\n Layer complexity reduction complete!\n")


if __name__ == "__main__":
    main()
