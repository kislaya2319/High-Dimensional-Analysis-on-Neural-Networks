#!/usr/bin/env python3
"""
CLI Script: Denoised Synthetic Dataset Generation via Layer-wise Spectral Projection.

Usage:
    python scripts/run_synthetic_generation.py --model_name gpt2 --prompt "Random matrix theory" --num_samples 3 --filter_noise
"""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gpt2_extractor import load_gpt2_model_and_tokenizer
from src.synthetic_gen.spectral_denoiser import generate_synthetic_dataset


DEFAULT_PROMPTS = [
    "In high-dimensional statistics, the empirical spectral density of random covariance matrices",
    "Deep neural network representations can be compressed by identifying redundant layers through",
    "The eigenvalues of correlation matrices in overparameterized models follow the Marchenko-Pastur law, which implies",
    "Autoregressive language models generate coherent sequences when hidden activation noise is minimized by",
]


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Data via Spectral Noise Denoising in GPT-2")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Hugging Face model identifier")
    parser.add_argument("--prompt", type=str, default=None, help="Custom text prompt (if omitted, uses default academic prompts)")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of synthetic samples per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens generated per sequence")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--filter_noise", action="store_true", default=True, help="Enable Marchenko-Pastur activation denoising hooks")
    parser.add_argument("--no_filter", dest="filter_noise", action="store_false", help="Disable spectral denoising (standard baseline)")
    parser.add_argument("--output_file", type=str, default="results/synthetic_dataset.jsonl", help="Filepath to save synthetic samples")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"\n=======================================================")
    print(f" Denoised Synthetic Dataset Generation ({args.model_name})")
    print(f" Spectral Activation Denoising: {'ENABLED' if args.filter_noise else 'DISABLED'}")
    print(f"=======================================================\n")

    print(f"[1/3] Loading model: {args.model_name} on {args.device}...")
    model, tokenizer = load_gpt2_model_and_tokenizer(args.model_name, device=args.device)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print(f"[2/3] Generating synthetic samples for {len(prompts)} prompt(s) (samples/prompt={args.num_samples})...")
    samples = generate_synthetic_dataset(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples_per_prompt=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        filter_noise=args.filter_noise,
        device=args.device,
    )

    print("\n" + "=" * 65)
    print("                SYNTHETIC DATASET SAMPLES")
    print("=" * 65)
    for sample in samples:
        print(f"\n[Sample #{sample.sample_id}] (Denoised: {sample.is_denoised}, {sample.num_tokens} tokens, {sample.generation_time_sec:.2f}s)")
        print(f"Prompt: \"{sample.prompt}\"")
        print(f"Output: {sample.full_text}")
        print("-" * 65)

    # Save to JSON Lines (.jsonl)
    print(f"\n[3/3] Exporting {len(samples)} synthetic samples to: {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")

    print(f" Synthetic dataset export complete!\n")


if __name__ == "__main__":
    main()
