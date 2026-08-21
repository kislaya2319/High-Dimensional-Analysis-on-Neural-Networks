"""
Spectral Activation Denoising & Synthetic Dataset Generation.

Filters Marchenko-Pastur bulk noise from hidden representation tensors across
GPT-2 transformer layers and produces high-quality synthetic text samples.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import time
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ..spectral.rmt_analysis import fit_marchenko_pastur


@dataclass
class SyntheticDataSample:
    """Container for a generated synthetic sequence with metadata."""
    sample_id: int
    prompt: str
    generated_text: str
    full_text: str
    is_denoised: bool
    num_tokens: int
    generation_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "is_denoised": self.is_denoised,
            "num_tokens": self.num_tokens,
            "generation_time_sec": self.generation_time_sec,
        }


def denoise_activation_tensor(
    hidden_states: torch.Tensor,
    shrinkage: float = 0.8
) -> torch.Tensor:
    """
    Applies RMT spectral projection on batch token activations H in R^(B x T x d).
    
    1. Flatten to N x d where N = B * T.
    2. Compute empirical covariance Sigma = (1/N) H^T H in R^(d x d).
    3. Decompose Sigma into V Lambda V^T.
    4. Fit Marchenko-Pastur bulk edge lambda_+.
    5. Construct projection matrix P = sum_{lambda_i > lambda_+} v_i v_i^T.
    6. Denoised H_clean = H @ (shrinkage * P + (1 - shrinkage) * I).
    """
    orig_shape = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Reshape to (N, d)
    flat_H = hidden_states.view(-1, orig_shape[-1]).to(torch.float32)
    N, d = flat_H.shape

    # If sequence length is too small to estimate covariance reliably, return unmodified
    if N <= 4:
        return hidden_states

    # Center activations
    mean = flat_H.mean(dim=0, keepdim=True)
    centered_H = flat_H - mean

    # Covariance Sigma: (d, d)
    cov = (centered_H.T @ centered_H) / float(N)
    
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    except Exception:
        # Fallback if eigendecomposition fails
        return hidden_states

    eigenvalues_np = eigenvalues.detach().cpu().numpy()
    eigenvalues_np = np.clip(eigenvalues_np, a_min=0.0, a_max=None)

    Q = max(float(N) / float(d), 1.0)
    _, _, lambda_plus = fit_marchenko_pastur(eigenvalues_np, Q)

    # Signal mask: keep eigenvalues above Marchenko-Pastur bulk cutoff
    signal_mask = eigenvalues > float(lambda_plus)

    # If no eigenvalues above threshold, keep top 10% principal components
    if signal_mask.sum() == 0:
        top_k = max(1, d // 10)
        signal_mask[-top_k:] = True

    # Construct signal projector P = V_signal @ V_signal^T
    V_signal = eigenvectors[:, signal_mask]  # (d, k)
    P_signal = V_signal @ V_signal.T         # (d, d)

    # Soft shrinkage blend: P_blend = shrinkage * P_signal + (1 - shrinkage) * I
    I = torch.eye(d, device=device, dtype=torch.float32)
    P_blend = (shrinkage * P_signal + (1.0 - shrinkage) * I)

    # Project centered representations and re-add mean
    H_denoised = (centered_H @ P_blend) + mean
    return H_denoised.view(orig_shape).to(dtype)


class SpectralActivationDenoiser:
    """
    Hook manager to dynamically intercept and denoise activations across GPT-2 layers.
    """
    def __init__(
        self,
        model: GPT2LMHeadModel,
        target_layers: Optional[List[int]] = None,
        shrinkage: float = 0.8
    ):
        self.model = model
        self.shrinkage = shrinkage
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        num_layers = len(model.transformer.h)
        self.target_layers = target_layers or list(range(num_layers))

    def _hook_fn(self, module: nn.Module, input: Any, output: Any) -> Any:
        # GPT-2 Block output is a tuple (hidden_states, present, attentions...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            denoised_h = denoise_activation_tensor(hidden_states, shrinkage=self.shrinkage)
            return (denoised_h,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return denoise_activation_tensor(output, shrinkage=self.shrinkage)
        return output

    def attach(self) -> None:
        """Attaches denoising hooks to specified transformer blocks."""
        self.remove()
        for idx in self.target_layers:
            block = self.model.transformer.h[idx]
            handle = block.register_forward_hook(self._hook_fn)
            self.handles.append(handle)

    def remove(self) -> None:
        """Removes all active hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def attach_spectral_hooks(
    model: GPT2LMHeadModel,
    target_layers: Optional[List[int]] = None,
    shrinkage: float = 0.8
) -> SpectralActivationDenoiser:
    """Helper to construct and attach spectral denoiser."""
    denoiser = SpectralActivationDenoiser(model, target_layers=target_layers, shrinkage=shrinkage)
    denoiser.attach()
    return denoiser


def generate_synthetic_dataset(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompts: List[str],
    num_samples_per_prompt: int = 1,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    filter_noise: bool = True,
    device: str = "cpu"
) -> List[SyntheticDataSample]:
    """
    Generates synthetic dataset samples by sampling from GPT-2 with or without spectral denoising.
    """
    model.eval()
    model.to(device)

    dataset: List[SyntheticDataSample] = []
    sample_id = 0

    denoiser = SpectralActivationDenoiser(model) if filter_noise else None

    with denoiser if filter_noise else torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            for _ in range(num_samples_per_prompt):
                sample_id += 1
                t0 = time.time()

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_time = time.time() - t0
                full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                # Extract only generated portion
                gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                sample = SyntheticDataSample(
                    sample_id=sample_id,
                    prompt=prompt,
                    generated_text=gen_text,
                    full_text=full_text,
                    is_denoised=filter_noise,
                    num_tokens=int(output_ids.shape[1]),
                    generation_time_sec=float(gen_time),
                )
                dataset.append(sample)

    return dataset
