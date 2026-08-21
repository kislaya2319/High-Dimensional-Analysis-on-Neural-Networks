"""
Automated unit tests for High-Dimensional Spectral Analysis, RMT, and Layer Pruning.
Uses standard Python unittest framework for maximum compatibility.
"""

import unittest
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.spectral.rmt_analysis import (
    compute_correlation_matrix,
    compute_eigenvalues,
    fit_marchenko_pastur,
    marchenko_pastur_pdf,
    compute_spectral_metrics,
    filter_noise_eigenvalues,
    analyze_matrix_spectrum,
)
from src.models.gpt2_extractor import (
    extract_layer_weights,
    create_pruned_gpt2_model,
    count_parameters,
)
from src.pruning.layer_reduction import (
    compute_layer_importance_scores,
    select_layers_to_prune,
    compute_layer_similarity_matrix,
    apply_layerwise_svd_noise_filter,
)
from src.synthetic_gen.spectral_denoiser import (
    denoise_activation_tensor,
    SpectralActivationDenoiser,
)


class TestRMTAnalysis(unittest.TestCase):
    def test_correlation_matrix_shape(self):
        # 100 x 300 matrix (M=100, N=300, Q=3.0)
        W = np.random.randn(100, 300).astype(np.float32)
        C = compute_correlation_matrix(W, center=True, normalize=False)
        self.assertEqual(C.shape, (100, 100))
        # Check symmetry
        self.assertTrue(np.allclose(C, C.T, atol=1e-5))

    def test_marchenko_pastur_bounds(self):
        # Generate pure Gaussian noise matrix
        np.random.seed(42)
        M, N = 200, 800
        Q = float(N) / float(M)
        W = np.random.randn(M, N).astype(np.float32) / np.sqrt(N)
        eigenvalues = compute_eigenvalues(W, center=True)

        sigma_sq, l_minus, l_plus = fit_marchenko_pastur(eigenvalues, Q)
        self.assertGreaterEqual(l_minus, 0.0)
        self.assertGreater(l_plus, l_minus)

        # For pure noise, most eigenvalues should lie near the MP bounds
        within_bulk = np.mean((eigenvalues >= l_minus * 0.7) & (eigenvalues <= l_plus * 1.3))
        self.assertGreater(within_bulk, 0.85)

    def test_marchenko_pastur_pdf(self):
        x = np.linspace(0.1, 3.0, 100)
        pdf = marchenko_pastur_pdf(x, Q=2.0, sigma_sq=1.0)
        self.assertEqual(len(pdf), 100)
        self.assertTrue(np.all(pdf >= 0.0))

    def test_spectral_metrics(self):
        eigenvalues = np.array([10.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        metrics = compute_spectral_metrics(eigenvalues, lambda_plus=1.5)
        
        self.assertGreater(metrics["stable_rank"], 1.0)
        self.assertGreaterEqual(metrics["effective_rank"], 1.0)
        self.assertEqual(metrics["signal_eigenvalue_count"], 2)  # 10.0 and 2.0
        self.assertTrue(0.0 <= metrics["signal_energy_ratio"] <= 1.0)
        self.assertTrue(np.isclose(metrics["signal_energy_ratio"] + metrics["noise_energy_ratio"], 1.0))

    def test_filter_noise_eigenvalues(self):
        W = np.random.randn(50, 100).astype(np.float32)
        W_denoised = filter_noise_eigenvalues(W)
        self.assertEqual(W_denoised.shape, W.shape)
        self.assertLessEqual(np.linalg.norm(W_denoised), np.linalg.norm(W) + 1e-3)


class TestModelExtractionAndPruning(unittest.TestCase):
    def setUp(self):
        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=4,
            n_head=4,
        )
        self.dummy_gpt2 = GPT2LMHeadModel(config)
        self.dummy_gpt2.eval()

    def test_extract_layer_weights(self):
        weights = extract_layer_weights(self.dummy_gpt2, 0)
        self.assertEqual(weights.layer_idx, 0)
        self.assertEqual(weights.attn_qkv.shape, (3 * 64, 64))
        self.assertEqual(weights.attn_proj.shape, (64, 64))
        self.assertEqual(weights.mlp_fc.shape, (4 * 64, 64))

    def test_create_pruned_gpt2_model(self):
        orig_layers = len(self.dummy_gpt2.transformer.h)
        self.assertEqual(orig_layers, 4)
        
        pruned_model = create_pruned_gpt2_model(self.dummy_gpt2, layers_to_remove=[1, 2])
        self.assertEqual(len(pruned_model.transformer.h), 2)
        self.assertEqual(pruned_model.config.n_layer, 2)

        # Test forward pass with random input
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            out = pruned_model(input_ids)
            self.assertEqual(out.logits.shape, (2, 16, 1000))

    def test_parameter_count(self):
        counts = count_parameters(self.dummy_gpt2)
        self.assertGreater(counts["total_parameters"], 0)
        self.assertGreater(counts["size_mb"], 0)

    def test_layer_importance_scores(self):
        scores = compute_layer_importance_scores(self.dummy_gpt2)
        self.assertEqual(len(scores), 4)
        self.assertTrue(0 <= scores[0].mean_signal_energy_ratio <= 1.0)

    def test_select_layers_to_prune(self):
        scores = compute_layer_importance_scores(self.dummy_gpt2)
        to_prune = select_layers_to_prune(scores, num_layers_to_prune=2, strategy="min_signal_energy")
        self.assertEqual(len(to_prune), 2)
        self.assertNotIn(0, to_prune)
        self.assertNotIn(3, to_prune)

    def test_layer_similarity_matrix(self):
        sim = compute_layer_similarity_matrix(self.dummy_gpt2)
        self.assertEqual(sim.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(sim), 1.0, atol=1e-3))

    def test_apply_svd_noise_filter(self):
        model = apply_layerwise_svd_noise_filter(self.dummy_gpt2)
        input_ids = torch.randint(0, 1000, (1, 8))
        out = model(input_ids)
        self.assertEqual(out.logits.shape, (1, 8, 1000))


class TestSpectralDenoiser(unittest.TestCase):
    def setUp(self):
        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=4,
            n_head=4,
        )
        self.dummy_gpt2 = GPT2LMHeadModel(config)
        self.dummy_gpt2.eval()

    def test_denoise_activation_tensor(self):
        H = torch.randn(2, 16, 64)
        H_denoised = denoise_activation_tensor(H, shrinkage=0.8)
        self.assertEqual(H_denoised.shape, H.shape)

    def test_spectral_denoiser_hook(self):
        denoiser = SpectralActivationDenoiser(self.dummy_gpt2, target_layers=[0, 1])
        denoiser.attach()
        self.assertEqual(len(denoiser.handles), 2)

        input_ids = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out = self.dummy_gpt2(input_ids)
            self.assertEqual(out.logits.shape, (2, 8, 1000))

        denoiser.remove()
        self.assertEqual(len(denoiser.handles), 0)


if __name__ == "__main__":
    unittest.main()
