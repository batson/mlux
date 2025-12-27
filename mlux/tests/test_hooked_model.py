"""
Tests for HookedModel - the core mlux functionality.

Run with: pytest mlux/tests/test_hooked_model.py -v
"""

import pytest
import mlx.core as mx
from mlx_lm import load

from mlux import HookedModel


# Use a smaller model for faster tests
MODEL_NAME = "mlx-community/gemma-2-2b-it-4bit"


@pytest.fixture(scope="module")
def hooked_model():
    """Shared HookedModel instance for tests."""
    return HookedModel.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def original_model():
    """Original mlx_lm model for comparison."""
    model, tokenizer = load(MODEL_NAME)
    return model, tokenizer


class TestLogitEquivalence:
    """Test that HookedModel produces identical outputs to the original model."""

    def test_simple_prompt(self, hooked_model, original_model):
        """Basic equivalence test."""
        model, tok = original_model
        prompt = "Hello world"
        tokens = mx.array([tok.encode(prompt)])

        orig_logits = model(tokens)
        hooked_logits = hooked_model.forward(tokens)
        mx.eval(orig_logits, hooked_logits)

        diff = mx.max(mx.abs(orig_logits - hooked_logits)).item()
        assert diff < 1e-6, f"Logit diff {diff} exceeds threshold"

    @pytest.mark.parametrize("prompt", [
        "The capital of France is",
        "def fibonacci(n):",
        "In 1776, the United States",
        "The quick brown fox jumps over the lazy dog.",
    ])
    def test_various_prompts(self, hooked_model, original_model, prompt):
        """Test equivalence across different prompt types."""
        model, tok = original_model
        tokens = mx.array([tok.encode(prompt)])

        orig_logits = model(tokens)
        hooked_logits = hooked_model.forward(tokens)
        mx.eval(orig_logits, hooked_logits)

        diff = mx.max(mx.abs(orig_logits - hooked_logits)).item()
        assert diff < 1e-6, f"Logit diff {diff} for prompt: {prompt[:30]}..."


class TestCaching:
    """Test activation caching functionality."""

    def test_cache_mlp_outputs(self, hooked_model):
        """Test caching all MLP outputs."""
        output, cache = hooked_model.run_with_cache(
            "Hello world",
            hooks=lambda p: p.endswith(".mlp"),
        )

        n_layers = hooked_model.config["n_layers"]
        assert len(cache) == n_layers, f"Expected {n_layers} MLP outputs"

        # Check all have correct hidden dim
        for key, val in cache.items():
            assert val.ndim == 3, f"Expected 3D tensor for {key}"

    def test_cache_attention_outputs(self, hooked_model):
        """Test caching attention outputs."""
        output, cache = hooked_model.run_with_cache(
            "Hello world",
            hooks=lambda p: "self_attn" in p and p.count(".") == 3,
        )

        n_layers = hooked_model.config["n_layers"]
        assert len(cache) == n_layers

    def test_cache_specific_layers(self, hooked_model):
        """Test caching specific layer outputs."""
        hooks = [
            "model.layers.0.mlp",
            "model.layers.5.mlp",
            "model.layers.10.mlp",
        ]
        output, cache = hooked_model.run_with_cache("Test", hooks=hooks)

        assert len(cache) == 3
        for hook in hooks:
            assert hook in cache

    def test_cache_qkv_projections(self, hooked_model):
        """Test caching Q, K, V projections."""
        output, cache = hooked_model.run_with_cache(
            "Hello world",
            hooks=[
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
        )

        q = cache["model.layers.0.self_attn.q_proj"]
        k = cache["model.layers.0.self_attn.k_proj"]
        v = cache["model.layers.0.self_attn.v_proj"]

        config = hooked_model.config
        expected_q_dim = config["n_heads"] * config["d_head"]
        expected_k_dim = config["n_kv_heads"] * config["d_head"]

        assert q.shape[2] == expected_q_dim
        assert k.shape[2] == expected_k_dim
        assert v.shape[2] == expected_k_dim

    def test_cache_residual_stream(self, hooked_model):
        """Test caching full layer outputs (residual stream)."""
        n_layers = hooked_model.config["n_layers"]
        output, cache = hooked_model.run_with_cache(
            "Hello world",
            hooks=lambda p: p.startswith("model.layers.") and p.count(".") == 2,
        )

        assert len(cache) == n_layers

        # Stack and verify shape
        resid_stack = mx.stack([cache[f"model.layers.{i}"] for i in range(n_layers)])
        assert resid_stack.shape[0] == n_layers


class TestInterventions:
    """Test intervention/hook functionality."""

    def test_zero_ablation(self, hooked_model):
        """Test that zero ablation changes outputs."""
        normal_output = hooked_model.forward("The capital of France is")
        mx.eval(normal_output)

        def zero_ablate(inputs, output, wrapper):
            return mx.zeros_like(output)

        ablated_output = hooked_model.run_with_hooks(
            "The capital of France is",
            hooks=[("model.layers.10.mlp", zero_ablate)],
        )
        mx.eval(ablated_output)

        diff = mx.max(mx.abs(normal_output - ablated_output)).item()
        assert diff > 0.1, f"Expected significant diff from ablation, got {diff}"

    def test_activation_patching(self, hooked_model):
        """Test patching activations from one prompt to another."""
        # Cache from source prompt
        _, source_cache = hooked_model.run_with_cache(
            "The capital of Germany is",
            hooks=["model.layers.15.self_attn"],
        )
        source_attn = source_cache["model.layers.15.self_attn"]

        # Patch into target prompt
        def patch_source(inputs, output, wrapper):
            return source_attn

        patched_output = hooked_model.run_with_hooks(
            "The capital of France is",
            hooks=[("model.layers.15.self_attn", patch_source)],
        )
        mx.eval(patched_output)

        normal_output = hooked_model.forward("The capital of France is")
        mx.eval(normal_output)

        diff = mx.max(mx.abs(normal_output - patched_output)).item()
        assert diff > 0.01, f"Expected diff from patching, got {diff}"

    def test_multiple_interventions(self, hooked_model):
        """Test multiple simultaneous interventions."""
        normal_output = hooked_model.forward("Hello world")
        mx.eval(normal_output)

        def scale_half(inputs, output, wrapper):
            return output * 0.5

        multi_output = hooked_model.run_with_hooks(
            "Hello world",
            hooks=[
                ("model.layers.5.mlp", scale_half),
                ("model.layers.10.mlp", scale_half),
                ("model.layers.15.mlp", scale_half),
            ],
        )
        mx.eval(multi_output)

        diff = mx.max(mx.abs(normal_output - multi_output)).item()
        assert diff > 0.01, "Expected diff from multiple interventions"

    def test_model_restoration(self, hooked_model):
        """Test that model is properly restored after hooks."""
        # Run with cache
        output1, _ = hooked_model.run_with_cache(
            "Test", hooks=["model.layers.0.self_attn"]
        )

        # Run with hooks
        def dummy(inputs, output, wrapper):
            return output

        output2 = hooked_model.run_with_hooks(
            "Test", hooks=[("model.layers.5.mlp", dummy)]
        )

        # Run normal
        output3 = hooked_model.forward("Test")

        mx.eval(output1, output2, output3)

        diff12 = mx.max(mx.abs(output1 - output2)).item()
        diff23 = mx.max(mx.abs(output2 - output3)).item()

        assert diff12 < 1e-6, f"Model not restored after cache: {diff12}"
        assert diff23 < 1e-6, f"Model not restored after hooks: {diff23}"


class TestAttentionPatterns:
    """Test attention pattern computation."""

    def test_pattern_shape(self, hooked_model):
        """Test attention pattern shapes are correct."""
        patterns = hooked_model.get_attention_patterns(
            "The cat sat on the mat",
            layers=[0, 10, 20],
        )

        assert len(patterns) == 3

        config = hooked_model.config
        for layer, pattern in patterns.items():
            mx.eval(pattern)
            # [batch, n_heads, seq, seq]
            assert pattern.shape[1] == config["n_heads"]
            assert pattern.shape[2] == pattern.shape[3]  # Square

    def test_patterns_sum_to_one(self, hooked_model):
        """Test that attention patterns sum to 1 (valid softmax)."""
        patterns = hooked_model.get_attention_patterns("Hello", layers=[0])

        pattern = patterns[0]
        mx.eval(pattern)

        # Sum over key dimension should be ~1
        row_sums = mx.sum(pattern[0, 0, :, :], axis=-1)
        mx.eval(row_sums)

        assert mx.allclose(row_sums, mx.ones_like(row_sums), atol=0.01).item()


class TestModelConfig:
    """Test model configuration extraction."""

    def test_config_keys(self, hooked_model):
        """Test that config has expected keys."""
        config = hooked_model.config
        expected_keys = ["n_layers", "n_heads", "n_kv_heads", "d_head"]
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"
            assert config[key] is not None, f"Config key {key} is None"

    def test_gemma_config(self, hooked_model):
        """Test Gemma-specific config values."""
        config = hooked_model.config
        # Gemma 2-2B specific values
        assert config["n_layers"] == 26
        assert config["n_heads"] == 8
        assert config["n_kv_heads"] == 4
        assert config["d_head"] == 256
        assert config["softcap"] == 50.0
