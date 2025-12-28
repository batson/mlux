"""
Tests for multi-model support.

These tests verify that mlux works across different model architectures.
Run with: pytest mlux/tests/test_multi_model.py -v

Note: These tests require downloading models and may be slow.
Mark with @pytest.mark.slow if you want to skip in quick test runs.
"""

import pytest
import mlx.core as mx

from mlux import HookedModel


class TestGPT2:
    """Tests for GPT-2 architecture support."""

    @pytest.fixture(scope="class")
    def gpt2_model(self):
        return HookedModel.from_pretrained("mlx-community/gpt2-base-mlx")

    def test_config_detection(self, gpt2_model):
        """Test that GPT-2 config is correctly detected."""
        config = gpt2_model.config
        assert config["n_layers"] == 12
        assert config["n_heads"] == 12
        assert config["n_kv_heads"] == 12  # MHA, not GQA
        assert config["d_head"] == 64
        assert config["layer_prefix"] == "model.h"
        assert config["qkv_style"] == "combined"

    def test_forward_pass(self, gpt2_model):
        """Test basic forward pass."""
        output = gpt2_model.forward("Hello world")
        mx.eval(output)
        assert output.shape[0] == 1  # batch
        assert output.shape[2] == 50257  # vocab size

    def test_caching(self, gpt2_model):
        """Test caching works with GPT-2 structure."""
        output, cache = gpt2_model.run_with_cache(
            "Hello",
            hooks=["model.h.0.attn.c_attn", "model.h.0.mlp"],
        )
        assert "model.h.0.attn.c_attn" in cache
        assert "model.h.0.mlp" in cache

    def test_attention_patterns(self, gpt2_model):
        """Test attention pattern computation for GPT-2."""
        patterns = gpt2_model.get_attention_patterns("Hello world", layers=[0, 5, 11])

        assert len(patterns) == 3
        for layer, pattern in patterns.items():
            mx.eval(pattern)
            # GPT-2 has 12 heads
            assert pattern.shape[1] == 12


class TestLlama:
    """Tests for Llama architecture support."""

    @pytest.fixture(scope="class")
    def llama_model(self):
        return HookedModel.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def test_config_detection(self, llama_model):
        """Test that Llama config is correctly detected."""
        config = llama_model.config
        assert config["n_layers"] == 16
        assert config["n_heads"] == 32
        assert config["n_kv_heads"] == 8  # GQA
        assert config["layer_prefix"] == "model.layers"
        assert config["qkv_style"] == "separate"

    def test_attention_patterns(self, llama_model):
        """Test attention pattern computation for Llama."""
        patterns = llama_model.get_attention_patterns("Hello", layers=[0, 8, 15])

        assert len(patterns) == 3
        for layer, pattern in patterns.items():
            mx.eval(pattern)
            assert pattern.shape[1] == 32  # 32 heads


class TestCrossModelConsistency:
    """Tests that verify consistent behavior across models."""

    @pytest.mark.parametrize("model_name", [
        "mlx-community/gemma-2-2b-it-4bit",
        "mlx-community/gpt2-base-mlx",
    ])
    def test_available_hooks(self, model_name):
        """Test that available_hooks returns non-empty list for all models."""
        hooked = HookedModel.from_pretrained(model_name)
        hooks = hooked.available_hooks()
        assert len(hooks) > 0

    @pytest.mark.parametrize("model_name", [
        "mlx-community/gemma-2-2b-it-4bit",
        "mlx-community/gpt2-base-mlx",
    ])
    def test_config_complete(self, model_name):
        """Test that all models have complete config."""
        hooked = HookedModel.from_pretrained(model_name)
        config = hooked.config

        required = ["n_layers", "n_heads", "n_kv_heads", "d_head"]
        for key in required:
            assert config[key] is not None, f"{model_name} missing {key}"
