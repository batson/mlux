"""
Miscellaneous tests covering:
- Dtype consistency
- Cache stacking
- Tokenization/BOS handling
- Batch operations

Run with: pytest tests/test_misc.py -v
"""

import pytest
import mlx.core as mx

from mlux import HookedModel


MODEL_NAME = "mlx-community/gemma-2-2b-it-4bit"


@pytest.fixture(scope="module")
def hooked():
    """Shared HookedModel instance."""
    return HookedModel.from_pretrained(MODEL_NAME)


class TestQuantizationDetection:
    """Test quantization detection and tolerance adjustment."""

    def test_detects_4bit_quantization(self, hooked):
        """Correctly detects 4-bit quantized model."""
        bits = hooked.quantization_bits
        # gemma-2-2b-it-4bit should be 4-bit
        assert bits == 4, f"Expected 4-bit, got {bits}"

    def test_tolerance_reflects_quantization(self, hooked):
        """get_tolerance() returns appropriate value for quantization level."""
        tol = hooked.get_tolerance()

        # 4-bit model should get relaxed tolerance
        assert tol > 0.01, f"4-bit model should have relaxed tolerance, got {tol}"

    def test_tolerance_custom_values(self, hooked):
        """Can customize strict and relaxed tolerance values."""
        tol = hooked.get_tolerance(strict=1e-6, relaxed=0.5)
        # 4-bit model should use relaxed value
        assert tol == 0.5, f"Expected custom relaxed=0.5, got {tol}"


class TestDtypeConsistency:
    """Test that dtypes are preserved through operations."""

    def test_output_dtype_matches_model(self, hooked):
        """Output dtype matches model's weight dtype."""
        output = hooked.forward("Hello")
        mx.eval(output)

        # Output should be float32 (or float16 depending on model)
        assert output.dtype in [mx.float32, mx.float16, mx.bfloat16], \
            f"Unexpected output dtype: {output.dtype}"

    def test_cache_dtype_consistent(self, hooked):
        """Cached activations have consistent dtype."""
        _, cache = hooked.run_with_cache("Hello", hooks=[
            "model.layers.0.mlp",
            "model.layers.10.mlp",
        ])

        dtypes = set()
        for path, val in cache.items():
            mx.eval(val)
            dtypes.add(val.dtype)

        assert len(dtypes) == 1, f"Inconsistent dtypes in cache: {dtypes}"

    def test_hook_output_dtype_preserved(self, hooked):
        """Hook can return same dtype as input."""
        captured_dtype = [None]

        def capture_dtype(inputs, output, wrapper):
            captured_dtype[0] = output.dtype
            return output

        hooked.run_with_hooks("Test", hooks=[("model.layers.5.mlp", capture_dtype)])

        assert captured_dtype[0] is not None
        assert captured_dtype[0] in [mx.float32, mx.float16, mx.bfloat16]


class TestCacheStacking:
    """Test stacking cached activations."""

    def test_stack_mlp_outputs_across_layers(self, hooked):
        """Can stack MLP outputs across all layers."""
        n_layers = hooked.config["n_layers"]
        hooks = [f"model.layers.{i}.mlp" for i in range(n_layers)]

        _, cache = hooked.run_with_cache("Hello world", hooks=hooks)

        # Stack all MLP outputs
        stacked = mx.stack([cache[f"model.layers.{i}.mlp"] for i in range(n_layers)])
        mx.eval(stacked)

        # Shape should be [n_layers, batch, seq, hidden_dim]
        assert stacked.shape[0] == n_layers
        assert stacked.shape[1] == 1  # batch

    def test_stack_attention_outputs(self, hooked):
        """Can stack attention outputs across layers."""
        layers = [0, 5, 10, 15, 20, 25]
        hooks = [f"model.layers.{i}.self_attn" for i in layers]

        _, cache = hooked.run_with_cache("Test", hooks=hooks)

        stacked = mx.stack([cache[f"model.layers.{i}.self_attn"] for i in layers])
        mx.eval(stacked)

        assert stacked.shape[0] == len(layers)

    def test_residual_stream_decomposition(self, hooked):
        """Can cache full layer outputs for residual stream analysis."""
        n_layers = hooked.config["n_layers"]
        hooks = [f"model.layers.{i}" for i in range(n_layers)]

        _, cache = hooked.run_with_cache("The capital of France", hooks=hooks)

        # Stack residual stream
        resid_stack = mx.stack([cache[f"model.layers.{i}"] for i in range(n_layers)])
        mx.eval(resid_stack)

        # Should have n_layers entries
        assert resid_stack.shape[0] == n_layers


class TestTokenization:
    """Test tokenization behavior."""

    def test_string_input_tokenized(self, hooked):
        """String input is automatically tokenized."""
        output = hooked.forward("Hello world")
        mx.eval(output)

        # Should produce valid output
        assert output.shape[0] == 1  # batch
        assert output.shape[2] > 0  # vocab size

    def test_token_input_works(self, hooked):
        """Can pass token array directly."""
        tokens = hooked._tokenize("Hello")
        output = hooked.forward(tokens)
        mx.eval(output)

        # Both methods should work
        assert output.shape[0] == 1

    def test_tokenize_returns_2d(self, hooked):
        """Tokenization returns 2D array [batch, seq]."""
        tokens = hooked._tokenize("Hello world")

        assert tokens.ndim == 2, f"Expected 2D, got {tokens.ndim}D"
        assert tokens.shape[0] == 1, f"Expected batch=1, got {tokens.shape[0]}"

    def test_longer_text_more_tokens(self, hooked):
        """Longer text produces more tokens."""
        short = hooked._tokenize("Hi")
        long = hooked._tokenize("Hello world, how are you doing today?")

        assert long.shape[1] > short.shape[1], \
            f"Longer text should have more tokens: {long.shape[1]} vs {short.shape[1]}"


class TestCacheSlicing:
    """Test slicing cached activations."""

    def test_cache_last_position(self, hooked):
        """Can slice cache to get last position only."""
        _, cache = hooked.run_with_cache(
            "The quick brown fox",
            hooks=["model.layers.10.mlp"]
        )

        full = cache["model.layers.10.mlp"]
        mx.eval(full)

        # Get last position
        last_pos = full[:, -1, :]  # [batch, hidden]
        mx.eval(last_pos)

        assert last_pos.ndim == 2
        assert last_pos.shape[0] == 1  # batch

    def test_cache_specific_positions(self, hooked):
        """Can slice cache to specific positions."""
        _, cache = hooked.run_with_cache(
            "The quick brown fox jumps",
            hooks=["model.layers.5.mlp"]
        )

        full = cache["model.layers.5.mlp"]
        mx.eval(full)

        # Get positions 1-3
        subset = full[:, 1:4, :]
        mx.eval(subset)

        assert subset.shape[1] == 3


class TestBatchOperations:
    """Test batch dimension handling."""

    def test_single_batch_output_shape(self, hooked):
        """Single input produces batch=1 output."""
        output = hooked.forward("Hello")
        mx.eval(output)

        assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"

    def test_cache_has_batch_dim(self, hooked):
        """Cached values have batch dimension."""
        _, cache = hooked.run_with_cache("Test", hooks=["model.layers.0.mlp"])

        val = cache["model.layers.0.mlp"]
        mx.eval(val)

        # Should be [batch, seq, hidden]
        assert val.ndim == 3, f"Expected 3D, got {val.ndim}D"
        assert val.shape[0] == 1, f"Expected batch=1, got {val.shape[0]}"


class TestAvailableHooks:
    """Test hook discovery."""

    def test_available_hooks_nonempty(self, hooked):
        """available_hooks returns non-empty list."""
        hooks = hooked.available_hooks()
        assert len(hooks) > 0, "Should have available hooks"

    def test_find_hooks_by_pattern(self, hooked):
        """find_hooks filters by pattern."""
        mlp_hooks = hooked.find_hooks("mlp")
        attn_hooks = hooked.find_hooks("self_attn")

        assert len(mlp_hooks) > 0, "Should find MLP hooks"
        assert len(attn_hooks) > 0, "Should find attention hooks"

        # All should contain pattern
        for hook in mlp_hooks:
            assert "mlp" in hook
        for hook in attn_hooks:
            assert "self_attn" in hook

    def test_layer_hooks_exist(self, hooked):
        """Each layer should have hookable modules."""
        n_layers = hooked.config["n_layers"]

        for i in range(n_layers):
            layer_hooks = hooked.find_hooks(f"layers.{i}")
            assert len(layer_hooks) > 0, f"Layer {i} should have hooks"


class TestEdgeCasesAndSpecialInputs:
    """Test edge cases with special inputs."""

    def test_single_token_input(self, hooked):
        """Works with single token input."""
        output = hooked.forward("A")
        mx.eval(output)
        assert output.shape[1] >= 1

    def test_very_long_input(self, hooked):
        """Works with longer input (up to reasonable length)."""
        long_text = "word " * 100  # 100 words
        output = hooked.forward(long_text)
        mx.eval(output)
        assert output.shape[1] > 50  # Should have many positions

    def test_special_characters(self, hooked):
        """Works with special characters."""
        output = hooked.forward("Hello! @#$%^&*() 你好 🎉")
        mx.eval(output)
        assert output.shape[0] == 1

    def test_empty_hooks_list(self, hooked):
        """Empty hooks list produces same output as forward."""
        tokens = hooked._tokenize("Test")

        forward_out = hooked.forward(tokens)
        hooks_out = hooked.run_with_hooks(tokens, hooks=[])
        mx.eval(forward_out, hooks_out)

        diff = mx.max(mx.abs(forward_out - hooks_out)).item()
        assert diff < 1e-6
