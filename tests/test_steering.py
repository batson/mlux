"""
Tests for steering - correctness of steering operations.

Run with: pytest mlux/tests/test_steering.py -v
"""

import pytest
import mlx.core as mx

from mlux import HookedModel
from mlux.steering import (
    create_steering_hook,
    create_additive_hook,
    prefill_with_cache,
    generate_from_cache,
)


MODEL_NAME = "mlx-community/gemma-2-2b-it-4bit"


@pytest.fixture(scope="module")
def hooked():
    """Shared HookedModel instance."""
    return HookedModel.from_pretrained(MODEL_NAME)


def get_hidden_dim(hooked):
    """Get hidden dimension from model config."""
    # Try common config attribute names
    model = hooked.model.model if hasattr(hooked.model, 'model') else hooked.model
    if hasattr(model, 'args'):
        args = model.args
        if hasattr(args, 'hidden_size'):
            return args.hidden_size
        if hasattr(args, 'dim'):
            return args.dim
    # Fallback: infer from embedding weight
    if hasattr(model, 'embed_tokens'):
        return model.embed_tokens.weight.shape[-1]
    raise ValueError("Could not determine hidden dimension")


class TestZeroVectorEquivalence:
    """Test that steering with zero vector equals unsteered output."""

    def test_zero_steering_matches_baseline(self, hooked):
        """
        Steering with zero vector at T=0 should give identical output
        to sampling without steering.
        """
        prompt = "The capital of France is"
        layer = 10
        d_model = get_hidden_dim(hooked)

        # Create zero steering vector
        zero_vector = mx.zeros(d_model)
        hook = create_steering_hook(zero_vector, alpha=1.0)
        hook_path = f"model.layers.{layer}"

        # Get tokens
        tokens = hooked._tokenize(prompt)

        # Unsteered forward pass
        baseline_logits = hooked.model(tokens)
        mx.eval(baseline_logits)

        # Steered with zero vector
        steered_logits = hooked.run_with_hooks(
            tokens, hooks=[(hook_path, hook)]
        )
        mx.eval(steered_logits)

        # Should be identical
        diff = mx.max(mx.abs(baseline_logits - steered_logits)).item()
        assert diff < 1e-5, f"Zero steering should match baseline, got diff {diff}"

    def test_zero_steering_same_argmax(self, hooked):
        """Zero steering should produce same greedy token."""
        prompt = "The quick brown fox"
        layer = 10
        d_model = get_hidden_dim(hooked)

        zero_vector = mx.zeros(d_model)
        hook = create_steering_hook(zero_vector, alpha=1.0)
        hook_path = f"model.layers.{layer}"

        tokens = hooked._tokenize(prompt)

        baseline_logits = hooked.model(tokens)
        steered_logits = hooked.run_with_hooks(tokens, hooks=[(hook_path, hook)])
        mx.eval(baseline_logits, steered_logits)

        baseline_token = mx.argmax(baseline_logits[:, -1, :], axis=-1).item()
        steered_token = mx.argmax(steered_logits[:, -1, :], axis=-1).item()

        assert baseline_token == steered_token, "Zero steering changed prediction"


class TestPositionSpecificCausality:
    """Test that position-specific steering respects causality."""

    def test_cache_unchanged_before_steering_position(self, hooked):
        """
        If we steer at position 5, the KV cache at positions 0-4
        should be identical to no steering.

        Note: Steering at layer N's output affects layer N+1's cache, not N's.
        So we steer at layer-1 and check layer's cache.
        """
        from mlx_lm.models.cache import make_prompt_cache

        prompt = "The quick brown fox jumps over the lazy dog"
        steer_layer = 9  # Steer at layer 9
        check_layer = 10  # Check cache at layer 10 (affected by layer 9's output)
        d_model = get_hidden_dim(hooked)
        steer_position = 5

        tokens = hooked._tokenize(prompt)
        seq_len = tokens.shape[1]
        assert seq_len > steer_position, "Need longer prompt for this test"

        hook_path = f"model.layers.{steer_layer}"

        # Build deltas that only affect position 5
        # Use smaller values to avoid fp16 overflow
        random_vector = mx.random.normal((d_model,)) * 0.1
        deltas = mx.zeros((1, seq_len, d_model))
        # Add steering only at position 5
        pos_mask = (mx.arange(seq_len) == steer_position).reshape(1, seq_len, 1)
        deltas = pos_mask * random_vector.reshape(1, 1, -1)

        hook_steered = create_additive_hook(deltas)

        # Run with no steering
        cache_baseline = make_prompt_cache(hooked.model)
        hooked.model(tokens, cache=cache_baseline)
        mx.eval([c.state for c in cache_baseline])

        # Run with position-5 steering
        cache_steered = make_prompt_cache(hooked.model)
        hooked.run_with_hooks(tokens, hooks=[(hook_path, hook_steered)], cache=cache_steered)
        mx.eval([c.state for c in cache_steered])

        # Compare KV cache at check_layer (affected by steering at steer_layer)
        # Cache is per-layer, each has keys/values of shape [batch, n_kv_heads, seq, d_head]
        baseline_keys = cache_baseline[check_layer].keys
        steered_keys = cache_steered[check_layer].keys
        mx.eval(baseline_keys, steered_keys)

        # Positions 0-4 should be identical (causality)
        # The KV at position i only depends on tokens 0..i
        for pos in range(steer_position):
            baseline_k = baseline_keys[:, :, pos, :]
            steered_k = steered_keys[:, :, pos, :]
            diff = mx.max(mx.abs(baseline_k - steered_k)).item()
            assert diff < 1e-4, f"KV at position {pos} differs (diff={diff}), but steering was at {steer_position}"

        # Position 5+ may differ (that's expected)
        # We don't need to test this, but let's verify steering did something
        steered_k5 = steered_keys[:, :, steer_position, :]
        baseline_k5 = baseline_keys[:, :, steer_position, :]
        diff_at_steer = mx.max(mx.abs(baseline_k5 - steered_k5)).item()
        assert diff_at_steer > 1e-5, f"Steering at position {steer_position} had no effect (diff={diff_at_steer})"


class TestSteeringEffects:
    """Test that non-zero steering actually changes outputs."""

    def test_nonzero_steering_changes_logits(self, hooked):
        """Non-zero steering should change the output logits."""
        prompt = "Hello world"
        layer = 10
        d_model = get_hidden_dim(hooked)

        # Random steering vector (use moderate scale to avoid fp16 overflow)
        steering_vec = mx.random.normal((d_model,)) * 0.5
        hook = create_steering_hook(steering_vec, alpha=1.0)
        hook_path = f"model.layers.{layer}"

        tokens = hooked._tokenize(prompt)

        baseline_logits = hooked.model(tokens)
        steered_logits = hooked.run_with_hooks(tokens, hooks=[(hook_path, hook)])
        mx.eval(baseline_logits, steered_logits)

        diff = mx.max(mx.abs(baseline_logits - steered_logits)).item()
        assert diff > 0.1, f"Steering should change logits, got diff {diff}"

    def test_alpha_scaling(self, hooked):
        """Larger alpha should produce larger changes."""
        prompt = "Test"
        layer = 10
        d_model = get_hidden_dim(hooked)

        # Use smaller random values to avoid fp16 overflow
        steering_vec = mx.random.normal((d_model,)) * 0.1
        hook_path = f"model.layers.{layer}"
        tokens = hooked._tokenize(prompt)

        baseline_logits = hooked.model(tokens)
        mx.eval(baseline_logits)

        diffs = []
        for alpha in [0.5, 1.0, 2.0]:
            hook = create_steering_hook(steering_vec, alpha=alpha)
            steered = hooked.run_with_hooks(tokens, hooks=[(hook_path, hook)])
            mx.eval(steered)
            diff = mx.mean(mx.abs(baseline_logits - steered)).item()
            diffs.append(diff)

        # Diffs should increase with alpha
        assert diffs[0] < diffs[1] < diffs[2], f"Diffs should scale with alpha: {diffs}"


class TestCacheGeneration:
    """Test cache-based generation workflow."""

    def test_prefill_creates_cache(self, hooked):
        """prefill_with_cache should return a usable cache and logits."""
        prompt = "Hello"
        cache, logits = prefill_with_cache(hooked, prompt)

        assert cache is not None
        assert len(cache) > 0
        # Check cache has content
        assert cache[0].offset > 0
        # Check logits shape
        assert logits.ndim == 3

    def test_generate_from_cache_produces_tokens(self, hooked):
        """generate_from_cache should produce text."""
        prompt = "The capital of France is"
        cache, logits = prefill_with_cache(hooked, prompt)
        output = generate_from_cache(hooked, cache, max_tokens=10, temperature=0, initial_logits=logits)

        assert len(output) > 0, "Should generate some text"
        assert isinstance(output, str)
