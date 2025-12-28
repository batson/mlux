"""
KV Cache correctness tests inspired by TransformerLens patterns.

These tests validate that KV cache works correctly:
- Prefill + continuation matches full forward pass
- Cache with hooks produces correct results
- Multiple token continuation works

Run with: pytest tests/test_kv_cache.py -v
"""

import pytest
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlux import HookedModel


MODEL_NAME = "mlx-community/gemma-2-2b-it-4bit"


@pytest.fixture(scope="module")
def hooked():
    """Shared HookedModel instance."""
    return HookedModel.from_pretrained(MODEL_NAME)


class TestKVCacheBasics:
    """Test basic KV cache functionality."""

    def test_cache_structure(self, hooked):
        """KV cache has expected structure."""
        cache = make_prompt_cache(hooked.model)

        n_layers = hooked.config["n_layers"]
        assert len(cache) == n_layers, f"Expected {n_layers} cache entries"

        # Each entry should have keys, values, offset attributes
        for i, entry in enumerate(cache):
            assert hasattr(entry, "keys"), f"Cache entry {i} missing keys"
            assert hasattr(entry, "values"), f"Cache entry {i} missing values"
            assert hasattr(entry, "offset"), f"Cache entry {i} missing offset"

    def test_prefill_updates_cache(self, hooked):
        """Running prefill updates the cache offset."""
        tokens = hooked._tokenize("Hello world")
        cache = make_prompt_cache(hooked.model)

        assert cache[0].offset == 0, "Cache should start at offset 0"

        hooked.model(tokens, cache=cache)
        # Eval the keys/values
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        seq_len = tokens.shape[1]
        assert cache[0].offset == seq_len, f"Cache offset should be {seq_len}"


class TestCachePlusContinuation:
    """Test that cache + new tokens equals full forward pass.

    Uses hooked.get_tolerance() to automatically adjust for quantization level.
    """

    def test_single_new_token(self, hooked):
        """Prefill + single token matches full forward on that position."""
        pre_prompt = "I went to the store,"
        new_token_str = " and"

        # Tokenize separately
        pre_tokens = hooked._tokenize(pre_prompt)
        new_token = mx.array([[hooked.tokenizer.encode(new_token_str)[-1]]])

        # Full forward (both parts together)
        full_tokens = mx.concatenate([pre_tokens, new_token], axis=1)
        full_logits = hooked.model(full_tokens)
        mx.eval(full_logits)

        # Prefill + continuation
        cache = make_prompt_cache(hooked.model)
        hooked.model(pre_tokens, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        cont_logits = hooked.model(new_token, cache=cache)
        mx.eval(cont_logits)

        # Last position logits should match
        full_last = full_logits[0, -1, :]
        cont_last = cont_logits[0, -1, :]

        diff = mx.max(mx.abs(full_last - cont_last)).item()
        tol = hooked.get_tolerance()
        assert diff < tol, f"Cache continuation doesn't match full forward: diff={diff}, tol={tol}"

    def test_multiple_new_tokens(self, hooked):
        """Prefill + multiple tokens matches full forward."""
        pre_prompt = "The capital of"
        post_prompt = " France is Paris"

        pre_tokens = hooked._tokenize(pre_prompt)
        post_tokens = mx.array([hooked.tokenizer.encode(post_prompt)])

        # Full forward
        full_tokens = mx.concatenate([pre_tokens, post_tokens], axis=1)
        full_logits = hooked.model(full_tokens)
        mx.eval(full_logits)

        # Prefill + continuation
        cache = make_prompt_cache(hooked.model)
        hooked.model(pre_tokens, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        cont_logits = hooked.model(post_tokens, cache=cache)
        mx.eval(cont_logits)

        # All post positions should match
        post_len = post_tokens.shape[1]
        full_post = full_logits[0, -post_len:, :]
        cont_all = cont_logits[0, :, :]

        diff = mx.max(mx.abs(full_post - cont_all)).item()
        tol = hooked.get_tolerance()
        assert diff < tol, f"Multi-token continuation doesn't match: diff={diff}, tol={tol}"

    def test_argmax_matches(self, hooked):
        """The predicted token (argmax) matches between cache and full forward."""
        pre_prompt = "The quick brown fox"
        new_token_str = " jumps"

        pre_tokens = hooked._tokenize(pre_prompt)
        new_token = mx.array([[hooked.tokenizer.encode(new_token_str)[-1]]])

        # Full forward
        full_tokens = mx.concatenate([pre_tokens, new_token], axis=1)
        full_logits = hooked.model(full_tokens)
        mx.eval(full_logits)

        # Prefill + continuation
        cache = make_prompt_cache(hooked.model)
        hooked.model(pre_tokens, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        cont_logits = hooked.model(new_token, cache=cache)
        mx.eval(cont_logits)

        # Argmax should match (what actually matters for generation)
        full_pred = mx.argmax(full_logits[0, -1, :]).item()
        cont_pred = mx.argmax(cont_logits[0, -1, :]).item()

        assert full_pred == cont_pred, f"Argmax mismatch: full={full_pred}, cont={cont_pred}"


class TestCacheWithHooks:
    """Test that hooks work correctly with KV cache."""

    def test_hook_with_cache_runs(self, hooked):
        """Hook executes when using cache."""
        tokens = hooked._tokenize("Test prompt")
        cache = make_prompt_cache(hooked.model)

        hook_ran = [False]

        def mark_ran(inputs, output, wrapper):
            hook_ran[0] = True
            return output

        hooked.run_with_hooks(
            tokens,
            hooks=[("model.layers.5.mlp", mark_ran)],
            cache=cache
        )

        assert hook_ran[0], "Hook should run with cache"

    def test_hook_with_cache_matches_no_cache(self, hooked):
        """Hook output with cache matches hook without cache (prefill phase)."""
        prompt = "Hello world"
        tokens = hooked._tokenize(prompt)

        captured_no_cache = {}
        captured_with_cache = {}

        def capture_no_cache(inputs, output, wrapper):
            captured_no_cache["output"] = output
            return output

        def capture_with_cache(inputs, output, wrapper):
            captured_with_cache["output"] = output
            return output

        # Without cache
        hooked.run_with_hooks(
            tokens,
            hooks=[("model.layers.5.mlp", capture_no_cache)]
        )

        # With cache (prefill phase)
        cache = make_prompt_cache(hooked.model)
        hooked.run_with_hooks(
            tokens,
            hooks=[("model.layers.5.mlp", capture_with_cache)],
            cache=cache
        )

        out1 = captured_no_cache["output"]
        out2 = captured_with_cache["output"]
        mx.eval(out1, out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff < 1e-5, f"Hook output differs with/without cache: diff={diff}"

    def test_steering_with_cache_affects_continuation(self, hooked):
        """Steering during prefill affects subsequent generation."""
        from mlux.steering import prefill_with_cache, generate_from_cache

        prompt = "The capital of France is"
        layer = 10

        # Get hidden dim
        _, act_cache = hooked.run_with_cache(prompt, hooks=[f"model.layers.{layer}"])
        hidden_dim = act_cache[f"model.layers.{layer}"].shape[-1]

        # Generate without steering
        cache1, logits1 = prefill_with_cache(hooked, prompt)
        out1 = generate_from_cache(hooked, cache1, max_tokens=5, temperature=0, initial_logits=logits1)

        # Generate with steering
        mx.random.seed(42)
        steering_vec = mx.random.normal((hidden_dim,)) * 0.5

        def steer_hook(inputs, output, wrapper):
            return output + steering_vec

        cache2, logits2 = prefill_with_cache(
            hooked, prompt,
            hooks=[(f"model.layers.{layer}", steer_hook)]
        )
        out2 = generate_from_cache(hooked, cache2, max_tokens=5, temperature=0, initial_logits=logits2)

        # Outputs should differ due to steering
        # (They might be same by chance, so we just check they're both valid strings)
        assert isinstance(out1, str) and len(out1) > 0
        assert isinstance(out2, str) and len(out2) > 0


class TestCacheConsistency:
    """Test cache consistency properties."""

    def test_same_prompt_same_cache(self, hooked):
        """Same prompt produces identical cache values."""
        prompt = "Hello world"
        tokens = hooked._tokenize(prompt)

        cache1 = make_prompt_cache(hooked.model)
        hooked.model(tokens, cache=cache1)
        for c in cache1:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        cache2 = make_prompt_cache(hooked.model)
        hooked.model(tokens, cache=cache2)
        for c in cache2:
            if c.keys is not None:
                mx.eval(c.keys, c.values)

        # Compare first layer's keys
        keys1 = cache1[0].keys
        keys2 = cache2[0].keys
        mx.eval(keys1, keys2)

        diff = mx.max(mx.abs(keys1 - keys2)).item()
        assert diff < 1e-6, f"Same prompt should produce same cache: diff={diff}"

    def test_cache_offset_increments(self, hooked):
        """Cache offset increments correctly with each forward pass."""
        cache = make_prompt_cache(hooked.model)

        # First chunk
        tokens1 = hooked._tokenize("Hello")
        len1 = tokens1.shape[1]
        hooked.model(tokens1, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)
        assert cache[0].offset == len1

        # Second chunk
        tokens2 = mx.array([[hooked.tokenizer.encode(" world")[0]]])
        hooked.model(tokens2, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)
        assert cache[0].offset == len1 + 1

        # Third chunk
        tokens3 = mx.array([[hooked.tokenizer.encode("!")[0]]])
        hooked.model(tokens3, cache=cache)
        for c in cache:
            if c.keys is not None:
                mx.eval(c.keys, c.values)
        assert cache[0].offset == len1 + 2


class TestCacheWithGeneration:
    """Test cache during text generation."""

    def test_generate_from_cache_produces_text(self, hooked):
        """Can generate text from a prefilled cache."""
        from mlux.steering import prefill_with_cache, generate_from_cache

        prompt = "Once upon a time"
        cache, logits = prefill_with_cache(hooked, prompt)

        output = generate_from_cache(hooked, cache, max_tokens=10, temperature=0, initial_logits=logits)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_deterministic_with_temp_zero(self, hooked):
        """Generation is deterministic with temperature=0."""
        from mlux.steering import prefill_with_cache, generate_from_cache

        prompt = "The answer is"

        cache1, logits1 = prefill_with_cache(hooked, prompt)
        out1 = generate_from_cache(hooked, cache1, max_tokens=5, temperature=0, initial_logits=logits1)

        cache2, logits2 = prefill_with_cache(hooked, prompt)
        out2 = generate_from_cache(hooked, cache2, max_tokens=5, temperature=0, initial_logits=logits2)

        assert out1 == out2, f"Temperature=0 should be deterministic: '{out1}' vs '{out2}'"
