"""
Advanced hook tests inspired by TransformerLens patterns.

These tests validate mlux's module wrapping approach:
- Hook execution counts
- Model restoration after operations
- Multiple hooks on same module
- Hook ordering behavior
- Error handling and cleanup

Run with: pytest tests/test_hooks_advanced.py -v
"""

import pytest
import mlx.core as mx

from mlux import HookedModel


MODEL_NAME = "mlx-community/gemma-2-2b-it-4bit"


@pytest.fixture(scope="module")
def hooked():
    """Shared HookedModel instance."""
    return HookedModel.from_pretrained(MODEL_NAME)


class Counter:
    """Simple counter to track hook invocations."""
    def __init__(self):
        self.count = 0
        self.captured_shapes = []

    def inc(self, inputs, output, wrapper):
        self.count += 1
        self.captured_shapes.append(output.shape)
        return output


class TestHookExecutionCount:
    """Test that hooks execute the expected number of times."""

    def test_hook_runs_once_per_forward(self, hooked):
        """A hook on a single module runs exactly once per forward pass."""
        counter = Counter()

        hooked.run_with_hooks(
            "Hello world",
            hooks=[("model.layers.0.mlp", counter.inc)]
        )

        assert counter.count == 1, f"Hook ran {counter.count} times, expected 1"

    def test_multiple_hooks_all_run(self, hooked):
        """Multiple hooks on different modules all execute."""
        counters = [Counter() for _ in range(3)]

        hooks = [
            ("model.layers.0.mlp", counters[0].inc),
            ("model.layers.5.mlp", counters[1].inc),
            ("model.layers.10.mlp", counters[2].inc),
        ]

        hooked.run_with_hooks("Test", hooks=hooks)

        for i, c in enumerate(counters):
            assert c.count == 1, f"Hook {i} ran {c.count} times, expected 1"

    def test_hook_runs_on_each_call(self, hooked):
        """Hook count accumulates across multiple forward passes."""
        counter = Counter()

        for i in range(3):
            hooked.run_with_hooks(
                f"Test {i}",
                hooks=[("model.layers.0.mlp", counter.inc)]
            )

        assert counter.count == 3, f"Hook ran {counter.count} times, expected 3"


class TestModelRestoration:
    """Test that model is properly restored after hook operations."""

    def test_model_unchanged_after_run_with_hooks(self, hooked):
        """Model produces same output before and after run_with_hooks."""
        tokens = hooked._tokenize("Test prompt")

        # Baseline before hooks
        baseline1 = hooked.model(tokens)
        mx.eval(baseline1)

        # Run with hooks
        def modify_output(inputs, output, wrapper):
            return output * 2.0  # Modify output

        hooked.run_with_hooks(tokens, hooks=[("model.layers.5.mlp", modify_output)])

        # Baseline after hooks
        baseline2 = hooked.model(tokens)
        mx.eval(baseline2)

        diff = mx.max(mx.abs(baseline1 - baseline2)).item()
        assert diff < 1e-6, f"Model changed after run_with_hooks: diff={diff}"

    def test_model_unchanged_after_run_with_cache(self, hooked):
        """Model produces same output before and after run_with_cache."""
        tokens = hooked._tokenize("Test prompt")

        baseline1 = hooked.model(tokens)
        mx.eval(baseline1)

        # Run with cache
        hooked.run_with_cache(tokens, hooks=["model.layers.0.mlp", "model.layers.5.mlp"])

        baseline2 = hooked.model(tokens)
        mx.eval(baseline2)

        diff = mx.max(mx.abs(baseline1 - baseline2)).item()
        assert diff < 1e-6, f"Model changed after run_with_cache: diff={diff}"

    def test_sequential_operations_dont_leak(self, hooked):
        """Multiple sequential operations don't affect each other."""
        tokens = hooked._tokenize("The capital of France is")

        # Operation 1: cache
        _, cache1 = hooked.run_with_cache(tokens, hooks=["model.layers.0.mlp"])

        # Operation 2: hooks with modification
        def zero_out(inputs, output, wrapper):
            return mx.zeros_like(output)

        hooked.run_with_hooks(tokens, hooks=[("model.layers.10.mlp", zero_out)])

        # Operation 3: cache again - should match operation 1
        _, cache2 = hooked.run_with_cache(tokens, hooks=["model.layers.0.mlp"])

        val1 = cache1["model.layers.0.mlp"]
        val2 = cache2["model.layers.0.mlp"]
        mx.eval(val1, val2)

        diff = mx.max(mx.abs(val1 - val2)).item()
        assert diff < 1e-6, f"Cache values differ after intervening operation: diff={diff}"


class TestMultipleHooksOnSameModule:
    """Test behavior when multiple hooks target the same module.

    mlux warns when duplicate paths are provided and only runs the last hook
    for each path. Users should combine operations into a single hook function.
    """

    def test_duplicate_hooks_warns(self, hooked):
        """Duplicate hook paths trigger a warning."""
        import warnings

        hooks = [
            ("model.layers.5.mlp", lambda i, o, w: o),
            ("model.layers.5.mlp", lambda i, o, w: o),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hooked.run_with_hooks("Test", hooks=hooks)

            # Should have exactly one warning about duplicates
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert "Duplicate paths" in str(w[0].message)
            assert "model.layers.5.mlp" in str(w[0].message)

    def test_later_hook_overwrites_earlier(self, hooked):
        """When two hooks target the same path, later one wins."""
        import warnings

        counters = [Counter(), Counter()]

        hooks = [
            ("model.layers.5.mlp", counters[0].inc),
            ("model.layers.5.mlp", counters[1].inc),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress the expected warning
            hooked.run_with_hooks("Test", hooks=hooks)

        # Due to dict(hooks), only the second hook runs
        assert counters[0].count == 0, "First hook should not run (overwritten)"
        assert counters[1].count == 1, "Second hook should run"

    def test_different_modules_both_run(self, hooked):
        """Hooks on different modules both execute correctly."""
        counters = [Counter(), Counter()]

        hooks = [
            ("model.layers.5.mlp", counters[0].inc),
            ("model.layers.10.mlp", counters[1].inc),
        ]

        hooked.run_with_hooks("Test", hooks=hooks)

        assert counters[0].count == 1, "First hook should run"
        assert counters[1].count == 1, "Second hook should run"

    def test_no_warning_for_unique_paths(self, hooked):
        """No warning when all paths are unique."""
        import warnings

        hooks = [
            ("model.layers.5.mlp", lambda i, o, w: o),
            ("model.layers.10.mlp", lambda i, o, w: o),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hooked.run_with_hooks("Test", hooks=hooks)

            # Filter to only UserWarnings about duplicates
            dup_warnings = [x for x in w if "Duplicate" in str(x.message)]
            assert len(dup_warnings) == 0, "Should not warn for unique paths"


class TestPreHookBehavior:
    """Test pre-hooks that modify inputs before module execution."""

    def test_pre_hook_captures_input(self, hooked):
        """Pre-hook can capture the input to a module."""
        captured = {}

        def capture_input(args, kwargs, wrapper):
            captured["input_shape"] = args[0].shape
            return args, kwargs

        hooked.run_with_hooks(
            "Hello world",
            pre_hooks=[("model.layers.5.self_attn.o_proj", capture_input)]
        )

        assert "input_shape" in captured
        # o_proj input should have shape [batch, seq, n_heads * d_head]
        assert len(captured["input_shape"]) == 3

    def test_pre_hook_can_modify_input(self, hooked):
        """Pre-hook can modify input, affecting module output."""
        tokens = hooked._tokenize("Test")

        baseline = hooked.forward(tokens)
        mx.eval(baseline)

        def zero_input(args, kwargs, wrapper):
            return (mx.zeros_like(args[0]),), kwargs

        modified = hooked.run_with_hooks(
            tokens,
            pre_hooks=[("model.layers.10.self_attn.o_proj", zero_input)]
        )
        mx.eval(modified)

        diff = mx.max(mx.abs(baseline - modified)).item()
        assert diff > 0.01, f"Pre-hook should change output: diff={diff}"


class TestCacheOperations:
    """Test cache-related functionality."""

    def test_cache_matches_hook_output(self, hooked):
        """Cached value matches what a hook would capture."""
        captured = {}

        def capture(inputs, output, wrapper):
            captured["output"] = output
            return output

        tokens = hooked._tokenize("Test")

        # Run with cache
        _, cache = hooked.run_with_cache(tokens, hooks=["model.layers.5.mlp"])

        # Run with hook to capture
        hooked.run_with_hooks(tokens, hooks=[("model.layers.5.mlp", capture)])

        cached_val = cache["model.layers.5.mlp"]
        hooked_val = captured["output"]
        mx.eval(cached_val, hooked_val)

        diff = mx.max(mx.abs(cached_val - hooked_val)).item()
        assert diff < 1e-6, f"Cache doesn't match hook capture: diff={diff}"

    def test_cache_multiple_layers(self, hooked):
        """Can cache multiple layers at once."""
        layers = [0, 5, 10, 15, 20]
        hooks = [f"model.layers.{i}.mlp" for i in layers]

        _, cache = hooked.run_with_cache("Test", hooks=hooks)

        assert len(cache) == len(layers), f"Expected {len(layers)} cached items"
        for hook in hooks:
            assert hook in cache, f"Missing {hook} in cache"

    def test_cache_with_predicate(self, hooked):
        """Can cache using predicate function."""
        _, cache = hooked.run_with_cache(
            "Test",
            hooks=lambda p: p.endswith(".mlp") and "layers.0" in p
        )

        assert len(cache) == 1
        assert "model.layers.0.mlp" in cache


class TestHookWithGeneration:
    """Test hooks work correctly during token generation."""

    def test_hook_during_prefill(self, hooked):
        """Hooks work during prefill phase of generation."""
        from mlux.steering import prefill_with_cache, generate_from_cache

        counter = Counter()

        def count_hook(inputs, output, wrapper):
            counter.count += 1
            return output

        cache, logits = prefill_with_cache(
            hooked, "Hello",
            hooks=[("model.layers.10", count_hook)]
        )

        assert counter.count == 1, "Hook should run once during prefill"
        assert cache is not None
        assert logits is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_hooks_list(self, hooked):
        """Empty hooks list produces unmodified output."""
        tokens = hooked._tokenize("Test")

        baseline = hooked.forward(tokens)
        with_empty = hooked.run_with_hooks(tokens, hooks=[])
        mx.eval(baseline, with_empty)

        diff = mx.max(mx.abs(baseline - with_empty)).item()
        assert diff < 1e-6, "Empty hooks should not change output"

    def test_hook_returning_none_keeps_original(self, hooked):
        """Hook returning None keeps original output."""
        def return_none(inputs, output, wrapper):
            return None  # Should keep original

        tokens = hooked._tokenize("Test")

        baseline = hooked.forward(tokens)
        with_none_hook = hooked.run_with_hooks(
            tokens, hooks=[("model.layers.5.mlp", return_none)]
        )
        mx.eval(baseline, with_none_hook)

        diff = mx.max(mx.abs(baseline - with_none_hook)).item()
        assert diff < 1e-6, "None-returning hook should not change output"

    def test_long_sequence(self, hooked):
        """Hooks work with longer sequences."""
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 20

        counter = Counter()
        hooked.run_with_hooks(
            long_prompt,
            hooks=[("model.layers.0.mlp", counter.inc)]
        )

        assert counter.count == 1
        # Check captured shape has longer sequence dimension
        assert counter.captured_shapes[0][1] > 50, "Sequence should be long"
