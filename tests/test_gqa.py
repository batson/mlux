"""
Grouped Query Attention (GQA) tests.

These tests validate that GQA works correctly:
- Attention patterns have full n_heads (not n_kv_heads)
- K/V heads are properly repeated for each query head group
- GQA vs MHA model detection
- Head patterns are consistent

Run with: pytest tests/test_gqa.py -v
"""

import pytest
import mlx.core as mx

from mlux import HookedModel


# Models with different attention configurations
GQA_MODEL = "mlx-community/gemma-2-2b-it-4bit"  # n_heads=8, n_kv_heads=4
MHA_MODEL = "mlx-community/gpt2-base-mlx"  # n_heads=12, n_kv_heads=12


@pytest.fixture(scope="module")
def gqa_model():
    """GQA model (Gemma - has fewer KV heads than query heads)."""
    return HookedModel.from_pretrained(GQA_MODEL)


@pytest.fixture(scope="module")
def mha_model():
    """MHA model (GPT-2 - same number of KV heads and query heads)."""
    return HookedModel.from_pretrained(MHA_MODEL)


class TestGQADetection:
    """Test that GQA vs MHA is correctly detected."""

    def test_gemma_is_gqa(self, gqa_model):
        """Gemma is correctly identified as GQA (n_kv_heads < n_heads)."""
        config = gqa_model.config
        assert config["n_kv_heads"] < config["n_heads"], \
            f"Gemma should be GQA: n_kv_heads={config['n_kv_heads']}, n_heads={config['n_heads']}"

    def test_gpt2_is_mha(self, mha_model):
        """GPT-2 is correctly identified as MHA (n_kv_heads == n_heads)."""
        config = mha_model.config
        assert config["n_kv_heads"] == config["n_heads"], \
            f"GPT-2 should be MHA: n_kv_heads={config['n_kv_heads']}, n_heads={config['n_heads']}"

    def test_gemma_kv_ratio(self, gqa_model):
        """Gemma has correct KV head ratio."""
        config = gqa_model.config
        ratio = config["n_heads"] // config["n_kv_heads"]
        assert ratio == 2, f"Gemma 2-2B should have 2 query heads per KV head, got {ratio}"


class TestAttentionPatternShapes:
    """Test that attention patterns have correct shapes."""

    def test_gqa_attention_has_full_heads(self, gqa_model):
        """GQA attention patterns have n_heads (not n_kv_heads) in head dimension."""
        config = gqa_model.config
        patterns = gqa_model.get_attention_patterns("Hello world", layers=[0, 10, 20])

        for layer, pattern in patterns.items():
            mx.eval(pattern)
            # Shape should be [batch, n_heads, seq, seq]
            assert pattern.shape[1] == config["n_heads"], \
                f"Layer {layer}: expected {config['n_heads']} heads, got {pattern.shape[1]}"

    def test_mha_attention_has_all_heads(self, mha_model):
        """MHA attention patterns have n_heads in head dimension."""
        config = mha_model.config
        patterns = mha_model.get_attention_patterns("Hello world", layers=[0, 5, 11])

        for layer, pattern in patterns.items():
            mx.eval(pattern)
            assert pattern.shape[1] == config["n_heads"], \
                f"Layer {layer}: expected {config['n_heads']} heads, got {pattern.shape[1]}"

    def test_attention_pattern_is_square(self, gqa_model):
        """Attention patterns are square (seq x seq)."""
        patterns = gqa_model.get_attention_patterns("The quick brown fox", layers=[5])

        for layer, pattern in patterns.items():
            mx.eval(pattern)
            assert pattern.shape[2] == pattern.shape[3], \
                f"Pattern should be square: got {pattern.shape[2]} x {pattern.shape[3]}"


class TestGQAHeadExpansion:
    """Test that K/V heads are correctly expanded for GQA."""

    def test_q_and_kv_projections_different_dims(self, gqa_model):
        """Q projection has more dimensions than K/V projections in GQA."""
        config = gqa_model.config

        _, cache = gqa_model.run_with_cache("Hello", hooks=[
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ])

        q_out = cache["model.layers.0.self_attn.q_proj"]
        k_out = cache["model.layers.0.self_attn.k_proj"]
        v_out = cache["model.layers.0.self_attn.v_proj"]
        mx.eval(q_out, k_out, v_out)

        # Q should have n_heads * d_head dimensions
        expected_q_dim = config["n_heads"] * config["d_head"]
        # K/V should have n_kv_heads * d_head dimensions
        expected_kv_dim = config["n_kv_heads"] * config["d_head"]

        assert q_out.shape[-1] == expected_q_dim, \
            f"Q dim: expected {expected_q_dim}, got {q_out.shape[-1]}"
        assert k_out.shape[-1] == expected_kv_dim, \
            f"K dim: expected {expected_kv_dim}, got {k_out.shape[-1]}"
        assert v_out.shape[-1] == expected_kv_dim, \
            f"V dim: expected {expected_kv_dim}, got {v_out.shape[-1]}"

    def test_mha_q_and_kv_same_dims(self, mha_model):
        """MHA model has same dimensions for Q, K, V projections."""
        config = mha_model.config

        _, cache = mha_model.run_with_cache("Hello", hooks=[
            "model.h.0.attn.c_attn",  # GPT-2 uses combined QKV projection
        ])

        qkv_out = cache["model.h.0.attn.c_attn"]
        mx.eval(qkv_out)

        # Combined QKV should have 3 * n_heads * d_head dimensions
        expected_dim = 3 * config["n_heads"] * config["d_head"]
        assert qkv_out.shape[-1] == expected_dim, \
            f"QKV dim: expected {expected_dim}, got {qkv_out.shape[-1]}"


class TestGQAConsistency:
    """Test consistency properties of GQA attention."""

    def test_attention_rows_sum_to_one(self, gqa_model):
        """Each attention row (query) sums to 1 (valid softmax)."""
        patterns = gqa_model.get_attention_patterns("Hello world", layers=[0, 10])

        for layer, pattern in patterns.items():
            mx.eval(pattern)
            # Sum over key dimension (last axis)
            row_sums = mx.sum(pattern, axis=-1)
            mx.eval(row_sums)

            # All rows should sum to 1
            ones = mx.ones_like(row_sums)
            diff = mx.max(mx.abs(row_sums - ones)).item()
            assert diff < 0.01, f"Layer {layer}: rows don't sum to 1, max diff={diff}"

    def test_attention_is_causal(self, gqa_model):
        """Attention is causal (no attending to future tokens)."""
        patterns = gqa_model.get_attention_patterns("The quick brown fox", layers=[5])

        pattern = patterns[5]
        mx.eval(pattern)

        # Check upper triangle is zero (no future attention)
        seq_len = pattern.shape[2]
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # pattern[batch, head, query_pos, key_pos]
                # For query at position i, attention to position j>i should be ~0
                future_attn = pattern[0, :, i, j]
                mx.eval(future_attn)
                max_future = mx.max(mx.abs(future_attn)).item()
                assert max_future < 0.01, \
                    f"Position {i} attends to future position {j}: {max_future}"

    def test_all_heads_have_different_patterns(self, gqa_model):
        """Different attention heads have different patterns (learned distinct roles)."""
        patterns = gqa_model.get_attention_patterns(
            "The quick brown fox jumps over the lazy dog",
            layers=[10]
        )

        pattern = patterns[10]  # [batch, heads, seq, seq]
        mx.eval(pattern)

        n_heads = pattern.shape[1]

        # Compare pairs of heads - they shouldn't be identical
        identical_pairs = 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                head_i = pattern[0, i, :, :]
                head_j = pattern[0, j, :, :]
                diff = mx.max(mx.abs(head_i - head_j)).item()
                if diff < 0.001:
                    identical_pairs += 1

        # Allow some similar heads but not all
        max_identical = n_heads // 2
        assert identical_pairs <= max_identical, \
            f"Too many identical head pairs: {identical_pairs}"


class TestGQAWithHooks:
    """Test hooks work correctly with GQA models."""

    def test_can_hook_qkv_separately(self, gqa_model):
        """Can hook Q, K, V projections separately in GQA model."""
        hooks = [
            "model.layers.5.self_attn.q_proj",
            "model.layers.5.self_attn.k_proj",
            "model.layers.5.self_attn.v_proj",
        ]

        _, cache = gqa_model.run_with_cache("Test", hooks=hooks)

        assert len(cache) == 3, f"Expected 3 cached values, got {len(cache)}"
        for hook in hooks:
            assert hook in cache, f"Missing {hook} in cache"

    def test_can_modify_attention_input(self, gqa_model):
        """Can modify attention input via pre-hook."""
        config = gqa_model.config
        d_head = config["d_head"]

        tokens = gqa_model._tokenize("Test prompt")

        baseline = gqa_model.forward(tokens)
        mx.eval(baseline)

        # Zero out first head's contribution
        def zero_first_head(args, kwargs, wrapper):
            x = args[0]  # [batch, seq, n_heads * d_head]
            zeroed = mx.concatenate([
                mx.zeros_like(x[:, :, :d_head]),
                x[:, :, d_head:]
            ], axis=-1)
            return (zeroed,), kwargs

        modified = gqa_model.run_with_hooks(
            tokens,
            pre_hooks=[("model.layers.10.self_attn.o_proj", zero_first_head)]
        )
        mx.eval(modified)

        diff = mx.max(mx.abs(baseline - modified)).item()
        assert diff > 0.01, f"Zeroing head should change output: diff={diff}"
