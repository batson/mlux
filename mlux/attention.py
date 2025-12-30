"""
Attention pattern computation from cached Q/K.

Since we can't hook inside the attention computation, we compute
attention patterns post-hoc from cached Q and K projections.
"""

import mlx.core as mx
from typing import Optional


def compute_attention_patterns(
    q: mx.array,
    k: mx.array,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    softcap: Optional[float] = None,
) -> mx.array:
    """
    Compute attention patterns from Q and K projections.

    Args:
        q: Q projection output [batch, seq, n_heads * d_head]
        k: K projection output [batch, seq, n_kv_heads * d_head]
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads (for GQA). Defaults to n_heads.
        softcap: Attention logit softcapping value (e.g., 50 for Gemma 2)

    Returns:
        Attention patterns [batch, n_heads, seq, seq]
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads

    batch, seq_len, _ = q.shape
    d_head = q.shape[-1] // n_heads

    # Reshape Q: [batch, seq, n_heads, d_head] -> [batch, n_heads, seq, d_head]
    q = q.reshape(batch, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)

    # Reshape K: [batch, seq, n_kv_heads, d_head] -> [batch, n_kv_heads, seq, d_head]
    k = k.reshape(batch, seq_len, n_kv_heads, d_head).transpose(0, 2, 1, 3)

    # Handle GQA: repeat K heads to match Q heads
    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = mx.repeat(k, repeats, axis=1)

    # Compute attention scores: [batch, n_heads, seq, seq]
    scale = d_head ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Apply softcapping if specified (Gemma 2 style)
    if softcap is not None:
        scores = mx.tanh(scores / softcap) * softcap

    # Apply causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    scores = scores + mask

    # Softmax to get patterns
    patterns = mx.softmax(scores, axis=-1)

    return patterns


def get_attention_info(model) -> dict:
    """
    Extract attention configuration from a model.

    Returns dict with n_layers, n_heads, n_kv_heads, d_head, softcap, layer_prefix.
    """
    info = {
        "n_layers": None,
        "n_heads": None,
        "n_kv_heads": None,
        "d_head": None,
        "softcap": None,
        "layer_prefix": "model.layers",  # Default for Llama/Gemma/Qwen style
        "qkv_style": "separate",  # "separate" for q_proj/k_proj/v_proj, "combined" for c_attn
    }

    # Try standard structure: model.model.args (Llama, Gemma, Qwen)
    if hasattr(model, "model") and hasattr(model.model, "args"):
        args = model.model.args
        if hasattr(args, "num_hidden_layers"):
            info["n_layers"] = args.num_hidden_layers
        if hasattr(args, "num_attention_heads"):
            info["n_heads"] = args.num_attention_heads
        if hasattr(args, "num_key_value_heads"):
            info["n_kv_heads"] = args.num_key_value_heads
        if hasattr(args, "head_dim") and args.head_dim is not None:
            info["d_head"] = args.head_dim
        elif hasattr(args, "hidden_size") and info["n_heads"]:
            info["d_head"] = args.hidden_size // info["n_heads"]

        # Try to get softcap from attention layer
        if hasattr(model.model, "layers"):
            layer0 = model.model.layers[0]
            if hasattr(layer0, "self_attn"):
                attn = layer0.self_attn
                if hasattr(attn, "attn_logit_softcapping"):
                    info["softcap"] = attn.attn_logit_softcapping

    # Try GPT-2 style: model.args directly, model.h for layers
    elif hasattr(model, "args"):
        args = model.args
        if hasattr(args, "n_layer"):
            info["n_layers"] = args.n_layer
        if hasattr(args, "n_head"):
            info["n_heads"] = args.n_head
        if hasattr(args, "num_key_value_heads"):
            info["n_kv_heads"] = args.num_key_value_heads
        else:
            info["n_kv_heads"] = info["n_heads"]  # GPT-2 uses MHA, not GQA
        if hasattr(args, "n_embd") and info["n_heads"]:
            info["d_head"] = args.n_embd // info["n_heads"]

        # GPT-2 uses model.h for layers and combined c_attn
        info["layer_prefix"] = "model.h"
        info["qkv_style"] = "combined"

    return info


class AttentionPatternHelper:
    """
    Helper class to compute attention patterns from a HookedModel.

    Example:
        helper = AttentionPatternHelper(hooked_model)
        patterns = helper.get_patterns("Hello world", layers=[0, 5, 10])
        # patterns[0] is [batch, n_heads, seq, seq] for layer 0
    """

    def __init__(self, hooked_model):
        self.model = hooked_model
        self.info = get_attention_info(hooked_model.model)

    def get_patterns(
        self,
        input,
        layers: list[int],
    ) -> dict[int, mx.array]:
        """
        Compute attention patterns for specified layers.

        Args:
            input: String or token array
            layers: List of layer indices

        Returns:
            Dict mapping layer index -> attention pattern [batch, heads, seq, seq]
        """
        layer_prefix = self.info["layer_prefix"]
        qkv_style = self.info["qkv_style"]

        # Build hooks based on model style
        hooks = []
        for layer in layers:
            if qkv_style == "combined":
                # GPT-2 style: combined c_attn
                hooks.append(f"{layer_prefix}.{layer}.attn.c_attn")
            else:
                # Llama/Gemma/Qwen style: separate projections
                hooks.append(f"{layer_prefix}.{layer}.self_attn.q_proj")
                hooks.append(f"{layer_prefix}.{layer}.self_attn.k_proj")

        # Run with cache
        _, cache = self.model.run_with_cache(input, hooks=hooks)

        # Compute patterns
        patterns = {}
        for layer in layers:
            if qkv_style == "combined":
                # GPT-2: split combined QKV output
                qkv = cache[f"{layer_prefix}.{layer}.attn.c_attn"]
                # c_attn outputs [batch, seq, 3 * n_embd], split into Q, K, V
                n_embd = self.info["n_heads"] * self.info["d_head"]
                q = qkv[:, :, :n_embd]
                k = qkv[:, :, n_embd:2*n_embd]
                # v = qkv[:, :, 2*n_embd:]  # Not needed for patterns
            else:
                q = cache[f"{layer_prefix}.{layer}.self_attn.q_proj"]
                k = cache[f"{layer_prefix}.{layer}.self_attn.k_proj"]

            pattern = compute_attention_patterns(
                q, k,
                n_heads=self.info["n_heads"],
                n_kv_heads=self.info["n_kv_heads"],
                softcap=self.info["softcap"],
            )
            patterns[layer] = pattern

        return patterns
