#!/usr/bin/env python3
"""
Validation tests for head patching mechanism.

Tests that the pre-hook infrastructure correctly:
1. Preserves output when self-replacing (no-op)
2. Changes output when zeroing heads
3. Changes output when injecting noise
4. Changes output when transplanting from different context

Key insight from experiments:
- Attention patterns alone DON'T identify causally important heads
- Effect-based identification (measure logit change when ablating) is required
- Heads with high attention may actually SUPPRESS the prediction

This file includes:
1. Basic patching validation tests
2. Effect-based head identification (correct method)
3. GPT-2 induction head validation
"""

import sys
sys.path.insert(0, '.')

import mlx.core as mx
from mlux import HookedModel
from mlux.attention import get_attention_info


def make_tokens(hooked, letters):
    """Create token array from letters."""
    return mx.array([[hooked.tokenizer.encode(f" {l}")[0] for l in letters]])


def get_attention_patterns(hooked, tokens, layer):
    """Compute attention patterns for a layer."""
    info = get_attention_info(hooked.model)
    n_heads = info["n_heads"]
    n_kv_heads = info.get("n_kv_heads", n_heads)
    d_head = info["d_head"]
    softcap = info.get("softcap")

    q_path = f"model.layers.{layer}.self_attn.q_proj"
    k_path = f"model.layers.{layer}.self_attn.k_proj"

    captured = {}

    def capture_q(inputs, output, wrapper):
        captured["q"] = output
        return output

    def capture_k(inputs, output, wrapper):
        captured["k"] = output
        return output

    hooked.run_with_hooks(tokens, hooks=[(q_path, capture_q), (k_path, capture_k)])

    seq = tokens.shape[1]
    q = captured["q"].reshape(-1, seq, n_heads, d_head)
    k = captured["k"].reshape(-1, seq, n_kv_heads, d_head)
    k_expanded = mx.repeat(k, n_heads // n_kv_heads, axis=2)

    scores = mx.einsum("bihd,bjhd->bhij", q, k_expanded) / (d_head ** 0.5)
    if softcap:
        scores = softcap * mx.tanh(scores / softcap)

    mask = mx.triu(mx.full((seq, seq), float('-inf')), k=1)
    attn = mx.softmax(scores + mask, axis=-1)
    mx.eval(attn)
    return attn[0]  # [n_heads, seq, seq]


def find_induction_heads(hooked, n_layers):
    """Find heads that attend to earlier occurrences of repeated tokens."""
    info = get_attention_info(hooked.model)
    n_heads = info["n_heads"]

    # Use pattern with exact token repetition
    tokens = make_tokens(hooked, ["A", "B", "C", "A"])
    token_ids = [int(t) for t in tokens[0]]

    last_pos = len(token_ids) - 1
    earlier_pos = [i for i in range(last_pos) if token_ids[i] == token_ids[last_pos]]

    if not earlier_pos:
        return []

    induction_scores = {}
    for layer in range(n_layers):
        attn = get_attention_patterns(hooked, tokens, layer)
        last_attn = attn[:, last_pos, :]

        for h in range(n_heads):
            score = sum(float(last_attn[h, pos].item()) for pos in earlier_pos)
            induction_scores[(layer, h)] = score

    sorted_heads = sorted(induction_scores.items(), key=lambda x: x[1], reverse=True)
    return [(l, h, s) for (l, h), s in sorted_heads if s > 0.1][:10]


def get_head_activation(hooked, tokens, layer, head):
    """Get specific head's activation at last token position."""
    info = get_attention_info(hooked.model)
    n_heads = info["n_heads"]
    d_head = info["d_head"]

    o_proj_path = f"model.layers.{layer}.self_attn.o_proj"
    captured = {}

    def hook(args, kwargs, wrapper):
        captured["act"] = args[0][:, -1, :].reshape(n_heads, d_head)[head]
        return args, kwargs

    hooked.run_with_hooks(tokens, pre_hooks=[(o_proj_path, hook)])
    mx.eval(captured["act"])
    return captured["act"]


def patch_head_and_run(hooked, tokens, layer, head, patch_value):
    """Patch a specific head at last token and run forward pass."""
    info = get_attention_info(hooked.model)
    n_heads = info["n_heads"]
    d_head = info["d_head"]

    o_proj_path = f"model.layers.{layer}.self_attn.o_proj"

    def hook(args, kwargs, wrapper):
        x = args[0]
        b, s, dim = x.shape
        x_flat = x.reshape(b, s, n_heads, d_head)

        parts = []
        for h in range(n_heads):
            head_act = x_flat[:, :, h, :]
            if h == head:
                new_head = mx.concatenate([
                    head_act[:, :-1, :],
                    patch_value.reshape(1, 1, d_head)
                ], axis=1)
                parts.append(new_head)
            else:
                parts.append(head_act)

        result = mx.stack(parts, axis=2)
        return (result.reshape(b, s, dim),), kwargs

    output = hooked.run_with_hooks(tokens, pre_hooks=[(o_proj_path, hook)])
    mx.eval(output)
    return output


def test_head_patching(model_name="mlx-community/gemma-2-2b-it-4bit"):
    """Run validation tests for head patching."""
    print("="*70)
    print("Head Patching Validation Tests")
    print("="*70)

    hooked = HookedModel.from_pretrained(model_name)
    info = get_attention_info(hooked.model)
    n_layers = info["n_layers"]
    d_head = info["d_head"]

    print(f"\nModel: {model_name}")
    print(f"Architecture: {n_layers} layers, {info['n_heads']} heads, d_head={d_head}")

    # Find induction heads
    print("\n" + "-"*70)
    print("Finding Induction Heads")
    print("-"*70)

    induction_heads = find_induction_heads(hooked, n_layers)
    print(f"Top induction heads (attention to earlier same-token):")
    for l, h, s in induction_heads[:5]:
        print(f"  L{l}H{h}: {s:.3f}")

    # Run patching validation
    print("\n" + "-"*70)
    print("Patching Validation Tests")
    print("-"*70)

    context = make_tokens(hooked, ["A", "B", "C", "D", "A"])
    baseline_out = hooked.model(context)
    mx.eval(baseline_out)
    baseline_probs = mx.softmax(baseline_out[0, -1, :], axis=-1)

    # Use top induction head for testing
    if induction_heads:
        test_layer, test_head, _ = induction_heads[0]
    else:
        test_layer, test_head = 20, 0

    print(f"\nTesting on L{test_layer}H{test_head}")

    results = {}

    # Test 1: Self-replacement
    orig_act = get_head_activation(hooked, context, test_layer, test_head)
    self_out = patch_head_and_run(hooked, context, test_layer, test_head, orig_act)
    self_probs = mx.softmax(self_out[0, -1, :], axis=-1)
    self_diff = float(mx.max(mx.abs(baseline_probs - self_probs)).item())
    results["self_replacement"] = self_diff
    print(f"  Self-replacement: diff={self_diff:.8f} {'PASS' if self_diff < 0.0001 else 'FAIL'}")

    # Test 2: Zero ablation
    zero_out = patch_head_and_run(hooked, context, test_layer, test_head, mx.zeros((d_head,)))
    zero_probs = mx.softmax(zero_out[0, -1, :], axis=-1)
    zero_diff = float(mx.max(mx.abs(baseline_probs - zero_probs)).item())
    results["zero_ablation"] = zero_diff
    print(f"  Zero ablation: diff={zero_diff:.4f} {'PASS' if zero_diff > 0.001 else 'FAIL'}")

    # Test 3: Random noise
    mx.random.seed(42)
    noise = mx.random.normal((d_head,)) * 10
    noise_out = patch_head_and_run(hooked, context, test_layer, test_head, noise)
    noise_probs = mx.softmax(noise_out[0, -1, :], axis=-1)
    noise_diff = float(mx.max(mx.abs(baseline_probs - noise_probs)).item())
    results["random_noise"] = noise_diff
    print(f"  Random noise: diff={noise_diff:.4f} {'PASS' if noise_diff > 0.01 else 'FAIL'}")

    # Test 4: Cross-context transplant
    ctx2 = make_tokens(hooked, ["X", "Y", "Z", "W", "X"])
    ctx2_act = get_head_activation(hooked, ctx2, test_layer, test_head)
    cross_out = patch_head_and_run(hooked, context, test_layer, test_head, ctx2_act)
    cross_probs = mx.softmax(cross_out[0, -1, :], axis=-1)
    cross_diff = float(mx.max(mx.abs(baseline_probs - cross_probs)).item())
    results["cross_context"] = cross_diff
    print(f"  Cross-context: diff={cross_diff:.4f} {'PASS' if cross_diff > 0.001 else 'FAIL'}")

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    all_pass = (
        results["self_replacement"] < 0.0001 and
        results["zero_ablation"] > 0.001 and
        results["random_noise"] > 0.01 and
        results["cross_context"] > 0.001
    )

    if all_pass:
        print("\n✓ ALL TESTS PASSED - Head patching mechanism validated!")
    else:
        print("\n✗ Some tests failed - check implementation")

    return results, induction_heads


def test_gpt2_induction_validation():
    """
    Validate head patching using GPT-2 with probability-space metrics.

    Uses sequence "J A M P R X J A M P R X" where:
    - Model should predict X after second R (position 10 -> 11)
    - We measure three metrics per head:
      1. Attention score (how much head attends to first X)
      2. Direct effect (head output dotted with ∇log P(X))
      3. Ablation effect (ΔP(X) when head is removed)

    Key finding: These three metrics identify DIFFERENT heads as important.
    Only ~1 head typically appears in all three top-10 lists.
    """
    print("="*70)
    print("GPT-2 Induction Head Validation (Probability-Space Metrics)")
    print("="*70)

    hooked = HookedModel.from_pretrained("gpt2")
    info = get_attention_info(hooked.model)
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    d_head = info["d_head"]

    print(f"\nModel: GPT-2 ({n_layers} layers, {n_heads} heads, d_head={d_head})")

    # Create repeated sequence: J A M P R X J A M P R X
    letters = ["J", "A", "M", "P", "R", "X", "J", "A", "M", "P", "R", "X"]
    tokens = mx.array([[hooked.tokenizer.encode(f" {l}")[0] for l in letters]])
    x_token_id = int(tokens[0, 5].item())
    seq_len = 11  # Up to second R

    print(f"Sequence: {' '.join(letters[:seq_len])}")
    print(f"Target: predict X (token {x_token_id}) at position {seq_len}")

    # Get model components for direct effect computation
    model = hooked.model.model
    wte = model.wte.weight  # Embedding/unembedding matrix

    # Baseline prediction
    baseline_out = hooked.model(tokens[:, :seq_len])
    mx.eval(baseline_out)
    baseline_logits = baseline_out[0, -1, :]
    baseline_probs = mx.softmax(baseline_logits, axis=-1)
    mx.eval(baseline_probs)
    baseline_p_x = float(baseline_probs[x_token_id].item())

    print(f"\nBaseline P(' X') = {baseline_p_x:.4f}")

    # Gradient of log P(X) w.r.t. logits: ∂log P(X)/∂logit = e_X - P
    grad_logprob = -baseline_probs.astype(mx.float32)
    grad_logprob = grad_logprob.at[x_token_id].add(1.0)
    mx.eval(grad_logprob)

    # =========================================================================
    # Metric 1: Attention scores (R position attending to first X)
    # =========================================================================
    print("\n" + "-"*70)
    print("Computing attention patterns...")
    attention_scores = {}

    for layer in range(n_layers):
        q_path = f"model.h.{layer}.attn.c_attn"
        captured = {}

        def make_capture():
            def hook(inputs, output, wrapper):
                captured["qkv"] = output
                return output
            return hook

        hooked.run_with_hooks(tokens[:, :seq_len], hooks=[(q_path, make_capture())])

        qkv = captured["qkv"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(1, seq_len, n_heads, d_head)
        k = k.reshape(1, seq_len, n_heads, d_head)
        scores = mx.einsum("bihd,bjhd->bhij", q, k) / (d_head ** 0.5)
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        attn = mx.softmax(scores + mask, axis=-1)
        mx.eval(attn)

        for h in range(n_heads):
            # Attention from pos 10 (second R) to pos 5 (first X)
            attention_scores[(layer, h)] = float(attn[0, h, 10, 5].item())

    # =========================================================================
    # Metric 2: Direct effect (head output · ∇log P(X))
    # =========================================================================
    print("Computing direct effects...")
    direct_effects = {}

    for layer in range(n_layers):
        c_proj_path = f"model.h.{layer}.attn.c_proj"
        c_proj_weight = model.h[layer].attn.c_proj.weight

        for head in range(n_heads):
            captured = {}

            def make_capture(target_h):
                def hook(args, kwargs, wrapper):
                    x = args[0]
                    batch, seq, dim = x.shape
                    x_flat = x.reshape(batch, seq, n_heads, d_head)
                    captured["head_out"] = x_flat[:, :, target_h, :]
                    return args, kwargs
                return hook

            hooked.run_with_hooks(tokens[:, :seq_len],
                                  pre_hooks=[(c_proj_path, make_capture(head))])

            head_out = captured["head_out"][0, -1, :]  # [d_head]
            mx.eval(head_out)

            # Project through c_proj slice and unembed
            head_proj = c_proj_weight[:, head * d_head:(head + 1) * d_head]
            head_contribution = head_out @ head_proj.T  # [d_model]
            head_logits = head_contribution @ wte.T  # [vocab]
            mx.eval(head_logits)

            # Direct effect = dot with log prob gradient
            effect = float(mx.sum(head_logits * grad_logprob).item())
            direct_effects[(layer, head)] = effect

    # =========================================================================
    # Metric 3: Ablation effect (ΔP(X) when head removed)
    # =========================================================================
    print("Computing ablation effects...")
    ablation_effects = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            c_proj_path = f"model.h.{layer}.attn.c_proj"

            def make_ablate(target_h):
                def hook(args, kwargs, wrapper):
                    x = args[0]
                    batch, seq, dim = x.shape
                    x_flat = x.reshape(batch, seq, n_heads, d_head)
                    parts = []
                    for hh in range(n_heads):
                        if hh == target_h:
                            parts.append(mx.zeros((batch, seq, d_head)))
                        else:
                            parts.append(x_flat[:, :, hh, :])
                    return (mx.stack(parts, axis=2).reshape(batch, seq, dim),), kwargs
                return hook

            ablated = hooked.run_with_hooks(tokens[:, :seq_len],
                                            pre_hooks=[(c_proj_path, make_ablate(head))])
            mx.eval(ablated)
            ablated_probs = mx.softmax(ablated[0, -1, :], axis=-1)
            mx.eval(ablated_probs)
            ablated_p_x = float(ablated_probs[x_token_id].item())

            # Positive = head supports X (ablating hurts)
            ablation_effects[(layer, head)] = baseline_p_x - ablated_p_x

    # =========================================================================
    # Results
    # =========================================================================
    by_attention = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    by_direct = sorted(direct_effects.items(), key=lambda x: x[1], reverse=True)[:10]
    by_ablation = sorted(ablation_effects.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n" + "="*70)
    print("TOP 10 BY ATTENTION (R → X)")
    print("="*70)
    print(f"{'Head':<8} {'Attn':>8} {'Direct':>10} {'Ablation':>10}")
    print("-"*40)
    for (l, h), attn in by_attention:
        direct = direct_effects[(l, h)]
        ablation = ablation_effects[(l, h)]
        print(f"L{l:2d}H{h:2d}   {attn:>7.1%}   {direct:>+9.2f}   {ablation:>+9.4f}")

    print("\n" + "="*70)
    print("TOP 10 BY DIRECT EFFECT (head output · ∇log P(X))")
    print("="*70)
    print(f"{'Head':<8} {'Direct':>10} {'Attn':>8} {'Ablation':>10}")
    print("-"*40)
    for (l, h), direct in by_direct:
        attn = attention_scores[(l, h)]
        ablation = ablation_effects[(l, h)]
        print(f"L{l:2d}H{h:2d}   {direct:>+9.2f}   {attn:>7.1%}   {ablation:>+9.4f}")

    print("\n" + "="*70)
    print("TOP 10 BY ABLATION EFFECT (ΔP(X) when removed)")
    print("="*70)
    print(f"{'Head':<8} {'Ablation':>10} {'Attn':>8} {'Direct':>10}")
    print("-"*40)
    for (l, h), ablation in by_ablation:
        attn = attention_scores[(l, h)]
        direct = direct_effects[(l, h)]
        print(f"L{l:2d}H{h:2d}   {ablation:>+9.4f}   {attn:>7.1%}   {direct:>+9.2f}")

    # Overlap analysis
    attn_set = set(x[0] for x in by_attention)
    direct_set = set(x[0] for x in by_direct)
    ablation_set = set(x[0] for x in by_ablation)

    print("\n" + "="*70)
    print("OVERLAP ANALYSIS")
    print("="*70)
    print(f"Attention ∩ Direct: {len(attn_set & direct_set)} heads")
    print(f"Attention ∩ Ablation: {len(attn_set & ablation_set)} heads")
    print(f"Direct ∩ Ablation: {len(direct_set & ablation_set)} heads")
    all_three = attn_set & direct_set & ablation_set
    print(f"All three: {len(all_three)} heads - {all_three if all_three else 'none'}")

    # Validation: ablate top 5 by ablation effect
    print("\n" + "-"*70)
    print("Validation: Ablating top 5 heads by ablation effect")
    print("-"*70)

    top_5 = [(l, h) for (l, h), _ in by_ablation[:5]]
    print(f"Ablating: {top_5}")

    hooks = []
    for layer, head in top_5:
        c_proj_path = f"model.h.{layer}.attn.c_proj"

        def make_ablate(target_l, target_h):
            def hook(args, kwargs, wrapper):
                x = args[0]
                batch, seq, dim = x.shape
                x_flat = x.reshape(batch, seq, n_heads, d_head)
                parts = []
                for hh in range(n_heads):
                    if hh == target_h:
                        parts.append(mx.zeros((batch, seq, d_head)))
                    else:
                        parts.append(x_flat[:, :, hh, :])
                return (mx.stack(parts, axis=2).reshape(batch, seq, dim),), kwargs
            return hook

        hooks.append((c_proj_path, make_ablate(layer, head)))

    ablated = hooked.run_with_hooks(tokens[:, :seq_len], pre_hooks=hooks)
    mx.eval(ablated)
    final_probs = mx.softmax(ablated[0, -1, :], axis=-1)
    final_p_x = float(final_probs[x_token_id].item())

    reduction = baseline_p_x - final_p_x
    passed = reduction > 0.3

    print(f"\nBaseline P(X) = {baseline_p_x:.4f}")
    print(f"After ablation P(X) = {final_p_x:.4f}")
    print(f"Reduction: {reduction:.4f} ({reduction/baseline_p_x:.1%} of baseline)")

    print("\n" + "="*70)
    if passed:
        print(f"PASSED: Ablating top heads reduces P(X) by {reduction:.2%}")
    else:
        print(f"FAILED: Expected >30% reduction, got {reduction:.2%}")
    print("="*70)

    return {
        "baseline_p_x": baseline_p_x,
        "final_p_x": final_p_x,
        "reduction": reduction,
        "top_5_heads": top_5,
        "passed": passed,
        "by_attention": by_attention,
        "by_direct": by_direct,
        "by_ablation": by_ablation,
        "overlap_all_three": all_three,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", action="store_true", help="Run GPT-2 validation")
    parser.add_argument("--gemma", action="store_true", help="Run Gemma validation")
    args = parser.parse_args()

    if args.gpt2:
        test_gpt2_induction_validation()
    elif args.gemma:
        test_head_patching()
    else:
        # Default: run both
        print("\n" + "#"*70)
        print("# RUNNING GPT-2 EFFECT-BASED VALIDATION")
        print("#"*70 + "\n")
        test_gpt2_induction_validation()

        print("\n\n" + "#"*70)
        print("# RUNNING GEMMA BASIC PATCHING VALIDATION")
        print("#"*70 + "\n")
        test_head_patching()
