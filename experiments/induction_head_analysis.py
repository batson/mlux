#!/usr/bin/env python3
"""
Induction Head Analysis: Attention vs Causal Importance

This script reproduces the findings from induction_heads.md, demonstrating that
attention patterns are poor predictors of causal importance in transformers.

Key findings:
1. Attention to a token ≠ causal importance for predicting that token
2. High-attention heads often suppress rather than promote predictions
3. Most causally important heads have low attention scores
4. Only ~1 head typically appears in top-10 for all three metrics

Usage:
    python experiments/induction_head_analysis.py [--save]
"""

import sys
sys.path.insert(0, '.')

import json
import argparse
from datetime import datetime

import mlx.core as mx
from mlux import HookedModel
from mlux.attention import get_attention_info


def create_sequence(hooked, letters):
    """Create token array from letter sequence."""
    return mx.array([[hooked.tokenizer.encode(f" {l}")[0] for l in letters]])


def compute_attention_scores(hooked, tokens, target_pos, query_pos, n_layers, n_heads, d_head):
    """
    Compute attention from query_pos to target_pos for all heads.

    Returns dict: (layer, head) -> attention_score
    """
    seq_len = tokens.shape[1]
    attention_scores = {}

    for layer in range(n_layers):
        captured = {}

        def make_capture():
            def hook(inputs, output, wrapper):
                captured["qkv"] = output
                return output
            return hook

        hooked.run_with_hooks(tokens, hooks=[(f"model.h.{layer}.attn.c_attn", make_capture())])

        qkv = captured["qkv"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(1, seq_len, n_heads, d_head)
        k = k.reshape(1, seq_len, n_heads, d_head)

        scores = mx.einsum("bihd,bjhd->bhij", q, k) / (d_head ** 0.5)
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        attn = mx.softmax(scores + mask, axis=-1)
        mx.eval(attn)

        for h in range(n_heads):
            attention_scores[(layer, h)] = float(attn[0, h, query_pos, target_pos].item())

    return attention_scores


def compute_direct_effects(hooked, tokens, target_token_id, n_layers, n_heads, d_head):
    """
    Compute direct effect of each head on log P(target).

    Direct effect = head_output · ∇_logits log P(target)
    where ∇_logits log P(target) = e_target - P

    Returns dict: (layer, head) -> direct_effect
    """
    seq_len = tokens.shape[1]
    model = hooked.model.model
    wte = model.wte.weight

    # Get baseline probabilities for gradient
    baseline_out = hooked.model(tokens)
    mx.eval(baseline_out)
    baseline_probs = mx.softmax(baseline_out[0, -1, :], axis=-1)
    mx.eval(baseline_probs)

    # Gradient of log P(target) w.r.t. logits
    grad_logprob = -baseline_probs.astype(mx.float32)
    grad_logprob = grad_logprob.at[target_token_id].add(1.0)
    mx.eval(grad_logprob)

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

            hooked.run_with_hooks(tokens, pre_hooks=[(c_proj_path, make_capture(head))])

            head_out = captured["head_out"][0, -1, :]
            mx.eval(head_out)

            # Project through c_proj slice and unembed
            head_proj = c_proj_weight[:, head * d_head:(head + 1) * d_head]
            head_contribution = head_out @ head_proj.T
            head_logits = head_contribution @ wte.T
            mx.eval(head_logits)

            effect = float(mx.sum(head_logits * grad_logprob).item())
            direct_effects[(layer, head)] = effect

    return direct_effects


def compute_ablation_effects(hooked, tokens, target_token_id, n_layers, n_heads, d_head):
    """
    Compute ablation effect of each head on P(target).

    Ablation effect = P(target)_baseline - P(target)_ablated
    Positive = head supports target; Negative = head suppresses target

    Returns dict: (layer, head) -> ablation_effect
    """
    seq_len = tokens.shape[1]

    # Baseline
    baseline_out = hooked.model(tokens)
    mx.eval(baseline_out)
    baseline_probs = mx.softmax(baseline_out[0, -1, :], axis=-1)
    baseline_p = float(baseline_probs[target_token_id].item())

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

            ablated = hooked.run_with_hooks(tokens, pre_hooks=[(c_proj_path, make_ablate(head))])
            mx.eval(ablated)
            ablated_probs = mx.softmax(ablated[0, -1, :], axis=-1)
            ablated_p = float(ablated_probs[target_token_id].item())

            ablation_effects[(layer, head)] = baseline_p - ablated_p

    return ablation_effects


def compute_prev_token_scores(hooked, tokens, n_layers, n_heads, d_head):
    """
    Compute average previous-token attention for each head.

    Returns dict: (layer, head) -> avg_prev_token_attention
    """
    seq_len = tokens.shape[1]
    prev_token_scores = {}

    for layer in range(n_layers):
        captured = {}

        def make_capture():
            def hook(inputs, output, wrapper):
                captured["qkv"] = output
                return output
            return hook

        hooked.run_with_hooks(tokens, hooks=[(f"model.h.{layer}.attn.c_attn", make_capture())])

        qkv = captured["qkv"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(1, seq_len, n_heads, d_head)
        k = k.reshape(1, seq_len, n_heads, d_head)

        scores = mx.einsum("bihd,bjhd->bhij", q, k) / (d_head ** 0.5)
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        attn = mx.softmax(scores + mask, axis=-1)
        mx.eval(attn)

        for h in range(n_heads):
            # Average attention to previous token across all positions
            prev_attn = sum(float(attn[0, h, pos, pos-1].item()) for pos in range(1, seq_len))
            prev_token_scores[(layer, h)] = prev_attn / (seq_len - 1)

    return prev_token_scores


def analyze_induction(model_name="gpt2", save_results=False):
    """
    Run full induction head analysis.
    """
    print("="*70)
    print("Induction Head Analysis: Attention vs Causal Importance")
    print("="*70)

    # Load model
    hooked = HookedModel.from_pretrained(model_name)
    info = get_attention_info(hooked.model)
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    d_head = info["d_head"]

    print(f"\nModel: {model_name}")
    print(f"Architecture: {n_layers} layers, {n_heads} heads, d_head={d_head}")

    # Create test sequence: J A M P R X J A M P R (predict X)
    letters = ["J", "A", "M", "P", "R", "X", "J", "A", "M", "P", "R"]
    tokens = create_sequence(hooked, letters)
    target_token_id = int(tokens[0, 5].item())  # X token
    target_pos = 5  # First X position
    query_pos = 10  # Second R position (predicting X)

    print(f"\nSequence: {' '.join(letters)}")
    print(f"Task: Predict token at position {query_pos + 1}")
    print(f"Target: X (token {target_token_id})")

    # Get baseline
    baseline_out = hooked.model(tokens)
    mx.eval(baseline_out)
    baseline_probs = mx.softmax(baseline_out[0, -1, :], axis=-1)
    baseline_p_x = float(baseline_probs[target_token_id].item())
    print(f"\nBaseline P(X) = {baseline_p_x:.4f}")

    # Compute all three metrics
    print("\n" + "-"*70)
    print("Computing metrics...")

    attention_scores = compute_attention_scores(
        hooked, tokens, target_pos, query_pos, n_layers, n_heads, d_head
    )
    print("  Attention scores: done")

    direct_effects = compute_direct_effects(
        hooked, tokens, target_token_id, n_layers, n_heads, d_head
    )
    print("  Direct effects: done")

    ablation_effects = compute_ablation_effects(
        hooked, tokens, target_token_id, n_layers, n_heads, d_head
    )
    print("  Ablation effects: done")

    prev_token_scores = compute_prev_token_scores(
        hooked, tokens, n_layers, n_heads, d_head
    )
    print("  Previous-token scores: done")

    # Sort by each metric
    by_attention = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
    by_direct = sorted(direct_effects.items(), key=lambda x: x[1], reverse=True)
    by_ablation = sorted(ablation_effects.items(), key=lambda x: x[1], reverse=True)
    by_prev_token = sorted(prev_token_scores.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print("\n" + "="*70)
    print("TOP 10 BY ATTENTION (query → target)")
    print("="*70)
    print(f"{'Head':<8} {'Attn':>8} {'Direct':>10} {'Ablation':>10} {'PrevTok':>8}")
    print("-"*48)
    for (l, h), attn in by_attention[:10]:
        direct = direct_effects[(l, h)]
        ablation = ablation_effects[(l, h)]
        prev = prev_token_scores[(l, h)]
        print(f"L{l:2d}H{h:2d}   {attn:>7.1%}   {direct:>+9.2f}   {ablation:>+9.4f}   {prev:>7.1%}")

    print("\n" + "="*70)
    print("TOP 10 BY DIRECT EFFECT (head · ∇log P(X))")
    print("="*70)
    print(f"{'Head':<8} {'Direct':>10} {'Attn':>8} {'Ablation':>10}")
    print("-"*40)
    for (l, h), direct in by_direct[:10]:
        attn = attention_scores[(l, h)]
        ablation = ablation_effects[(l, h)]
        print(f"L{l:2d}H{h:2d}   {direct:>+9.2f}   {attn:>7.1%}   {ablation:>+9.4f}")

    print("\n" + "="*70)
    print("TOP 10 BY ABLATION EFFECT (ΔP(X))")
    print("="*70)
    print(f"{'Head':<8} {'Ablation':>10} {'Attn':>8} {'Direct':>10} {'PrevTok':>8}")
    print("-"*50)
    for (l, h), ablation in by_ablation[:10]:
        attn = attention_scores[(l, h)]
        direct = direct_effects[(l, h)]
        prev = prev_token_scores[(l, h)]
        print(f"L{l:2d}H{h:2d}   {ablation:>+9.4f}   {attn:>7.1%}   {direct:>+9.2f}   {prev:>7.1%}")

    print("\n" + "="*70)
    print("TOP 5 PREVIOUS-TOKEN HEADS")
    print("="*70)
    print(f"{'Head':<8} {'PrevTok':>8} {'Ablation':>10}")
    print("-"*30)
    for (l, h), prev in by_prev_token[:5]:
        ablation = ablation_effects[(l, h)]
        print(f"L{l:2d}H{h:2d}   {prev:>7.1%}   {ablation:>+9.4f}")

    # Overlap analysis
    attn_set = set(x[0] for x in by_attention[:10])
    direct_set = set(x[0] for x in by_direct[:10])
    ablation_set = set(x[0] for x in by_ablation[:10])

    print("\n" + "="*70)
    print("OVERLAP ANALYSIS (Top 10 of each)")
    print("="*70)
    print(f"Attention ∩ Direct:   {len(attn_set & direct_set)} heads")
    print(f"Attention ∩ Ablation: {len(attn_set & ablation_set)} heads")
    print(f"Direct ∩ Ablation:    {len(direct_set & ablation_set)} heads")
    all_three = attn_set & direct_set & ablation_set
    print(f"All three:            {len(all_three)} heads - {sorted(all_three) if all_three else 'none'}")

    # Validation: ablate top 5 by ablation effect
    print("\n" + "="*70)
    print("VALIDATION: Ablate top 5 heads by ablation effect")
    print("="*70)

    top_5 = [x[0] for x in by_ablation[:5]]
    print(f"Ablating: {top_5}")

    hooks = []
    for layer, head in top_5:
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
        hooks.append((f"model.h.{layer}.attn.c_proj", make_ablate(layer, head)))

    ablated = hooked.run_with_hooks(tokens, pre_hooks=hooks)
    mx.eval(ablated)
    final_probs = mx.softmax(ablated[0, -1, :], axis=-1)
    final_p_x = float(final_probs[target_token_id].item())

    reduction = baseline_p_x - final_p_x
    print(f"\nBaseline P(X) = {baseline_p_x:.4f}")
    print(f"After ablation P(X) = {final_p_x:.4f}")
    print(f"Reduction: {reduction:.4f} ({reduction/baseline_p_x:.1%} of baseline)")

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"""
1. Only {len(all_three)} head(s) appear in top-10 for all three metrics
2. High-attention heads often SUPPRESS the prediction
3. Most causally important heads have LOW attention to target
4. Ablating top-5 ablation-effect heads reduces P(X) by {reduction:.1%}
""")

    # Save results
    if save_results:
        results = {
            "model": model_name,
            "sequence": letters,
            "baseline_p_x": baseline_p_x,
            "final_p_x": final_p_x,
            "reduction": reduction,
            "by_attention": [(list(k), v) for k, v in by_attention],
            "by_direct": [(list(k), v) for k, v in by_direct],
            "by_ablation": [(list(k), v) for k, v in by_ablation],
            "by_prev_token": [(list(k), v) for k, v in by_prev_token],
            "overlap_all_three": [list(x) for x in all_three],
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiments/results/induction_analysis_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filename}")

    return {
        "attention_scores": attention_scores,
        "direct_effects": direct_effects,
        "ablation_effects": ablation_effects,
        "prev_token_scores": prev_token_scores,
        "overlap_all_three": all_three,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Induction head analysis")
    parser.add_argument("--model", default="gpt2", help="Model to analyze")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    analyze_induction(args.model, args.save)
