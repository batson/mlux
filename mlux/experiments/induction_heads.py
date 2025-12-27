#!/usr/bin/env python3
"""
Induction Head Detection Experiment

Induction heads implement in-context learning by completing patterns:
    "...A B ... A" -> predicts "B"

They work via:
1. Previous token head: copies info from previous token into current position
2. Induction head: attends to positions where the *previous* token matches current token

Detection methodology:
1. Create repeated random sequence: [seq] [seq]
2. Look for attention to the "induction diagonal" (offset = seq_len - 1)
3. Validate with actual prediction accuracy

Usage:
    python -m mlux.experiments.induction_heads
    python -m mlux.experiments.induction_heads --model mlx-community/gpt2-base-mlx
"""

import argparse

import mlx.core as mx
import numpy as np

from mlux import HookedModel


def create_repeated_sequence(tokenizer, seq_len: int = 20, seed: int = 42) -> mx.array:
    """
    Create a repeated sequence of random tokens.
    Returns tokens like: [random_seq] [random_seq]
    """
    np.random.seed(seed)
    vocab_size = tokenizer.vocab_size

    # Generate random token IDs (avoiding special tokens, typically 0-1000)
    random_tokens = np.random.randint(1000, min(vocab_size, 30000), size=seq_len)

    # Repeat the sequence
    repeated = np.concatenate([random_tokens, random_tokens])

    return mx.array([repeated.tolist()])


def detect_induction_heads(
    model_name: str = "mlx-community/gemma-2-2b-it-4bit",
    seq_len: int = 20,
    threshold: float = 0.3,
):
    """
    Detect induction heads by looking for attention to the induction diagonal.

    An induction head at position i (in the second half) should attend strongly
    to position (i - seq_len + 1), which is one position after where the same
    token appeared in the first half.

    Args:
        model_name: HuggingFace model name
        seq_len: Length of the random sequence (will be repeated)
        threshold: Minimum score to display in results

    Returns:
        List of (layer, head, score) tuples sorted by score
    """
    print("\n" + "=" * 60)
    print("Induction Head Detection")
    print("=" * 60)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)

    tokens = create_repeated_sequence(hooked.tokenizer, seq_len=seq_len)
    total_len = tokens.shape[1]

    print(f"\nSequence length: {seq_len} tokens, repeated -> {total_len} total")

    sample_tokens = tokens[0, :5].tolist()
    sample_text = [hooked.tokenizer.decode([t]) for t in sample_tokens]
    print(f"First 5 tokens: {sample_text}")

    n_layers = hooked.config["n_layers"]
    all_layers = list(range(n_layers))

    print(f"\nComputing attention patterns for {n_layers} layers...")
    patterns = hooked.get_attention_patterns(tokens, layers=all_layers)

    # Analyze induction diagonal
    induction_scores = []

    print(f"\nAnalyzing induction diagonal (offset = {seq_len - 1})...")
    print(f"{'Layer':>5} {'Head':>5} {'Score':>8}  Pattern")
    print("-" * 50)

    for layer in all_layers:
        pattern = patterns[layer]
        mx.eval(pattern)
        n_heads = pattern.shape[1]

        for head in range(n_heads):
            attn = pattern[0, head]

            # Extract induction diagonal scores
            diagonal_scores = []
            for i in range(seq_len, total_len):
                target_pos = i - seq_len + 1
                if target_pos >= 0:
                    score = attn[i, target_pos].item()
                    diagonal_scores.append(score)

            avg_score = np.mean(diagonal_scores) if diagonal_scores else 0
            induction_scores.append((layer, head, avg_score))

            if avg_score > threshold:
                bar = "#" * int(avg_score * 40)
                print(f"{layer:>5} {head:>5} {avg_score:>8.3f}  {bar}")

    induction_scores.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'=' * 60}")
    print("Top 10 Induction Head Candidates")
    print(f"{'=' * 60}")
    print(f"{'Rank':>4} {'Layer':>5} {'Head':>5} {'Score':>8}")
    print("-" * 30)

    for rank, (layer, head, score) in enumerate(induction_scores[:10], 1):
        print(f"{rank:>4} {layer:>5} {head:>5} {score:>8.3f}")

    return induction_scores


def visualize_induction_pattern(
    model_name: str = "mlx-community/gemma-2-2b-it-4bit",
    layer: int = None,
    head: int = None,
    seq_len: int = 10,
):
    """
    Visualize attention pattern for a specific head on repeated sequence.
    If layer/head not specified, uses the top induction head.
    """
    print("\n" + "=" * 60)
    print("Attention Pattern Visualization")
    print("=" * 60)

    hooked = HookedModel.from_pretrained(model_name)

    if layer is None or head is None:
        print("Finding top induction head...")
        scores = detect_induction_heads(model_name, seq_len=20, threshold=0.0)
        layer, head, _ = scores[0]
        print(f"Using Layer {layer}, Head {head}")

    tokens = create_repeated_sequence(hooked.tokenizer, seq_len=seq_len, seed=42)
    total_len = tokens.shape[1]

    patterns = hooked.get_attention_patterns(tokens, layers=[layer])
    pattern = patterns[layer][0, head]
    mx.eval(pattern)

    print(f"\nLayer {layer}, Head {head}")
    print("Query positions (rows) -> Key positions (cols)")
    print(f"Sequence: [A0...A{seq_len-1}] [A0...A{seq_len-1}] (repeated)")
    print()

    # Header
    print("     ", end="")
    for j in range(total_len):
        marker = "." if j < seq_len else "'"
        print(f" {j%10}{marker}", end="")
    print()

    # Pattern
    for i in range(total_len):
        marker = " " if i < seq_len else "'"
        print(f" {i:2d}{marker} ", end="")
        for j in range(total_len):
            val = pattern[i, j].item()
            if val > 0.5:
                char = "#"
            elif val > 0.3:
                char = "+"
            elif val > 0.1:
                char = "."
            elif val > 0.05:
                char = ":"
            else:
                char = " "

            # Highlight induction diagonal
            is_induction = (i >= seq_len) and (j == i - seq_len + 1)
            if is_induction and val > 0.1:
                char = "*"

            print(f"  {char}", end="")
        print()

    print()
    print("Legend: # >0.5  + >0.3  . >0.1  : >0.05  * = induction diagonal")


def validate_induction_copying(
    model_name: str = "mlx-community/gemma-2-2b-it-4bit",
    seq_len: int = 20,
):
    """
    Validate that the model actually copies tokens in repeated sequences.

    Returns:
        Accuracy of next-token prediction in the second half
    """
    print("\n" + "=" * 60)
    print("Induction Copying Validation")
    print("=" * 60)

    hooked = HookedModel.from_pretrained(model_name)
    tokens = create_repeated_sequence(hooked.tokenizer, seq_len=seq_len, seed=42)

    logits = hooked.forward(tokens)
    mx.eval(logits)

    correct = 0
    total = 0

    print("\nChecking predictions in second half of repeated sequence:")
    for i in range(seq_len, tokens.shape[1] - 1):
        pred = mx.argmax(logits[0, i, :]).item()
        target = tokens[0, i + 1].item()

        is_correct = pred == target
        if is_correct:
            correct += 1
        total += 1

        if i < seq_len + 5:
            pred_str = hooked.tokenizer.decode([pred])
            target_str = hooked.tokenizer.decode([target])
            status = "OK" if is_correct else "X "
            print(f"  {status} Position {i}: pred={repr(pred_str)}, target={repr(target_str)}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nIn-context copying accuracy: {accuracy:.1%} ({correct}/{total})")

    return accuracy


def run_experiment(model_name: str = "mlx-community/gemma-2-2b-it-4bit"):
    """Run the complete induction head experiment."""
    print("\n" + "=" * 70)
    print("   INDUCTION HEAD DETECTION EXPERIMENT")
    print("=" * 70)

    scores = detect_induction_heads(model_name, seq_len=20, threshold=0.2)

    top_candidates = [(l, h, s) for l, h, s in scores if s > 0.2]
    print(f"\nFound {len(top_candidates)} candidate induction heads (score > 0.2)")

    if top_candidates:
        best_layer, best_head, best_score = scores[0]
        visualize_induction_pattern(model_name, best_layer, best_head, seq_len=10)

    accuracy = validate_induction_copying(model_name, seq_len=20)

    print("\n" + "=" * 70)
    print("   Experiment Complete")
    print("=" * 70)

    return scores, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect induction heads in a language model")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    args = parser.parse_args()

    run_experiment(args.model)
