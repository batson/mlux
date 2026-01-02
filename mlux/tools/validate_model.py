#!/usr/bin/env python3
"""
Model validation script for mlux.

Runs three integration tests on a model to verify it works correctly:
1. Logit lens on "The capital of France is"
2. Activation patching swapping Josh and Alex
3. Free generation on "The capital of France is"

Usage:
    python -m mlux.tools.validate_model mlx-community/LFM2-350M-4bit
    python -m mlux.tools.validate_model mlx-community/gemma-2-2b-it-4bit
"""

import argparse
import sys

import mlx.core as mx

from mlux import HookedModel
from mlux.tools.logit_lens import LogitLens
from mlux.steering import prefill_with_cache, generate_from_cache_stream


def test_logit_lens(hooked: HookedModel) -> bool:
    """Test logit lens on the default prompt from logit lens explorer."""
    print("\n1. LOGIT LENS")
    print("-" * 60)

    try:
        lens = LogitLens(hooked)
        prompt = 'Le contraire de "petit" est "'
        print(f'Prompt: {repr(prompt)}')
        print()

        results = lens.get_layer_predictions(prompt, token_idx=-1, top_k=3)

        for r in results:
            layer = r["layer"]
            top_pred = r["predictions"][0]
            print(
                f'  L{layer:2d}: {repr(top_pred["token"]):12s} ({top_pred["logit"]:+.1f})'
            )

        print("\n✓ Logit lens passed")
        return True
    except Exception as e:
        print(f"\n✗ Logit lens failed: {e}")
        return False


def _find_last_occurrence(tokens: list[str], name: str) -> int:
    """Find the last token position containing the name."""
    for i in range(len(tokens) - 1, -1, -1):
        if name.lower() in tokens[i].lower():
            return i
    return -1


def test_activation_patching(hooked: HookedModel) -> bool:
    """Test activation patching with prompts from patching explorer."""
    print("\n2. ACTIVATION PATCHING")
    print("-" * 60)

    try:
        # Use the default prompts from patching_explorer.py
        source_text = "Setup: Josh has a yellow book. Jesse has a black book. Alex has a green book.\nAnswer: The color of Josh's book is"
        target_text = "Setup: Jesse has a black book. Alex has a green book. Josh has a yellow book.\nAnswer: The color of Alex's book is"
        print(f'Source: "{source_text[:50]}..."')
        print(f'Target: "{target_text[:50]}..."')
        print()

        n_layers = hooked.config["n_layers"]
        layer_prefix = hooked.config.get("layer_prefix", "model.layers")

        # Tokenize
        source_ids = hooked.tokenizer.encode(source_text)
        target_ids = hooked.tokenizer.encode(target_text)
        source_arr = mx.array([source_ids])
        target_arr = mx.array([target_ids])

        # Decode tokens for finding name positions
        source_tokens = [hooked.tokenizer.decode([t]) for t in source_ids]
        target_tokens = [hooked.tokenizer.decode([t]) for t in target_ids]

        # Find the last occurrence of "Josh" in source and "Alex" in target
        source_pos = _find_last_occurrence(source_tokens, "Josh")
        target_pos = _find_last_occurrence(target_tokens, "Alex")

        if source_pos < 0 or target_pos < 0:
            print(f"  Could not find name positions (source={source_pos}, target={target_pos})")
            print("  Falling back to second-to-last token")
            source_pos = len(source_ids) - 2
            target_pos = len(target_ids) - 2

        print(f"  Patching position: source[{source_pos}]='{source_tokens[source_pos]}' -> target[{target_pos}]='{target_tokens[target_pos]}'")
        print()

        # Get source activations
        hook_paths = [f"{layer_prefix}.{i}" for i in range(n_layers)]
        _, source_cache = hooked.run_with_cache(source_arr, hooks=hook_paths)

        # Get baseline
        baseline_logits = hooked.forward(target_text)
        mx.eval(baseline_logits)
        baseline_probs = mx.softmax(baseline_logits[0, -1, :], axis=-1)
        top_idx = mx.argmax(baseline_probs).item()
        top_token = hooked.tokenizer.decode([top_idx])
        print(f"Baseline prediction: {repr(top_token)}")
        print()

        # Patch at each layer
        for layer_idx in range(n_layers):
            hook_path = f"{layer_prefix}.{layer_idx}"
            source_act = source_cache[hook_path]

            def create_patch_hook(src_act, src_pos, tgt_pos):
                def hook_fn(inputs, output, wrapper):
                    batch, seq, d = output.shape
                    src_vec = src_act[0, src_pos, :].reshape(1, 1, -1)
                    parts = []
                    if tgt_pos > 0:
                        parts.append(output[:, :tgt_pos, :])
                    parts.append(src_vec)
                    if tgt_pos < seq - 1:
                        parts.append(output[:, tgt_pos + 1 :, :])
                    return mx.concatenate(parts, axis=1)

                return hook_fn

            hook_fn = create_patch_hook(source_act, source_pos, target_pos)
            patched_output = hooked.run_with_hooks(target_arr, hooks=[(hook_path, hook_fn)])
            mx.eval(patched_output)

            patched_probs = mx.softmax(patched_output[0, -1, :], axis=-1)
            top_idx = mx.argmax(patched_probs).item()
            top_token = hooked.tokenizer.decode([top_idx])

            print(f"  L{layer_idx:2d}: {repr(top_token)}")

        print("\n✓ Activation patching passed")
        return True
    except Exception as e:
        print(f"\n✗ Activation patching failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generation(hooked: HookedModel) -> bool:
    """Test free generation on 'The capital of France is'."""
    print("\n3. FREE GENERATION")
    print("-" * 60)

    try:
        prompt = "The capital of France is"
        print(f'Prompt: "{prompt}"')
        print()

        # Use base model generation (no chat template)
        cache, logits = prefill_with_cache(hooked, prompt)

        # Generate a few tokens
        tokens = []
        for token in generate_from_cache_stream(
            hooked, cache, max_tokens=20, temperature=0.7, initial_logits=logits
        ):
            tokens.append(token)
            if len(tokens) >= 20:
                break

        completion = "".join(tokens)
        print(f'Completion: "{completion}"')

        print("\n✓ Generation passed")
        return True
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_model(model_name: str) -> bool:
    """Run all validation tests on a model."""
    print("=" * 60)
    print(f"VALIDATING: {model_name}")
    print("=" * 60)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)
    print(f"Loaded: {hooked}")

    # Show config summary
    cfg = hooked.config
    print(f"\nConfig:")
    print(f"  Layers: {cfg['n_layers']}")
    print(f"  Heads: {cfg['n_heads']} (KV: {cfg['n_kv_heads']})")
    print(f"  Head dim: {cfg['d_head']}")
    if cfg.get("layer_types"):
        types = cfg["layer_types"]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Layer types: {type_counts}")
    print(f"  Quantization: {hooked.quantization_bits}-bit")

    # Run tests
    results = []
    results.append(("Logit Lens", test_logit_lens(hooked)))
    results.append(("Activation Patching", test_activation_patching(hooked)))
    results.append(("Generation", test_generation(hooked)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n✓ All tests passed for {model_name}")
    else:
        print(f"\n✗ Some tests failed for {model_name}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate a model works with mlux")
    parser.add_argument(
        "model",
        nargs="?",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name to validate",
    )
    args = parser.parse_args()

    success = validate_model(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
