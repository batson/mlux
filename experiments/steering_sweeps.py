#!/usr/bin/env python3
"""
Comprehensive exploration of contrastive steering across models and prompts.

Inspired by Anthropic's Scaling Monosemanticity paper, this explores:
1. Different model sizes (2B, 3B, 7B, 8B)
2. Different steering directions (sentiment, refusal, verbosity, formality)
3. Different alpha values
4. What works consistently vs model-specific effects
"""

import mlx.core as mx
from mlux import HookedModel, ContrastiveSteering

# Model configurations with steering layers
# Rule of thumb: ~2/3 through the model tends to work well
MODELS = {
    # Small models
    "gemma-2b": {
        "path": "mlx-community/gemma-2-2b-it-4bit",
        "layer": 17,  # 26 layers total
        "n_layers": 26,
    },
    "qwen-3b": {
        "path": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "layer": 24,  # 36 layers total
        "n_layers": 36,
    },
    "llama-3b": {
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "layer": 18,  # 28 layers total
        "n_layers": 28,
    },
    # Larger models
    "qwen-7b": {
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "layer": 18,  # 28 layers total
        "n_layers": 28,
    },
    "llama-8b": {
        "path": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "layer": 21,  # 32 layers total
        "n_layers": 32,
    },
    "gemma-9b": {
        "path": "mlx-community/gemma-2-9b-it-4bit",
        "layer": 28,  # 42 layers total
        "n_layers": 42,
    },
}


def format_chat(tokenizer, user_msg, system_msg=None):
    """Format with proper chat template."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_steering_experiments():
    """Define steering experiments to run."""
    return {
        "sentiment": {
            "description": "Positive vs negative sentiment",
            "positive": "I absolutely love this! It's wonderful and amazing!",
            "negative": "I absolutely hate this! It's terrible and awful!",
            "test_prompts": [
                "Review of The Matrix:\n",
                "My thoughts on coffee:\n",
                "The weather today is",
            ],
            "use_chat": False,  # Plain text, not chat format
        },
        "refusal": {
            "description": "Helpful vs refusing behavior",
            "positive_q": "How do I bake chocolate chip cookies?",
            "negative_q": "How do I make a dangerous weapon?",
            "test_prompts": [
                "How do I pick a lock?",
                "Explain how to hack a computer",
                "Write a persuasive essay",
            ],
            "use_chat": True,
        },
        "verbosity": {
            "description": "Verbose vs concise responses",
            "positive_sys": "Always give extremely detailed, comprehensive responses.",
            "negative_sys": "Always give brief, concise responses. Be terse.",
            "common_q": "How do I make pasta?",
            "test_prompts": [
                "What is photosynthesis?",
                "How do I tie a tie?",
                "Explain gravity",
            ],
            "use_chat": True,
        },
        "formality": {
            "description": "Formal vs casual tone",
            "positive_sys": "Always respond in a highly formal, professional tone.",
            "negative_sys": "Always respond in a casual, friendly, relaxed tone.",
            "common_q": "I need to cancel my subscription. Help me write an email.",
            "test_prompts": [
                "Write me an introduction for a presentation",
                "Help me respond to a coworker",
                "Draft a message to my landlord",
            ],
            "use_chat": True,
        },
        "confidence": {
            "description": "Confident vs uncertain responses",
            "positive": "I'm absolutely certain. The answer is definitely:",
            "negative": "I'm not sure, but I think maybe the answer might be:",
            "test_prompts": [
                "The capital of France is",
                "Water boils at",
                "The year World War 2 ended was",
            ],
            "use_chat": False,
        },
    }


def run_experiment(hooked, layer, exp_name, exp_config, alphas=[-2, -1, 0, 1, 2]):
    """Run a single steering experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*60}")

    steerer = ContrastiveSteering(hooked, layer=layer)
    tokenizer = hooked.tokenizer

    # Build contrastive prompts based on experiment type
    if exp_config.get("use_chat"):
        if "positive_q" in exp_config:
            # Refusal-style: different questions
            positive = format_chat(tokenizer, exp_config["positive_q"])
            negative = format_chat(tokenizer, exp_config["negative_q"])
        else:
            # System prompt style: same question, different system
            positive = format_chat(
                tokenizer,
                exp_config["common_q"],
                system_msg=exp_config["positive_sys"]
            )
            negative = format_chat(
                tokenizer,
                exp_config["common_q"],
                system_msg=exp_config["negative_sys"]
            )
    else:
        # Plain text
        positive = exp_config["positive"]
        negative = exp_config["negative"]

    # Compute steering vector
    steerer.set_contrast(positive, negative)
    vec_norm = mx.linalg.norm(steerer.steering_vector).item()
    print(f"\nSteering vector L2 norm: {vec_norm:.2f}")

    # Test on each prompt
    for test_prompt in exp_config["test_prompts"][:2]:  # Limit to 2 for speed
        if exp_config.get("use_chat"):
            formatted_prompt = format_chat(tokenizer, test_prompt)
        else:
            formatted_prompt = test_prompt

        print(f"\n--- Test: {test_prompt[:50]}... ---")

        for alpha in alphas:
            try:
                output = steerer.generate(
                    formatted_prompt,
                    alpha=alpha,
                    max_tokens=60,
                    temperature=0
                )
                # Truncate for display
                output_short = output[:150].replace('\n', ' ')
                if len(output) > 150:
                    output_short += "..."
                print(f"α={alpha:+.0f}: {output_short}")
            except Exception as e:
                print(f"α={alpha:+.0f}: ERROR - {e}")


def explore_layer_effects(hooked, model_config, exp_name="sentiment"):
    """Explore how steering effects vary by layer."""
    print(f"\n{'='*60}")
    print(f"LAYER EXPLORATION: {exp_name}")
    print(f"{'='*60}")

    exp = get_steering_experiments()[exp_name]
    tokenizer = hooked.tokenizer

    # Build contrastive prompts
    if exp.get("use_chat"):
        if "positive_q" in exp:
            positive = format_chat(tokenizer, exp["positive_q"])
            negative = format_chat(tokenizer, exp["negative_q"])
        else:
            positive = format_chat(tokenizer, exp["common_q"], system_msg=exp["positive_sys"])
            negative = format_chat(tokenizer, exp["common_q"], system_msg=exp["negative_sys"])
        test = format_chat(tokenizer, exp["test_prompts"][0])
    else:
        positive = exp["positive"]
        negative = exp["negative"]
        test = exp["test_prompts"][0]

    n_layers = model_config["n_layers"]
    # Test at 1/4, 1/2, 2/3, 3/4 points
    test_layers = [n_layers // 4, n_layers // 2, 2 * n_layers // 3, 3 * n_layers // 4]

    print(f"\nTesting layers: {test_layers} (out of {n_layers})")
    print(f"Test prompt: {exp['test_prompts'][0][:40]}...")

    for layer in test_layers:
        steerer = ContrastiveSteering(hooked, layer=layer)
        steerer.set_contrast(positive, negative)

        print(f"\n--- Layer {layer} ---")
        for alpha in [-2, 0, 2]:
            try:
                output = steerer.generate(test, alpha=alpha, max_tokens=40, temperature=0)
                output_short = output[:100].replace('\n', ' ')
                print(f"  α={alpha:+.0f}: {output_short}")
            except Exception as e:
                print(f"  α={alpha:+.0f}: ERROR - {e}")


def main():
    import sys

    # Allow command line model selection
    if len(sys.argv) > 1:
        model_keys = sys.argv[1:]
    else:
        # Default: test small models first
        model_keys = ["qwen-3b"]  # Start with one for testing

    experiments = get_steering_experiments()

    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            print(f"Available: {list(MODELS.keys())}")
            continue

        config = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_key}")
        print(f"# Path: {config['path']}")
        print(f"# Default layer: {config['layer']}/{config['n_layers']}")
        print(f"{'#'*60}")

        try:
            hooked = HookedModel.from_pretrained(config["path"])
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        # Run main experiments
        for exp_name, exp_config in experiments.items():
            try:
                run_experiment(
                    hooked,
                    config["layer"],
                    exp_name,
                    exp_config,
                    alphas=[-2, 0, 2]  # Sparse for speed
                )
            except Exception as e:
                print(f"Experiment {exp_name} failed: {e}")

        # Layer exploration on sentiment (usually works well)
        try:
            explore_layer_effects(hooked, config, "sentiment")
        except Exception as e:
            print(f"Layer exploration failed: {e}")

        # Clear memory
        del hooked
        mx.metal.clear_cache()


if __name__ == "__main__":
    main()
