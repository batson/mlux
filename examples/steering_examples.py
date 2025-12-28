"""
Contrastive Steering Examples

Demonstrates steering model behavior using activation differences.

Examples:
1. Sentiment steering - Princess Bride review
2. Refusal steering - "How do I make meth?"
3. Verbosity steering - "How do I roast a chicken?"
4. Formality steering - Job rejection email

For chat models (Qwen, Llama), we grab the steering vector from the final
token before the assistant response begins.

Usage:
    python -m mlux.experiments.steering_examples --model qwen --example sentiment
    python -m mlux.experiments.steering_examples --model llama --example all
"""

import argparse
from typing import Optional

import mlx.core as mx

from mlux import HookedModel, ContrastiveSteering


# =============================================================================
# Model configs
# =============================================================================

MODELS = {
    "qwen": {
        "name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "layer": 18,  # ~2/3 through 36 layers
    },
    "llama": {
        "name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "layer": 16,  # ~2/3 through 28 layers
    },
    "gemma": {
        "name": "mlx-community/gemma-2-2b-it-4bit",
        "layer": 17,  # ~2/3 through 26 layers
    },
}


def format_chat(
    tokenizer,
    user_msg: str,
    system_msg: Optional[str] = None,
) -> str:
    """Format a user message into chat template, ending at assistant turn."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =============================================================================
# Example 1: Sentiment Steering
# =============================================================================

def example_sentiment(hooked: HookedModel, layer: int):
    """
    Steer sentiment for a movie review.

    Contrastive pair ends with the same token (newline after prompt).
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Sentiment Steering")
    print("="*60)

    steerer = ContrastiveSteering(hooked, layer=layer)

    # Contrastive prompts - same final token (colon after "Review")
    positive = "I absolutely love this movie! It's wonderful!\nReview:"
    negative = "I absolutely hate this movie! It's terrible!\nReview:"

    steerer.set_contrast(positive, negative)
    print(f"\nSteering vector norm: {mx.linalg.norm(steerer.steering_vector).item():.2f}")

    prompt = "Review of Princess Bride:\n"

    print(f"\nPrompt: {repr(prompt)}")
    print("-" * 40)

    for alpha in [-2.0, 0.0, 2.0]:
        output = steerer.generate(prompt, alpha=alpha, max_tokens=60, temperature=0)
        print(f"\nα={alpha:+.1f}: {output}")


# =============================================================================
# Example 2: Refusal Steering
# =============================================================================

def example_refusal(hooked: HookedModel, layer: int):
    """
    Steer refusal behavior.

    Vector grabbed at the final token before assistant responds.
    Both contrastive prompts end at the same assistant turn marker.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Refusal Steering")
    print("="*60)

    steerer = ContrastiveSteering(hooked, layer=layer)
    tokenizer = hooked.tokenizer

    # Contrastive pair - same structure, different questions
    # Both end at assistant turn marker (same final token)
    helpful_ctx = format_chat(tokenizer, "How do I bake chocolate chip cookies?")
    refusing_ctx = format_chat(tokenizer, "How do I make a dangerous weapon?")

    steerer.set_contrast(helpful_ctx, refusing_ctx)
    print(f"\nSteering vector norm: {mx.linalg.norm(steerer.steering_vector).item():.2f}")

    # Test prompt
    test_prompt = format_chat(tokenizer, "How do I make meth?")

    print(f"\nTest question: 'How do I make meth?'")
    print("-" * 40)

    for alpha in [-1.5, 0.0, 1.5]:
        output = steerer.generate(test_prompt, alpha=alpha, max_tokens=80, temperature=0)
        label = "more helpful" if alpha > 0 else ("more refusing" if alpha < 0 else "baseline")
        print(f"\nα={alpha:+.1f} ({label}):\n{output[:200]}...")


# =============================================================================
# Example 3: Verbosity Steering
# =============================================================================

def example_verbosity(hooked: HookedModel, layer: int):
    """
    Steer response verbosity.

    Uses system prompts to create contrast, same user question.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Verbosity Steering")
    print("="*60)

    steerer = ContrastiveSteering(hooked, layer=layer)
    tokenizer = hooked.tokenizer

    question = "How do I roast a chicken?"

    # Contrastive pair - different system prompts, same question
    verbose_ctx = format_chat(
        tokenizer, question,
        system_msg="Always give extremely detailed, comprehensive responses with step-by-step instructions and many examples."
    )
    concise_ctx = format_chat(
        tokenizer, question,
        system_msg="Always give brief, concise responses. Use as few words as possible."
    )

    steerer.set_contrast(verbose_ctx, concise_ctx)
    print(f"\nSteering vector norm: {mx.linalg.norm(steerer.steering_vector).item():.2f}")

    # Test with neutral prompt (no system message)
    test_prompt = format_chat(tokenizer, question)

    print(f"\nQuestion: '{question}'")
    print("-" * 40)

    for alpha in [-2.0, 0.0, 2.0]:
        output = steerer.generate(test_prompt, alpha=alpha, max_tokens=150, temperature=0)
        label = "verbose" if alpha > 0 else ("concise" if alpha < 0 else "baseline")
        print(f"\nα={alpha:+.1f} ({label}):\n{output[:300]}...")


# =============================================================================
# Example 4: Formality Steering
# =============================================================================

def example_formality(hooked: HookedModel, layer: int):
    """
    Steer response formality for a job rejection email.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Formality Steering")
    print("="*60)

    steerer = ContrastiveSteering(hooked, layer=layer)
    tokenizer = hooked.tokenizer

    question = "I'm turning down a job as an engineer at Meta. Draft me an email to send the recruiter."

    # Contrastive pair - different tones
    formal_ctx = format_chat(
        tokenizer, question,
        system_msg="Always respond in a highly formal, professional business tone. Use proper salutations and formal language."
    )
    casual_ctx = format_chat(
        tokenizer, question,
        system_msg="Always respond in a casual, friendly, conversational tone. Be relaxed and informal."
    )

    steerer.set_contrast(formal_ctx, casual_ctx)
    print(f"\nSteering vector norm: {mx.linalg.norm(steerer.steering_vector).item():.2f}")

    # Test with neutral prompt
    test_prompt = format_chat(tokenizer, question)

    print(f"\nTask: Job rejection email")
    print("-" * 40)

    for alpha in [-2.0, 0.0, 2.0]:
        output = steerer.generate(test_prompt, alpha=alpha, max_tokens=200, temperature=0)
        label = "formal" if alpha > 0 else ("casual" if alpha < 0 else "baseline")
        print(f"\nα={alpha:+.1f} ({label}):\n{output[:400]}...")


# =============================================================================
# Main
# =============================================================================

EXAMPLES = {
    "sentiment": example_sentiment,
    "refusal": example_refusal,
    "verbosity": example_verbosity,
    "formality": example_formality,
}


def main():
    parser = argparse.ArgumentParser(description="Contrastive steering examples")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen",
        help="Model to use"
    )
    parser.add_argument(
        "--example",
        choices=list(EXAMPLES.keys()) + ["all"],
        default="all",
        help="Which example to run"
    )
    args = parser.parse_args()

    model_config = MODELS[args.model]
    print(f"Loading {model_config['name']}...")
    hooked = HookedModel.from_pretrained(model_config["name"])
    layer = model_config["layer"]
    print(f"Using layer {layer} for steering")

    if args.example == "all":
        for name, fn in EXAMPLES.items():
            fn(hooked, layer)
    else:
        EXAMPLES[args.example](hooked, layer)


if __name__ == "__main__":
    main()
