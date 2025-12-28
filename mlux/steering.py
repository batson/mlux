"""
Contrastive Steering - steer model generations using activation interventions.

Two main workflows:

1. Uniform steering (all tokens, same vector):
   >>> steerer = ContrastiveSteering(hooked, layer=12)
   >>> steerer.set_contrast(positive="I love this!", negative="I hate this!")
   >>> output = steerer.generate("The movie was", alpha=2.0)

2. Position-specific steering (prompt only, then unsteered generation):
   >>> deltas = steerer.build_deltas(prompt, positions=[0, 5], alpha=2.0)
   >>> cache, logits = steerer.prefill_steered(prompt, deltas)
   >>> output = steerer.generate_from_cache(cache, logits, max_tokens=50)
"""

from typing import Optional, Union
import mlx.core as mx

from .hooked_model import HookedModel
from .hook_wrapper import HookFn


# =============================================================================
# Hook Factories
# =============================================================================

def create_steering_hook(vector: mx.array, alpha: float = 1.0) -> HookFn:
    """
    Create hook that adds vector * alpha to all positions.

    Args:
        vector: Steering vector, shape [d_model]
        alpha: Scaling factor

    Returns:
        Hook function for use with run_with_hooks
    """
    scaled = alpha * vector.reshape(1, 1, -1)

    def hook_fn(inputs, output, wrapper):
        # Cast to output dtype to avoid precision issues
        delta = scaled.astype(output.dtype)
        return output + delta

    return hook_fn


def create_additive_hook(deltas: mx.array) -> HookFn:
    """
    Create hook that adds position-specific deltas.

    Args:
        deltas: Array to add, shape [1, seq, d_model] or [seq, d_model]

    Returns:
        Hook function for use with run_with_hooks
    """
    if deltas.ndim == 2:
        deltas = deltas[None, :, :]  # [seq, d] -> [1, seq, d]

    def hook_fn(inputs, output, wrapper):
        # Cast to output dtype to avoid precision issues
        delta = deltas.astype(output.dtype)
        return output + delta

    return hook_fn


# =============================================================================
# Core Functions
# =============================================================================

def compute_steering_vector(
    hooked: HookedModel,
    positive: str,
    negative: str,
    layer: int,
) -> mx.array:
    """
    Compute steering vector from contrastive prompts.

    Returns difference of final-token activations: positive - negative.

    Args:
        hooked: HookedModel instance
        positive: Prompt for desired direction
        negative: Prompt for opposite direction
        layer: Layer to extract from

    Returns:
        Steering vector, shape [d_model]
    """
    hook_path = f"model.layers.{layer}"

    _, pos_cache = hooked.run_with_cache(positive, hooks=[hook_path])
    _, neg_cache = hooked.run_with_cache(negative, hooks=[hook_path])

    pos_act = pos_cache[hook_path][:, -1, :]  # [1, d_model]
    neg_act = neg_cache[hook_path][:, -1, :]

    return (pos_act - neg_act).squeeze(0)  # [d_model]


def prefill_with_cache(
    hooked: HookedModel,
    prompt: Union[str, mx.array],
    hooks: list[tuple[str, HookFn]] = None,
) -> tuple[list, mx.array]:
    """
    Run prefill (optionally with hooks) and return cache + logits.

    Args:
        hooked: HookedModel instance
        prompt: Input prompt
        hooks: Optional list of (path, hook_fn) tuples

    Returns:
        (cache, logits) - KV cache and logits from last token
    """
    from mlx_lm.models.cache import make_prompt_cache

    tokens = hooked._tokenize(prompt)
    cache = make_prompt_cache(hooked.model)

    if hooks:
        logits = hooked.run_with_hooks(tokens, hooks=hooks, cache=cache)
    else:
        logits = hooked.model(tokens, cache=cache)
        mx.eval(logits)

    mx.eval([c.state for c in cache])
    return cache, logits


def generate_from_cache(
    hooked: HookedModel,
    cache: list,
    max_tokens: int = 100,
    temperature: float = 0.0,
    hooks: list[tuple[str, HookFn]] = None,
    initial_logits: mx.array = None,
) -> str:
    """
    Generate tokens continuing from a prefilled cache.

    Args:
        hooked: HookedModel instance
        cache: KV cache from prefill_with_cache
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        hooks: Optional hooks to apply during generation
        initial_logits: Logits from prefill (required for first token)

    Returns:
        Generated text
    """
    tokens = list(generate_from_cache_stream(
        hooked, cache, max_tokens, temperature, hooks, initial_logits
    ))
    return "".join(tokens)


def generate_from_cache_stream(
    hooked: HookedModel,
    cache: list,
    max_tokens: int = 100,
    temperature: float = 0.0,
    hooks: list[tuple[str, HookFn]] = None,
    initial_logits: mx.array = None,
):
    """
    Generate tokens continuing from a prefilled cache, yielding each token.

    Args:
        hooked: HookedModel instance
        cache: KV cache from prefill_with_cache
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        hooks: Optional hooks to apply during generation
        initial_logits: Logits from prefill (required for first token)

    Yields:
        Token strings as they are generated
    """
    if initial_logits is None:
        raise ValueError("initial_logits required - use prefill_with_cache which returns (cache, logits)")

    eos_tokens = _get_eos_tokens(hooked.tokenizer)

    # Sample first token from prefill logits
    y = _sample_token(initial_logits[:, -1, :], temperature)
    mx.eval(y)

    if y.item() in eos_tokens:
        return

    yield hooked.tokenizer.decode([y.item()])

    # Continue generation
    for _ in range(max_tokens - 1):
        if hooks:
            logits = hooked.run_with_hooks(y[:, None], hooks=hooks, cache=cache)
        else:
            logits = hooked.model(y[:, None], cache=cache)
            mx.eval(logits)

        y = _sample_token(logits[:, -1, :], temperature)
        mx.eval(y)

        if y.item() in eos_tokens:
            break
        yield hooked.tokenizer.decode([y.item()])


def generate_with_steering(
    hooked: HookedModel,
    prompt: str,
    vector: mx.array,
    layer: Union[int, list[int]],
    alpha: float = 1.0,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """
    Generate with uniform steering applied to all tokens.

    Args:
        hooked: HookedModel instance
        prompt: Input prompt
        vector: Steering vector [d_model]
        layer: Layer(s) to apply steering - int or list of ints for multi-layer
        alpha: Steering strength
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text (not including prompt)
    """
    # Support multi-layer steering
    if isinstance(layer, int):
        layers = [layer]
    else:
        layers = layer

    hooks = []
    for l in layers:
        hook_path = f"model.layers.{l}"
        hook = create_steering_hook(vector, alpha)
        hooks.append((hook_path, hook))

    cache, logits = prefill_with_cache(hooked, prompt, hooks=hooks)
    return generate_from_cache(hooked, cache, max_tokens, temperature, hooks=hooks, initial_logits=logits)


# =============================================================================
# Helper Functions
# =============================================================================

def suggest_alpha(vector: mx.array, target_effect: str = "moderate") -> float:
    """
    Suggest an alpha value based on vector norm.

    Based on calibration experiments:
    - Very large norms (>100) need small alphas (0.3-0.5)
    - Medium norms (50-100) can use moderate alphas (0.5-1.0)
    - Small norms (<50) can use larger alphas (1.0-2.0)

    Args:
        vector: Steering vector
        target_effect: "subtle" (0.5x), "moderate" (1x), "strong" (1.5x)

    Returns:
        Suggested alpha value
    """
    norm = mx.sqrt(mx.sum(vector**2)).item()

    # Base formula: alpha ~= 50 / norm (keeps effect roughly constant)
    base_alpha = 50.0 / max(norm, 1.0)

    # Clamp to reasonable range
    base_alpha = max(0.1, min(base_alpha, 2.0))

    # Scale by target effect
    multipliers = {"subtle": 0.5, "moderate": 1.0, "strong": 1.5}
    multiplier = multipliers.get(target_effect, 1.0)

    return base_alpha * multiplier


def _sample_token(logits: mx.array, temperature: float) -> mx.array:
    """Sample a token from logits."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits / temperature)


def _get_eos_tokens(tokenizer) -> set:
    """Get EOS token IDs from tokenizer."""
    eos = set()
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'eos_token_ids'):
        eos.update(tokenizer.eos_token_ids)
    return eos


# =============================================================================
# High-Level Interface
# =============================================================================

class ContrastiveSteering:
    """
    High-level interface for contrastive activation steering.

    Example (uniform steering):
        >>> steerer = ContrastiveSteering(hooked, layer=12)
        >>> steerer.set_contrast("I love this!", "I hate this!")
        >>> output = steerer.generate("The movie was", alpha=2.0)

    Example (position-specific, then unsteered):
        >>> steerer.set_contrast("happy", "sad")
        >>> deltas = steerer.build_deltas("The movie was really", positions=[3, 4], alpha=3.0)
        >>> cache, logits = steerer.prefill_steered("The movie was really", deltas)
        >>> output = steerer.generate_from_cache(cache, logits, max_tokens=20)
    """

    def __init__(self, hooked: HookedModel, layer: int):
        self.hooked = hooked
        self.layer = layer
        self.steering_vector: Optional[mx.array] = None

    def set_contrast(self, positive: str, negative: str) -> mx.array:
        """Compute and store steering vector from contrastive prompts."""
        self.steering_vector = compute_steering_vector(
            self.hooked, positive, negative, self.layer
        )
        return self.steering_vector

    def set_vector(self, vector: mx.array):
        """Directly set the steering vector."""
        self.steering_vector = vector

    # -------------------------------------------------------------------------
    # Workflow 1: Uniform steering (all tokens)
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        alpha: float = 1.0,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate with uniform steering applied to all tokens."""
        if self.steering_vector is None:
            raise ValueError("Call set_contrast() or set_vector() first")

        return generate_with_steering(
            self.hooked, prompt, self.steering_vector, self.layer,
            alpha=alpha, max_tokens=max_tokens, temperature=temperature,
        )

    # -------------------------------------------------------------------------
    # Workflow 2: Position-specific steering, then unsteered generation
    # -------------------------------------------------------------------------

    def build_deltas(
        self,
        prompt: str,
        positions: list[int],
        alpha: float = 1.0,
    ) -> mx.array:
        """
        Build a deltas array that steers only at specified positions.

        Args:
            prompt: The prompt (to determine sequence length)
            positions: Token positions to steer
            alpha: Steering strength

        Returns:
            Deltas array of shape [1, seq_len, d_model]
        """
        if self.steering_vector is None:
            raise ValueError("Call set_contrast() or set_vector() first")

        tokens = self.hooked._tokenize(prompt)
        seq_len = tokens.shape[1]
        d_model = self.steering_vector.shape[0]

        # Build position mask and multiply
        pos_mask = mx.zeros((1, seq_len, 1))
        for pos in positions:
            if 0 <= pos < seq_len:
                pos_mask = pos_mask + (mx.arange(seq_len) == pos).reshape(1, seq_len, 1)

        scaled_vec = alpha * self.steering_vector.reshape(1, 1, -1)
        return pos_mask * scaled_vec

    def prefill_steered(self, prompt: str, deltas: mx.array) -> tuple[list, mx.array]:
        """
        Prefill with position-specific steering, return cache and logits.

        Args:
            prompt: Input prompt
            deltas: Steering deltas [1, seq, d_model]

        Returns:
            (cache, logits) for use with generate_from_cache
        """
        hook_path = f"model.layers.{self.layer}"
        hook = create_additive_hook(deltas)
        return prefill_with_cache(self.hooked, prompt, hooks=[(hook_path, hook)])

    def generate_from_cache(
        self,
        cache: list,
        logits: mx.array,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate from cache without any steering."""
        return generate_from_cache(
            self.hooked, cache, max_tokens, temperature, hooks=None, initial_logits=logits
        )
