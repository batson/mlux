"""
mlux - Light inside MLX models

A minimal interpretability library for MLX models, inspired by TransformerLens and Penzai.

Uses module wrapping for perfect logit equivalence - no forward pass rewriting required.

Quick Start:
    >>> from mlux import HookedModel
    >>> hooked = HookedModel.from_pretrained("mlx-community/gemma-2-2b-it-4bit")
    >>>
    >>> # Cache activations
    >>> output, cache = hooked.run_with_cache(
    ...     "Hello world",
    ...     hooks=["model.layers.0.self_attn", "model.layers.5.mlp"]
    ... )
    >>>
    >>> # Run with intervention
    >>> def double_mlp(inputs, output, wrapper):
    ...     return output * 2
    >>> output = hooked.run_with_hooks(
    ...     "Hello world",
    ...     hooks=[("model.layers.5.mlp", double_mlp)]
    ... )
    >>>
    >>> # Get attention patterns
    >>> patterns = hooked.get_attention_patterns("Hello world", layers=[0, 5, 10])

Supported Models:
    - Gemma 2 (gemma-2-2b, gemma-2-9b, etc.)
    - Llama 3.x (Llama-3.2-1B, Llama-3.2-3B, etc.)
    - Qwen 2.5 (Qwen2.5-3B, Qwen2.5-7B, etc.)
    - GPT-2 (gpt2-base-mlx)
    - Any mlx-lm compatible model
"""

__version__ = "0.2.0"

from .hooked_model import HookedModel
from .hook_wrapper import HookWrapper, HookFn
from .attention import (
    compute_attention_patterns,
    get_attention_info,
    AttentionPatternHelper,
)
from .utils import (
    wrap_modules,
    unwrap_modules,
    find_modules,
    collect_activations,
)

__all__ = [
    # Version
    "__version__",
    # Main class
    "HookedModel",
    # Hook system
    "HookWrapper",
    "HookFn",
    # Attention
    "compute_attention_patterns",
    "get_attention_info",
    "AttentionPatternHelper",
    # Utilities
    "wrap_modules",
    "unwrap_modules",
    "find_modules",
    "collect_activations",
]
