"""
HookedModel - High-level interface for model interpretability.

Uses the module wrapping approach for perfect logit equivalence.
"""

from typing import Any, Callable, Optional, Union
import mlx.core as mx
import mlx.nn as nn

from .hook_wrapper import HookWrapper, HookFn, PreHookFn
from .utils import wrap_modules, unwrap_modules, find_modules, collect_activations, clear_all_caches


class HookedModel:
    """
    A model wrapper that provides hook points via module wrapping.

    This approach preserves perfect logit equivalence by wrapping modules
    rather than rewriting the forward pass.

    Example:
        >>> model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")
        >>> hooked = HookedModel(model, tokenizer)
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
    """

    def __init__(self, model: nn.Module, tokenizer: Any = None):
        """
        Args:
            model: The MLX model (typically from mlx_lm.load)
            tokenizer: Optional tokenizer for string inputs
        """
        self.model = model
        self.tokenizer = tokenizer
        self._wrappers: dict[str, HookWrapper] = {}

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "HookedModel":
        """Load a pretrained model."""
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx_lm required. Install with: pip install mlx-lm")

        model, tokenizer = load(model_name, **kwargs)
        return cls(model, tokenizer)

    def _tokenize(self, input: Union[str, mx.array]) -> mx.array:
        """Convert string to tokens if needed."""
        if isinstance(input, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string input")
            return mx.array([self.tokenizer.encode(input)])
        return input

    def forward(self, input: Union[str, mx.array]) -> mx.array:
        """Run forward pass."""
        tokens = self._tokenize(input)
        return self.model(tokens)

    def run_with_cache(
        self,
        input: Union[str, mx.array],
        hooks: Union[list[str], Callable[[str], bool]],
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """
        Run forward pass and cache specified activations.

        Args:
            input: Token IDs or string
            hooks: Either:
                - List of module paths to cache: ["model.layers.0.self_attn"]
                - Predicate function: lambda path: "self_attn" in path

        Returns:
            (output, cache) where cache maps path -> captured output
        """
        tokens = self._tokenize(input)

        # Convert list to predicate
        if isinstance(hooks, list):
            hook_set = set(hooks)
            predicate = lambda p, m: p in hook_set
        else:
            predicate = lambda p, m: hooks(p)

        # Wrap matching modules
        wrappers = wrap_modules(self.model, predicate)
        self._wrappers.update(wrappers)

        try:
            # Run forward
            output = self.model(tokens)
            mx.eval(output)

            # Collect cached values
            cache = collect_activations(wrappers)
            mx.eval(*cache.values())

            return output, cache
        finally:
            # Unwrap modules
            unwrap_modules(self.model)
            self._wrappers = {}

    def run_with_hooks(
        self,
        input: Union[str, mx.array],
        hooks: list[tuple[str, HookFn]] = None,
        pre_hooks: list[tuple[str, PreHookFn]] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        """
        Run forward pass with intervention hooks.

        Args:
            input: Token IDs or string
            hooks: List of (module_path, post_hook_fn) tuples - modify outputs
            pre_hooks: List of (module_path, pre_hook_fn) tuples - modify inputs
            cache: Optional KV cache for efficient generation (from mlx_lm.models.cache)

        Returns:
            Model output after interventions

        Example (post-hook to double MLP output):
            >>> def double_mlp(inputs, output, wrapper):
            ...     return output * 2
            >>> output = hooked.run_with_hooks(
            ...     "Hello", hooks=[("model.layers.5.mlp", double_mlp)]
            ... )

        Example (pre-hook to patch attention head):
            >>> def patch_head(args, kwargs, wrapper):
            ...     x = args[0]  # [batch, seq, n_heads * d_head]
            ...     x = x.at[:, :, :d_head].set(mean_activation)
            ...     return (x,) + args[1:], kwargs
            >>> output = hooked.run_with_hooks(
            ...     "Hello", pre_hooks=[("model.layers.5.self_attn.o_proj", patch_head)]
            ... )
        """
        tokens = self._tokenize(input)
        hooks = hooks or []
        pre_hooks = pre_hooks or []

        # Create path -> hook_fn mappings
        hook_map = dict(hooks)
        pre_hook_map = dict(pre_hooks)
        all_paths = set(hook_map.keys()) | set(pre_hook_map.keys())

        # Wrap with specific hooks
        for path in all_paths:
            def make_predicate(p):
                return lambda path, m: path == p

            wrappers = wrap_modules(
                self.model,
                make_predicate(path),
                hook_fn=hook_map.get(path),
                pre_hook_fn=pre_hook_map.get(path)
            )
            self._wrappers.update(wrappers)

        try:
            if cache is not None:
                output = self.model(tokens, cache=cache)
            else:
                output = self.model(tokens)
            mx.eval(output)
            return output
        finally:
            unwrap_modules(self.model)
            self._wrappers = {}

    def available_hooks(self) -> list[str]:
        """List all module paths that can be hooked."""
        return [p for p, m in find_modules(self.model, lambda p, m: True)]

    def find_hooks(self, pattern: str) -> list[str]:
        """Find hook paths matching a pattern."""
        return [p for p, m in find_modules(self.model, lambda p, m: pattern in p)]

    @property
    def config(self) -> dict:
        """
        Get model configuration info.

        Returns dict with n_layers, n_heads, n_kv_heads, d_head, softcap.
        """
        from .attention import get_attention_info
        return get_attention_info(self.model)

    def get_attention_patterns(
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
        from .attention import AttentionPatternHelper
        helper = AttentionPatternHelper(self)
        return helper.get_patterns(input, layers)
