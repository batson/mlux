"""
HookedModel - High-level interface for model interpretability.

Uses the module wrapping approach for perfect logit equivalence.
"""

from typing import Any, Callable, Optional, Union
import warnings
import mlx.core as mx
import mlx.nn as nn

from .hook_wrapper import HookWrapper, HookFn, PreHookFn
from .utils import wrap_modules, unwrap_modules, find_modules, collect_activations, clear_all_caches


def _detect_quantization_bits(model: nn.Module) -> Optional[int]:
    """
    Detect quantization bits from model weights.

    Returns:
        Number of bits (e.g., 4, 8) if quantized, None if full precision.
    """
    try:
        # Navigate to a layer's linear module
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return None

        if len(layers) == 0:
            return None

        layer = layers[0]

        # Check MLP/feed_forward or attention for quantized linear
        # Different architectures use different names: mlp, feed_forward, self_attn, conv
        possible_submodules = []
        for attr in ['mlp', 'feed_forward', 'self_attn', 'conv']:
            if hasattr(layer, attr):
                possible_submodules.append(getattr(layer, attr))

        for submodule in possible_submodules if possible_submodules else [layer]:
            if submodule is None:
                continue
            for name, mod in submodule.items() if hasattr(submodule, 'items') else []:
                if hasattr(mod, 'bits'):
                    return mod.bits

        # Alternative: check if any module is QuantizedLinear
        for name, mod in layer.items() if hasattr(layer, 'items') else []:
            if 'Quantized' in type(mod).__name__ and hasattr(mod, 'bits'):
                return mod.bits

    except Exception:
        pass

    return None


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

    def __repr__(self) -> str:
        # Get model type/family
        model_type = getattr(self.model, 'model_type', None)
        if model_type is None:
            model_type = type(self.model).__name__

        # Get config info
        try:
            cfg = self.config
            arch = f"{cfg['n_layers']}L/{cfg['n_heads']}H/{cfg['d_head']}D"
        except Exception:
            arch = "?"

        # Quantization and dtype
        bits = self.quantization_bits
        dtype = self._detect_dtype()
        if bits:
            precision = f"{bits}bit/{dtype}"
        else:
            precision = dtype

        return f"HookedModel({model_type}, {arch}, {precision})"

    def _detect_dtype(self) -> str:
        """Detect dtype from first linear layer's weight or scales."""
        try:
            for child in self._iter_modules(self.model):
                # QuantizedLinear has scales, regular Linear has weight
                if hasattr(child, 'scales'):
                    return str(child.scales.dtype).replace('mlx.core.', '')
                if isinstance(child, nn.Linear):
                    return str(child.weight.dtype).replace('mlx.core.', '')
        except Exception:
            pass
        return "?"

    def _iter_modules(self, module):
        """Iterate over all child modules, handling lists."""
        for name, child in module.children().items():
            if isinstance(child, list):
                for item in child:
                    yield item
                    if hasattr(item, 'children'):
                        yield from self._iter_modules(item)
            else:
                yield child
                if hasattr(child, 'children'):
                    yield from self._iter_modules(child)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "HookedModel":
        """Load a pretrained model."""
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx_lm required. Install with: pip install mlx-lm")

        model, tokenizer = load(model_name, **kwargs)
        return cls(model, tokenizer)

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        """
        Tokenize text with consistent BOS handling across transformers versions.

        Args:
            text: Input text to tokenize
            add_bos: Whether to prepend BOS token (default True)

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for string input")

        # Always encode without special tokens, then manually add BOS if needed.
        # This ensures consistent behavior across transformers versions.
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if add_bos and self.tokenizer.bos_token_id is not None:
            if not token_ids or token_ids[0] != self.tokenizer.bos_token_id:
                token_ids = [self.tokenizer.bos_token_id] + token_ids

        return token_ids

    def _tokenize(self, input: Union[str, mx.array], add_bos: bool = True) -> mx.array:
        """Convert string to token array if needed."""
        if isinstance(input, str):
            return mx.array([self.tokenize(input, add_bos=add_bos)])
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

        # Check for duplicate paths and warn
        hook_paths = [path for path, _ in hooks]
        pre_hook_paths = [path for path, _ in pre_hooks]

        for paths, hook_type in [(hook_paths, "hooks"), (pre_hook_paths, "pre_hooks")]:
            seen = set()
            duplicates = []
            for path in paths:
                if path in seen:
                    duplicates.append(path)
                seen.add(path)

            if duplicates:
                warnings.warn(
                    f"Duplicate paths in {hook_type}: {duplicates}. "
                    f"Only the last hook for each path will run. "
                    f"To apply multiple operations, combine them into a single hook function.",
                    UserWarning,
                    stacklevel=2
                )

        # Create path -> hook_fn mappings (later hooks overwrite earlier ones)
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

    @property
    def quantization_bits(self) -> Optional[int]:
        """
        Get quantization bits if model is quantized.

        Returns:
            Number of bits (4, 8, etc.) if quantized, None if full precision.
        """
        return _detect_quantization_bits(self.model)

    def get_tolerance(self, strict: float = 1e-5, relaxed: float = 0.15) -> float:
        """
        Get appropriate numerical tolerance based on model quantization.

        Args:
            strict: Tolerance for full-precision models
            relaxed: Tolerance for quantized models

        Returns:
            Appropriate tolerance value
        """
        bits = self.quantization_bits
        if bits is None:
            return strict
        elif bits <= 4:
            return relaxed
        elif bits <= 8:
            return relaxed / 2  # 8-bit is more precise than 4-bit
        else:
            return strict

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
