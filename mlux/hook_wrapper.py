"""
HookWrapper - A module wrapper that captures inputs/outputs and optionally modifies them.

Supports both pre-hooks (modify inputs before module runs) and post-hooks (modify outputs).
This enables causal interventions on specific components like attention heads.
"""

from typing import Any, Callable, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

# Post-hook function signature: (inputs, outputs, wrapper) -> Optional[modified_outputs]
# Return None to keep original outputs, or return modified outputs
HookFn = Callable[[tuple, Any, "HookWrapper"], Optional[Any]]

# Pre-hook function signature: (args, kwargs, wrapper) -> (modified_args, modified_kwargs)
# Must return the (potentially modified) args and kwargs tuple
PreHookFn = Callable[[tuple, dict, "HookWrapper"], Tuple[tuple, dict]]


class HookWrapper(nn.Module):
    """
    Wraps an MLX module to capture and optionally modify its inputs/outputs.

    The wrapper is transparent - it passes all calls through to the wrapped module.
    By default, no modification occurs and the model produces identical outputs.

    Supports two types of hooks:
    - pre_hook_fn: Called BEFORE the module runs, can modify inputs
    - hook_fn (post-hook): Called AFTER the module runs, can modify outputs

    This enables causal interventions like patching specific attention head outputs.

    Args:
        wrapped: The module to wrap
        name: Identifier for this hook point (e.g., "layer0.self_attn.o_proj")
        hook_fn: Optional post-hook function called after forward pass
        pre_hook_fn: Optional pre-hook function called before forward pass

    Example (capture only):
        >>> linear = nn.Linear(4, 4)
        >>> wrapped = HookWrapper(linear, "my_linear")
        >>> out = wrapped(x)  # Same as linear(x), but captures input/output
        >>> print(wrapped.last_input, wrapped.last_output)

    Example (pre-hook to patch attention head):
        >>> def patch_head(args, kwargs, wrapper):
        ...     x = args[0]  # [batch, seq, n_heads * d_head]
        ...     # Modify head 0's slice
        ...     x = x.at[:, :, :d_head].set(mean_activation)
        ...     return (x,) + args[1:], kwargs
        >>> wrapped = HookWrapper(o_proj, "o_proj", pre_hook_fn=patch_head)
    """

    def __init__(
        self,
        wrapped: nn.Module,
        name: str = "",
        hook_fn: Optional[HookFn] = None,
        pre_hook_fn: Optional[PreHookFn] = None,
    ):
        super().__init__()
        self.wrapped = wrapped
        self.hook_name = name  # Renamed to avoid confusion with nn.Module internals
        self.hook_fn = hook_fn  # Post-hook (after module)
        self.pre_hook_fn = pre_hook_fn  # Pre-hook (before module)
        self.last_input: Optional[tuple] = None
        self.last_output: Optional[Any] = None
        self.capture_enabled: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        """Forward pass through wrapped module, with optional pre/post hooks."""
        # Apply pre-hook if present (modify inputs before module runs)
        if self.pre_hook_fn is not None:
            args, kwargs = self.pre_hook_fn(args, kwargs, self)

        # Capture input (after pre-hook modification)
        if self.capture_enabled:
            self.last_input = args

        # Call original module
        output = self.wrapped(*args, **kwargs)

        # Capture output
        if self.capture_enabled:
            self.last_output = output

        # Apply post-hook if present (modify outputs after module runs)
        if self.hook_fn is not None:
            modified = self.hook_fn(args, output, self)
            if modified is not None:
                output = modified
                if self.capture_enabled:
                    self.last_output = output

        return output

    def clear_cache(self) -> None:
        """Clear captured values."""
        self.last_input = None
        self.last_output = None

    def set_hook(self, hook_fn: Optional[HookFn]) -> None:
        """Set or clear the post-hook function."""
        self.hook_fn = hook_fn

    def set_pre_hook(self, pre_hook_fn: Optional[PreHookFn]) -> None:
        """Set or clear the pre-hook function."""
        self.pre_hook_fn = pre_hook_fn

    def enable_capture(self, enabled: bool = True) -> None:
        """Enable or disable input/output capture."""
        self.capture_enabled = enabled

    def __repr__(self) -> str:
        return f"HookWrapper({self.hook_name!r}, wrapped={type(self.wrapped).__name__})"

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped module for transparency."""
        # This is called when the attribute isn't found via __getattribute__.
        # MLX stores:
        #   - Child modules in children()
        #   - Arrays in parameters() (accessed via parent's __getattr__)
        #
        # Order:
        # 1. Try parent's __getattr__ (handles arrays stored as parameters)
        # 2. If name == "wrapped", return the wrapped module
        # 3. Otherwise forward to wrapped module (e.g., use_sliding for Llama)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        children = self.children()
        if "wrapped" in children:
            wrapped = children["wrapped"]
            if name == "wrapped":
                return wrapped
            return getattr(wrapped, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
