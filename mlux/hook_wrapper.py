"""
HookWrapper - A module wrapper that captures inputs/outputs and optionally modifies them.

This is the core primitive. By wrapping modules instead of rewriting forward passes,
we preserve perfect logit equivalence with the original model.
"""

from typing import Any, Callable, Optional
import mlx.core as mx
import mlx.nn as nn

# Hook function signature: (inputs, outputs, wrapper) -> Optional[modified_outputs]
# Return None to keep original outputs, or return modified outputs
HookFn = Callable[[tuple, Any, "HookWrapper"], Optional[Any]]


class HookWrapper(nn.Module):
    """
    Wraps an MLX module to capture and optionally modify its inputs/outputs.

    The wrapper is transparent - it passes all calls through to the wrapped module.
    By default, no modification occurs and the model produces identical outputs.

    Args:
        wrapped: The module to wrap
        name: Identifier for this hook point (e.g., "layer0.self_attn")
        hook_fn: Optional function called on each forward pass

    Example:
        >>> linear = nn.Linear(4, 4)
        >>> wrapped = HookWrapper(linear, "my_linear")
        >>> out = wrapped(x)  # Same as linear(x), but captures input/output
        >>> print(wrapped.last_input, wrapped.last_output)
    """

    def __init__(
        self,
        wrapped: nn.Module,
        name: str = "",
        hook_fn: Optional[HookFn] = None,
    ):
        super().__init__()
        self.wrapped = wrapped
        self.hook_name = name  # Renamed to avoid confusion with nn.Module internals
        self.hook_fn = hook_fn
        self.last_input: Optional[tuple] = None
        self.last_output: Optional[Any] = None
        self.capture_enabled: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        """Forward pass through wrapped module, with optional hook."""
        # Capture input
        if self.capture_enabled:
            self.last_input = args

        # Call original module
        output = self.wrapped(*args, **kwargs)

        # Capture output
        if self.capture_enabled:
            self.last_output = output

        # Apply hook if present
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
        """Set or clear the hook function."""
        self.hook_fn = hook_fn

    def enable_capture(self, enabled: bool = True) -> None:
        """Enable or disable input/output capture."""
        self.capture_enabled = enabled

    def __repr__(self) -> str:
        return f"HookWrapper({self.hook_name!r}, wrapped={type(self.wrapped).__name__})"
