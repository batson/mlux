"""
Utilities for finding and wrapping modules in MLX models.

Provides Penzai-style selection and transformation of pytree nodes.
"""

from typing import Any, Callable, Type, Iterator
import mlx.nn as nn
from .hook_wrapper import HookWrapper, HookFn


def iter_modules(
    module: nn.Module,
    prefix: str = "",
) -> Iterator[tuple[str, nn.Module]]:
    """
    Iterate over all submodules with their paths.

    Yields:
        (path, module) tuples like ("layers.0.self_attn", Attention())
    """
    for name, child in module.children().items():
        path = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Module):
            yield path, child
            yield from iter_modules(child, path)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    item_path = f"{path}.{i}"
                    yield item_path, item
                    yield from iter_modules(item, item_path)


def find_modules(
    model: nn.Module,
    predicate: Callable[[str, nn.Module], bool],
) -> list[tuple[str, nn.Module]]:
    """
    Find all modules matching a predicate.

    Args:
        model: Root model to search
        predicate: Function(path, module) -> bool

    Returns:
        List of (path, module) tuples

    Example:
        # Find all attention modules
        attns = find_modules(model, lambda p, m: "self_attn" in p)

        # Find all Linear layers
        linears = find_modules(model, lambda p, m: isinstance(m, nn.Linear))
    """
    return [(p, m) for p, m in iter_modules(model) if predicate(p, m)]


def find_by_type(model: nn.Module, module_type: Type) -> list[tuple[str, nn.Module]]:
    """Find all modules of a specific type."""
    return find_modules(model, lambda p, m: isinstance(m, module_type))


def find_by_name(model: nn.Module, pattern: str) -> list[tuple[str, nn.Module]]:
    """Find modules whose path contains the pattern."""
    return find_modules(model, lambda p, m: pattern in p)


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute like 'layers.0.self_attn'."""
    parts = path.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    final = parts[-1]
    if final.isdigit():
        obj[int(final)] = value
    else:
        setattr(obj, final, value)


def wrap_modules(
    model: nn.Module,
    predicate: Callable[[str, nn.Module], bool],
    hook_fn: HookFn = None,
) -> dict[str, HookWrapper]:
    """
    Wrap matching modules with HookWrapper in-place.

    Args:
        model: Model to modify (modified in-place!)
        predicate: Function(path, module) -> bool to select which modules to wrap
        hook_fn: Optional hook function to attach to all wrappers

    Returns:
        Dict mapping path -> HookWrapper for all wrapped modules

    Example:
        # Wrap all attention modules
        wrappers = wrap_modules(model, lambda p, m: "self_attn" in p)

        # Run forward pass
        output = model(tokens)

        # Access captured values
        for name, wrapper in wrappers.items():
            print(f"{name}: {wrapper.last_output.shape}")
    """
    wrappers = {}

    # Find all matching modules first (before we start modifying)
    matches = list(find_modules(model, predicate))

    for path, module in matches:
        # Skip if already wrapped
        if isinstance(module, HookWrapper):
            continue

        wrapper = HookWrapper(wrapped=module, name=path, hook_fn=hook_fn)
        _set_nested_attr(model, path, wrapper)
        wrappers[path] = wrapper

    return wrappers


def unwrap_modules(model: nn.Module) -> None:
    """
    Remove all HookWrappers, restoring original modules.

    Args:
        model: Model to unwrap (modified in-place!)
    """
    wrappers = find_by_type(model, HookWrapper)

    for path, wrapper in wrappers:
        _set_nested_attr(model, path, wrapper.wrapped)


def collect_activations(wrappers: dict[str, HookWrapper]) -> dict[str, Any]:
    """
    Collect all captured outputs from wrappers.

    Args:
        wrappers: Dict of HookWrappers (from wrap_modules)

    Returns:
        Dict mapping path -> captured output
    """
    return {
        path: wrapper.last_output
        for path, wrapper in wrappers.items()
        if wrapper.last_output is not None
    }


def clear_all_caches(wrappers: dict[str, HookWrapper]) -> None:
    """Clear captured values from all wrappers."""
    for wrapper in wrappers.values():
        wrapper.clear_cache()
