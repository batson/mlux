"""
Utilities for finding and wrapping modules in MLX models.

Provides Penzai-style selection and transformation of pytree nodes.
"""

from typing import Any, Callable, Type, Iterator
import mlx.nn as nn
from .hook_wrapper import HookWrapper, HookFn, PreHookFn


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
    pre_hook_fn: PreHookFn = None,
) -> dict[str, HookWrapper]:
    """
    Wrap matching modules with HookWrapper in-place.

    Args:
        model: Model to modify (modified in-place!)
        predicate: Function(path, module) -> bool to select which modules to wrap
        hook_fn: Optional post-hook function (called after module)
        pre_hook_fn: Optional pre-hook function (called before module, can modify inputs)

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

    Example with pre-hook (patching attention head input):
        def patch_head(args, kwargs, wrapper):
            x = args[0]  # [batch, seq, n_heads * d_head]
            x = x.at[:, :, :d_head].set(mean_activation)
            return (x,) + args[1:], kwargs

        wrappers = wrap_modules(model,
            lambda p, m: p == "model.layers.5.self_attn.o_proj",
            pre_hook_fn=patch_head)
    """
    wrappers = {}

    # Find all matching modules first (before we start modifying)
    matches = list(find_modules(model, predicate))

    for path, module in matches:
        # Skip if already wrapped
        if isinstance(module, HookWrapper):
            continue

        wrapper = HookWrapper(
            wrapped=module,
            name=path,
            hook_fn=hook_fn,
            pre_hook_fn=pre_hook_fn
        )
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


def list_local_models() -> list[str]:
    """
    List locally cached HuggingFace models.

    Returns:
        List of model repo IDs (e.g., "mlx-community/gemma-2-2b-it-4bit")
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        return sorted([repo.repo_id for repo in cache.repos])
    except ImportError:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface-hub")
    except Exception:
        return []


DEFAULT_MODELS = [
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]


def get_cached_models() -> list[str]:
    """
    Get list of mlx-community models in HF cache.

    Returns:
        List of mlx-community model IDs (e.g., "mlx-community/gemma-2-2b-it-4bit")
    """
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    models = []
    try:
        for name in os.listdir(cache_dir):
            if name.startswith("models--mlx-community--"):
                model_id = name.replace("models--", "").replace("--", "/")
                models.append(model_id)
    except FileNotFoundError:
        pass
    return sorted(models)


def get_model_options() -> list[dict]:
    """
    Get model options for UI dropdowns.

    Always includes DEFAULT_MODELS, plus any other cached models.
    Non-cached defaults are marked with "(select to download)".

    Returns:
        List of dicts with 'id', 'display', and 'cached' keys.
    """
    cached = set(get_cached_models())

    options = []
    seen = set()

    # Add defaults first
    for model_id in DEFAULT_MODELS:
        is_cached = model_id in cached
        short_name = model_id.replace("mlx-community/", "")
        display = short_name if is_cached else f"{short_name} (select to download)"
        options.append({"id": model_id, "display": display, "cached": is_cached})
        seen.add(model_id)

    # Add remaining cached models
    for model_id in sorted(cached):
        if model_id not in seen:
            short_name = model_id.replace("mlx-community/", "")
            options.append({"id": model_id, "display": short_name, "cached": True})

    return options
