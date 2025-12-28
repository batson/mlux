# Bug Fixes and Infrastructure Improvements

## Bugs Fixed During Experiments

### 1. HookWrapper AttributeError for Llama Models
**Bug**: `AttributeError: 'HookWrapper' object has no attribute 'use_sliding'`

**Root Cause**: MLX stores child modules in `children()`, not `__dict__`. The `__getattr__` method was checking wrong location.

**Fix** (hook_wrapper.py):
```python
def __getattr__(self, name: str) -> Any:
    # First try parent's __getattr__ (handles arrays stored as parameters)
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
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

### 2. Steering Dtype Mismatch
**Bug**: Zero steering (alpha=0) caused non-zero diff in activations.

**Root Cause**: Steering vectors defaulted to float32 while model activations are float16.

**Fix** (steering.py):
```python
def hook_fn(inputs, output, wrapper):
    delta = scaled.astype(output.dtype)  # Cast to output dtype
    return output + delta
```

### 3. Causality Test Failure
**Bug**: Causality test expected steering at layer N to affect layer N's cache.

**Root Cause**: Steering layer N's output affects layer N+1's input, not N's cache.

**Fix**: Steer at layer N-1, check cache at layer N.

### 4. Logit Lens Showing Generic Tokens
**Bug**: Instruction-tuned models showing only generic tokens like "The", "," in logit lens.

**Root Cause**: Model expects chat-formatted input, not raw prompts.

**Fix**: Use `tokenizer.apply_chat_template()` for all prompts.

### 5. MLX Array Modification Syntax
**Bug**: `output.at[0, -1, :].set(value)` doesn't work in MLX.

**Root Cause**: MLX arrays are immutable, different syntax than JAX.

**Fix**: Use concatenation instead:
```python
if seq_len > 1:
    new_output = mx.concatenate([output[:, :-1, :], patch.reshape(1, 1, -1)], axis=1)
else:
    new_output = patch.reshape(1, 1, -1)
```

---

## Infrastructure Improvement Suggestions

### 1. Unified Model Wrapper
**Issue**: Different models have different attribute names (e.g., `model.model.layers` vs `model.layers`).

**Suggestion**: Create a unified accessor:
```python
class HookedModel:
    @property
    def layers(self):
        if hasattr(self.model, 'model'):
            return self.model.model.layers
        return self.model.layers
```

### 2. Alpha Auto-Calibration
**Issue**: Optimal alpha varies wildly between models (Gemma: ~0.2, Llama: ~1.0).

**Suggestion**: Add `auto_calibrate_alpha()` method that:
1. Computes steering vector norm
2. Returns alpha such that `alpha * norm ≈ target_magnitude`

### 3. Cross-Model Test Suite
**Issue**: Tests currently hardcode Gemma-specific dimensions (e.g., d_model=2048).

**Suggestion**: Dynamic dimension detection:
```python
def get_hidden_dim(hooked):
    test_hook = "model.layers.0"
    _, cache = hooked.run_with_cache("test", hooks=[test_hook])
    return cache[test_hook].shape[-1]
```

### 4. Logit Lens Caching
**Issue**: `get_layer_predictions()` runs full forward pass for each layer query.

**Suggestion**: Cache all layer activations in one pass, then project lazily.

### 5. Batch Processing for Multi-Example Experiments
**Issue**: Current experiments process one prompt at a time.

**Suggestion**: Add batch processing to reduce model loading overhead:
```python
def batch_get_activations(hooked, prompts: list[str], layer: int):
    # Tokenize all prompts, pad to same length
    # Run single forward pass
    # Return list of activations
```

### 6. Progress Bars for Long Experiments
**Issue**: No visibility into progress during multi-model experiments.

**Suggestion**: Add tqdm integration to experiment runners.

### 7. Result Persistence
**Issue**: Results saved to timestamped JSON files, hard to compare across runs.

**Suggestion**: SQLite database with schema for experiments, models, prompts, results.

---

## Bugs Fixed During Exp 9-11 (December 27, 2025)

### 6. Generation Loop Without KV Cache
**Bug**: Simple generation without steering produces degenerate output (repeating tokens).

**Root Cause**: Passing full token history each iteration forces model to recompute, can cause attention pattern issues.

**Fix**: Use KV cache for generation:
```python
from mlx_lm.models.cache import make_prompt_cache

cache = make_prompt_cache(model)
logits = model(tokens, cache=cache)
# Continue with single token inputs using cache
```

### 7. Zero-Shot Prompts Not Following Task Format
**Bug**: Zero-shot prompts like "cat →" don't elicit task completion.

**Root Cause**: Chat-tuned models expect conversational format, not raw continuation.

**Partial Fix**: Use chat template, but task format still underspecified:
```python
# Better: "Complete the following: cat →"
# Or: "What is the plural of cat? Answer with just the word:"
```

---

## Additional Infrastructure Suggestions (from Exp 9-11)

### 8. Attention Head Isolation for FV
**Issue**: Function Vectors paper uses per-head outputs, mlux hooks full layer.

**Suggestion**: Add hooks for attention head outputs before combination:
```python
# Current: hooks["model.layers.0.self_attn"] = full output
# Need: hooks["model.layers.0.self_attn.head.0"] = per-head output
```

### 9. Causal Mediation Analysis
**Issue**: No built-in support for causal mediation (patch from A→B, measure effect).

**Suggestion**: Add mediation analysis utilities:
```python
def causal_mediation(hooked, source_prompt, target_prompt, patch_layer):
    """Patch activations from source to target and measure effect change."""
    pass
```

### 10. Better Prompt Templates
**Issue**: Different models need different prompt formats for best results.

**Suggestion**: Model-specific prompt templates for common tasks:
```python
PROMPT_TEMPLATES = {
    "gemma": {
        "completion": "Complete: {input} →",
        "qa": "Q: {question}\nA:",
    },
    "llama": {...},
    "qwen": {...},
}
```
