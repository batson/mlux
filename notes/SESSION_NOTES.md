# Session Notes - Contrastive Steering Implementation

## Current Status: Fully Functional ✓

### Completed
1. **Core steering module** (`mlux/steering.py`) - fully implemented with:
   - `compute_steering_vector(hooked, positive, negative, layer)`
   - `create_steering_hook(vector, alpha)` - uniform steering with dtype handling
   - `create_additive_hook(deltas)` - position-specific steering with dtype handling
   - `prefill_with_cache()` - returns (cache, logits) for KV-cached generation
   - `generate_from_cache()` - continues from cache with optional hooks
   - `ContrastiveSteering` class - high-level interface

2. **HookWrapper fixed** (`mlux/hook_wrapper.py`):
   - `__getattr__` now properly handles MLX's module storage
   - First tries parent's `__getattr__` for array parameters
   - Then forwards to wrapped module for attributes like `use_sliding` (Llama)
   - Works correctly with Llama, Gemma, Qwen models

3. **Tests passing** - all 35 tests pass including:
   - Zero vector equivalence (dtype handling fixed)
   - Causality tests (layer selection corrected)
   - Steering effects tests

4. **New experiments module**:
   - `mlux/experiments/layer_sweep.py` - systematic layer testing
   - `mlux/experiments/steering_explorer.py` - interactive web UI
   - `mlux/experiments/alpha_calibration.py` - alpha-norm relationship

## Key Findings

### Dtype Handling
- MLX models often use float16 for activations
- Steering vectors created with `mx.zeros()` default to float32
- Adding float32 to float16 causes precision issues
- **Fix**: Cast steering delta to output.dtype before adding

### Optimal Layers
- Layer sweep on Gemma 2-2B (26 layers) found:
  - Layer 20 has strongest effect
  - Layer 16-20 range works well (60-80% through model)
  - Earlier layers (4-8) have minimal effect
  - Last few layers (22-25) have numerical issues (inf norms)

### Alpha Calibration
- Vector norm varies by layer and experiment (8-200 range seen)
- **Rule of thumb**: `max_safe_alpha ≈ 300 / norm`
  - norm ~150: alpha up to 2.0 is coherent
  - norm ~150, alpha 3.0+: outputs become gibberish/repetition
- For reliable results, use `alpha = 0.3 to 0.5` as starting point

### Model-Specific Notes
- **Gemma 2-2B**: Works well, chat template required for instruction models
- **Llama 3.2-3B**: Works after HookWrapper fix (use_sliding attribute)
- **Qwen 3B**: Previous experiments showed good results at layer 24/36

## Files Reference
- `mlux/steering.py` - main implementation
- `mlux/hook_wrapper.py` - fixed attribute forwarding
- `mlux/hooked_model.py` - run_with_hooks accepts cache param
- `mlux/experiments/layer_sweep.py` - layer testing script
- `mlux/experiments/steering_explorer.py` - interactive web UI (Flask)
- `mlux/experiments/alpha_calibration.py` - alpha optimization
- `mlux/tests/test_steering.py` - steering tests

## Running the Tools

### Interactive Steering Explorer
```bash
python -m mlux.experiments.steering_explorer --model mlx-community/gemma-2-2b-it-4bit
# Opens at http://127.0.0.1:5001
```

### Layer Sweep
```bash
python mlux/experiments/layer_sweep.py --alpha 0.3 --layers "5,10,15,20"
```

### Alpha Calibration
```bash
python mlux/experiments/alpha_calibration.py --output results.json
```

## Next Steps (Future Work)

### Immediate Experiments
1. Test larger models (Qwen 7B, Llama 8B, Gemma 9B)
2. Multi-layer steering (apply same vector at multiple layers)
3. Steering vector interpolation (blend multiple concepts)

### Research Directions
1. **Feature discovery**: Find interpretable directions via clustering/PCA
2. **Safety features**: Explore deception, sycophancy directions
3. **Golden Gate style**: Find specific concepts, steer model identity
4. **Persona steering**: Formal/casual, expert/novice

### Tooling Improvements
1. Streaming generation in explorer
2. Steering vector library (save/load)
3. Automatic optimal layer selection

## Models Tested
- ✓ mlx-community/gemma-2-2b-it-4bit (26 layers, d=2304)
- ✓ mlx-community/Llama-3.2-3B-Instruct-4bit (28 layers, d=3072)
- ✓ mlx-community/Qwen2.5-3B-Instruct-4bit (36 layers)

## Code Quality
- All 35 tests pass
- Proper dtype handling throughout
- Works with quantized (4-bit) models
- Clean separation of concerns
