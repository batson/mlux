# mlux

Interpretability library for MLX models. Light inside the model.

Inspired by TransformerLens and Penzai.

## Features

- **Zero logit diff**: Module wrapping preserves exact model outputs
- **Activation caching**: Cache any module's outputs during forward pass
- **Interventions**: Ablate, patch, or modify activations with hooks
- **Attention patterns**: Compute attention patterns from cached Q/K
- **Multi-model support**: Works with Gemma, Llama, Qwen, GPT-2, and any mlx-lm model

## Installation

```bash
pip install mlux
```

Or from source:
```bash
git clone https://github.com/jxnl/mlux
cd mlux
pip install -e .
```

## Quick Start

```python
from mlux import HookedModel

# Load any mlx-lm compatible model
hooked = HookedModel.from_pretrained("mlx-community/gemma-2-2b-it-4bit")

# Cache activations
output, cache = hooked.run_with_cache(
    "Hello world",
    hooks=["model.layers.0.mlp", "model.layers.5.self_attn"]
)

# Run with interventions
def zero_ablate(inputs, output, wrapper):
    return output * 0

output = hooked.run_with_hooks(
    "The capital of France is",
    hooks=[("model.layers.10.mlp", zero_ablate)]
)

# Get attention patterns
patterns = hooked.get_attention_patterns("Hello world", layers=[0, 5, 10])
# patterns[0].shape = [batch, n_heads, seq, seq]
```

## Experiments

### Induction Heads

```bash
python -m mlux.experiments.induction_heads
python -m mlux.experiments.induction_heads --model mlx-community/gpt2-base-mlx
```

### Binding Mechanisms

Replicate [arXiv:2510.06182](https://arxiv.org/abs/2510.06182):

```bash
python -m mlux.experiments.binding_mechanisms
python -m mlux.experiments.binding_mechanisms --compare
```

## Supported Models

Any mlx-lm compatible model: Gemma 2, Llama 3.x, Qwen 2.5, GPT-2, etc.

## How It Works

Unlike TransformerLens which rewrites the forward pass, mlux wraps modules in-place. This preserves perfect logit equivalence with the original model.

## License

MIT
