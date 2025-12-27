# mlux

Model illumination. An interpretability library for MLX models, so you can do small-model interpretability on your Apple laptops.

Inspired by TransformerLens and Penzai.

## Features

- **No rewrites**: Original model components are wrapped, so we don't need to rewrite the forward pass.
- **Activation caching**: Cache any module's outputs during forward pass
- **Interventions**: Ablate, patch, or modify activations with hooks
- **Attention patterns**: Compute attention patterns from cached Q/K
- **Multi-model support**: Works with Gemma, Llama, Qwen, GPT-2, and any mlx-lm model

## Installation

```bash
git clone https://github.com/batson/mlux
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

### Logit Lens

Interactive viewer showing what the model predicts at each layer. See how predictions evolve from garbage in early layers to the final answer.

```bash
pip install flask  # required for web UI
python -m mlux.experiments.logit_lens
python -m mlux.experiments.logit_lens --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Opens a web UI where you can:
- Enter any prompt
- See a grid of tokens (rows) × layers (columns)
- View top predictions at each layer for each token position
- Switch between residual stream, MLP output, and attention output probes

### Induction Heads

Find induction heads in any model. (This was a sanity check for attention patterns to see if they matched the previous literature.)

```bash
python -m mlux.experiments.induction_heads
python -m mlux.experiments.induction_heads --model mlx-community/gpt2-base-mlx
```

### Binding Mechanisms

An attempt to replicate [arXiv:2510.06182](https://arxiv.org/abs/2510.06182):

```bash
python -m mlux.experiments.binding_mechanisms
python -m mlux.experiments.binding_mechanisms --compare
python -m mlux.experiments.binding_mechanisms --causal  # causal intervention analysis
```

### Mean Ablation

Ablate the residual stream at each (layer, position) and measure the effect on next-token prediction. Uses mean activations (averaged across positions in the prompt) as the ablation target.

```bash
python -m mlux.experiments.ablation --web  # Web interface with colored heatmap
python -m mlux.experiments.ablation --prompt "John has a dog. Mary has a cat. What pet does John have?"
```

The web interface shows a blue-white-red heatmap where:
- **Red** = ablating this position hurts the prediction
- **Blue** = ablating this position helps the prediction (removes noise)
- **White** = no effect

## Supported Models

Any mlx-lm compatible model: Gemma 2, Llama 3.x, Qwen 2.5, GPT-2, etc.

## License

MIT
