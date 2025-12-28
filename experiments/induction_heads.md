# Induction Head Analysis: Attention vs Causal Importance

## Abstract

We investigate the relationship between attention patterns and causal importance in transformer language models. Using GPT-2 as a test case, we find that **attention patterns are poor predictors of causal importance**. Heads that strongly attend to a token often have minimal causal effect on predicting that token, while the most causally important heads frequently have low attention scores. We identify three distinct metrics for characterizing head behavior and show they identify largely non-overlapping sets of "important" heads.

## Background

**Induction heads** are attention heads that implement in-context learning by:
1. Attending to previous occurrences of the current token
2. Copying information about what followed that token
3. Using this to predict the next token

The canonical test is a repeated sequence like `A B ... A B` where the model should predict `B` after the second `A` by attending to the first `A` and copying what followed.

## Experimental Setup

**Model:** GPT-2 (12 layers, 12 heads, d_head=64)

**Task:** Given sequence `J A M P R X J A M P R`, predict the next token at position 10 (after second `R`). The correct prediction is `X`.

**Baseline:** P(X) = 87.5%

## Three Metrics for Head Importance

We measure each head using three different metrics:

### 1. Attention Score
How much does the head attend from the query position (second R) to the target position (first X)?

```
attention_score = attn[query_pos=10, key_pos=5]
```

### 2. Direct Effect
What is the head's direct contribution to log P(X), measured by projecting the head's output through the unembedding matrix and dotting with the gradient of log P(X)?

```
direct_effect = (head_output @ W_proj @ W_unembed) · ∇_logits log P(X)
```

where `∇_logits log P(X) = e_X - P` (one-hot for X minus probability distribution).

### 3. Ablation Effect
How much does P(X) change when the head is ablated (zeroed)?

```
ablation_effect = P(X)_baseline - P(X)_ablated
```

Positive values indicate the head supports X; negative values indicate suppression.

## Results

### Top 10 by Attention (R → X)

| Head | Attention | Direct | Ablation |
|------|-----------|--------|----------|
| L5H5 | 96.2% | +0.03 | +0.037 |
| L7H10 | 94.6% | +1.38 | -0.032 |
| L6H9 | 89.5% | +0.83 | -0.045 |
| L5H1 | 88.0% | +0.48 | +0.177 |
| L11H10 | 68.1% | -1.75 | -0.070 |
| L9H9 | 58.0% | +4.23 | -0.036 |
| L5H8 | 47.0% | +0.16 | -0.033 |
| L10H7 | 44.8% | -1.35 | -0.040 |
| L8H1 | 39.1% | +0.36 | +0.017 |
| L7H2 | 39.1% | +0.91 | +0.030 |

### Top 10 by Direct Effect

| Head | Direct | Attention | Ablation |
|------|--------|-----------|----------|
| L9H9 | +4.23 | 58.0% | -0.036 |
| L10H10 | +2.54 | 33.8% | +0.097 |
| L7H10 | +1.38 | 94.6% | -0.032 |
| L7H2 | +0.91 | 39.1% | +0.030 |
| L6H9 | +0.83 | 89.5% | -0.045 |
| L5H1 | +0.48 | 88.0% | +0.177 |
| L11H6 | +0.46 | 7.3% | +0.014 |
| L8H1 | +0.36 | 39.1% | +0.017 |
| L11H0 | +0.36 | 31.9% | +0.166 |
| L9H6 | +0.24 | 27.8% | -0.027 |

### Top 10 by Ablation Effect

| Head | Ablation | Attention | Direct |
|------|----------|-----------|--------|
| L1H3 | +0.461 | 4.5% | +0.09 |
| L0H10 | +0.383 | 3.0% | +0.03 |
| L7H6 | +0.353 | 7.7% | -0.18 |
| L0H9 | +0.222 | 7.4% | +0.11 |
| L2H5 | +0.177 | 2.2% | -0.07 |
| L5H1 | +0.177 | 88.0% | +0.48 |
| L11H0 | +0.166 | 31.9% | +0.36 |
| L1H1 | +0.162 | 6.4% | +0.03 |
| L6H7 | +0.161 | 3.1% | -0.07 |
| L2H8 | +0.154 | 6.5% | +0.08 |

## Overlap Analysis

| Comparison | Overlap |
|------------|---------|
| Attention ∩ Direct | 6 heads |
| Attention ∩ Ablation | **1 head** |
| Direct ∩ Ablation | 2 heads |
| All three | **1 head (L5H1)** |

## Key Findings

### 1. Attention ≠ Causal Importance

The correlation between attention to X and causal importance for predicting X is remarkably weak. Only **1 head (L5H1)** appears in the top 10 for both attention and ablation effect.

### 2. High-Attention Heads Often Suppress

Several heads with high attention to X actually have **negative** ablation effects, meaning ablating them *increases* P(X):
- L7H10: 94.6% attention, -0.032 ablation (suppresses X)
- L6H9: 89.5% attention, -0.045 ablation (suppresses X)
- L11H10: 68.1% attention, -0.070 ablation (suppresses X)

### 3. Most Important Heads Have Low Attention

The heads with the largest causal effects (L1H3, L0H10, L7H6) have very low attention to X (<8%). They work through indirect mechanisms.

### 4. Direct Effect ≠ Ablation Effect

A head can have a large direct logit contribution but small ablation effect. Example:
- L9H9: +4.23 direct effect, but -0.036 ablation effect

This occurs because the head's contribution is cancelled or compensated by other mechanisms.

### 5. Logit Effects Can Be Misleading

Early experiments found L0H7 had a -169.6 logit effect when ablated. However, this was a **uniform shift** affecting all tokens equally. The actual P(X) change was only +0.08 (ablating *helped* X slightly).

The correct metric is the **differential effect**: how much does the logit change for X relative to other tokens?

### 6. The One True Induction Head

**L5H1** is the only head that satisfies all classical criteria:
- High attention to X (88%)
- Positive direct effect (+0.48)
- Positive ablation effect (+0.177)

However, it's not even the most causally important head for this prediction.

### 7. BOS Heads Are Critical

The most important heads by ablation (L1H3, L7H6) primarily attend to **position 0** (the BOS/first token), not to X. These "BOS heads" provide baseline context that enables the induction circuit.

### 8. Previous Token Heads

L2H8 shows partial previous-token behavior (40% attention to previous position across the sequence). These heads are part of the induction circuit, moving information about "what follows each token."

## Implications

1. **Don't use attention patterns to identify important heads.** Attention shows where heads look, not what they do.

2. **Use probability-space metrics.** Ablation effects on P(X) are more meaningful than logit changes, which can be uniform shifts.

3. **Indirect effects dominate.** Early-layer heads often have larger causal effects than late-layer heads, despite contributing little directly to the output.

4. **Induction circuits are distributed.** The prediction relies on multiple head types (BOS heads, previous-token heads, induction heads) working together.

## Reproducibility

Run the analysis with:
```bash
python experiments/induction_head_analysis.py
```

Or run the validation test:
```bash
python experiments/test_head_patching_validation.py --gpt2
```

## References

- Olsson et al. (2022). "In-context Learning and Induction Heads"
- Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits"
- Todd et al. (2024). "Function Vectors in Large Language Models"
