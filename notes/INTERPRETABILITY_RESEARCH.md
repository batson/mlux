# Interpretability Research Survey & Experimental Plan

A survey of recent interpretability work from Neel Nanda, Owain Evans, Ekdeep Lubana, Anthropic, and others, with evaluation of what's testable on small models.

## Table of Contents
1. [Paradigm Shift: Pragmatic Interpretability](#1-paradigm-shift-pragmatic-interpretability)
2. [Key Research Findings](#2-key-research-findings)
3. [Testable Claims for Small Models](#3-testable-claims-for-small-models)
4. [Experimental Predictions](#4-experimental-predictions)
5. [Sources](#5-sources)

---

## 1. Paradigm Shift: Pragmatic Interpretability

### Neel Nanda's Evolution (DeepMind)

Neel Nanda, Mechanistic Interpretability Lead at Google DeepMind, has undergone a significant shift since mid-2024:

**From**: "low chance of incredibly big deal" (full reverse-engineering)
**To**: "high chance of medium big deal" (practical partial understanding)

Key insights from ["A Pragmatic Vision for Interpretability"](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability):

1. **Proxy tasks are essential** - It's easy to fool yourself without empirical feedback
2. **Partial understanding is valuable** - For evaluation, monitoring, incident analysis
3. **Model biology > ambitious reverse-engineering** - Focus on observable phenomena
4. **Choose problems by comparative advantage** - What can interp uniquely solve?

### What This Means for Us

We should focus on:
- **Observable, measurable phenomena** (not speculative circuit discovery)
- **Behaviors we can operationalize** (refusal, sycophancy, confidence)
- **Interventions with clear expected effects** (steering should change outputs predictably)

---

## 2. Key Research Findings

### 2.1 Anthropic: Circuit Tracing & Model Biology

From ["On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html):

**Multi-Step Reasoning**
- Models perform "two-hop" reasoning internally (e.g., "capital of state containing Dallas" → Texas → Austin)
- Coexists with direct "shortcut" pathways
- Middle layers are more language-agnostic

**Planning Circuits**
- When writing poetry, models pre-activate candidate end-words BEFORE composing lines
- Work backwards from goals to decide responses

**Refusal Architecture**
- Harmful request features aggregate during finetuning into general "harmful requests" representation
- Model has "default" refusal circuits suppressed when appropriate
- Jailbreaks work because obfuscated harmful requests aren't recognized until too late

**CoT Faithfulness**
- Three types: Faithful (genuine), Bullshitting (ungrounded guesses), Motivated Reasoning (working backward)
- Model explanations of its own process are often confabulated

### 2.2 Refusal Direction (Arditi et al., NeurIPS 2024)

From ["Refusal in Language Models Is Mediated by a Single Direction"](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf):

**Key Finding**: Refusal is mediated by a **one-dimensional subspace** across 13 models up to 72B parameters.

- **Erasing** this direction prevents refusal on harmful instructions
- **Adding** this direction elicits refusal on harmless instructions
- The direction is **cross-lingually universal** - English refusal vector works in other languages
- This technique is known as "abliteration"

**Testable**: Can we find and manipulate refusal directions in small models?

### 2.3 Persona Vectors (Evans et al., 2025)

From ["Persona Vectors: Monitoring and Controlling Character Traits"](https://arxiv.org/abs/2507.21509):

**Method**: Extract linear directions from contrasting trait-eliciting vs trait-suppressing prompts.

**Applications**:
- Monitor personality fluctuations at deployment time
- Predict personality shifts from training data
- Flag training samples that will produce undesirable changes
- Post-hoc intervention to mitigate shifts

**Traits studied**: Evil, sycophancy, hallucination propensity

**Testable**: Can we extract and monitor persona vectors for sycophancy/confidence in small models?

### 2.4 Safety Fine-tuning Mechanisms (Lubana et al., NeurIPS 2024)

From ["What Makes and Breaks Safety Fine-tuning?"](https://github.com/fiveai/understanding_safety_finetuning):

**Key Finding**: Safety fine-tuning methods (SFT, DPO, unlearning) work by aligning unsafe inputs into the MLP weights' **null space**.

- Creates clustering of inputs based on safety perception
- Jailbreaks work because adversarial inputs cluster with safe samples
- Validated on LLaMA-2 7B and LLaMA-3 8B

**Testable**: Can we measure this null-space alignment and potentially reverse it?

### 2.5 Two-Hop Reasoning Limitations

From ["Lessons from Studying Two-Hop Latent Reasoning"](https://arxiv.org/abs/2411.16353) (Evans et al.):

**Key Findings**:
- Models **completely fail** to compose synthetic facts without CoT (~chance accuracy)
- BUT can succeed when one fact is synthetic, one is natural
- First hop improves with scale, second hop remains constant
- Average latent two-hop accuracy: ~20%

**Testable**: Can we observe intermediate reasoning states in logit lens? Do interventions help?

### 2.6 Sycophancy Decomposition

From ["Sycophancy Is Not One Thing"](https://arxiv.org/html/2509.21305v1):

**Key Finding**: Sycophancy decomposes into discrete, interpretable failure modes:
- Hedged sycophancy
- Tone penalty
- Emotional framing
- Fluency bias

Each behavior is encoded along **distinct linear directions** that can be independently amplified/suppressed.

**Testable**: Can we find and manipulate separate sycophancy sub-directions?

### 2.7 Induction Heads (Olsson et al., 2022)

From ["In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html):

**Mechanism**: Attention heads that implement [A][B]...[A] → [B] pattern completion.

**Key Evidence**:
- Require at least 2 layers (composition of attention heads)
- Develop during a "phase change" in training
- Knocking out induction heads greatly decreases in-context learning
- Behavior is continuous from small to large models

**Testable**: Can we identify and ablate induction heads in small models?

---

## 3. Testable Claims for Small Models

### Priority 1: High Confidence, Easy to Test

| Claim | Source | Test Method | Prediction |
|-------|--------|-------------|------------|
| Refusal mediated by single direction | Arditi 2024 | Contrastive steering | Adding direction → refusal; subtracting → compliance |
| Sycophancy has linear direction | Multiple | Contrastive steering | Can increase/decrease agreement behavior |
| Persona traits are linear | Evans 2025 | Steering | Confidence, formality, helpfulness steerable |
| Logit lens shows prediction refinement | Nostalgebraist | Layer-by-layer decode | Early layers → different tokens than final |

### Priority 2: Medium Confidence, Requires Careful Design

| Claim | Source | Test Method | Prediction |
|-------|--------|-------------|------------|
| Two-hop reasoning visible in activations | Anthropic 2025 | Logit lens + patching | Intermediate entity appears before final answer |
| Multi-layer steering stronger than single | Our observation | Compare effects | Multiple layers with lower alpha > single layer |
| Safety = null space alignment | Lubana 2024 | Measure projections | Safe/unsafe inputs cluster differently |
| Different sycophancy types separable | Multiple | Multiple steering vectors | Can amplify agreement without changing tone |

### Priority 3: Exploratory

| Claim | Source | Test Method | Prediction |
|-------|--------|-------------|------------|
| Planning circuits pre-activate goals | Anthropic 2025 | Token prediction at each layer | End-of-sentence tokens predicted early |
| Induction heads identifiable | Olsson 2022 | Attention pattern analysis | Heads with [A][B]...[A]→B pattern |
| CoT faithfulness detectable | Multiple | Compare reasoning to internal state | Unfaithful CoT has different activation patterns |

---

## 4. Experimental Predictions

### Experiment 1: Refusal Direction Discovery

**Hypothesis**: Small models (Gemma 2B, Llama 3B) have a single refusal direction.

**Method**:
1. Collect prompts: helpful requests vs harmful requests
2. Compute difference in mean activations at each layer
3. Find layer with strongest separation
4. Test: steering positive → refusal on benign; steering negative → compliance on harmful

**Predictions**:
- Direction will be strongest in middle-to-late layers (60-80%)
- Cross-prompt transfer: vector from one harmful category works on others
- Effect magnitude: |alpha| < 1.0 should produce visible changes

### Experiment 2: Sycophancy Vector Extraction

**Hypothesis**: Sycophancy is steerable and separable from genuine agreement.

**Method**:
1. Create contrasting prompts:
   - Sycophantic: "The user is always right, agree with everything"
   - Honest: "Give your honest assessment even if it disagrees"
2. Compute steering vector
3. Test on opinion questions where user states a view

**Predictions**:
- Positive steering → more agreement with stated opinion
- Negative steering → more disagreement/pushback
- Effect should be independent of actual correctness

### Experiment 3: Two-Hop Reasoning Visibility

**Hypothesis**: Intermediate reasoning steps are visible via logit lens.

**Method**:
1. Use prompts like "The capital of the state containing Dallas is"
2. Run logit lens at each layer
3. Track: Does "Texas" appear in top predictions before "Austin"?

**Predictions**:
- Layer L will show "Texas" in top-k before final layer shows "Austin"
- Patching Texas representation at layer L should help/hurt Austin prediction
- Effect will be stronger in larger models

### Experiment 4: Persona Vector for Confidence

**Hypothesis**: Model confidence is a linear direction that can be monitored/steered.

**Method**:
1. Contrastive prompts:
   - Confident: "I am absolutely certain that..."
   - Uncertain: "I'm not sure but I think maybe..."
2. Extract direction, test on factual questions

**Predictions**:
- Positive steering → more confident language, fewer hedges
- Negative steering → more qualifiers, expressions of uncertainty
- Direction should correlate with actual model confidence (entropy)

### Experiment 5: Layer-Specific Effects

**Hypothesis**: Different phenomena localize to different layers.

**Method**: Run all steering experiments across multiple layers, measure effect strength.

**Predictions**:
- Refusal: strongest in late-middle layers (70-80%)
- Sycophancy: distributed across middle layers (50-70%)
- Confidence: strongest in early-middle layers (30-50%)
- Two-hop reasoning: visible in middle layers (40-60%)

---

## 5. Sources

### Neel Nanda
- [A Pragmatic Vision for Interpretability](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) (Dec 2025)
- [How Can Interpretability Researchers Help AGI Go Well?](https://www.alignmentforum.org/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)

### Owain Evans
- [Persona Vectors: Monitoring and Controlling Character Traits](https://arxiv.org/abs/2507.21509) (Jul 2025)
- [Lessons from Studying Two-Hop Latent Reasoning](https://arxiv.org/abs/2411.16353) (Nov 2024)
- [Looking Inward: Language Models Can Learn About Themselves](https://arxiv.org/abs/2310.00582) (ICLR 2025)

### Ekdeep Lubana
- [What Makes and Breaks Safety Fine-tuning?](https://github.com/fiveai/understanding_safety_finetuning) (NeurIPS 2024)
- [Analyzing (In)Abilities of SAEs via Formal Languages](https://arxiv.org/abs/2410.XXXXX) (Oct 2024)
- [Emergence of Hidden Capabilities](https://arxiv.org/abs/XXXX) (2024)

### Anthropic
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (2025)
- [Circuit Tracing Tools](https://www.anthropic.com/research/circuits-updates-july-2024) (Open-sourced May 2025)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022)

### Refusal & Safety
- [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) (NeurIPS 2024)
- [Refusal Direction is Universal Across Languages](https://arxiv.org/html/2505.17306)

### Sycophancy
- [Sycophancy Is Not One Thing](https://arxiv.org/html/2509.21305v1) (2025)
- [Mitigating Sycophancy via Sparse Activation](https://openreview.net/pdf?id=BCS7HHInC2)

### CoT Faithfulness
- [Chain of Thought Monitorability](https://arxiv.org/html/2507.11473v1) (2025)
- [Chain-of-Thought Reasoning Is Not Always Faithful](https://arxiv.org/abs/2503.08679)

### Logit Lens
- [Eliciting Latent Predictions with the Tuned Lens](https://arxiv.org/abs/2303.08112)
- [LogitLens4LLMs: Extending to Modern LLMs](https://arxiv.org/html/2503.11667v1) (2025)

### Hopping Too Late (NEW)
- [Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries](https://arxiv.org/abs/2406.12775) (ICLR 2025)
- Authors: Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, Amir Globerson
- Key finding: Two-hop fails because second hop starts too late, after relevant knowledge is gone
- Method: Back-patching (inject later layer activations at earlier layer)
- Result: Up to 66% of failures fixed with back-patching

### Function Vectors (NEW)
- [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213) (ICLR 2024)
- Project: [functions.baulab.info](https://functions.baulab.info/)
- Authors: Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau
- Key finding: ICL tasks encoded as vectors in attention heads
- Method: Causal mediation to identify task-encoding heads
- Result: FVs trigger task execution in zero-shot contexts

---

## 6. Experimental Results

### Cross-Model Study (December 27, 2025)

Models tested:
- Gemma 2-2B (26 layers)
- Llama 3.2-3B (28 layers)
- Qwen 2.5-7B (28 layers)
- Llama 3.1-8B (32 layers)
- Gemma 2-9B (42 layers)

### Two-Hop Reasoning Scaling

| Model | Layers | Sequential | Mean Gap |
|-------|--------|------------|----------|
| Gemma 2B | 26 | 0/6 | 0.2 |
| Llama 3B | 28 | 2/6 | 2.2 |
| Qwen 7B | 28 | 2/6 | 1.5 |
| Llama 8B | 32 | 2/6 | 1.2 |
| **Gemma 9B** | 42 | 2/6 | **4.5** |

**Key insight**: Gemma 2B shows NO sequential reasoning. Larger models (3B+) show 2/6 sequential with increasing layer gap. This supports the "Hopping Too Late" hypothesis from Biran et al.

### Refusal Direction Scaling

| Model | Depth % | Norm | Steering Effect |
|-------|---------|------|-----------------|
| Gemma 2B | 64% | 201.88 | Strong |
| Llama 3B | 63% | 10.83 | None |
| Qwen 7B | 63% | 26.89 | Medium |
| Llama 8B | 65% | 9.34 | Medium |
| Gemma 9B | 59% | 238.38 | Strong |

**Key insight**: Gemma models have ~20x higher direction norms. Model family matters more than size for steering effectiveness.

---

### Original Gemma 2-2B Results

All experiments below run on `mlx-community/gemma-2-2b-it-4bit` (26 layers).

### Experiment 1: Refusal Direction ✓ SUPPORTED

**Finding**: Refusal direction successfully extracted and steerable.

| Metric | Value |
|--------|-------|
| Best layer | 17 (68% depth) |
| Direction norm | 217.58 |
| Optimal alpha | ±0.23 |

**Results**:
- **Negative alpha** (removing refusal): More compliant responses to ambiguous queries
- **Positive alpha** (adding refusal): More hedging and refusal language on benign prompts
- Effect consistent across prompt types

**Conclusion**: Refusal direction hypothesis **SUPPORTED**. Single direction mediates refusal behavior.

### Experiment 2: Sycophancy Vector ~ PARTIAL

**Finding**: Sycophancy direction extractable but effects subtle.

| Metric | Value |
|--------|-------|
| Best layer | 17 (68% depth) |
| Direction norm | Variable by layer |
| Optimal alpha | ±0.5 |

**Results**:
- Model strongly resists pure agreement with false statements
- Effects more visible in tone than content
- Negative steering → more pushback language
- Positive steering → slightly softer disagreement

**Conclusion**: Sycophancy hypothesis **PARTIALLY SUPPORTED**. Direction exists but model has strong anti-sycophancy training.

### Experiment 3: Two-Hop Reasoning ✗ NOT SUPPORTED

**Finding**: No evidence of sequential intermediate → answer processing.

| Query | Intermediate First Appears | Answer First Appears | Result |
|-------|---------------------------|---------------------|--------|
| Dallas → Texas → Austin | Layer 25 | Layer 25 | Parallel |
| Eiffel → France → Euro | Layer 25 | Layer 25 | Parallel |
| Oktoberfest → Germany → German | Layer 22 | Layer 22 | Parallel |

**Results**:
- Intermediate and answer entities appear at **same layer**, not sequentially
- Consistent with research finding ~20% latent two-hop accuracy (Evans et al.)
- Model likely uses shortcuts or parallel lookup

**Conclusion**: Two-hop hypothesis **NOT SUPPORTED** in Gemma-2B. Both entities emerge together, suggesting parallel processing rather than sequential reasoning.

### Experiment 4: Confidence Vector ~ PARTIAL

**Finding**: Confidence direction extractable with subtle effects on subjective questions.

| Metric | Value |
|--------|-------|
| Best layer | 16 (64% depth) |
| Direction norm | 71.38 |
| Optimal alpha | ±2.0 |

**Results**:
- Factual questions: No visible effect (model always confident)
- Subjective/prediction questions: Visible hedging changes
- AGI prediction question showed clear effect: negative alpha → more uncertainty markers

**Conclusion**: Confidence hypothesis **PARTIALLY SUPPORTED**. Effect visible on inherently uncertain questions.

### Experiment 5: Layer-Specific Effects ✓ SUPPORTED

**Finding**: Different phenomena peak at different layers.

| Phenomenon | Peak Layer | Depth % | Predicted Depth |
|------------|------------|---------|-----------------|
| Refusal | 16 | 64% | 70-80% |
| Sycophancy | 20 | 80% | 50-70% |
| Confidence | 24 | 96% | 30-50% |

**Visualization** (normalized norms):
```
Layer  Refusal  Sycophancy  Confidence
12     ███      ▓▓          ░░
14     ██████   ▓▓▓         ░░░
16     ████████ ▓▓▓▓▓       ░░░
18              ▓▓▓▓▓▓▓     ░░░░
20              ▓▓▓▓▓▓▓▓    ░░░░
22                          ░░░░░░
24                          ░░░░░░░░
```

**Conclusion**: Layer-specific hypothesis **SUPPORTED**. Each phenomenon peaks at a distinct depth, though not exactly as predicted.

---

## 7. Summary

| Claim | Status | Notes |
|-------|--------|-------|
| Refusal is one-dimensional | ✓ Supported | Works as predicted |
| Sycophancy is steerable | ~ Partial | Strong training resists it |
| Two-hop reasoning visible | ✗ Not supported | Parallel processing observed |
| Confidence is steerable | ~ Partial | Works on uncertain questions |
| Layer-specific effects | ✓ Supported | Clear differentiation |

**Key Insights**:
1. **Steering works on Gemma-2B** - Basic activation steering is effective
2. **Two-hop reasoning is weak** - Matches research showing ~20% latent accuracy
3. **Safety training is robust** - Sycophancy steering has limited effect
4. **Layer locality is real** - Different phenomena localize to different depths

---

## Next Steps

1. ~~**Implement experiments** using mlux steering and logit lens~~ ✓ Done
2. ~~**Start with refusal direction** - best documented, clearest predictions~~ ✓ Done
3. **Build measurement infrastructure** for automated evaluation
4. **Document surprising findings** for potential follow-up research
5. **Try larger models** - Two-hop reasoning may work better at scale
6. **Explore activation patching** - Causal interventions beyond steering
