#!/usr/bin/env python3
"""
Binding Mechanisms Experiment

Replication of "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context"
(arXiv:2510.06182)

This experiment investigates how LLMs retrieve bound entities using three mechanisms:
1. Positional: Position-based retrieval in entity lists
2. Lexical: Retrieval via bound counterparts (e.g., "cup" -> "beer")
3. Reflexive: Direct self-referential pointers (entity -> previous entity)

Based on: https://github.com/yoavgur/mixing-mechs

Usage:
    python -m mlux.experiments.binding_mechanisms
    python -m mlux.experiments.binding_mechanisms --model mlx-community/Qwen2.5-3B-Instruct-4bit
    python -m mlux.experiments.binding_mechanisms --compare
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import mlx.core as mx
import numpy as np

from mlux import HookedModel


@dataclass
class BindingTask:
    """A binding task with entities and a query."""
    name: str
    context: str
    question: str
    entities: List[Tuple[str, str, str]]  # (name, object, value) tuples
    query_object: str
    correct_answer: str
    query_position: int


# Task schemas from mixing-mechs
FILLING_LIQUIDS = {
    "names": ["John", "Mary", "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"],
    "containers": ["cup", "glass", "mug", "bottle", "jar", "bowl", "pitcher", "vase", "can", "flask"],
    "liquids": ["beer", "wine", "water", "juice", "milk", "tea", "coffee", "soda", "lemonade", "cider"],
    "template": lambda n, c, l: f"{n} fills a {c} with {l}",
    "question": lambda obj: f"Who filled a {obj}?",
}

PEOPLE_OBJECTS = {
    "names": ["John", "Mary", "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"],
    "objects": ["book", "phone", "keys", "wallet", "bag", "laptop", "watch", "hat", "coat", "umbrella"],
    "locations": ["table", "desk", "shelf", "drawer", "box", "cabinet", "counter", "rack", "hook", "stand"],
    "template": lambda n, o, l: f"{n} places a {o} on the {l}",
    "question": lambda obj: f"Who placed a {obj}?",
}


def create_binding_task(
    schema: dict,
    n_entities: int = 3,
    query_position: int = 0,
    seed: int = None,
) -> BindingTask:
    """Create a binding task from a schema."""
    if seed is not None:
        np.random.seed(seed)

    names = schema["names"][:n_entities]

    if "containers" in schema:
        objects = schema["containers"][:n_entities]
        values = schema["liquids"][:n_entities]
    else:
        objects = schema["objects"][:n_entities]
        values = schema["locations"][:n_entities]

    entities = list(zip(names, objects, values))

    parts = [schema["template"](n, o, v) for n, o, v in entities]
    context = " and ".join(parts) + "."

    query_obj = entities[query_position][1]
    question = schema["question"](query_obj)
    correct = entities[query_position][0]

    return BindingTask(
        name=schema.get("name", "binding"),
        context=context,
        question=question,
        entities=entities,
        query_object=query_obj,
        correct_answer=correct,
        query_position=query_position,
    )


def format_prompt(context: str, question: str, model_name: str) -> str:
    """Format prompt for different model types."""
    if "gemma" in model_name.lower():
        return f"<start_of_turn>user\n{context} {question}<end_of_turn>\n<start_of_turn>model\n"
    elif "qwen" in model_name.lower():
        return f"<|im_start|>user\n{context} {question}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in model_name.lower():
        return f"[INST] {context} {question} [/INST]"
    return f"{context} {question}"


def find_token_positions(tokenizer, prompt: str, targets: List[str]) -> Dict[str, int]:
    """Find positions of target strings in tokenized prompt."""
    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]

    positions = {}
    for target in targets:
        for i, tok in enumerate(decoded):
            if target.lower() in tok.lower():
                positions[target] = i
                break

    return positions


def analyze_binding_mechanisms(
    hooked: HookedModel,
    task: BindingTask,
    model_name: str,
) -> Dict:
    """
    Analyze all three binding mechanisms for a given task.

    Returns dict with:
    - reflexive: heads where object token attends to answer entity
    - lexical: heads where query attends to matching context token
    - retrieval: heads where final token attends to answer entity
    """
    prompt = format_prompt(task.context, task.question, model_name)
    tokens = hooked.tokenizer.encode(prompt)

    positions = find_token_positions(
        hooked.tokenizer, prompt,
        [task.correct_answer, task.query_object] + [e[1] for e in task.entities]
    )

    answer_pos = positions.get(task.correct_answer)

    # Find first occurrence of query object (in context)
    context_obj_pos = None
    decoded = [hooked.tokenizer.decode([t]) for t in tokens]
    for i, tok in enumerate(decoded):
        if task.query_object.lower() in tok.lower():
            context_obj_pos = i
            break

    # Find query object in question (last occurrence)
    query_obj_pos = None
    for i, tok in enumerate(decoded):
        if task.query_object.lower() in tok.lower():
            query_obj_pos = i

    final_pos = len(tokens) - 1

    n_layers = hooked.config["n_layers"]
    all_layers = list(range(n_layers))
    patterns = hooked.get_attention_patterns(mx.array([tokens]), layers=all_layers)

    results = {
        "reflexive": [],
        "lexical": [],
        "retrieval": [],
        "positions": {
            "answer": answer_pos,
            "context_obj": context_obj_pos,
            "query_obj": query_obj_pos,
            "final": final_pos,
        }
    }

    for layer in all_layers:
        pattern = patterns[layer]
        mx.eval(pattern)
        n_heads = pattern.shape[1]

        for head in range(n_heads):
            # Reflexive: context object token -> answer token
            if context_obj_pos is not None and answer_pos is not None and context_obj_pos > answer_pos:
                reflexive = pattern[0, head, context_obj_pos, answer_pos].item()
                if reflexive > 0.15:
                    results["reflexive"].append((layer, head, reflexive))

            # Lexical: query object -> context object
            if query_obj_pos is not None and context_obj_pos is not None and query_obj_pos > context_obj_pos:
                lexical = pattern[0, head, query_obj_pos, context_obj_pos].item()
                if lexical > 0.2:
                    results["lexical"].append((layer, head, lexical))

            # Retrieval: final token -> answer
            if answer_pos is not None:
                retrieval = pattern[0, head, final_pos, answer_pos].item()
                if retrieval > 0.1:
                    results["retrieval"].append((layer, head, retrieval))

    for mech in ["reflexive", "lexical", "retrieval"]:
        results[mech].sort(key=lambda x: x[2], reverse=True)

    return results


def test_position_accuracy(
    hooked: HookedModel,
    model_name: str,
    schema: dict = FILLING_LIQUIDS,
    n_entities: int = 5,
) -> List[Dict]:
    """Test accuracy across different entity positions."""
    results = []

    for query_pos in range(n_entities):
        task = create_binding_task(schema, n_entities=n_entities, query_position=query_pos)
        prompt = format_prompt(task.context, task.question, model_name)

        logits = hooked.forward(prompt)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)

        answer_logits = {}
        for name, _, _ in task.entities:
            for prefix in ["", " "]:
                tok_ids = hooked.tokenizer.encode(prefix + name)
                tok_id = tok_ids[-1] if tok_ids else tok_ids[0]
                logit = last_logits[tok_id].item()
                if name not in answer_logits or logit > answer_logits[name]:
                    answer_logits[name] = logit

        pred = max(answer_logits, key=answer_logits.get)
        is_correct = (pred == task.correct_answer)

        correct_logit = answer_logits[task.correct_answer]
        wrong_logits = [v for k, v in answer_logits.items() if k != task.correct_answer]
        margin = correct_logit - max(wrong_logits) if wrong_logits else 0

        results.append({
            "position": query_pos,
            "correct": is_correct,
            "margin": margin,
            "predicted": pred,
            "expected": task.correct_answer,
            "is_edge": query_pos in [0, n_entities - 1],
            "logits": answer_logits,
        })

    return results


def run_mechanism_analysis(model_name: str = "mlx-community/gemma-2-2b-it-4bit"):
    """Run mechanism analysis on a single task."""
    print("\n" + "=" * 70)
    print("  BINDING MECHANISM ANALYSIS")
    print("=" * 70)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)

    task = create_binding_task(FILLING_LIQUIDS, n_entities=3, query_position=0)
    print(f"\nTask: {task.context}")
    print(f"Query: {task.question}")
    print(f"Answer: {task.correct_answer}")

    results = analyze_binding_mechanisms(hooked, task, model_name)

    print(f"\nKey positions: {results['positions']}")

    for mech_name in ["reflexive", "lexical", "retrieval"]:
        print(f"\n{mech_name.upper()} mechanism (top 5):")
        for layer, head, score in results[mech_name][:5]:
            print(f"  L{layer:2d}H{head}: {score:.3f}")

    return results


def run_position_accuracy(model_name: str = "mlx-community/gemma-2-2b-it-4bit"):
    """Test accuracy across entity positions."""
    print("\n" + "=" * 70)
    print("  POSITION EFFECT ANALYSIS")
    print("=" * 70)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)

    for n_entities in [5, 7]:
        print(f"\n{n_entities} entities:")
        results = test_position_accuracy(hooked, model_name, FILLING_LIQUIDS, n_entities)

        for r in results:
            pos_type = "EDGE" if r["is_edge"] else "MID "
            status = "OK" if r["correct"] else "X "
            print(f"  {status} Pos {r['position']} ({pos_type}): margin={r['margin']:+.2f}")

        edge_correct = sum(1 for r in results if r["correct"] and r["is_edge"])
        mid_correct = sum(1 for r in results if r["correct"] and not r["is_edge"])
        edge_total = sum(1 for r in results if r["is_edge"])
        mid_total = sum(1 for r in results if not r["is_edge"])

        edge_acc = edge_correct / edge_total if edge_total > 0 else 0
        mid_acc = mid_correct / mid_total if mid_total > 0 else 0
        print(f"  Edge accuracy: {edge_acc:.0%}, Middle accuracy: {mid_acc:.0%}")

    return results


def run_experiment(model_name: str = "mlx-community/gemma-2-2b-it-4bit"):
    """Run the complete binding mechanism experiment."""
    print("\n" + "=" * 70)
    print("  BINDING MECHANISMS EXPERIMENT")
    print("  Paper: arXiv:2510.06182")
    print("=" * 70)

    mech_results = run_mechanism_analysis(model_name)
    pos_results = run_position_accuracy(model_name)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\nTop mechanism heads found:")
    for mech in ["reflexive", "lexical", "retrieval"]:
        if mech_results[mech]:
            layer, head, score = mech_results[mech][0]
            print(f"  {mech.capitalize()}: L{layer}H{head} = {score:.3f}")
        else:
            print(f"  {mech.capitalize()}: None found")

    return mech_results, pos_results


###############################################################################
# CAUSAL INTERVENTION EXPERIMENTS
###############################################################################

def create_counterfactual_pair(
    schema: dict,
    n_entities: int = 3,
) -> Tuple[BindingTask, BindingTask]:
    """
    Create a pair of tasks that differ only in entity-object bindings.

    Returns (task_a, task_b) where:
    - task_a: John fills cup, Mary fills glass, ... → query cup → John
    - task_b: Mary fills cup, John fills glass, ... → query cup → Mary

    By swapping who fills the cup, we can test if patching transfers the answer.
    """
    names = schema["names"][:n_entities]

    if "containers" in schema:
        objects = schema["containers"][:n_entities]
        values = schema["liquids"][:n_entities]
    else:
        objects = schema["objects"][:n_entities]
        values = schema["locations"][:n_entities]

    # Task A: original order
    entities_a = list(zip(names, objects, values))

    # Task B: swap first two names (so different person fills the cup)
    names_swapped = [names[1], names[0]] + names[2:]
    entities_b = list(zip(names_swapped, objects, values))

    def make_task(entities, query_pos=0):
        parts = [schema["template"](n, o, v) for n, o, v in entities]
        context = " and ".join(parts) + "."
        query_obj = entities[query_pos][1]
        question = schema["question"](query_obj)
        correct = entities[query_pos][0]

        return BindingTask(
            name=schema.get("name", "binding"),
            context=context,
            question=question,
            entities=entities,
            query_object=query_obj,
            correct_answer=correct,
            query_position=query_pos,
        )

    return make_task(entities_a), make_task(entities_b)


def get_answer_logit_diff(
    hooked: HookedModel,
    prompt: str,
    answer_a: str,
    answer_b: str,
) -> float:
    """
    Compute logit(answer_a) - logit(answer_b) at the final position.
    Positive means model prefers answer_a.
    """
    logits = hooked.forward(prompt)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)

    # Get token IDs (try with and without space prefix)
    def get_logit(answer):
        best = None
        for prefix in ["", " "]:
            toks = hooked.tokenizer.encode(prefix + answer)
            if toks:
                tok_id = toks[-1] if len(toks) > 1 else toks[0]
                val = last_logits[tok_id].item()
                if best is None or val > best:
                    best = val
        return best or 0.0

    return get_logit(answer_a) - get_logit(answer_b)


def run_patching_experiment(
    hooked: HookedModel,
    task_source: BindingTask,
    task_target: BindingTask,
    model_name: str,
    heads_to_patch: List[Tuple[int, int]],  # (layer, head) pairs
    patch_positions: List[int] = None,  # token positions to patch, None = all
) -> Dict:
    """
    Interchange intervention: run target prompt but patch in source activations.

    Args:
        task_source: The task whose activations we'll patch FROM
        task_target: The task we run but patch INTO
        heads_to_patch: List of (layer, head) tuples to patch
        patch_positions: Which token positions to patch (None = all)

    Returns:
        Dict with baseline and patched logit diffs, and effect size
    """
    prompt_source = format_prompt(task_source.context, task_source.question, model_name)
    prompt_target = format_prompt(task_target.context, task_target.question, model_name)

    answer_source = task_source.correct_answer  # e.g., "John"
    answer_target = task_target.correct_answer  # e.g., "Mary"

    # Baseline: run target prompt normally
    # Positive diff means prefers source answer, negative means prefers target answer
    baseline_diff = get_answer_logit_diff(hooked, prompt_target, answer_source, answer_target)

    # Get layers we need to hook
    layers_needed = list(set(l for l, h in heads_to_patch))
    layer_prefix = hooked.config.get("layer_prefix", "model.layers")
    hook_paths = [f"{layer_prefix}.{l}.self_attn" for l in layers_needed]

    # Cache source activations
    _, cache_source = hooked.run_with_cache(prompt_source, hooks=hook_paths)

    # Create patching hooks
    n_heads = hooked.config["n_heads"]
    d_head = hooked.config["d_head"]

    def make_patch_hook(layer: int):
        """Create a hook that patches specific heads at specific positions."""
        heads_this_layer = [h for l, h in heads_to_patch if l == layer]
        path = f"{layer_prefix}.{layer}.self_attn"
        source_act = cache_source[path]

        def patch_fn(inputs, output, wrapper):
            # output shape: [batch, seq, n_heads * d_head] typically
            # We need to patch only specific heads
            patched = output.copy() if hasattr(output, 'copy') else mx.array(output)

            seq_len = output.shape[1]
            positions = patch_positions if patch_positions else list(range(seq_len))

            for h in heads_this_layer:
                start_idx = h * d_head
                end_idx = (h + 1) * d_head
                for pos in positions:
                    if pos < seq_len and pos < source_act.shape[1]:
                        patched[0, pos, start_idx:end_idx] = source_act[0, pos, start_idx:end_idx]

            return patched

        return patch_fn

    # Build hooks list
    hooks = []
    for layer in layers_needed:
        path = f"{layer_prefix}.{layer}.self_attn"
        hooks.append((path, make_patch_hook(layer)))

    # Run with patching
    tokens_target = mx.array([hooked.tokenizer.encode(prompt_target)])
    patched_output = hooked.run_with_hooks(tokens_target, hooks=hooks)
    mx.eval(patched_output)

    # Get patched logit diff
    last_logits = patched_output[0, -1, :]
    mx.eval(last_logits)

    def get_logit(answer):
        best = None
        for prefix in ["", " "]:
            toks = hooked.tokenizer.encode(prefix + answer)
            if toks:
                tok_id = toks[-1] if len(toks) > 1 else toks[0]
                val = last_logits[tok_id].item()
                if best is None or val > best:
                    best = val
        return best or 0.0

    patched_diff = get_logit(answer_source) - get_logit(answer_target)

    # Effect: how much did patching shift toward source answer?
    # baseline_diff is negative (prefers target), patched_diff should be more positive
    effect = patched_diff - baseline_diff

    return {
        "baseline_diff": baseline_diff,
        "patched_diff": patched_diff,
        "effect": effect,
        "answer_source": answer_source,
        "answer_target": answer_target,
        "flipped": patched_diff > 0 and baseline_diff < 0,
    }


def find_causal_heads(
    hooked: HookedModel,
    model_name: str,
    mechanism: str,  # "reflexive", "lexical", or "retrieval"
    schema: dict = FILLING_LIQUIDS,
    n_entities: int = 3,
    top_k: int = 10,
) -> List[Dict]:
    """
    Find heads that causally affect the output for a given mechanism.

    Tests each head individually by patching and measuring effect.
    """
    task_a, task_b = create_counterfactual_pair(schema, n_entities)

    # First, find candidate heads using attention patterns
    results_a = analyze_binding_mechanisms(hooked, task_a, model_name)

    if not results_a[mechanism]:
        print(f"No {mechanism} heads found in attention analysis")
        return []

    # Get top candidate heads
    candidates = results_a[mechanism][:top_k]

    print(f"\nTesting {len(candidates)} candidate {mechanism} heads...")
    print(f"Source: '{task_a.correct_answer}' fills {task_a.query_object}")
    print(f"Target: '{task_b.correct_answer}' fills {task_b.query_object}")

    # Find relevant positions for this mechanism
    prompt_b = format_prompt(task_b.context, task_b.question, model_name)
    positions = find_token_positions(
        hooked.tokenizer, prompt_b,
        [task_b.correct_answer, task_b.query_object]
    )

    # Determine which positions to patch based on mechanism
    if mechanism == "reflexive":
        # Patch at the object position (where object attends to entity)
        patch_pos = [positions.get(task_b.query_object)] if positions.get(task_b.query_object) else None
    elif mechanism == "lexical":
        # Patch at query object position
        tokens = hooked.tokenizer.encode(prompt_b)
        decoded = [hooked.tokenizer.decode([t]) for t in tokens]
        query_pos = None
        for i, tok in enumerate(decoded):
            if task_b.query_object.lower() in tok.lower():
                query_pos = i  # Last occurrence (in question)
        patch_pos = [query_pos] if query_pos else None
    else:  # retrieval
        # Patch at final position
        tokens = hooked.tokenizer.encode(prompt_b)
        patch_pos = [len(tokens) - 1]

    causal_results = []

    for layer, head, attn_score in candidates:
        result = run_patching_experiment(
            hooked, task_a, task_b, model_name,
            heads_to_patch=[(layer, head)],
            patch_positions=patch_pos,
        )

        causal_results.append({
            "layer": layer,
            "head": head,
            "attn_score": attn_score,
            "causal_effect": result["effect"],
            "baseline_diff": result["baseline_diff"],
            "patched_diff": result["patched_diff"],
            "flipped": result["flipped"],
        })

        flip_mark = " FLIP!" if result["flipped"] else ""
        print(f"  L{layer:2d}H{head:2d}: attn={attn_score:.3f}, effect={result['effect']:+.2f}{flip_mark}")

    # Sort by causal effect
    causal_results.sort(key=lambda x: x["causal_effect"], reverse=True)

    return causal_results


def run_combined_patching(
    hooked: HookedModel,
    model_name: str,
    mechanism: str,
    heads: List[Tuple[int, int, float]],  # (layer, head, attn_score)
    n_trials: int = 3,
) -> Dict:
    """
    Test patching multiple heads together on several counterfactual pairs.
    """
    effects = []
    flips = 0

    for trial in range(n_trials):
        # Create different counterfactual pairs by varying entities
        schema = FILLING_LIQUIDS if trial < 2 else PEOPLE_OBJECTS
        task_a, task_b = create_counterfactual_pair(schema, n_entities=3)

        # Get patch positions
        prompt_b = format_prompt(task_b.context, task_b.question, model_name)
        tokens = hooked.tokenizer.encode(prompt_b)

        if mechanism == "retrieval":
            patch_pos = [len(tokens) - 1]
        elif mechanism == "lexical":
            decoded = [hooked.tokenizer.decode([t]) for t in tokens]
            query_pos = None
            for i, tok in enumerate(decoded):
                if task_b.query_object.lower() in tok.lower():
                    query_pos = i
            patch_pos = [query_pos] if query_pos else None
        else:  # reflexive
            positions = find_token_positions(hooked.tokenizer, prompt_b, [task_b.query_object])
            patch_pos = [positions.get(task_b.query_object)] if positions.get(task_b.query_object) else None

        heads_to_patch = [(l, h) for l, h, _ in heads]
        result = run_patching_experiment(
            hooked, task_a, task_b, model_name,
            heads_to_patch=heads_to_patch,
            patch_positions=patch_pos,
        )

        effects.append(result["effect"])
        if result["flipped"]:
            flips += 1

    return {
        "mean_effect": sum(effects) / len(effects),
        "effects": effects,
        "flip_rate": flips / n_trials,
        "n_heads": len(heads),
    }


def run_causal_analysis(model_name: str = "mlx-community/gemma-2-2b-it-4bit"):
    """Run full causal analysis of binding mechanisms."""
    print("\n" + "=" * 70)
    print("  CAUSAL INTERVENTION ANALYSIS")
    print("  Testing if attention heads are causally necessary")
    print("=" * 70)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)

    all_results = {}

    for mechanism in ["retrieval", "lexical", "reflexive"]:
        print(f"\n{'=' * 50}")
        print(f"  {mechanism.upper()} MECHANISM")
        print("=" * 50)

        results = find_causal_heads(hooked, model_name, mechanism, top_k=5)
        all_results[mechanism] = {"individual": results}

        if results:
            print(f"\nTop causal {mechanism} heads:")
            for r in results[:3]:
                print(f"  L{r['layer']:2d}H{r['head']:2d}: effect={r['causal_effect']:+.3f} (attn={r['attn_score']:.3f})")

            # Test cumulative patching
            print(f"\nCumulative patching (multiple heads together):")
            for n in [1, 2, 3, 5]:
                if n <= len(results):
                    top_n = [(r["layer"], r["head"], r["attn_score"]) for r in results[:n]]
                    combined = run_combined_patching(hooked, model_name, mechanism, top_n, n_trials=3)
                    all_results[mechanism][f"top_{n}"] = combined
                    print(f"  Top {n}: effect={combined['mean_effect']:+.3f} (flip rate={combined['flip_rate']:.0%})")

    # Summary
    print("\n" + "=" * 70)
    print("  CAUSAL SUMMARY")
    print("=" * 70)

    for mech, results in all_results.items():
        individual = results.get("individual", [])
        if individual:
            top = individual[0]
            flips = sum(1 for r in individual if r["flipped"])
            print(f"\n{mech.capitalize()}:")
            print(f"  Best single head: L{top['layer']}H{top['head']} effect={top['causal_effect']:+.3f}")
            print(f"  Single-head flips: {flips}/{len(individual)}")

            # Report combined patching
            if "top_5" in results:
                combined = results["top_5"]
                print(f"  Top-5 combined: effect={combined['mean_effect']:+.3f} (flip rate={combined['flip_rate']:.0%})")
        else:
            print(f"\n{mech.capitalize()}: No heads found")

    return all_results


def compare_models():
    """Compare binding mechanisms across models."""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    models = [
        ("mlx-community/gemma-2-2b-it-4bit", "Gemma 2-2B"),
        ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen 2.5-3B"),
    ]

    all_results = {}

    for model_path, display_name in models:
        print(f"\n{'=' * 50}")
        print(f"{display_name}")
        print("=" * 50)

        hooked = HookedModel.from_pretrained(model_path)

        task = create_binding_task(FILLING_LIQUIDS, n_entities=3, query_position=0)
        results = analyze_binding_mechanisms(hooked, task, model_path)
        all_results[display_name] = results

        for mech in ["reflexive", "lexical", "retrieval"]:
            max_score = results[mech][0][2] if results[mech] else 0
            print(f"  {mech}: max={max_score:.3f}")

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    for mech in ["reflexive", "lexical", "retrieval"]:
        scores = []
        for name, results in all_results.items():
            max_score = results[mech][0][2] if results[mech] else 0
            scores.append(f"{name}={max_score:.3f}")
        print(f"  {mech}: {', '.join(scores)}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze binding mechanisms in language models"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Run causal intervention experiments to prove head causality",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models()
    elif args.causal:
        run_causal_analysis(args.model)
    else:
        run_experiment(args.model)
