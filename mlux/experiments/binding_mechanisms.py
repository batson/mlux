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
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        run_experiment(args.model)
