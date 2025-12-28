#!/usr/bin/env python3
"""
Mean Ablation Experiment

Ablate the residual stream at each (layer, position) by replacing it with
the mean activation (averaged across positions in the same prompt).
Shows how much each position at each layer contributes to next-token prediction.

Usage:
    python -m mlux.experiments.ablation
    python -m mlux.experiments.ablation --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python -m mlux.experiments.ablation --prompt "The capital of France is"
    python -m mlux.experiments.ablation --web  # Launch web interface
"""

import argparse
import json
from typing import Dict

import mlx.core as mx
import numpy as np

from mlux import HookedModel


def compute_mean_activations(
    hooked: HookedModel,
    tokens: mx.array,
) -> Dict[int, mx.array]:
    """
    Compute mean residual stream activation at each layer from the prompt itself.

    Using the same-prompt mean (averaged across positions) gives much better
    results than random tokens, which are too out-of-distribution.

    Returns dict mapping layer_idx -> mean activation vector [d_model].
    """
    n_layers = hooked.config["n_layers"]
    layer_prefix = hooked.config.get("layer_prefix", "model.layers")

    # Hook each transformer layer output
    hook_paths = [f"{layer_prefix}.{i}" for i in range(n_layers)]

    _, cache = hooked.run_with_cache(tokens, hooks=hook_paths)

    # Compute mean across positions for each layer
    means = {}
    for i in range(n_layers):
        act = cache[f"{layer_prefix}.{i}"]  # [batch, seq, d_model]
        means[i] = act[0].mean(axis=0)  # mean across seq positions
        mx.eval(means[i])

    return means


def run_ablation_sweep(
    hooked: HookedModel,
    prompt: str,
    mean_activations: Dict[int, mx.array],
) -> Dict:
    """
    Ablate each (layer, position) and measure effect on next-token probability.

    Returns dict with:
    - tokens: list of token strings
    - baseline_prob: probability of correct next token without ablation
    - effects: [n_layers+1, seq_len] array of probability drops (first row is embedding)
    """
    tokens = hooked.tokenizer.encode(prompt)
    token_strs = [hooked.tokenizer.decode([t]) for t in tokens]
    seq_len = len(tokens)

    n_layers = hooked.config["n_layers"]
    layer_prefix = hooked.config.get("layer_prefix", "model.layers")

    # Get baseline prediction
    logits = hooked.forward(prompt)
    mx.eval(logits)
    baseline_probs = mx.softmax(logits[0, -1, :], axis=-1)
    top_token = mx.argmax(baseline_probs).item()
    baseline_prob = baseline_probs[top_token].item()
    top_token_str = hooked.tokenizer.decode([top_token])

    print(f"\nPrompt: {prompt}")
    print(f"Baseline prediction: '{top_token_str}' (p={baseline_prob:.3f})")
    print(f"\nRunning ablation sweep over {n_layers} layers x {seq_len} positions...")

    tokens_arr = mx.array([tokens])

    # Layer ablations
    # effects[layer, pos] = probability drop when ablating that position at that layer
    layer_effects = np.zeros((n_layers, seq_len))

    # Pre-create all ablation hooks to avoid closure issues
    def create_ablate_hook(ablate_pos, mean_vec):
        """Create ablation hook with explicit closure capture."""
        def hook_fn(inputs, output, wrapper):
            batch, seq, d = output.shape
            parts = []
            if ablate_pos > 0:
                parts.append(output[:, :ablate_pos, :])
            parts.append(mean_vec.reshape(1, 1, -1))
            if ablate_pos < seq - 1:
                parts.append(output[:, ablate_pos + 1:, :])
            return mx.concatenate(parts, axis=1)
        return hook_fn

    # Ablate at each layer
    for layer_idx in range(n_layers):
        hook_path = f"{layer_prefix}.{layer_idx}"
        mean_act = mean_activations[layer_idx]

        # Ablate each position
        for pos in range(seq_len):
            hook_fn = create_ablate_hook(pos, mean_act)
            hooks = [(hook_path, hook_fn)]
            ablated_output = hooked.run_with_hooks(tokens_arr, hooks=hooks)
            mx.eval(ablated_output)

            ablated_logits = ablated_output[0, -1, :]
            # Handle potential NaN/Inf from unstable ablations
            if mx.any(mx.isnan(ablated_logits)).item() or mx.any(mx.isinf(ablated_logits)).item():
                ablated_prob = 0.0  # Treat as complete destruction
            else:
                ablated_probs = mx.softmax(ablated_logits, axis=-1)
                ablated_prob = ablated_probs[top_token].item()

            # Effect = how much probability dropped
            layer_effects[layer_idx, pos] = baseline_prob - ablated_prob

        max_effect = layer_effects[layer_idx].max()
        max_pos = layer_effects[layer_idx].argmax()
        print(f"  L{layer_idx:2d}: max effect = {max_effect:+.3f} at pos {max_pos} ('{token_strs[max_pos]}')")

    return {
        "tokens": token_strs,
        "baseline_prob": baseline_prob,
        "baseline_token": top_token_str,
        "effects": layer_effects,
        "layer_labels": [f"L{i}" for i in range(n_layers)],
    }


def print_ablation_grid(results: Dict, top_k: int = 10):
    """Print the most impactful ablation positions."""
    effects = results["effects"]
    tokens = results["tokens"]
    layer_labels = results.get("layer_labels", [f"L{i}" for i in range(effects.shape[0])])
    n_layers, seq_len = effects.shape

    print(f"\n{'=' * 60}")
    print("  TOP ABLATION EFFECTS")
    print(f"{'=' * 60}")
    print(f"\nBaseline: '{results['baseline_token']}' (p={results['baseline_prob']:.3f})")

    # Find top effects
    flat_idx = np.argsort(effects.flatten())[::-1][:top_k]

    print(f"\nTop {top_k} positions where ablation hurts most:")
    for rank, idx in enumerate(flat_idx):
        layer = idx // seq_len
        pos = idx % seq_len
        effect = effects[layer, pos]
        token = tokens[pos]
        label = layer_labels[layer]
        print(f"  {rank + 1}. {label:5s} pos {pos:2d} ('{token}'): -{effect:.3f}")

    # Print layer summary (sum across positions)
    print(f"\n{'=' * 60}")
    print("  LAYER IMPORTANCE (summed across positions)")
    print(f"{'=' * 60}")
    layer_importance = effects.sum(axis=1)
    max_layer_imp = max(abs(layer_importance.min()), layer_importance.max()) if layer_importance.max() > 0 else 1
    for layer in range(n_layers):
        bar_len = int(abs(layer_importance[layer]) / max_layer_imp * 30) if max_layer_imp > 0 else 0
        bar = "█" * bar_len
        label = layer_labels[layer]
        print(f"  {label:5s} {bar} {layer_importance[layer]:.3f}")

    # Print position summary (sum across layers)
    print(f"\n{'=' * 60}")
    print("  POSITION IMPORTANCE (summed across layers)")
    print(f"{'=' * 60}")
    pos_importance = effects.sum(axis=0)
    max_pos_imp = max(abs(pos_importance.min()), pos_importance.max()) if pos_importance.max() > 0 else 1
    for pos in range(seq_len):
        bar_len = int(abs(pos_importance[pos]) / max_pos_imp * 30) if max_pos_imp > 0 else 0
        bar = "█" * bar_len
        print(f"  {pos:2d} '{tokens[pos]:12s}' {bar} {pos_importance[pos]:.3f}")


def run_experiment(
    model_name: str = "mlx-community/gemma-2-2b-it-4bit",
    prompt: str = "The capital of France is",
):
    """Run the full ablation experiment."""
    print("\n" + "=" * 60)
    print("  MEAN ABLATION EXPERIMENT")
    print("=" * 60)

    print(f"\nLoading {model_name}...")
    hooked = HookedModel.from_pretrained(model_name)

    # Tokenize prompt
    tokens = hooked.tokenizer.encode(prompt)
    tokens_arr = mx.array([tokens])

    print(f"\nComputing mean activations from prompt ({len(tokens)} tokens)...")
    mean_activations = compute_mean_activations(hooked, tokens_arr)

    results = run_ablation_sweep(hooked, prompt, mean_activations)

    print_ablation_grid(results)

    return results


###############################################################################
# WEB INTERFACE
###############################################################################

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 95vw; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; font-size: 24px; }
        .controls {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .top-row {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
            min-width: 280px;
        }
        .input-row { display: flex; gap: 10px; align-items: flex-start; }
        textarea {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            min-height: 60px;
        }
        .btn-group { display: flex; flex-direction: column; gap: 5px; }
        button {
            padding: 10px 20px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            white-space: nowrap;
        }
        button:hover { background: #3a8eef; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        button.secondary {
            background: #e0e0e0;
            color: #333;
        }
        button.secondary:hover { background: #d0d0d0; }
        .info {
            margin-top: 10px;
            font-size: 13px;
            color: #666;
        }
        .baseline {
            font-weight: 500;
            color: #333;
        }
        .grid-container {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 0;
            text-align: center;
        }
        th {
            font-weight: 500;
            color: #666;
            padding: 4px 2px;
        }
        .row-header {
            text-align: right;
            padding-right: 8px;
            color: #666;
            font-weight: 500;
        }
        .cell {
            width: 28px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: rgba(0,0,0,0.3);
            cursor: default;
            position: relative;
            margin: 1px;
            border-radius: 2px;
        }
        .cell:hover {
            outline: 2px solid #333;
            z-index: 10;
        }
        .cell .value {
            opacity: 0;
            transition: opacity 0.1s;
        }
        .cell:hover .value {
            opacity: 1;
            color: #333;
            font-weight: 500;
            text-shadow: 0 0 2px white, 0 0 2px white, 0 0 2px white;
        }
        .token-header {
            max-width: 100px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 11px;
            text-align: left;
        }
        .legend {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            font-size: 12px;
            color: #666;
        }
        .legend-gradient {
            width: 150px;
            height: 16px;
            background: linear-gradient(to right, #2166ac, #67a9cf, #d1e5f0, #ffffff, #fddbc7, #ef8a62, #b2182b);
            border-radius: 2px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .status {
            font-size: 12px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ablation Viewer</h1>

        <div class="controls">
            <div class="top-row">
                <select id="model-select" onchange="swapModel()">
                    {% for m in cached_models %}
                    <option value="{{ m }}"{% if m == model_name %} selected{% endif %}>{{ m }}</option>
                    {% endfor %}
                </select>
                <span id="status" class="status"></span>
            </div>
            <div class="input-row">
                <textarea id="prompt" placeholder="Enter prompt...">X A V P Q R T X A V P Q R</textarea>
                <div class="btn-group">
                    <button id="run-btn" onclick="runAblation()">Run Ablation</button>
                    <button id="chat-btn" class="secondary" onclick="applyChatFormat()">Chat Format</button>
                </div>
            </div>
            <div class="info">
                <span id="baseline" class="baseline"></span>
            </div>
        </div>

        <div class="grid-container">
            <div id="grid" class="loading">Enter a prompt and click "Run Ablation"</div>
        </div>

        <div class="legend">
            <span>Helps prediction</span>
            <div class="legend-gradient"></div>
            <span>Hurts prediction</span>
        </div>
    </div>

    <script>
        let currentModel = '{{ model_name }}';
        let nLayers = {{ n_layers }};

        async function swapModel() {
            const select = document.getElementById('model-select');
            const newModel = select.value;
            if (newModel === currentModel) return;

            const status = document.getElementById('status');
            status.textContent = 'Loading model...';
            select.disabled = true;

            try {
                const response = await fetch('/swap_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: newModel})
                });
                const result = await response.json();
                if (result.success) {
                    currentModel = result.model_name;
                    nLayers = result.n_layers;
                    status.textContent = `${nLayers} layers`;
                } else {
                    status.textContent = 'Error: ' + result.error;
                    select.value = currentModel;
                }
            } catch (err) {
                status.textContent = 'Error loading model';
                select.value = currentModel;
            } finally {
                select.disabled = false;
            }
        }

        async function applyChatFormat() {
            const prompt = document.getElementById('prompt').value;
            try {
                const response = await fetch('/chat_template', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt})
                });
                const result = await response.json();
                if (result.formatted) {
                    document.getElementById('prompt').value = result.formatted;
                }
            } catch (err) {
                console.error('Failed to apply chat format:', err);
            }
        }

        function getColor(value, maxAbs) {
            if (maxAbs === 0) return 'rgb(255,255,255)';
            const normalized = Math.max(-1, Math.min(1, value / maxAbs));

            if (normalized > 0) {
                const r = 255;
                const g = Math.round(255 - normalized * 116);
                const b = Math.round(255 - normalized * 212);
                return `rgb(${r},${g},${b})`;
            } else {
                const factor = -normalized;
                const r = Math.round(255 - factor * 222);
                const g = Math.round(255 - factor * 88);
                const b = 255;
                return `rgb(${r},${g},${b})`;
            }
        }

        let allEffects = [];
        let allLabels = [];
        let tokens = [];

        function renderTable() {
            const grid = document.getElementById('grid');
            if (allEffects.length === 0) return;

            let maxAbs = 0;
            for (const row of allEffects) {
                for (const val of row) {
                    maxAbs = Math.max(maxAbs, Math.abs(val));
                }
            }

            // Transposed: rows = tokens, columns = layers
            let html = '<table>';
            html += '<tr><th></th>';
            for (let layer = 0; layer < allEffects.length; layer++) {
                const label = allLabels[layer] || `L${layer}`;
                html += `<th>${label}</th>`;
            }
            html += '</tr>';

            for (let pos = 0; pos < tokens.length; pos++) {
                const tok = tokens[pos].replace(/</g, '&lt;').replace(/>/g, '&gt;');
                html += `<tr><td class="row-header token-header" title="${tok}">${tok}</td>`;
                for (let layer = 0; layer < allEffects.length; layer++) {
                    const val = allEffects[layer][pos];
                    const color = getColor(val, maxAbs);
                    const displayVal = val >= 0 ? '+' + val.toFixed(2) : val.toFixed(2);
                    const label = allLabels[layer] || `L${layer}`;
                    html += `<td><div class="cell" style="background:${color}" title="${tok} ${label}: ${displayVal}"><span class="value">${displayVal}</span></div></td>`;
                }
                html += '</tr>';
            }

            html += '</table>';
            grid.innerHTML = html;
        }

        async function runAblation() {
            const prompt = document.getElementById('prompt').value;
            const btn = document.getElementById('run-btn');
            const grid = document.getElementById('grid');

            btn.disabled = true;
            btn.textContent = 'Running...';

            allEffects = [];
            allLabels = [];
            tokens = [];

            grid.innerHTML = '<div class="loading">Running ablation...</div>';

            try {
                const response = await fetch('/ablate_stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt})
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.type === 'init') {
                                    tokens = data.tokens;
                                    document.getElementById('baseline').textContent =
                                        `Baseline: "${data.baseline_token}" (p=${data.baseline_prob.toFixed(3)})`;
                                } else if (data.type === 'layer') {
                                    allEffects.push(data.effects);
                                    allLabels.push(data.label);
                                    renderTable();
                                } else if (data.type === 'done') {
                                    renderTable();
                                }
                            } catch (e) {
                                console.error('Parse error:', e, line);
                            }
                        }
                    }
                }

            } catch (err) {
                grid.innerHTML = '<div class="loading">Error: ' + err.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Ablation';
            }
        }

        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                runAblation();
            }
        });
    </script>
</body>
</html>
"""


def get_cached_models() -> list:
    """Get list of mlx-community models in the HuggingFace cache."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    models = []
    if os.path.exists(cache_dir):
        for name in os.listdir(cache_dir):
            if name.startswith("models--mlx-community--"):
                model_name = name.replace("models--", "").replace("--", "/")
                models.append(model_name)
    return sorted(models)


def create_app(model_name: str):
    """Create Flask app for ablation viewer."""
    from flask import Flask, request, jsonify, Response, render_template_string

    app = Flask(__name__)

    # Global state
    state = {"hooked": None, "model_name": model_name, "n_layers": 0}
    cached_models = get_cached_models()

    def load_model(name: str):
        print(f"Loading {name}...")
        state["hooked"] = HookedModel.from_pretrained(name)
        state["model_name"] = name
        state["n_layers"] = state["hooked"].config["n_layers"]
        print(f"Model loaded. {state['n_layers']} layers.")
        return state["n_layers"]

    load_model(model_name)

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            model_name=state["model_name"],
            n_layers=state["n_layers"],
            cached_models=cached_models if state["model_name"] in cached_models else [state["model_name"]] + cached_models
        )

    @app.route("/models")
    def models():
        cached = get_cached_models()
        if state["model_name"] not in cached:
            cached.insert(0, state["model_name"])
        return jsonify(cached)

    @app.route("/swap_model", methods=["POST"])
    def swap_model():
        data = request.json
        new_model = data.get("model")
        try:
            n_layers = load_model(new_model)
            return jsonify({"success": True, "model_name": state["model_name"], "n_layers": n_layers})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route("/chat_template", methods=["POST"])
    def chat_template():
        data = request.json
        prompt = data.get("prompt", "")
        hooked = state["hooked"]
        try:
            if hasattr(hooked.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted = hooked.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return jsonify({"formatted": formatted})
        except:
            pass
        return jsonify({"formatted": None})

    @app.route("/ablate_stream", methods=["POST"])
    def ablate_stream():
        """Stream ablation progress via SSE."""
        data = request.json
        prompt = data.get("prompt", "The capital of France is")

        def generate():
            hooked = state["hooked"]
            tokens = hooked.tokenizer.encode(prompt)
            tokens_arr = mx.array([tokens])
            token_strs = [hooked.tokenizer.decode([t]) for t in tokens]
            seq_len = len(tokens)

            n_layers = hooked.config["n_layers"]
            layer_prefix = hooked.config.get("layer_prefix", "model.layers")

            # Get baseline prediction
            logits = hooked.forward(prompt)
            mx.eval(logits)
            baseline_probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_token = mx.argmax(baseline_probs).item()
            baseline_prob = baseline_probs[top_token].item()
            top_token_str = hooked.tokenizer.decode([top_token])

            # Send initial info
            yield f"data: {json.dumps({'type': 'init', 'tokens': token_strs, 'baseline_prob': baseline_prob, 'baseline_token': top_token_str})}\n\n"

            # Compute mean activations
            mean_activations = compute_mean_activations(hooked, tokens_arr)

            # Layer ablations
            def create_ablate_hook(ablate_pos, mean_vec):
                def hook_fn(inputs, output, wrapper):
                    batch, seq, d = output.shape
                    parts = []
                    if ablate_pos > 0:
                        parts.append(output[:, :ablate_pos, :])
                    parts.append(mean_vec.reshape(1, 1, -1))
                    if ablate_pos < seq - 1:
                        parts.append(output[:, ablate_pos + 1:, :])
                    return mx.concatenate(parts, axis=1)
                return hook_fn

            for layer_idx in range(n_layers):
                hook_path = f"{layer_prefix}.{layer_idx}"
                mean_act = mean_activations[layer_idx]
                layer_effects = np.zeros(seq_len)

                for pos in range(seq_len):
                    hook_fn = create_ablate_hook(pos, mean_act)
                    hooks = [(hook_path, hook_fn)]
                    ablated_output = hooked.run_with_hooks(tokens_arr, hooks=hooks)
                    mx.eval(ablated_output)

                    ablated_logits = ablated_output[0, -1, :]
                    if mx.any(mx.isnan(ablated_logits)).item() or mx.any(mx.isinf(ablated_logits)).item():
                        ablated_prob = 0.0
                    else:
                        ablated_probs = mx.softmax(ablated_logits, axis=-1)
                        ablated_prob = ablated_probs[top_token].item()

                    layer_effects[pos] = baseline_prob - ablated_prob

                yield f"data: {json.dumps({'type': 'layer', 'layer': layer_idx, 'label': f'L{layer_idx}', 'effects': layer_effects.tolist()})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    return app


def run_web(model_name: str, port: int = 5002):
    """Run the web interface."""
    app = create_app(model_name)
    print(f"\nAblation Viewer running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean ablation experiment")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt to analyze",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port for web interface (default: 5002)",
    )
    args = parser.parse_args()

    if args.web:
        run_web(args.model, args.port)
    else:
        run_experiment(
            model_name=args.model,
            prompt=args.prompt,
        )
