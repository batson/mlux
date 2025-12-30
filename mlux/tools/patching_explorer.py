#!/usr/bin/env python3
"""
Activation Patching Explorer

Patch the residual stream from a source prompt into a target prompt at a selected
position, sweeping through layers to see when predictions change.

Usage:
    python -m mlux.tools.patching_explorer
    python -m mlux.tools.patching_explorer --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python -m mlux.tools.patching_explorer --port 5004
"""

import argparse
import json
import threading
import webbrowser

import mlx.core as mx

from mlux import HookedModel
from mlux.utils import get_cached_models


def create_app(model_name: str):
    """Create the Flask app for the activation patching explorer."""
    try:
        from flask import Flask, render_template_string, request, jsonify
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(__name__)

    cached_models = get_cached_models()

    state = {"model_name": model_name, "hooked": None, "n_layers": 0}

    def load_model(name: str):
        print(f"Loading {name}...")
        state["hooked"] = HookedModel.from_pretrained(name)
        state["model_name"] = name
        state["n_layers"] = state["hooked"].config["n_layers"]
        print(f"Model loaded. {state['n_layers']} layers.")

    load_model(model_name)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Activation Patching</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
            background: #fafaf8;
            color: #333;
        }
        .header {
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 16px;
        }
        h1 { margin: 0; font-size: 1.1em; font-weight: 600; color: #222; }
        .subtitle { color: #999; font-size: 0.8em; }

        .description {
            margin-bottom: 16px;
            color: #555;
            font-size: 12px;
            line-height: 1.5;
            max-width: 800px;
        }

        .prompts-container {
            display: flex;
            gap: 20px;
            margin-bottom: 16px;
        }
        .prompt-section {
            flex: 1;
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 4px;
            padding: 12px;
        }
        .prompt-section h3 {
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: 600;
            color: #666;
        }
        .prompt-section.source h3 { color: #2166ac; }
        .prompt-section.target h3 { color: #b2182b; }

        textarea {
            width: 100%;
            padding: 8px 10px;
            font-size: 13px;
            font-family: inherit;
            background: #fff;
            border: 1px solid #ddd;
            color: #333;
            border-radius: 4px;
            resize: vertical;
            min-height: 60px;
            line-height: 1.4;
        }
        textarea:focus { outline: none; border-color: #aaa; }

        .token-list {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }
        .token-chip {
            padding: 4px 8px;
            background: #f5f5f3;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.1s;
        }
        .token-chip:hover {
            background: #eee;
            border-color: #aaa;
        }
        .token-chip.selected {
            background: #333;
            color: #fff;
            border-color: #333;
        }
        .source .token-chip.selected {
            background: #2166ac;
            border-color: #2166ac;
        }
        .target .token-chip.selected {
            background: #b2182b;
            border-color: #b2182b;
        }

        .btn-row {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        .run-btn {
            padding: 8px 20px;
            background: #333;
            color: #fff;
            border: none;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 12px;
        }
        .run-btn:hover { background: #555; }
        .run-btn:disabled { background: #ccc; cursor: not-allowed; }

        .template-btn {
            padding: 8px 14px;
            border: 1px solid #b8a060;
            background: #faf6e8;
            color: #887430;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 11px;
        }
        .template-btn:hover { background: #f0e8d0; }

        #model-select {
            padding: 4px 8px;
            font-family: inherit;
            font-size: 11px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            color: #333;
        }

        .results-container {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 4px;
            overflow-x: auto;
            padding: 12px;
        }
        .results-table {
            border-collapse: collapse;
            font-size: 11px;
        }
        .results-table th, .results-table td {
            border: 1px solid #eee;
            padding: 6px 8px;
            text-align: center;
            vertical-align: top;
        }
        .results-table th {
            background: #f5f5f3;
            color: #666;
            font-weight: 500;
            font-size: 10px;
        }
        .results-table th.baseline-header {
            background: #f9f9f7;
            color: #b2182b;
            font-weight: 600;
        }
        .layer-header {
            min-width: 55px;
        }
        .pred-cell {
            min-width: 55px;
            white-space: nowrap;
        }
        .pred-top {
            color: #2a7c4f;
            font-weight: 500;
            font-size: 11px;
        }
        .pred-others {
            color: #888;
            font-size: 9px;
        }
        .pred-prob {
            color: #aaa;
            font-size: 8px;
        }
        .changed {
            background: #fff8e1;
        }
        .baseline-cell {
            background: #f9f9f7;
        }

        .loading {
            padding: 40px;
            text-align: center;
            color: #999;
        }
        .model-loading {
            margin-bottom: 12px;
            padding: 8px 10px;
            background: #f5f5f3;
            border-radius: 3px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #666;
        }
        .loading-bar {
            width: 120px;
            height: 4px;
            background: #ddd;
            border-radius: 2px;
            overflow: hidden;
        }
        .loading-progress {
            height: 100%;
            width: 30%;
            background: #666;
            border-radius: 2px;
            animation: loading 1s ease-in-out infinite;
        }
        @keyframes loading {
            0% { margin-left: 0; width: 30%; }
            50% { margin-left: 35%; width: 50%; }
            100% { margin-left: 70%; width: 30%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>activation patching</h1>
        <select id="model-select" onchange="modelChanged()">
            {% for m in cached_models %}
            <option value="{{ m }}"{% if m == model_name %} selected{% endif %}>{{ m.replace('mlx-community/', '') }}</option>
            {% endfor %}
        </select>
        <span class="subtitle">{{ n_layers }} layers</span>
    </div>

    <p class="description">
        Patch the residual stream activation from a <strong style="color:#2166ac">source</strong> prompt
        into a <strong style="color:#b2182b">target</strong> prompt at selected positions.
        Sweep through layers to see when the model's predictions change.
    </p>

    <div id="model-loading" class="model-loading" style="display:none">
        <div class="loading-bar"><div class="loading-progress"></div></div>
        <span>loading model...</span>
    </div>

    <div class="prompts-container">
        <div class="prompt-section source">
            <h3>Source Prompt (patch from)</h3>
            <textarea id="source-prompt" placeholder="Enter source prompt...">Setup: Josh has a yellow book. Jesse has a black book. Alex has a green book.
Answer: The color of Josh's book is</textarea>
            <div id="source-tokens" class="token-list"></div>
        </div>
        <div class="prompt-section target">
            <h3>Target Prompt (patch into)</h3>
            <textarea id="target-prompt" placeholder="Enter target prompt...">Setup: Jesse has a black book. Alex has a green book. Josh has a yellow book.
Answer: The color of Alex's book is</textarea>
            <div id="target-tokens" class="token-list"></div>
        </div>
    </div>

    <div class="btn-row">
        <button class="run-btn" id="tokenize-btn" onclick="tokenize()">Tokenize</button>
        <button class="run-btn" id="patch-btn" onclick="runPatching()" disabled>Run Patching</button>
        <button class="template-btn" onclick="applyChatFormat()">Chat Format</button>
    </div>

    <div class="results-container">
        <div id="results" class="loading">Enter prompts, tokenize, select positions, and run patching</div>
    </div>

    <script>
        let currentModel = '{{ model_name }}';
        let nLayers = {{ n_layers }};
        let sourceTokens = [];
        let targetTokens = [];
        let selectedSourcePos = -1;
        let selectedTargetPos = -1;

        function cleanToken(s) {
            return s.replace(/Ġ/g, '␣').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        async function modelChanged() {
            const select = document.getElementById('model-select');
            const newModel = select.value;
            if (newModel === currentModel) return;

            document.getElementById('model-loading').style.display = 'flex';
            select.disabled = true;

            try {
                const resp = await fetch('/swap_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: newModel})
                });
                const data = await resp.json();
                currentModel = data.model_name;
                nLayers = data.n_layers;
                document.querySelector('.subtitle').textContent = `${nLayers} layers`;

                // Reset state
                sourceTokens = [];
                targetTokens = [];
                selectedSourcePos = -1;
                selectedTargetPos = -1;
                document.getElementById('source-tokens').innerHTML = '';
                document.getElementById('target-tokens').innerHTML = '';
                document.getElementById('patch-btn').disabled = true;
                document.getElementById('results').innerHTML = '<div class="loading">Enter prompts, tokenize, select positions, and run patching</div>';
            } catch (e) {
                console.error('Failed to swap model:', e);
            } finally {
                document.getElementById('model-loading').style.display = 'none';
                select.disabled = false;
            }
        }

        async function applyChatFormat() {
            try {
                const resp = await fetch('/chat_template', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        source: document.getElementById('source-prompt').value,
                        target: document.getElementById('target-prompt').value
                    })
                });
                const data = await resp.json();
                if (data.source) document.getElementById('source-prompt').value = data.source;
                if (data.target) document.getElementById('target-prompt').value = data.target;
            } catch (e) {
                console.error('Failed to apply chat format:', e);
            }
        }

        async function tokenize() {
            const sourceText = document.getElementById('source-prompt').value;
            const targetText = document.getElementById('target-prompt').value;

            const resp = await fetch('/tokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source: sourceText, target: targetText})
            });
            const data = await resp.json();

            sourceTokens = data.source_tokens;
            targetTokens = data.target_tokens;
            selectedSourcePos = -1;
            selectedTargetPos = -1;

            renderTokens('source-tokens', sourceTokens, 'source');
            renderTokens('target-tokens', targetTokens, 'target');

            document.getElementById('patch-btn').disabled = true;
        }

        function renderTokens(containerId, tokens, type) {
            const container = document.getElementById(containerId);
            container.innerHTML = tokens.map((tok, i) =>
                `<span class="token-chip" data-type="${type}" data-pos="${i}" onclick="selectToken('${type}', ${i})">${cleanToken(tok)}</span>`
            ).join('');
        }

        function selectToken(type, pos) {
            // Deselect previous
            document.querySelectorAll(`.token-chip[data-type="${type}"]`).forEach(el => el.classList.remove('selected'));
            // Select new
            document.querySelector(`.token-chip[data-type="${type}"][data-pos="${pos}"]`).classList.add('selected');

            if (type === 'source') {
                selectedSourcePos = pos;
            } else {
                selectedTargetPos = pos;
            }

            // Enable patch button if both selected
            document.getElementById('patch-btn').disabled = !(selectedSourcePos >= 0 && selectedTargetPos >= 0);
        }

        async function runPatching() {
            if (selectedSourcePos < 0 || selectedTargetPos < 0) return;

            const btn = document.getElementById('patch-btn');
            btn.disabled = true;
            btn.textContent = 'Running...';
            document.getElementById('results').innerHTML = '<div class="loading">Running patching sweep...</div>';

            try {
                const resp = await fetch('/patch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        source: document.getElementById('source-prompt').value,
                        target: document.getElementById('target-prompt').value,
                        source_pos: selectedSourcePos,
                        target_pos: selectedTargetPos
                    })
                });
                const data = await resp.json();
                renderResults(data);
            } catch (e) {
                document.getElementById('results').innerHTML = `<div class="loading">Error: ${e.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Patching';
            }
        }

        function renderResults(data) {
            const {baseline, patched_layers} = data;
            const nLayers = patched_layers.length;
            const midpoint = Math.ceil(nLayers / 2);

            // Reversed order: largest layer first (next to baseline), then descending
            // First row: baseline + upper half of layers (n-1 down to midpoint)
            // Second row: lower half (midpoint-1 down to 0)
            let html = '<table class="results-table">';

            // First row header: baseline, then L(n-1), L(n-2), ... L(midpoint)
            html += '<thead><tr>';
            html += '<th class="baseline-header">baseline</th>';
            for (let i = nLayers - 1; i >= midpoint; i--) {
                html += `<th class="layer-header">L${i}</th>`;
            }
            html += '</tr></thead><tbody><tr>';

            // First row data
            html += `<td class="pred-cell baseline-cell">${formatPreds(baseline)}</td>`;
            let prevTop = baseline[0].token;
            for (let i = nLayers - 1; i >= midpoint; i--) {
                const layer = patched_layers[i];
                const isChanged = layer.preds[0].token !== prevTop;
                const cellClass = isChanged ? 'pred-cell changed' : 'pred-cell';
                html += `<td class="${cellClass}">${formatPreds(layer.preds)}</td>`;
            }
            html += '</tr></tbody></table>';

            // Second row: L(midpoint-1), L(midpoint-2), ... L0
            // Add left margin to align with layer columns (skip baseline column)
            html += '<table class="results-table" style="margin-top: 8px; margin-left: 63px;">';
            html += '<thead><tr>';
            for (let i = midpoint - 1; i >= 0; i--) {
                html += `<th class="layer-header">L${i}</th>`;
            }
            html += '</tr></thead><tbody><tr>';

            for (let i = midpoint - 1; i >= 0; i--) {
                const layer = patched_layers[i];
                const isChanged = layer.preds[0].token !== prevTop;
                const cellClass = isChanged ? 'pred-cell changed' : 'pred-cell';
                html += `<td class="${cellClass}">${formatPreds(layer.preds)}</td>`;
            }
            html += '</tr></tbody></table>';

            document.getElementById('results').innerHTML = html;
        }

        function formatPreds(preds) {
            const top = preds[0];
            const others = preds.slice(1);

            const topDisplay = cleanToken(JSON.stringify(top.token).slice(1,-1));
            const topProb = (top.prob * 100).toFixed(1);

            const othersHtml = others.map(p => {
                const display = cleanToken(JSON.stringify(p.token).slice(1,-1));
                const prob = (p.prob * 100).toFixed(1);
                return `${display} ${prob}%`;
            }).join('<br>');

            return `<div class="pred-top">${topDisplay}</div><div class="pred-prob">${topProb}%</div><div class="pred-others">${othersHtml}</div>`;
        }

        // Auto-tokenize on Enter
        document.getElementById('source-prompt').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); tokenize(); }
        });
        document.getElementById('target-prompt').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); tokenize(); }
        });
    </script>
</body>
</html>
'''

    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            model_name=state["model_name"],
            n_layers=state["n_layers"],
            cached_models=cached_models if state["model_name"] in cached_models else [state["model_name"]] + cached_models
        )

    @app.route('/swap_model', methods=['POST'])
    def swap_model():
        data = request.json
        new_model = data.get('model', '')
        if new_model and new_model != state["model_name"]:
            load_model(new_model)
        return jsonify({
            "model_name": state["model_name"],
            "n_layers": state["n_layers"]
        })

    @app.route('/chat_template', methods=['POST'])
    def chat_template():
        data = request.json
        source = data.get('source', '')
        target = data.get('target', '')
        hooked = state["hooked"]

        result = {"source": None, "target": None}
        try:
            if hasattr(hooked.tokenizer, 'apply_chat_template'):
                for key, text in [('source', source), ('target', target)]:
                    messages = [{"role": "user", "content": text}]
                    formatted = hooked.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    result[key] = formatted
        except:
            pass
        return jsonify(result)

    @app.route('/tokenize', methods=['POST'])
    def tokenize():
        data = request.json
        source_text = data.get('source', '')
        target_text = data.get('target', '')
        hooked = state["hooked"]

        source_ids = hooked.tokenizer.encode(source_text)
        target_ids = hooked.tokenizer.encode(target_text)

        source_tokens = [hooked.tokenizer.decode([t]) for t in source_ids]
        target_tokens = [hooked.tokenizer.decode([t]) for t in target_ids]

        return jsonify({
            "source_tokens": source_tokens,
            "target_tokens": target_tokens
        })

    @app.route('/patch', methods=['POST'])
    def patch():
        data = request.json
        source_text = data.get('source', '')
        target_text = data.get('target', '')
        source_pos = data.get('source_pos', 0)
        target_pos = data.get('target_pos', 0)

        hooked = state["hooked"]
        n_layers = state["n_layers"]
        layer_prefix = hooked.config.get("layer_prefix", "model.layers")

        # Tokenize
        source_ids = hooked.tokenizer.encode(source_text)
        target_ids = hooked.tokenizer.encode(target_text)
        source_arr = mx.array([source_ids])
        target_arr = mx.array([target_ids])

        # Get source activations at each layer
        hook_paths = [f"{layer_prefix}.{i}" for i in range(n_layers)]
        _, source_cache = hooked.run_with_cache(source_arr, hooks=hook_paths)

        # Get baseline prediction on target (no patching)
        baseline_logits = hooked.forward(target_text)
        mx.eval(baseline_logits)
        baseline_probs = mx.softmax(baseline_logits[0, -1, :], axis=-1)
        baseline_preds = get_top_k_preds(hooked, baseline_probs, k=3)

        # Patch at each layer and get predictions
        patched_layers = []

        def create_patch_hook(source_act, src_pos, tgt_pos):
            """Create a hook that patches source activation into target at specified position."""
            def hook_fn(inputs, output, wrapper):
                # output: [batch, seq, d_model]
                # Replace target_pos with source activation from src_pos
                batch, seq, d = output.shape

                # Get the source activation vector
                src_vec = source_act[0, src_pos, :].reshape(1, 1, -1)

                # Build patched output
                parts = []
                if tgt_pos > 0:
                    parts.append(output[:, :tgt_pos, :])
                parts.append(src_vec)
                if tgt_pos < seq - 1:
                    parts.append(output[:, tgt_pos + 1:, :])

                return mx.concatenate(parts, axis=1)
            return hook_fn

        for layer_idx in range(n_layers):
            hook_path = f"{layer_prefix}.{layer_idx}"
            source_act = source_cache[hook_path]

            hook_fn = create_patch_hook(source_act, source_pos, target_pos)
            hooks = [(hook_path, hook_fn)]

            patched_output = hooked.run_with_hooks(target_arr, hooks=hooks)
            mx.eval(patched_output)

            patched_logits = patched_output[0, -1, :]

            if mx.any(mx.isnan(patched_logits)).item() or mx.any(mx.isinf(patched_logits)).item():
                # Handle unstable patching
                preds = [{"token": "???", "prob": 0.0}] * 3
            else:
                patched_probs = mx.softmax(patched_logits, axis=-1)
                preds = get_top_k_preds(hooked, patched_probs, k=3)

            patched_layers.append({
                "layer": layer_idx,
                "preds": preds
            })

        return jsonify({
            "baseline": baseline_preds,
            "patched_layers": patched_layers
        })

    def get_top_k_preds(hooked, probs, k=3):
        """Get top-k predictions with tokens and probabilities."""
        top_k_indices = mx.argpartition(probs, -k)[-k:]
        top_k_probs = probs[top_k_indices]

        # Sort by probability descending
        sort_idx = mx.argsort(top_k_probs)[::-1]
        top_k_indices = top_k_indices[sort_idx]
        top_k_probs = top_k_probs[sort_idx]

        mx.eval(top_k_indices, top_k_probs)

        preds = []
        for i in range(k):
            token_id = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            token_str = hooked.tokenizer.decode([token_id])
            preds.append({"token": token_str, "prob": prob})

        return preds

    return app


def main():
    parser = argparse.ArgumentParser(description="Activation Patching Explorer")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5004,
        help="Port to run the server on (default: 5004)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    args = parser.parse_args()

    app = create_app(args.model)
    url = f"http://{args.host}:{args.port}"
    print(f"\nActivation Patching Explorer running at {url}")
    print("Press Ctrl+C to stop\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
