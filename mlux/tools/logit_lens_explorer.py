#!/usr/bin/env python3
"""
Interactive Logit Lens Viewer

Visualize what tokens a model predicts at each layer by projecting
intermediate activations through the unembedding matrix.

Usage:
    python -m mlux.tools.logit_lens_explorer
    python -m mlux.tools.logit_lens_explorer --model mlx-community/Llama-3.2-1B-Instruct-4bit
    python -m mlux.tools.logit_lens_explorer --port 5001
"""

import argparse
import threading
import webbrowser

from mlux import HookedModel
from mlux.utils import get_cached_models
from .logit_lens import LogitLens


def create_app(model_name: str):
    """Create the Flask app for the logit lens viewer."""
    try:
        from flask import Flask, render_template_string, request, jsonify
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(__name__)

    # Get cached models for dropdown
    cached_models = get_cached_models()

    # Model state (mutable)
    state = {"model_name": model_name, "lens": None}

    def load_model(name: str):
        print(f"Loading {name}...")
        hooked = HookedModel.from_pretrained(name)
        state["lens"] = LogitLens(hooked)
        state["model_name"] = name
        print(f"Model loaded. {state['lens'].n_layers} layers.")

    load_model(model_name)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Logit Lens</title>
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
        .input-row {
            display: flex;
            gap: 8px;
            align-items: flex-start;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        #prompt-input {
            flex: 1;
            min-width: 300px;
            padding: 8px 10px;
            font-size: 13px;
            font-family: inherit;
            background: #fff;
            border: 1px solid #ddd;
            color: #333;
            border-radius: 4px;
            resize: vertical;
            min-height: 36px;
            max-height: 150px;
            line-height: 1.4;
        }
        #prompt-input:focus { outline: none; border-color: #aaa; }
        .probe-btn {
            padding: 5px 10px;
            border: 1px solid #ccc;
            background: #fff;
            color: #666;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 11px;
        }
        .probe-btn:hover { border-color: #999; color: #333; }
        .probe-btn.active { background: #333; color: #fff; border-color: #333; }
        .template-btn {
            padding: 5px 10px;
            border: 1px solid #b8a060;
            background: #faf6e8;
            color: #887430;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 11px;
        }
        .template-btn:hover { background: #f0e8d0; }
        .run-btn {
            padding: 5px 14px;
            background: #333;
            color: #fff;
            border: none;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 11px;
        }
        .run-btn:hover { background: #555; }

        .grid-container {
            overflow-x: auto;
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 4px;
        }
        .grid-table {
            border-collapse: collapse;
            font-size: 11px;
        }
        .grid-table th, .grid-table td {
            border: 1px solid #eee;
            padding: 4px 6px;
            text-align: left;
            white-space: nowrap;
        }
        .grid-table th {
            background: #f5f5f3;
            color: #666;
            font-weight: 500;
            font-size: 10px;
        }
        .corner-cell {
            position: sticky;
            left: 0;
            z-index: 2;
        }
        .layer-header {
            text-align: center;
            padding: 6px 4px !important;
        }
        .token-cell {
            background: #f9f9f7;
            color: #555;
            font-weight: 500;
            position: sticky;
            left: 0;
            z-index: 1;
            min-width: 80px;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pred-cell {
            background: #fff;
            cursor: pointer;
            min-width: 65px;
        }
        .pred-cell:hover {
            background: #f0f0ee;
        }
        .pred-cell.selected {
            background: #e8e8e4;
            outline: 1px solid #999;
            outline-offset: -1px;
        }
        .pred-top {
            color: #2a7c4f;
        }
        .pred-others {
            color: #aaa;
            font-size: 9px;
        }
        .loading {
            padding: 40px;
            text-align: center;
            color: #999;
        }
        .info-bar {
            margin-top: 12px;
            padding: 8px 10px;
            background: #f5f5f3;
            border-radius: 3px;
            font-size: 11px;
            color: #666;
        }
        .info-bar strong { color: #333; }
        #model-select {
            padding: 4px 8px;
            font-family: inherit;
            font-size: 11px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            color: #333;
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
        <h1>logit lens</h1>
        <select id="model-select" onchange="modelChanged()">
            {% for m in cached_models %}
            <option value="{{ m }}"{% if m == model_name %} selected{% endif %}>{{ m.replace('mlx-community/', '') }}</option>
            {% endfor %}
        </select>
        <span class="subtitle">{{ n_layers }} layers</span>
    </div>

    <div id="model-loading" class="model-loading" style="display:none">
        <div class="loading-bar"><div class="loading-progress"></div></div>
        <span>loading model...</span>
    </div>

    <div class="input-row">
        <textarea id="prompt-input" placeholder="enter prompt... (shift+enter for newline)" rows="1">The capital of France is</textarea>
        <button class="probe-btn active" data-probe="resid">resid</button>
        <button class="probe-btn" data-probe="mlp_out">mlp</button>
        <button class="probe-btn" data-probe="attn_out">attn</button>
        <button class="template-btn" id="chat-format-btn" onclick="insertTemplate()">chat format</button>
        <button class="run-btn" onclick="runLens()">run</button>
    </div>

    <div class="grid-container">
        <div id="grid-content">
            <div class="loading">enter a prompt and click run</div>
        </div>
    </div>

    <div id="info-bar" class="info-bar" style="display:none"></div>

    <script>
        let currentData = null;
        let currentProbe = 'resid';
        let currentModel = '{{ model_name }}';
        let nLayers = {{ n_layers }};

        async function insertTemplate() {
            const btn = document.getElementById('chat-format-btn');
            btn.disabled = true;
            btn.textContent = '...';

            try {
                const resp = await fetch('/chat_template');
                const data = await resp.json();
                if (data.supported && data.template) {
                    const input = document.getElementById('prompt-input');
                    input.value = data.template;
                    input.style.height = 'auto';
                    input.style.height = input.scrollHeight + 'px';
                } else {
                    alert('No chat template available for this model');
                }
            } finally {
                btn.disabled = false;
                btn.textContent = 'chat format';
            }
        }

        async function modelChanged() {
            const select = document.getElementById('model-select');
            const newModel = select.value;
            if (newModel === currentModel) return;

            // Show loading
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

                // Re-run lens with new model
                await runLens();
            } catch (e) {
                console.error('Failed to swap model:', e);
            } finally {
                document.getElementById('model-loading').style.display = 'none';
                select.disabled = false;
            }
        }

        document.querySelectorAll('.probe-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.probe-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentProbe = btn.dataset.probe;
                runLens();
            });
        });

        document.getElementById('prompt-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                runLens();
            }
        });

        async function runLens() {
            const text = document.getElementById('prompt-input').value;
            if (!text) return;

            document.getElementById('grid-content').innerHTML = '<div class="loading">loading...</div>';
            document.getElementById('info-bar').style.display = 'none';

            const resp = await fetch('/predict_all', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text, probe_type: currentProbe})
            });

            currentData = await resp.json();
            renderGrid();
        }

        function renderGrid() {
            const {tokens, grid, n_layers} = currentData;

            let html = '<table class="grid-table"><thead><tr><th class="corner-cell"></th>';
            // Layer headers - reversed (latest first)
            for (let l = n_layers - 1; l >= 0; l--) {
                html += `<th class="layer-header">${l}</th>`;
            }
            html += '</tr></thead><tbody>';

            // Helper to clean up tokenizer artifacts (Ġ = space in BPE vocab)
            function cleanToken(s) {
                return s.replace(/Ġ/g, '␣').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            }

            // Token rows
            tokens.forEach((tok, ti) => {
                const display = cleanToken(tok.display || tok.text);
                html += `<tr><td class="token-cell" title="${display}">${display}</td>`;

                // Iterate layers in reverse
                for (let li = n_layers - 1; li >= 0; li--) {
                    const preds = grid[ti][li];
                    const top = preds[0];
                    const topDisplay = cleanToken(JSON.stringify(top.token).slice(1,-1));
                    const others = preds.slice(1).map(p =>
                        cleanToken(JSON.stringify(p.token).slice(1,-1))
                    ).join(', ');

                    html += `<td class="pred-cell" data-ti="${ti}" data-li="${li}" onclick="selectCell(${ti},${li})">
                        <div class="pred-top">${topDisplay}</div>
                        <div class="pred-others">${others}</div>
                    </td>`;
                }
                html += '</tr>';
            });

            html += '</tbody></table>';
            document.getElementById('grid-content').innerHTML = html;
        }

        function selectCell(ti, li) {
            document.querySelectorAll('.pred-cell.selected').forEach(el => el.classList.remove('selected'));
            const cell = document.querySelector(`[data-ti="${ti}"][data-li="${li}"]`);
            cell.classList.add('selected');

            const tok = currentData.tokens[ti];
            const preds = currentData.grid[ti][li];
            const cleanTok = (s) => s.replace(/Ġ/g, '␣');
            const info = preds.map((p,i) => `${i+1}. "${cleanTok(p.token)}" (${p.logit.toFixed(1)})`).join('  ');

            const bar = document.getElementById('info-bar');
            bar.innerHTML = `<strong>token ${ti}:</strong> "${cleanTok(tok.display)}"  <strong>layer ${li}:</strong> ${info}`;
            bar.style.display = 'block';
        }

        window.onload = () => runLens();
    </script>
</body>
</html>
'''

    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            model_name=state["model_name"],
            n_layers=state["lens"].n_layers,
            cached_models=cached_models
        )

    @app.route('/tokenize', methods=['POST'])
    def tokenize():
        data = request.json
        text = data.get('text', '')
        tokens = state["lens"].tokenize_with_info(text)
        return jsonify(tokens)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        text = data.get('text', '')
        token_idx = data.get('token_idx', 0)
        probe_type = data.get('probe_type', 'resid')

        results = state["lens"].get_layer_predictions(
            text,
            token_idx,
            probe_type=probe_type,
            top_k=5
        )
        return jsonify(results)

    @app.route('/predict_all', methods=['POST'])
    def predict_all():
        data = request.json
        text = data.get('text', '')
        probe_type = data.get('probe_type', 'resid')

        results = state["lens"].get_all_predictions(text, probe_type=probe_type, top_k=3)
        return jsonify(results)

    @app.route('/swap_model', methods=['POST'])
    def swap_model():
        data = request.json
        new_model = data.get('model', '')
        if new_model and new_model != state["model_name"]:
            load_model(new_model)
        return jsonify({
            "model_name": state["model_name"],
            "n_layers": state["lens"].n_layers
        })

    @app.route('/chat_template', methods=['GET'])
    def chat_template():
        tokenizer = state["lens"].tokenizer
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                msgs = [
                    {'role': 'user', 'content': 'USERTEXT'},
                    {'role': 'assistant', 'content': 'ASSTRESPONSE'}
                ]
                template = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                # Remove trailing eos tokens that some templates add
                template = template.rstrip()
                return jsonify({"template": template, "supported": True})
            except Exception:
                pass
        return jsonify({"template": None, "supported": False})

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive Logit Lens Viewer")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5003,
        help="Port to run the server on (default: 5003)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    args = parser.parse_args()

    app = create_app(args.model)
    url = f"http://{args.host}:{args.port}"
    print(f"\nLogit Lens Viewer running at {url}")
    print("Press Ctrl+C to stop\n")

    # Open browser after short delay (so server is ready)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
