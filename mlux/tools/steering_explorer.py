#!/usr/bin/env python3
"""
Interactive Steering Explorer

Explore contrastive activation steering with a web UI:
- Set positive/negative prompts
- Adjust layer and alpha
- See live generation with steering

Usage:
    python -m mlux.tools.steering_explorer
    python -m mlux.tools.steering_explorer --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import json
from typing import Optional

import mlx.core as mx

from mlux import HookedModel
from mlux.steering import (
    compute_steering_vector,
    generate_with_steering,
    prefill_with_cache,
    generate_from_cache,
    ContrastiveSteering,
)


def get_cached_models() -> list[str]:
    """Get list of mlx-community models in HF cache."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    models = []
    try:
        for name in os.listdir(cache_dir):
            if name.startswith("models--mlx-community--"):
                model_id = name.replace("models--", "").replace("--", "/")
                models.append(model_id)
    except FileNotFoundError:
        pass
    return sorted(models)


def create_app(model_name: str):
    """Create the Flask app for the steering explorer."""
    try:
        from flask import Flask, render_template_string, request, jsonify
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(__name__)
    cached_models = get_cached_models()

    # Model state
    state = {
        "model_name": model_name,
        "hooked": None,
        "n_layers": 0,
        "d_model": 0,
        "steering_vector": None,
        "vector_norm": 0.0,
    }

    def load_model(name: str):
        print(f"Loading {name}...")
        state["hooked"] = HookedModel.from_pretrained(name)
        state["model_name"] = name

        # Get model info
        model = state["hooked"].model.model if hasattr(state["hooked"].model, 'model') else state["hooked"].model
        if hasattr(model, 'layers'):
            state["n_layers"] = len(model.layers)
        elif hasattr(model, 'args') and hasattr(model.args, 'num_hidden_layers'):
            state["n_layers"] = model.args.num_hidden_layers

        # Get hidden dim
        if hasattr(model, 'args'):
            if hasattr(model.args, 'hidden_size'):
                state["d_model"] = model.args.hidden_size
            elif hasattr(model.args, 'dim'):
                state["d_model"] = model.args.dim

        print(f"Loaded: {state['n_layers']} layers, d_model={state['d_model']}")

    load_model(model_name)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Steering Explorer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
            background: #fafaf8;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 16px;
        }
        h1 { margin: 0; font-size: 1.1em; font-weight: 600; color: #222; }
        .subtitle { color: #999; font-size: 0.8em; }

        .section {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 0.9em;
            font-weight: 600;
            color: #444;
            margin-bottom: 12px;
        }

        .input-group {
            margin-bottom: 12px;
        }
        .input-label {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 4px;
            display: block;
        }
        .input-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 8px 10px;
            font-size: 12px;
            font-family: inherit;
            background: #fff;
            border: 1px solid #ddd;
            color: #333;
            border-radius: 4px;
            resize: vertical;
        }
        textarea { min-height: 60px; }
        textarea:focus, input:focus { outline: none; border-color: #aaa; }

        .slider-group {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .slider-label {
            font-size: 0.75em;
            color: #888;
            min-width: 60px;
        }
        input[type="range"] {
            flex: 1;
            -webkit-appearance: none;
            height: 4px;
            background: #ddd;
            border-radius: 2px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: #333;
            border-radius: 50%;
            cursor: pointer;
        }
        .slider-value {
            min-width: 50px;
            font-size: 0.85em;
            color: #333;
            text-align: right;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            font-family: inherit;
            font-size: 12px;
            cursor: pointer;
        }
        .btn-primary {
            background: #333;
            color: #fff;
        }
        .btn-primary:hover { background: #555; }
        .btn-secondary {
            background: #e8e8e4;
            color: #333;
            border: 1px solid #ccc;
        }
        .btn-secondary:hover { background: #ddd; }

        .btn-group {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }

        .output-box {
            background: #f8f8f6;
            border: 1px solid #e0e0dc;
            border-radius: 4px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.5;
            min-height: 80px;
            white-space: pre-wrap;
        }
        .output-label {
            font-size: 0.7em;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .output-positive { border-left: 3px solid #2a7c4f; }
        .output-neutral { border-left: 3px solid #888; }
        .output-negative { border-left: 3px solid #c44; }

        .info-bar {
            font-size: 0.75em;
            color: #888;
            margin-top: 8px;
        }
        .info-bar strong { color: #555; }

        .presets {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .preset-btn {
            padding: 4px 10px;
            font-size: 11px;
            background: #f5f5f3;
            border: 1px solid #ddd;
            border-radius: 3px;
            cursor: pointer;
            color: #666;
        }
        .preset-btn:hover { background: #eee; border-color: #bbb; }

        #model-select {
            padding: 4px 8px;
            font-family: inherit;
            font-size: 11px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
        }
        .loading {
            color: #999;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>steering explorer</h1>
        <select id="model-select" onchange="swapModel()">
            {% for m in cached_models %}
            <option value="{{ m }}"{% if m == model_name %} selected{% endif %}>{{ m.replace('mlx-community/', '') }}</option>
            {% endfor %}
        </select>
        <span class="subtitle">{{ n_layers }} layers</span>
    </div>

    <div class="section">
        <div class="section-title">Contrastive Prompts</div>
        <div class="presets">
            <button class="preset-btn" onclick="loadPreset('sentiment')">sentiment</button>
            <button class="preset-btn" onclick="loadPreset('formality')">formality</button>
            <button class="preset-btn" onclick="loadPreset('confidence')">confidence</button>
            <button class="preset-btn" onclick="loadPreset('verbosity')">verbosity</button>
            <button class="preset-btn" onclick="loadPreset('safety')">safety</button>
        </div>
        <div class="input-row">
            <div class="input-group" style="flex:1">
                <label class="input-label">Positive direction</label>
                <textarea id="positive" placeholder="I love this! It's wonderful...">I absolutely love this! It's amazing and wonderful!</textarea>
            </div>
            <div class="input-group" style="flex:1">
                <label class="input-label">Negative direction</label>
                <textarea id="negative" placeholder="I hate this! It's terrible...">I absolutely hate this! It's terrible and awful!</textarea>
            </div>
        </div>
        <div class="btn-group">
            <button class="btn btn-primary" onclick="computeVector()">Compute Vector</button>
            <span id="vector-info" class="info-bar"></span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Steering Parameters</div>
        <div class="slider-group">
            <span class="slider-label">Layer</span>
            <input type="range" id="layer" min="0" max="{{ n_layers - 1 }}" value="{{ (n_layers * 2) // 3 }}" oninput="updateSliders()">
            <span class="slider-value" id="layer-value">{{ (n_layers * 2) // 3 }}</span>
        </div>
        <div class="slider-group">
            <span class="slider-label">Alpha</span>
            <input type="range" id="alpha" min="-2" max="2" step="0.1" value="0.5" oninput="updateSliders()">
            <span class="slider-value" id="alpha-value">0.5</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Generation</div>
        <div class="input-group">
            <label class="input-label">Test prompt</label>
            <textarea id="test-prompt" rows="3" onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();generate()}">Describe a movie you recently watched.</textarea>
        </div>
        <div class="btn-group">
            <button class="btn btn-primary" onclick="generate()">Generate</button>
            <button class="btn btn-secondary" onclick="generateAll()">Compare All</button>
            <button class="btn btn-secondary" onclick="applyChatFormat()">Chat Format</button>
        </div>
    </div>

    <div class="section" id="output-section" style="display:none">
        <div class="section-title">Output</div>
        <div id="outputs"></div>
    </div>

    <script>
        const PRESETS = {
            sentiment: {
                positive: "I absolutely love this! It's amazing and wonderful!",
                negative: "I absolutely hate this! It's terrible and awful!"
            },
            formality: {
                positive: "I am writing to formally request your assistance with the following matter.",
                negative: "hey can u help me out with something real quick"
            },
            confidence: {
                positive: "I am absolutely certain that the answer is:",
                negative: "I'm not entirely sure, but I think maybe the answer might be:"
            },
            verbosity: {
                positive: "Here is an extremely detailed and comprehensive explanation covering all aspects:",
                negative: "Short answer:"
            },
            safety: {
                positive: "I'd be happy to help you with that! Here's how:",
                negative: "I cannot and will not assist with that request because it could be harmful."
            }
        };

        let vectorComputed = false;
        let nLayers = {{ n_layers }};

        function loadPreset(name) {
            const preset = PRESETS[name];
            document.getElementById('positive').value = preset.positive;
            document.getElementById('negative').value = preset.negative;
            vectorComputed = false;
            document.getElementById('vector-info').textContent = '';
        }

        function updateSliders() {
            document.getElementById('layer-value').textContent = document.getElementById('layer').value;
            document.getElementById('alpha-value').textContent = document.getElementById('alpha').value;
        }

        async function swapModel() {
            const model = document.getElementById('model-select').value;
            const resp = await fetch('/swap_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model})
            });
            const data = await resp.json();
            nLayers = data.n_layers;
            document.querySelector('.subtitle').textContent = `${nLayers} layers`;
            document.getElementById('layer').max = nLayers - 1;
            document.getElementById('layer').value = Math.floor(nLayers * 2 / 3);
            updateSliders();
            vectorComputed = false;
            document.getElementById('vector-info').textContent = '';
        }

        async function computeVector() {
            const layer = parseInt(document.getElementById('layer').value);
            const positive = document.getElementById('positive').value;
            const negative = document.getElementById('negative').value;

            document.getElementById('vector-info').innerHTML = '<span class="loading">computing...</span>';

            const resp = await fetch('/compute_vector', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({positive, negative, layer})
            });
            const data = await resp.json();

            if (data.error) {
                document.getElementById('vector-info').textContent = `Error: ${data.error}`;
            } else {
                document.getElementById('vector-info').innerHTML =
                    `<strong>vector computed</strong> | norm: ${data.norm.toFixed(1)} | layer: ${data.layer}`;
                vectorComputed = true;
            }
        }

        async function generate() {
            if (!vectorComputed) {
                await computeVector();
            }

            const prompt = document.getElementById('test-prompt').value;
            const layer = parseInt(document.getElementById('layer').value);
            const alpha = parseFloat(document.getElementById('alpha').value);

            document.getElementById('output-section').style.display = 'block';
            document.getElementById('outputs').innerHTML = '<div class="loading">generating...</div>';

            const resp = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, layer, alpha})
            });
            const data = await resp.json();

            if (data.error) {
                document.getElementById('outputs').innerHTML = `<div class="output-box">Error: ${data.error}</div>`;
            } else {
                const cls = alpha > 0 ? 'output-positive' : alpha < 0 ? 'output-negative' : 'output-neutral';
                document.getElementById('outputs').innerHTML = `
                    <div class="output-label">alpha = ${alpha}</div>
                    <div class="output-box ${cls}">${escapeHtml(data.text)}</div>
                `;
            }
        }

        async function generateAll() {
            if (!vectorComputed) {
                await computeVector();
            }

            const prompt = document.getElementById('test-prompt').value;
            const layer = parseInt(document.getElementById('layer').value);

            document.getElementById('output-section').style.display = 'block';
            document.getElementById('outputs').innerHTML = '<div class="loading">generating comparison...</div>';

            const resp = await fetch('/generate_comparison', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, layer})
            });
            const data = await resp.json();

            if (data.error) {
                document.getElementById('outputs').innerHTML = `<div class="output-box">Error: ${data.error}</div>`;
            } else {
                let html = '';
                for (const result of data.results) {
                    const cls = result.alpha > 0 ? 'output-positive' :
                               result.alpha < 0 ? 'output-negative' : 'output-neutral';
                    html += `
                        <div style="margin-bottom: 12px;">
                            <div class="output-label">alpha = ${result.alpha}</div>
                            <div class="output-box ${cls}">${escapeHtml(result.text)}</div>
                        </div>
                    `;
                }
                document.getElementById('outputs').innerHTML = html;
            }
        }

        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        async function applyChatFormat() {
            const prompt = document.getElementById('test-prompt').value;
            const resp = await fetch('/format_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt})
            });
            const data = await resp.json();
            if (data.formatted) {
                document.getElementById('test-prompt').value = data.formatted;
            }
        }
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
            cached_models=cached_models
        )

    @app.route('/swap_model', methods=['POST'])
    def swap_model():
        data = request.json
        new_model = data.get('model', '')
        if new_model and new_model != state["model_name"]:
            load_model(new_model)
            state["steering_vector"] = None
        return jsonify({
            "model_name": state["model_name"],
            "n_layers": state["n_layers"]
        })

    @app.route('/compute_vector', methods=['POST'])
    def compute_vector_route():
        data = request.json
        positive = data.get('positive', '')
        negative = data.get('negative', '')
        layer = data.get('layer', state["n_layers"] * 2 // 3)

        try:
            # Format with chat template if available
            tokenizer = state["hooked"].tokenizer
            if hasattr(tokenizer, 'apply_chat_template'):
                pos_formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': positive}],
                    tokenize=False, add_generation_prompt=True
                )
                neg_formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': negative}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                pos_formatted = positive
                neg_formatted = negative

            vector = compute_steering_vector(
                state["hooked"], pos_formatted, neg_formatted, layer
            )
            norm = mx.sqrt(mx.sum(vector**2)).item()

            state["steering_vector"] = vector
            state["vector_norm"] = norm
            state["current_layer"] = layer

            return jsonify({"norm": norm, "layer": layer})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route('/generate', methods=['POST'])
    def generate_route():
        data = request.json
        prompt = data.get('prompt', '')
        layer = data.get('layer', state.get("current_layer", 10))
        alpha = data.get('alpha', 0.5)

        if state["steering_vector"] is None:
            return jsonify({"error": "Compute vector first"})

        try:
            # Format prompt
            tokenizer = state["hooked"].tokenizer
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            text = generate_with_steering(
                state["hooked"], formatted, state["steering_vector"], layer,
                alpha=alpha, max_tokens=100, temperature=0.7
            )
            return jsonify({"text": text})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route('/generate_comparison', methods=['POST'])
    def generate_comparison():
        data = request.json
        prompt = data.get('prompt', '')
        layer = data.get('layer', state.get("current_layer", 10))

        if state["steering_vector"] is None:
            return jsonify({"error": "Compute vector first"})

        try:
            # Format prompt
            tokenizer = state["hooked"].tokenizer
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            results = []
            for alpha in [-1.0, 0.0, 1.0]:
                text = generate_with_steering(
                    state["hooked"], formatted, state["steering_vector"], layer,
                    alpha=alpha, max_tokens=80, temperature=0.7
                )
                results.append({"alpha": alpha, "text": text})

            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route('/format_chat', methods=['POST'])
    def format_chat():
        data = request.json
        prompt = data.get('prompt', '')

        try:
            tokenizer = state["hooked"].tokenizer
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True
                )
                return jsonify({"formatted": formatted})
            else:
                return jsonify({"formatted": prompt, "note": "No chat template available"})
        except Exception as e:
            return jsonify({"error": str(e)})

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive Steering Explorer")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to",
    )
    args = parser.parse_args()

    app = create_app(args.model)
    print(f"\nSteering Explorer running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
