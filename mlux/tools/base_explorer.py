#!/usr/bin/env python3
"""
Base Model Explorer - Simple completion generation for base models.

A minimal interactive UI for generating completions from base (non-instruct) models.
Useful for interpretability experiments where you want raw model behavior.

Usage:
    python -m mlux.tools.base_explorer
    python -m mlux.tools.base_explorer --model mlx-community/Qwen2.5-7B-4bit
"""

import argparse
import json
import threading
import webbrowser

import mlx.core as mx

from mlux import HookedModel
from mlux.steering import prefill_with_cache, generate_from_cache_stream


# Base models (no instruction tuning)
BASE_MODELS = [
    "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
    "mlx-community/Qwen2.5-7B-4bit",
    "mlx-community/gemma-2-9b-4bit",
    "mlx-community/Meta-Llama-3.1-8B-4bit",
]


def get_cached_base_models() -> list[str]:
    """Get list of base models in HF cache."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    models = []
    try:
        for name in os.listdir(cache_dir):
            if name.startswith("models--mlx-community--"):
                model_id = name.replace("models--", "").replace("--", "/")
                # Filter to base models (no instruct/-it suffix)
                lower = model_id.lower()
                if not any(x in lower for x in ['instruct', '-it-', '-it_', '-it.']) and not lower.endswith('-it') and not lower.endswith('-it-4bit'):
                    models.append(model_id)
    except FileNotFoundError:
        pass
    return sorted(models)


def create_app(model_name: str):
    """Create the Flask app for the base model explorer."""
    try:
        from flask import Flask, render_template_string, request, jsonify, Response
    except ImportError:
        raise ImportError("Flask required. Install with: pip install mlux[viewer]")

    app = Flask(__name__)
    cached_models = get_cached_base_models()

    # Add target base models even if not cached (for download)
    for m in BASE_MODELS:
        if m not in cached_models:
            cached_models.append(m)
    cached_models = sorted(cached_models)

    # Model state
    state = {
        "model_name": model_name,
        "hooked": None,
        "n_layers": 0,
        "d_model": 0,
    }

    # Lock to prevent concurrent model access (MLX is not thread-safe)
    generation_lock = threading.Lock()

    def load_model(name: str):
        print(f"Loading {name}...")
        state["hooked"] = HookedModel.from_pretrained(name)
        state["model_name"] = name

        # Get model info
        cfg = state["hooked"].config
        state["n_layers"] = cfg.get("n_layers", 0)
        state["d_model"] = cfg.get("n_heads", 0) * cfg.get("d_head", 0)

        print(f"Loaded: {state['hooked']}")

    load_model(model_name)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Base Model Explorer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
            background: #fafaf8;
            color: #333;
            max-width: 800px;
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

        .input-group { margin-bottom: 12px; }
        .input-label {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 4px;
            display: block;
        }
        textarea {
            width: 100%;
            padding: 10px 12px;
            font-size: 13px;
            font-family: inherit;
            background: #fff;
            border: 1px solid #ddd;
            color: #333;
            border-radius: 4px;
            resize: vertical;
            min-height: 100px;
        }
        textarea:focus { outline: none; border-color: #aaa; }

        .slider-group {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .slider-label {
            font-size: 0.75em;
            color: #888;
            min-width: 80px;
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
            min-width: 40px;
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
        .btn-primary { background: #333; color: #fff; }
        .btn-primary:hover { background: #555; }
        .btn-primary:disabled { background: #999; cursor: not-allowed; }
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
            line-height: 1.6;
            min-height: 100px;
            white-space: pre-wrap;
        }
        .prompt-text { color: #666; }
        .completion-text { color: #000; }

        #model-select {
            padding: 4px 8px;
            font-family: inherit;
            font-size: 11px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
        }
        .loading { color: #999; font-style: italic; }

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
    </style>
</head>
<body>
    <div class="header">
        <h1>base model explorer</h1>
        <select id="model-select" onchange="swapModel()">
            {% for m in cached_models %}
            <option value="{{ m }}"{% if m == model_name %} selected{% endif %}>{{ m.replace('mlx-community/', '') }}</option>
            {% endfor %}
        </select>
        <span class="subtitle" id="model-info">{{ n_layers }} layers</span>
    </div>

    <div class="section">
        <div class="section-title">Prompt</div>
        <div class="presets">
            <button class="preset-btn" onclick="loadPreset('factual')">factual</button>
            <button class="preset-btn" onclick="loadPreset('story')">story</button>
            <button class="preset-btn" onclick="loadPreset('code')">code</button>
            <button class="preset-btn" onclick="loadPreset('reasoning')">reasoning</button>
            <button class="preset-btn" onclick="loadPreset('induction')">induction</button>
            <button class="preset-btn" onclick="loadPreset('fool')">fool</button>
        </div>
        <div class="input-group">
            <textarea id="prompt" placeholder="Enter a prompt...">The capital of France is</textarea>
        </div>
        <div class="slider-group">
            <span class="slider-label">Max tokens</span>
            <input type="range" id="max-tokens" min="10" max="500" step="10" value="200" oninput="updateSlider('max-tokens')">
            <span class="slider-value" id="max-tokens-value">200</span>
        </div>
        <div class="slider-group">
            <span class="slider-label">Temperature</span>
            <input type="range" id="temperature" min="0" max="1.5" step="0.1" value="1.0" oninput="updateSlider('temperature')">
            <span class="slider-value" id="temperature-value">1.0</span>
        </div>
        <div class="btn-group">
            <button class="btn btn-primary" id="generate-btn" onclick="generate()">Generate</button>
            <button class="btn btn-secondary" onclick="clearOutput()">Clear</button>
        </div>
    </div>

    <div class="section" id="output-section" style="display:none">
        <div class="section-title">Output</div>
        <div class="output-box" id="output"></div>
    </div>

    <script>
        const PRESETS = {
            factual: "The capital of France is",
            story: "Once upon a time, in a small village nestled between two mountains,",
            code: 'def fibonacci(n):\\n    """Return the nth Fibonacci number."""\\n',
            reasoning: "Question: If a train travels 120 miles in 2 hours, what is its average speed?\\nAnswer: Let me solve this step by step.",
            induction: "Q M T X B R F J K Q M T X B R F J K Q M T X B R",
            fool: 'absolute fool!"'
        };

        function loadPreset(name) {
            document.getElementById('prompt').value = PRESETS[name];
        }

        function updateSlider(id) {
            document.getElementById(id + '-value').textContent = document.getElementById(id).value;
        }

        async function swapModel() {
            const model = document.getElementById('model-select').value;
            document.getElementById('model-info').textContent = 'loading...';

            const resp = await fetch('/swap_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model})
            });
            const data = await resp.json();

            if (data.error) {
                document.getElementById('model-info').textContent = 'error loading';
            } else {
                document.getElementById('model-info').textContent = `${data.n_layers} layers`;
            }
        }

        function clearOutput() {
            document.getElementById('output-section').style.display = 'none';
            document.getElementById('output').innerHTML = '';
        }

        let isGenerating = false;

        async function generate() {
            if (isGenerating) return;

            const btn = document.getElementById('generate-btn');
            const prompt = document.getElementById('prompt').value;
            const maxTokens = parseInt(document.getElementById('max-tokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);

            // Disable button during generation
            isGenerating = true;
            btn.textContent = 'Generating...';
            btn.disabled = true;

            document.getElementById('output-section').style.display = 'block';
            document.getElementById('output').innerHTML =
                '<span class="prompt-text">' + escapeHtml(prompt) + '</span><span class="completion-text" id="completion"></span>';

            try {
                const resp = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, max_tokens: maxTokens, temperature})
                });

                const reader = resp.body.getReader();
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
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                document.getElementById('completion').textContent += data.token;
                            } else if (data.error) {
                                document.getElementById('completion').textContent = '\\nError: ' + data.error;
                            }
                        }
                    }
                }
            } catch (e) {
                document.getElementById('completion').textContent = '\\nError: ' + e.message;
            } finally {
                // Re-enable button
                isGenerating = false;
                btn.textContent = 'Generate';
                btn.disabled = false;
            }
        }

        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        // Allow Ctrl+Enter to generate
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                generate();
            }
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
            cached_models=cached_models
        )

    @app.route('/swap_model', methods=['POST'])
    def swap_model():
        data = request.json
        new_model = data.get('model', '')
        if new_model and new_model != state["model_name"]:
            try:
                load_model(new_model)
                return jsonify({
                    "model_name": state["model_name"],
                    "n_layers": state["n_layers"]
                })
            except Exception as e:
                return jsonify({"error": str(e)})
        return jsonify({
            "model_name": state["model_name"],
            "n_layers": state["n_layers"]
        })

    @app.route('/generate', methods=['POST'])
    def generate():
        # Reject if already generating (prevents segfault from concurrent MLX access)
        if not generation_lock.acquire(blocking=False):
            return Response(
                f"data: {json.dumps({'error': 'Generation already in progress'})}\n\n",
                mimetype='text/event-stream'
            )

        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)

        def stream():
            try:
                # Prefill
                cache, logits = prefill_with_cache(state["hooked"], prompt)

                # Stream tokens
                for token in generate_from_cache_stream(
                    state["hooked"], cache, max_tokens, temperature, initial_logits=logits
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                generation_lock.release()

        return Response(stream(), mimetype='text/event-stream')

    return app


def main():
    parser = argparse.ArgumentParser(description="Base Model Explorer")
    parser.add_argument(
        "--model",
        default="mlx-community/Meta-Llama-3.1-8B-4bit",
        help="Model name (default: Meta-Llama-3.1-8B-4bit)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to",
    )
    args = parser.parse_args()

    app = create_app(args.model)
    url = f"http://{args.host}:{args.port}"
    print(f"\nBase Model Explorer running at {url}")
    print("Press Ctrl+C to stop\n")

    # Open browser after short delay (so server is ready)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
