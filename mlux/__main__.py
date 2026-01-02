#!/usr/bin/env python3
"""
mlux Command Center

Launch and manage mlux explorers from a central interface.

Usage:
    python -m mlux
    python -m mlux --port 5050
"""

import subprocess
import sys
import threading
import webbrowser

from mlux.tools import EXPLORERS, get_explorer
from mlux.tools.explorer_utils import add_lifecycle_routes

# Server reference for shutdown (single-element list for mutability)
_server_ref = []


def create_app():
    """Create the Flask app for the command center."""
    try:
        from flask import Flask, render_template_string, request, jsonify
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(__name__)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>mlux</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
            background: #fafaf8;
            color: #333;
            max-width: 700px;
            margin: 0 auto;
        }
        h1 {
            margin: 0 0 24px 0;
            font-size: 1.2em;
            font-weight: 600;
            color: #222;
        }

        .explorer-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .explorer-card {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 6px;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .explorer-info {
            flex: 1;
        }
        .explorer-name {
            font-weight: 600;
            color: #222;
            margin-bottom: 4px;
        }
        .explorer-desc {
            font-size: 0.85em;
            color: #666;
        }
        .explorer-status {
            font-size: 0.75em;
            color: #999;
            margin-top: 4px;
        }
        .explorer-status.running {
            color: #2a7c4f;
        }

        .explorer-actions {
            display: flex;
            gap: 8px;
        }
        .btn {
            padding: 8px 14px;
            border: none;
            border-radius: 4px;
            font-family: inherit;
            font-size: 11px;
            cursor: pointer;
            text-decoration: none;
        }
        .btn-launch {
            background: #333;
            color: #fff;
        }
        .btn-launch:hover { background: #555; }
        .btn-open {
            background: #e8e8e4;
            color: #333;
            border: 1px solid #ccc;
        }
        .btn-open:hover { background: #ddd; }
        .btn-stop {
            background: #fff;
            color: #b44;
            border: 1px solid #dcc;
        }
        .btn-stop:hover { background: #fef5f5; border-color: #b44; }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <h1>mlux</h1>

    <div class="explorer-list" id="explorer-list">
        {% for e in explorers %}
        <div class="explorer-card" data-id="{{ e.id }}">
            <div class="explorer-info">
                <div class="explorer-name">{{ e.name }}</div>
                <div class="explorer-desc">{{ e.description }}</div>
                <div class="explorer-status" id="status-{{ e.id }}">port {{ e.port }}</div>
            </div>
            <div class="explorer-actions">
                <button class="btn btn-launch" id="launch-{{ e.id }}" onclick="launch('{{ e.id }}')">Launch</button>
                <a class="btn btn-open" id="open-{{ e.id }}" href="http://localhost:{{ e.port }}" target="_blank" style="display:none">Open</a>
                <button class="btn btn-stop" id="stop-{{ e.id }}" onclick="stop('{{ e.id }}')" style="display:none">Stop</button>
            </div>
        </div>
        {% endfor %}
    </div>

    <script>
        const explorers = {{ explorers_json | safe }};

        async function checkStatus(id) {
            const explorer = explorers.find(e => e.id === id);
            const statusEl = document.getElementById(`status-${id}`);
            const launchBtn = document.getElementById(`launch-${id}`);
            const openBtn = document.getElementById(`open-${id}`);
            const stopBtn = document.getElementById(`stop-${id}`);

            try {
                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), 1000);
                const resp = await fetch(`http://localhost:${explorer.port}/health`, {
                    method: 'GET',
                    mode: 'cors',
                    signal: controller.signal
                });
                clearTimeout(timeout);
                if (resp.ok) {
                    const data = await resp.json();
                    statusEl.textContent = `running on port ${explorer.port} (${data.model || 'unknown model'})`;
                    statusEl.classList.add('running');
                    launchBtn.style.display = 'none';
                    openBtn.style.display = 'inline-block';
                    stopBtn.style.display = 'inline-block';
                    return true;
                }
            } catch (e) {
                // Expected when explorer not running - silently ignore
            }

            statusEl.textContent = `port ${explorer.port}`;
            statusEl.classList.remove('running');
            launchBtn.style.display = 'inline-block';
            launchBtn.disabled = false;
            launchBtn.textContent = 'Launch';
            openBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            return false;
        }

        // Track explorers that are currently launching
        const launching = new Set();

        async function launch(id) {
            const btn = document.getElementById(`launch-${id}`);
            const statusEl = document.getElementById(`status-${id}`);
            btn.disabled = true;
            btn.textContent = 'Launching...';
            launching.add(id);
            statusEl.textContent = 'starting...';

            try {
                const resp = await fetch('/launch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id})
                });
                const data = await resp.json();

                if (data.success) {
                    // Poll until ready (model loading can take a while)
                    pollUntilReady(id, 60);
                } else {
                    alert('Failed to launch: ' + (data.error || 'unknown error'));
                    launching.delete(id);
                    btn.disabled = false;
                    btn.textContent = 'Launch';
                }
            } catch (e) {
                alert('Error: ' + e.message);
                launching.delete(id);
                btn.disabled = false;
                btn.textContent = 'Launch';
            }
        }

        async function pollUntilReady(id, attemptsLeft) {
            if (attemptsLeft <= 0) {
                launching.delete(id);
                checkStatus(id);
                return;
            }
            const isRunning = await checkStatus(id);
            if (isRunning) {
                launching.delete(id);
            } else if (launching.has(id)) {
                // Still launching, keep polling
                const btn = document.getElementById(`launch-${id}`);
                const statusEl = document.getElementById(`status-${id}`);
                btn.disabled = true;
                btn.textContent = 'Launching...';
                btn.style.display = 'inline-block';
                statusEl.textContent = 'loading model...';
                setTimeout(() => pollUntilReady(id, attemptsLeft - 1), 2000);
            }
        }

        async function stop(id) {
            const explorer = explorers.find(e => e.id === id);
            const stopBtn = document.getElementById(`stop-${id}`);
            stopBtn.disabled = true;
            stopBtn.textContent = 'Stopping...';

            try {
                // Call the explorer's shutdown endpoint
                await fetch(`http://localhost:${explorer.port}/shutdown`, {
                    method: 'POST',
                    mode: 'cors'
                });
            } catch (e) {
                // Ignore errors (connection reset expected)
            }

            // Wait and check status
            setTimeout(() => {
                checkStatus(id);
                stopBtn.disabled = false;
                stopBtn.textContent = 'Stop';
            }, 1000);
        }

        // Check all statuses on load
        explorers.forEach(e => checkStatus(e.id));

        // Periodically refresh status (skip explorers that are launching)
        setInterval(() => {
            explorers.forEach(e => {
                if (!launching.has(e.id)) {
                    checkStatus(e.id);
                }
            });
        }, 5000);
    </script>
</body>
</html>
'''

    @app.route('/')
    def index():
        import json
        return render_template_string(
            HTML_TEMPLATE,
            explorers=EXPLORERS,
            explorers_json=json.dumps(EXPLORERS)
        )

    @app.route('/launch', methods=['POST'])
    def launch():
        data = request.json
        explorer_id = data.get('id')
        explorer = get_explorer(explorer_id)

        if not explorer:
            return jsonify({"success": False, "error": "Unknown explorer"})

        # Check if already running
        try:
            import requests as req
            resp = req.get(f"http://localhost:{explorer['port']}/health", timeout=1)
            if resp.ok:
                return jsonify({"success": True, "message": "Already running"})
        except Exception:
            pass

        # Launch as subprocess
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", explorer['module'], "--no-browser"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return jsonify({"success": True, "pid": proc.pid})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # Add /health and /shutdown routes
    state = {"model_name": "command-center"}
    add_lifecycle_routes(app, state, "command-center", _server_ref)

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mlux Command Center")
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run command center on (default: 5050)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    args = parser.parse_args()

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"\nmlux Command Center running at {url}")
    print("Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # Use werkzeug for consistent server behavior
    from werkzeug.serving import make_server
    server = make_server(args.host, args.port, app, threaded=True)
    _server_ref.append(server)
    server.serve_forever()


if __name__ == "__main__":
    main()
