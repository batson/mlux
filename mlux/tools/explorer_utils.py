"""
Shared utilities for explorer apps.

Provides common functionality for health checks, shutdown, and server management.
"""

import threading
import time
import webbrowser


def add_lifecycle_routes(app, state: dict, explorer_id: str, server_ref: list):
    """
    Add /health and /shutdown routes to a Flask app.

    Args:
        app: Flask app instance
        state: Dict with at least "model_name" key
        explorer_id: Explorer identifier (e.g., "logit-lens")
        server_ref: Single-element list to hold server reference (for shutdown)
    """
    from flask import jsonify, make_response, request

    def add_cors(response):
        """Add CORS headers for command center access."""
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.route('/health', methods=['GET', 'OPTIONS'])
    def health():
        if request.method == 'OPTIONS':
            response = make_response()
            return add_cors(response)
        response = jsonify({
            "status": "ok",
            "model": state.get("model_name", "unknown"),
            "explorer": explorer_id
        })
        return add_cors(response)

    @app.route('/shutdown', methods=['POST', 'OPTIONS'])
    def shutdown():
        if request.method == 'OPTIONS':
            response = make_response()
            return add_cors(response)
        def do_shutdown():
            time.sleep(0.5)
            if server_ref and server_ref[0]:
                server_ref[0].shutdown()
        threading.Thread(target=do_shutdown).start()
        response = jsonify({"status": "shutting down"})
        return add_cors(response)


def run_explorer(
    app,
    name: str,
    host: str,
    port: int,
    server_ref: list,
    open_browser: bool = True
):
    """
    Run an explorer with proper server lifecycle.

    Args:
        app: Flask app instance
        name: Display name for the explorer
        host: Host to bind to
        port: Port to bind to
        server_ref: Single-element list to store server reference
        open_browser: Whether to open browser on start
    """
    from werkzeug.serving import make_server

    url = f"http://{host}:{port}"
    print(f"\n{name} running at {url}")
    print("Press Ctrl+C to stop\n")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = make_server(host, port, app, threaded=True)
    server_ref.append(server)
    server.serve_forever()


def create_arg_parser(description: str, default_port: int):
    """Create standard argument parser for explorers."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Model name (default: gemma-2-2b-it-4bit)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to run the server on (default: {default_port})",
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
    return parser
