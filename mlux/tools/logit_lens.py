"""
Logit Lens implementation for MLX models.

Projects intermediate activations through the final layer norm and
unembedding matrix to see what tokens the model would predict at each layer.

Usage:
    from mlux import HookedModel
    from mlux.tools.logit_lens import LogitLens

    hooked = HookedModel.from_pretrained("mlx-community/gemma-2-2b-it-4bit")
    lens = LogitLens(hooked)

    # Get predictions at each layer for a specific token position
    results = lens.get_layer_predictions("The capital of France is", token_idx=-1)

    # Get predictions for all tokens at all layers
    grid = lens.get_all_predictions("The capital of France is")
"""

import mlx.core as mx
import mlx.nn as nn

from mlux import HookedModel


class LogitLens:
    """
    Logit lens implementation for MLX models.

    Projects intermediate activations through the final layer norm and
    unembedding matrix to see what tokens the model would predict at each layer.
    """

    def __init__(self, hooked: HookedModel):
        self.hooked = hooked
        self.model = hooked.model
        self.tokenizer = hooked.tokenizer
        self.config = hooked.config
        self.n_layers = self.config["n_layers"]

        # Find the components we need
        self._final_norm = self._find_final_norm()
        self._unembed_fn = self._find_unembed_fn()

    def _find_final_norm(self) -> nn.Module:
        """Find the final layer norm (before lm_head)."""
        # Try common names
        for name in ["model.norm", "norm", "ln_f", "model.ln_f"]:
            try:
                parts = name.split(".")
                obj = self.model
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise ValueError("Could not find final layer norm")

    def _find_unembed_fn(self):
        """
        Find the unembedding function (lm_head or tied embeddings).

        Returns a callable that takes hidden states and returns logits.
        """
        # Try separate lm_head first
        for name in ["lm_head", "model.lm_head"]:
            try:
                parts = name.split(".")
                obj = self.model
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue

        # Try tied embeddings (Gemma, some Llama configs)
        for name in ["model.embed_tokens", "embed_tokens"]:
            try:
                parts = name.split(".")
                obj = self.model
                for part in parts:
                    obj = getattr(obj, part)
                # Check if it has as_linear method (for tied embeddings)
                if hasattr(obj, "as_linear"):
                    return obj.as_linear
            except AttributeError:
                continue

        raise ValueError("Could not find lm_head or tied embeddings")

    def _get_hook_paths(self, probe_type: str) -> list[str]:
        """Get hook paths for the specified probe type."""
        layer_prefix = self.config.get("layer_prefix", "model.layers")
        paths = []

        for i in range(self.n_layers):
            if probe_type == "resid":
                # Hook the full layer to get residual stream output
                paths.append(f"{layer_prefix}.{i}")
            elif probe_type == "mlp_out":
                paths.append(f"{layer_prefix}.{i}.mlp")
            elif probe_type == "attn_out":
                paths.append(f"{layer_prefix}.{i}.self_attn")
            else:
                raise ValueError(f"Unknown probe type: {probe_type}")

        return paths

    def _project_to_logits(self, hidden: mx.array) -> mx.array:
        """Project hidden states through final norm and unembedding."""
        normed = self._final_norm(hidden)
        logits = self._unembed_fn(normed)
        return logits

    def get_layer_predictions(
        self,
        text: str,
        token_idx: int,
        probe_type: str = "resid",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Get top-k token predictions at each layer for a specific token position.

        Args:
            text: Input text
            token_idx: Which token position to analyze
            probe_type: "resid", "mlp_out", or "attn_out"
            top_k: Number of top predictions per layer

        Returns:
            List of dicts with layer predictions
        """
        hooks = self._get_hook_paths(probe_type)

        # Run with cache to get all layer activations
        output, cache = self.hooked.run_with_cache(text, hooks=hooks)

        results = []
        layer_prefix = self.config.get("layer_prefix", "model.layers")

        for i in range(self.n_layers):
            if probe_type == "resid":
                path = f"{layer_prefix}.{i}"
            elif probe_type == "mlp_out":
                path = f"{layer_prefix}.{i}.mlp"
            elif probe_type == "attn_out":
                path = f"{layer_prefix}.{i}.self_attn"

            hidden = cache[path]

            # Get hidden state for this token position
            # Shape: [batch, seq, hidden] -> [hidden]
            token_hidden = hidden[0, token_idx, :]

            # Project to logits
            # Need to add batch dim for layer norm
            token_hidden = token_hidden.reshape(1, 1, -1)
            logits = self._project_to_logits(token_hidden)
            logits = logits[0, 0, :]  # [vocab]

            mx.eval(logits)

            # Get top-k
            top_indices = mx.argsort(logits)[-top_k:][::-1]
            top_logits = logits[top_indices]
            mx.eval(top_indices, top_logits)

            predictions = []
            for j in range(top_k):
                token_id = int(top_indices[j].item())
                token_str = self.tokenizer.decode([token_id])
                logit_val = float(top_logits[j].item())
                predictions.append({
                    "token": token_str,
                    "token_id": token_id,
                    "logit": logit_val,
                })

            results.append({
                "layer": i,
                "probe_type": probe_type,
                "predictions": predictions,
            })

        return results

    def get_all_predictions(
        self,
        text: str,
        probe_type: str = "resid",
        top_k: int = 3,
    ) -> dict:
        """
        Get predictions for ALL tokens at ALL layers at once.

        Returns:
            Dict with 'tokens' list and 'grid' (tokens x layers x top_k predictions)
        """
        hooks = self._get_hook_paths(probe_type)
        output, cache = self.hooked.run_with_cache(text, hooks=hooks)

        tokens = self.tokenize_with_info(text)
        n_tokens = len(tokens)
        layer_prefix = self.config.get("layer_prefix", "model.layers")

        # grid[token_idx][layer_idx] = list of predictions
        grid = []

        for token_idx in range(n_tokens):
            token_layers = []
            for layer_idx in range(self.n_layers):
                if probe_type == "resid":
                    path = f"{layer_prefix}.{layer_idx}"
                elif probe_type == "mlp_out":
                    path = f"{layer_prefix}.{layer_idx}.mlp"
                elif probe_type == "attn_out":
                    path = f"{layer_prefix}.{layer_idx}.self_attn"

                hidden = cache[path]
                token_hidden = hidden[0, token_idx, :].reshape(1, 1, -1)
                logits = self._project_to_logits(token_hidden)[0, 0, :]
                mx.eval(logits)

                top_indices = mx.argsort(logits)[-top_k:][::-1]
                top_logits = logits[top_indices]
                mx.eval(top_indices, top_logits)

                preds = []
                for j in range(top_k):
                    token_id = int(top_indices[j].item())
                    preds.append({
                        "token": self.tokenizer.decode([token_id]),
                        "logit": float(top_logits[j].item()),
                    })
                token_layers.append(preds)
            grid.append(token_layers)

        return {"tokens": tokens, "grid": grid, "n_layers": self.n_layers}

    def tokenize_with_info(self, text: str) -> list[dict]:
        """Tokenize text and return token info for display."""
        token_ids = self.tokenizer.encode(text)
        tokens = []
        for i, tid in enumerate(token_ids):
            token_str = self.tokenizer.decode([tid])
            tokens.append({
                "idx": i,
                "token_id": tid,
                "text": token_str,
                "display": repr(token_str)[1:-1],  # Show escapes
            })
        return tokens
