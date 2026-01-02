"""
mlux tools - Interactive explorers for model interpretability.

Each explorer can be run standalone or launched from the command center.
"""

# Explorer registry for command center discovery
EXPLORERS = [
    {
        "id": "logit-lens",
        "name": "Logit Lens",
        "module": "mlux.tools.logit_lens_explorer",
        "port": 5001,
        "description": "See what the model would say if you stopped it halfway",
    },
    {
        "id": "patching",
        "name": "Activation Patching",
        "module": "mlux.tools.patching_explorer",
        "port": 5002,
        "description": "Swap activations from one prompt into another",
    },
    {
        "id": "steering",
        "name": "Contrastive Steering",
        "module": "mlux.tools.contrastive_steering_explorer",
        "port": 5003,
        "description": "Steer model behavior with differences between activation vectors",
    },
    {
        "id": "ablation",
        "name": "Residual Stream Ablation",
        "module": "mlux.tools.ablation_explorer",
        "port": 5004,
        "description": "Trace information flow",
    },
    {
        "id": "base",
        "name": "Base Model Generation",
        "module": "mlux.tools.base_explorer",
        "port": 5005,
        "description": "Free generation with base models",
    },
]


def get_explorer(explorer_id: str) -> dict | None:
    """Get explorer config by ID."""
    for explorer in EXPLORERS:
        if explorer["id"] == explorer_id:
            return explorer
    return None
