"""
Interpretability experiments for mlux.

Available experiments:
- induction_heads: Detect induction heads for in-context learning
- binding_mechanisms: Analyze entity binding mechanisms (arXiv:2510.06182)

Usage:
    python -m mlux.experiments.induction_heads
    python -m mlux.experiments.binding_mechanisms
"""

from .induction_heads import (
    detect_induction_heads,
    visualize_induction_pattern,
    validate_induction_copying,
    run_experiment as run_induction_experiment,
)

from .binding_mechanisms import (
    analyze_binding_mechanisms,
    test_position_accuracy,
    run_experiment as run_binding_experiment,
    BindingTask,
    FILLING_LIQUIDS,
    PEOPLE_OBJECTS,
)

__all__ = [
    # Induction heads
    "detect_induction_heads",
    "visualize_induction_pattern",
    "validate_induction_copying",
    "run_induction_experiment",
    # Binding mechanisms
    "analyze_binding_mechanisms",
    "test_position_accuracy",
    "run_binding_experiment",
    "BindingTask",
    "FILLING_LIQUIDS",
    "PEOPLE_OBJECTS",
]
