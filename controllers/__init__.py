"""
controllers/__init__.py — public surface of the controllers package.

Per RL_lab_implementation_guide.md §9, "Each seat in the Environment owns
a Controller." This package is the home for every concrete Controller
listed there, plus the interface itself and the documented mask-only
bridge (see base_controller.py's module docstring for the KNOWN GAP this
bridge exists to cross, and what it does NOT solve).
"""

from controllers.base_controller import (
    Controller,
    RandomController,
    RuleBasedController,
    RuleBasedStrategy,
    LegalMaskControllerAdapter,
)
from controllers.policy_controller import (
    FrozenPolicyController,
    LearningPolicyController,
)
from controllers.human_controller import HumanController

__all__ = [
    "Controller",
    "RandomController",
    "RuleBasedController",
    "RuleBasedStrategy",
    "LegalMaskControllerAdapter",
    "FrozenPolicyController",
    "LearningPolicyController",
    "HumanController",
]