"""
test_stage1_imports.py — Stage 1: Project Skeleton

Verifies that every rl_lab sub-package imports without error and that
the top-level package exposes its version string.

These tests contain no poker logic and require no Engine dependency.
They exist solely to confirm the skeleton is correctly structured.
"""

import importlib
import pytest


PACKAGES = [
    "scenarios",
    "envs",
    "observations",
    "actions",
    "agents",
    "models",
    "trainers",
    "replay",
    "evaluation",
    "experiments",
    "utils",
    "configs",
    "tests",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package: str) -> None:
    """Every rl_lab sub-package must import without raising."""
    mod = importlib.import_module(package)
    assert mod is not None

def test_version_string() -> None:
    """The top-level package must expose a __version__ string."""
    import _version
    
    assert hasattr(_version, "__version__"), "_version.py is missing the __version__ attribute"
    assert isinstance(_version.__version__, str)
    assert len(_version.__version__) > 0
    
def test_package_docstrings() -> None:
    """Every sub-package must have a non-empty module docstring."""
    for package in PACKAGES:
        mod = importlib.import_module(package)
        assert mod.__doc__ is not None and len(mod.__doc__.strip()) > 0, (
            f"{package} is missing a module docstring"
        )