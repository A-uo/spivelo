"""Compatibility module aliases for the project-level HVGK replacement."""

from __future__ import annotations

from ._constants import REGISTRY_KEYS
from ._hvgk_pyro_module import HVGKPyroModule

CELL_TYPE_REGISTRY_KEY = REGISTRY_KEYS.CELL_TYPE_KEY

# Backward-compatible name, now mapped to the hierarchical Bayesian Pyro module.
VELOVAE = HVGKPyroModule

