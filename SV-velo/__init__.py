"""velovi."""

import logging
import warnings

from rich.console import Console  # noqa
from rich.logging import RichHandler  # noqa

from ._constants import REGISTRY_KEYS  # noqa
from ._hvgk_model import HVGK  # noqa
from ._model import VELOVI  # noqa
from ._module import CELL_TYPE_REGISTRY_KEY, VELOVAE  # noqa
# Plotting/preprocessing dependencies are loaded only when requested.
def __getattr__(name):
    if name in __all__ and name not in {"VELOVI", "HVGK", "VELOVAE", "CELL_TYPE_REGISTRY_KEY", "REGISTRY_KEYS"}:
        from importlib import import_module
        value = getattr(import_module("._utils", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

package_name = "velovi"
__version__ = "0.2.0+hvgk.20260927.inference2"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
ch.setFormatter(logging.Formatter("velovi: %(message)s"))
logger.addHandler(ch)
logger.propagate = False

__all__ = [
    "VELOVI",
    "HVGK",
    "VELOVAE",
    "CELL_TYPE_REGISTRY_KEY",
    "REGISTRY_KEYS",
    "get_permutation_scores",
    "preprocess_data",
    "get_training_data",
    "get_max_modules",
    "get_training_data_cell2fate_style",
    "get_max_modules_cell2fate_style",
    "plot_rna_velocity_flow",
    "plot_selected_cell_velocity",
    "plot_selected_spatial_probability_velocity_flow",
    "plot_spatial_probability_with_velocity",
]
