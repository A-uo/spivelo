"""Registry keys used across the velovi package."""

from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    U_KEY: str = "U"
    CELL_TYPE_KEY: str = "cell_type"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
