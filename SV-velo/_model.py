"""Project-level entrypoint now backed by the hierarchical Bayesian HVGK model.

`VELOVI` is kept as a compatibility alias, but its implementation is fully
non-VAE and delegates to :class:`velovi._hvgk_model.HVGK`.
"""

from __future__ import annotations

from ._hvgk_model import HVGK


class VELOVI(HVGK):
    """Compatibility class name for users migrating from old VELOVI.

    This class is intentionally a direct subclass of HVGK and no longer uses
    an encoder-decoder VAE backbone.
    """

    pass

