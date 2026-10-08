"""Input validation without importing the training stack."""
import numpy as np
from scipy import sparse


def validate_counts(x, name="counts"):
    values = x.data if sparse.issparse(x) else np.asarray(x)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name}: counts must be finite and non-negative.")
    if not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError(f"{name}: expected raw integer counts; do not round normalized or smoothed data.")
    return x
