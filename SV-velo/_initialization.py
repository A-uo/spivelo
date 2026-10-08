"""Label-free initial values; these change the guide start, never the objective."""
import numpy as np
import torch
from scipy import sparse
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.decomposition import PCA


def initial_values(u, s, n_modules, strategy="expression", seed=0, reverse=False):
    if strategy not in {"prior", "random", "expression"}:
        raise ValueError("initialization must be prior, random, or expression")
    if strategy == "prior":
        return {}
    u = u.toarray() if sparse.issparse(u) else np.asarray(u)
    s = s.toarray() if sparse.issparse(s) else np.asarray(s)
    if u.shape != s.shape or u.ndim != 2 or min(u.shape) < 2:
        raise ValueError("Initialization requires matching cell-by-gene arrays with >=2 cells/genes")
    if not (np.isfinite(u).all() and np.isfinite(s).all()) or (u < 0).any() or (s < 0).any():
        raise ValueError("Initialization requires finite nonnegative counts")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        rank = rng.permutation(len(s)).astype(float) + 1
    else:
        # This representation is ONLY used to initialize. The likelihood retains raw counts.
        blocks = []
        for x in (u, s):
            depth = x.sum(axis=1, keepdims=True)
            target = np.median(depth[depth > 0]) if (depth > 0).any() else 1.
            blocks.append(np.log1p(x / np.maximum(depth, 1.) * target))
        x = np.concatenate(blocks, axis=1)
        if np.max(x.std(axis=0)) < 1e-12:
            raise ValueError("No expression variation; use prior/random initialization for a null-data check")
        score = PCA(n_components=1, svd_solver="full").fit_transform(x)[:, 0]
        rank = rankdata(score, method="average")
    if reverse:
        rank = len(s) + 1 - rank
    # Match the existing log-time prior at its default hyperparameter means.
    # This is an initial value, not a constraint forcing a uniform final distribution.
    q = (rank - .5) / len(s)
    t = np.exp(-.7 + (.5 * np.sqrt(2/np.pi)) * ndtri(q))[:, None]
    # Break the identical module/gene loading initialization without encoding cell labels.
    loading = (3. / n_modules) * np.exp(rng.normal(0, .35, (n_modules, s.shape[1])) - .35**2/2)
    return {"t_c": torch.tensor(t, dtype=torch.float32),
            "g_fg": torch.tensor(loading, dtype=torch.float32)}
