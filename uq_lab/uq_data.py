"""uq_data.py -- datasets and human-label targets.

Targets: AU proxy = human soft-label entropy. Train labels (hard) may be used for probes and the class/residual
split; HUMAN SOFT LABELS MUST NEVER be used to fit or tune anything that produces EU (surrogate-validity rule).
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.special import digamma

CIFAR10_ROOT = "/Users/tanmoy/research/data/cifar-10-batches-py"
CIFAR10H_COUNTS = "/Users/tanmoy/research/Credal_Sets/Wasserstein/data/cifar10h/cifar10h-counts.npy"


def load_cifar10(root: str = CIFAR10_ROOT, train: bool = True):
    """Return (images uint8 [N,32,32,3], labels int [N]) in the original CIFAR-10 batch order.
    CIFAR-10H rows follow the TEST order, which is test_batch order (= torchvision order)."""
    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    xs, ys = [], []
    for f in files:
        with open(Path(root) / f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        ys.append(np.asarray(d[b"labels"]))
    return np.concatenate(xs).astype(np.uint8), np.concatenate(ys).astype(np.int64)


def load_cifar10h(path: str = CIFAR10H_COUNTS, test_labels=None):
    """Return counts [10000,10]. Source: github.com/jcpeterson/cifar-10h (CC BY-NC-SA 4.0).
    If test_labels is given, assert the human-majority label agrees with CIFAR-10 for the vast majority of items
    (agreement near 10% means the row order is wrong)."""
    counts = np.load(path).astype(np.int64)
    assert counts.shape == (10000, 10), counts.shape
    if test_labels is not None:
        agree = float(np.mean(counts.argmax(1) == np.asarray(test_labels)))
        assert agree > 0.9, f"CIFAR-10H row order looks wrong: majority agreement {agree:.3f}"
    return counts


def load_dcic(name: str, root: str):
    """DCIC benchmark (Zenodo 10.5281/zenodo.8115942). Not used in the submitted experiments."""
    raise NotImplementedError


def human_entropy(counts: np.ndarray, method: str = "miller_madow", alpha: float = 1.0) -> np.ndarray:
    """Per-item entropy in nats. method in {'plugin','miller_madow','dirichlet'}.
    Miller-Madow adds (K_obs-1)/(2n); 'dirichlet' = posterior-mean entropy under a symmetric Dir(alpha) prior."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(1)
    if method == "dirichlet":
        a = counts + alpha
        A = a.sum(1, keepdims=True)
        return (digamma(A[:, 0] + 1) - (a / A * digamma(a + 1)).sum(1))
    p = counts / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    if method == "plugin":
        return H
    if method == "miller_madow":
        k_obs = (counts > 0).sum(1)
        return H + (k_obs - 1) / (2 * n)
    raise ValueError(method)


def split_half_ceiling(counts, n_rep: int = 100, seed: int = 0, method: str = "plugin") -> dict:
    """Noise ceiling for any model-derived AU: Spearman between entropies of two random annotator halves,
    Spearman-Brown corrected r_full = 2r/(1+r). With counts only, votes are split without replacement
    (multivariate hypergeometric), which is exact when annotators contribute one vote per item."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.sum(1)
    rs = []
    for _ in range(n_rep):
        half = np.stack([rng.multivariate_hypergeometric(c, m // 2) for c, m in zip(counts, n)])
        r = spearmanr(human_entropy(half, method), human_entropy(counts - half, method)).correlation
        rs.append(r)
    rs = np.asarray(rs)
    full = 2 * rs / (1 + rs)
    return {"r_half": float(rs.mean()), "ceiling": float(full.mean()),
            "ci": (float(np.quantile(full, 0.025)), float(np.quantile(full, 0.975)))}


def n_annotations(counts: np.ndarray) -> np.ndarray:
    """Votes per item; keep as a covariate and filter on n >= n_min where the entropy estimate is too noisy."""
    return np.asarray(counts).sum(1)
