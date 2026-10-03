"""uq_stats.py -- agreement statistics with the confound controls."""
import itertools
import math

import numpy as np
from scipy.stats import rankdata, spearmanr


def spearman(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def partial_spearman(a, b, controls: np.ndarray) -> float:
    """Rank-transform, residualise a and b on the (ranked) controls, correlate residuals."""
    C = np.atleast_2d(np.asarray(controls, np.float64))
    if C.shape[0] != len(a):
        C = C.T
    Z = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in C.T])
    ra, rb = rankdata(a), rankdata(b)
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    ea, eb = res(ra), res(rb)
    return float(ea @ eb / np.sqrt((ea @ ea) * (eb @ eb)))


def stratified_agreement(a, b, strata: np.ndarray, n_bins: int = 10) -> dict:
    """Spearman within quantile strata of `strata` (e.g. human-entropy deciles); ties are broken by rank."""
    a, b = np.asarray(a), np.asarray(b)
    q = np.floor(rankdata(strata, method="ordinal") / (len(a) + 1) * n_bins).astype(int)
    per = [spearman(a[q == i], b[q == i]) for i in range(n_bins) if (q == i).sum() > 10]
    return {"per_stratum": per, "mean": float(np.nanmean(per))}


def bootstrap_items(fn, *arrays, n_boot: int = 1000, seed: int = 0) -> dict:
    """Resample ITEMS (probes fixed within a replicate). Returns point estimate and percentile 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = fn(*arrays)
    bs = []
    for _ in range(n_boot):
        j = rng.integers(0, n, n)
        bs.append(fn(*[np.asarray(x)[j] for x in arrays]))
    return {"est": float(est), "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975))}


def ceiling_normalise(value: float, ceiling: float) -> float:
    return float(value / ceiling)


def incremental_auroc(target_binary, base_scores: np.ndarray, new_score: np.ndarray, seed: int = 0) -> dict:
    """Cross-validated AUROC gain from adding new_score to base_scores in a logistic model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    base = np.column_stack([np.atleast_2d(base_scores).reshape(len(target_binary), -1)])
    full = np.column_stack([base, new_score])
    clf = LogisticRegression(max_iter=1000)
    pb = cross_val_predict(clf, base, target_binary, cv=5, method="predict_proba")[:, 1]
    pf = cross_val_predict(clf, full, target_binary, cv=5, method="predict_proba")[:, 1]
    a0, a1 = roc_auc_score(target_binary, pb), roc_auc_score(target_binary, pf)
    return {"base": float(a0), "full": float(a1), "gain": float(a1 - a0)}


def shapley_three_source(cell_values: dict) -> dict:
    """cell_values: {(r,h,d) in {0,1}^3: value}. Value of a coalition S = mean over cells that differ from the
    reference (0,0,0) only on factors in S ... formally v(S) = cell with factors in S switched to 1. Returns Shapley
    shares of {representation, head, data} of v(111) - v(000) and the order r->h->d telescoping terms."""
    names = ("representation", "head", "data")
    v = lambda S: cell_values[tuple(int(i in S) for i in range(3))]
    total = v({0, 1, 2}) - v(set())
    phi = {}
    for i in range(3):
        others = [j for j in range(3) if j != i]
        s = 0.0
        for r in range(3):
            for S in itertools.combinations(others, r):
                w = math.factorial(len(S)) * math.factorial(3 - len(S) - 1) / math.factorial(3)
                s += w * (v(set(S) | {i}) - v(set(S)))
        phi[names[i]] = s
    tele = {"representation": v({0}) - v(set()), "head": v({0, 1}) - v({0}), "data": v({0, 1, 2}) - v({0, 1})}
    shares = {k: (phi[k] / total if total != 0 else np.nan) for k in phi}
    return {"phi": phi, "shares": shares, "telescoping": tele, "total": total}
