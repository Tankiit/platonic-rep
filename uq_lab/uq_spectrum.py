"""uq_spectrum.py -- spectra, effective dimension, transformations, class/residual split.

Do NOT reuse the branch's participation ratio as 'effective rank': it is dominated by the top of the spectrum.
The relevant quantity is d_eff(rho) = sum_i lambda_i / (lambda_i + rho).
"""
import numpy as np


def covariance_eigs(F: np.ndarray, center: bool = True):
    """Return (eigvals descending [d], eigvecs [d,d]) of the d x d feature covariance, float64 on CPU."""
    X = np.asarray(F, dtype=np.float64)
    if center:
        X = X - X.mean(0, keepdims=True)
    ev, U = np.linalg.eigh(X.T @ X / len(X))
    return np.clip(ev[::-1], 0, None), U[:, ::-1]


def d_eff(eigvals: np.ndarray, rho) -> float:
    """sum lambda/(lambda+rho); vectorised over an array of rho."""
    ev = np.asarray(eigvals, np.float64)
    rho = np.asarray(rho, np.float64)
    out = (ev[None, :] / (ev[None, :] + rho.reshape(-1, 1))).sum(1)
    return float(out[0]) if rho.ndim == 0 else out


def d2_eff(eigvals, rho):
    """Second-order effective dimension sum (lambda/(lambda+rho))^2: the variance normaliser of the EU-agreement
    identity (Theorem 1)."""
    ev = np.asarray(eigvals, np.float64)
    return float(((ev / (ev + rho)) ** 2).sum())


def participation_ratio(eigvals: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2. Kept ONLY as a contrast (top-heavy, like CKA)."""
    ev = np.asarray(eigvals, np.float64)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def fit_power_law(eigvals: np.ndarray, i_lo: int, i_hi: int, n_boot: int = 200, seed: int = 0) -> dict:
    """Fit lambda_k ~ k^-alpha on ranks [i_lo, i_hi) in log-log. 'Good fit' pre-set: R^2 >= 0.95 and CI width <= 0.2."""
    ev = np.asarray(eigvals, np.float64)
    k = np.arange(i_lo, i_hi) + 1
    y = np.log(ev[i_lo:i_hi])
    x = np.log(k)
    slope, icpt = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (slope * x + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        j = rng.integers(0, len(x), len(x))
        bs.append(-np.polyfit(x[j], y[j], 1)[0])
    ci = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    return {"alpha": float(-slope), "ci": ci, "r2": float(r2), "range": (i_lo, i_hi),
            "good": bool(r2 >= 0.95 and ci[1] - ci[0] <= 0.2)}


def random_orthogonal(d, seed=0):
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def rotate_and_scale(F: np.ndarray, c: float, seed: int = 0) -> np.ndarray:
    """phi -> c * Q phi with Q random orthogonal (Prop. 1). Leaves linear CKA and cosine mKNN unchanged."""
    F = np.asarray(F, np.float64)
    return c * F @ random_orthogonal(F.shape[1], seed).T


def tail_reshape(F: np.ndarray, energy_frac: float = 0.95, s: float = 2.0) -> np.ndarray:
    """Scale the eigendirections OUTSIDE the smallest set carrying `energy_frac` of the variance by s (Prop. 3).
    Expressed in the eigenbasis then rotated back; the mean is preserved."""
    F = np.asarray(F, np.float64)
    mu = F.mean(0, keepdims=True)
    ev, U = covariance_eigs(F)
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), energy_frac) + 1)
    scale = np.ones_like(ev)
    scale[k:] = s
    return ((F - mu) @ U * scale) @ U.T + mu


def linear_map(F: np.ndarray, A: np.ndarray) -> np.ndarray:
    """phi -> A phi for a non-orthogonal invertible A (GL case)."""
    return np.asarray(F, np.float64) @ np.asarray(A).T


def class_residual_split(F_train, y_train, F_other):
    """Return (F_B, F_W): projection of F_other onto the span of the CENTRED CLASS MEANS (fit on TRAIN with hard
    labels only) and the orthogonal residual. Never uses human soft labels."""
    F_train = np.asarray(F_train, np.float64)
    mu = F_train.mean(0)
    K = int(y_train.max()) + 1
    Mc = np.stack([F_train[y_train == k].mean(0) - mu for k in range(K)])     # [K,d]
    Q, _ = np.linalg.qr(Mc.T)                                                 # [d,K]
    X = np.asarray(F_other, np.float64) - mu
    FB = X @ Q @ Q.T
    return FB, X - FB


def nc1(F: np.ndarray, y: np.ndarray) -> float:
    """Within-class variability collapse tr(Sigma_W Sigma_B^+) / K."""
    F = np.asarray(F, np.float64)
    K = int(y.max()) + 1
    mu = F.mean(0)
    means = np.stack([F[y == k].mean(0) for k in range(K)])
    Sb = (means - mu).T @ (means - mu) / K
    R = F - means[y]
    Sw = R.T @ R / len(F)
    return float(np.trace(Sw @ np.linalg.pinv(Sb)) / K)
