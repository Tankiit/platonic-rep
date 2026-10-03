"""uq_align.py -- alignment measures, per-item versions, calibration. PRIMAL forms (d x d), never N x N Grams."""
import numpy as np


def _c(F):
    F = np.asarray(F, dtype=np.float64)
    return F - F.mean(0, keepdims=True)


def linear_cka_primal(FA, FB) -> float:
    """||A^T B||_F^2 / (||A^T A||_F ||B^T B||_F) with centred A, B. Invariant to orthogonal maps and isotropic scale."""
    A, B = _c(FA), _c(FB)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def linear_predictivity(FA, FB, ridge: float = 1e-3, test_frac: float = 0.3, seed: int = 0) -> float:
    """Held-out R^2 (variance-weighted) of predicting FB from FA by ridge least squares (ridge relative to mean
    eigenvalue). Invariant to invertible linear maps of FA as ridge -> 0."""
    rng = np.random.default_rng(seed)
    n = len(FA)
    idx = rng.permutation(n)
    te, tr = idx[: int(test_frac * n)], idx[int(test_frac * n):]
    A, B = np.asarray(FA, np.float64), np.asarray(FB, np.float64)
    ma, mb = A[tr].mean(0), B[tr].mean(0)
    Atr, Btr = A[tr] - ma, B[tr] - mb
    G = Atr.T @ Atr
    Wt = np.linalg.solve(G + ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Atr.T @ Btr)
    pred = (A[te] - ma) @ Wt
    res = ((B[te] - mb) - pred)
    return float(1 - (res ** 2).sum() / ((B[te] - B[te].mean(0)) ** 2).sum())


def _knn(F, k, metric):
    from sklearn.neighbors import NearestNeighbors
    F = np.asarray(F, np.float32)
    if metric == "cosine":
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(F)
    return nn.kneighbors(F, return_distance=False)[:, 1:]


def per_item_mknn(FA, FB, k: int, metric: str = "cosine") -> np.ndarray:
    """[N] overlap |N_A(i) & N_B(i)| / k (cosine neighbourhoods: invariant to rotation and isotropic scale)."""
    na, nb = _knn(FA, k, metric), _knn(FB, k, metric)
    return np.array([len(set(a) & set(b)) / k for a, b in zip(na, nb)])


def global_mknn(FA, FB, k: int) -> float:
    return float(per_item_mknn(FA, FB, k).mean())


def s_rho(FA, FB, rho: float, center: bool = True) -> float:
    """Cosine between smoothers H = Phi (Phi^T Phi + rho I)^-1 Phi^T, via the primal trick (all d x d):
        <H_A, H_B>_F = tr( M_A (Phi_A^T Phi_B) M_B (Phi_B^T Phi_A) ),   ||H||_F^2 = tr( M G M G ).
    NOTE rho is on the UNNORMALISED Gram Phi^T Phi; the population index at rho_pop uses rho = n * rho_pop
    (see s_rho_pop). Limits: rho -> inf -> linear CKA; rho -> 0 -> mean squared canonical correlation scaled."""
    A = _c(FA) if center else np.asarray(FA, np.float64)
    B = _c(FB) if center else np.asarray(FB, np.float64)
    Ga, Gb, Gab = A.T @ A, B.T @ B, A.T @ B
    Ma = np.linalg.inv(Ga + rho * np.eye(Ga.shape[0]))
    Mb = np.linalg.inv(Gb + rho * np.eye(Gb.shape[0]))
    num = np.trace(Ma @ Gab @ Mb @ Gab.T)
    na = np.sqrt(np.trace(Ma @ Ga @ Ma @ Ga))
    nb = np.sqrt(np.trace(Mb @ Gb @ Mb @ Gb))
    return float(num / (na * nb))


def s_rho_pop(FA, FB, rho_pop: float, center: bool = True) -> float:
    """S_rho with rho on the covariance scale Phi^T Phi / n (the scale of blr_var_rho)."""
    return s_rho(FA, FB, rho_pop * len(FA), center)


def permutation_null(metric_fn, FA, FB, n_perm: int, seed: int = 0) -> dict:
    """Permute item correspondence; return {'obs','null_mean','null_sd','z','p'}."""
    rng = np.random.default_rng(seed)
    obs = metric_fn(FA, FB)
    null = np.array([metric_fn(FA, FB[rng.permutation(len(FB))]) for _ in range(n_perm)])
    sd = null.std() + 1e-12
    return {"obs": float(obs), "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((obs - null.mean()) / sd), "p": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def random_projection_floor(F, d_out: int, seed: int = 0):
    """Misalignment floor: a Gaussian random projection of the same features."""
    rng = np.random.default_rng(seed)
    return np.asarray(F, np.float64) @ rng.standard_normal((F.shape[1], d_out)) / np.sqrt(d_out)
