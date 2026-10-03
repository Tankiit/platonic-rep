"""uq_heads.py -- probes, bootstrap ensembles, closed-form BLR, EU/AU summaries.

Closed forms (verified in test_uq_theory.py reference tests), lam = sigma2 / tau2:
  primal:  v(x) = sigma2 * phi^T (Phi^T Phi + lam I)^-1 phi
  dual:    v(x) = tau2 * [ k_xx - k_x^T (K + lam I)^-1 k_x ],  K = Phi Phi^T, k_x = Phi phi(x)
Labels do not enter v(x) in the Gaussian case.
"""
import numpy as np


class Heads:
    """M linear-softmax heads sharing one feature standardisation (mu, sd)."""
    def __init__(self, W, b, mu, sd):
        self.W, self.b, self.mu, self.sd = W, b, mu, sd     # W [M,d,K], b [M,K]


def standardise_stats(F_tr):
    mu = F_tr.mean(0)
    sd = F_tr.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def _train_weighted_softmax(X, y, weights, K, wd, n_steps, device, seed=0):
    """Full-batch L-BFGS on M independent weighted softmax regressions in one tensor.
    X [N,d] shared, weights [M,N]. Objective per member: sum_i w_i CE_i / sum_i w_i + wd/2 ||W||^2."""
    import torch
    torch.manual_seed(seed)
    dev = device or torch.device("cpu")
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    wt = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    wt = wt / wt.sum(1, keepdim=True)
    M, (N, d) = wt.shape[0], Xt.shape
    W = (0.01 * torch.randn(d, M * K, device=dev)).requires_grad_()
    b = torch.zeros(M * K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=n_steps, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)
    onehot = torch.nn.functional.one_hot(yt, K).float()

    def closure():
        opt.zero_grad()
        logits = (Xt @ W + b).view(N, M, K)
        nll = -(onehot[:, None, :] * torch.log_softmax(logits, -1)).sum(-1)       # [N,M]
        loss = (wt.T * nll).sum() + 0.5 * wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    Wn = W.detach().cpu().numpy().reshape(d, M, K).transpose(1, 0, 2)
    bn = b.detach().cpu().numpy().reshape(M, K)
    return Wn, bn


def fit_bootstrap_heads(F_tr, y_tr, M: int, frac: float, subset_seed: int, wd: float,
                        n_steps: int = 200, poisson: bool = True, device=None, K: int | None = None,
                        subset_idx=None, stats=None):
    """Train M logistic heads IN ONE TENSOR on shared features with Poisson(1) sample weights.
    Subsample `frac` of the training set first (or use `subset_idx`, e.g. disjoint subsets for E2).
    Features are standardised ONCE with full-train statistics (identical across members and across fractions)."""
    rng = np.random.default_rng(subset_seed)
    K = K or int(y_tr.max()) + 1
    mu, sd = stats if stats is not None else standardise_stats(F_tr)
    if subset_idx is None:
        n = int(round(frac * len(F_tr)))
        subset_idx = np.sort(rng.choice(len(F_tr), n, replace=False))
    X = (F_tr[subset_idx] - mu) / sd
    y = y_tr[subset_idx]
    weights = rng.poisson(1.0, size=(M, len(X))).astype(np.float32) if poisson else np.ones((M, len(X)), np.float32)
    W, b = _train_weighted_softmax(X, y, weights, K, wd, n_steps, device, seed=subset_seed)
    return Heads(W, b, mu, sd)


def predict_probs(heads, F_te) -> np.ndarray:
    """[M, N, K] class probabilities."""
    X = (F_te - heads.mu) / heads.sd
    logits = np.einsum("nd,mdk->mnk", X, heads.W, optimize=True) + heads.b[:, None, :]
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)


def _entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis)


def eu_au_from_probs(probs: np.ndarray, interval=(0.05, 0.95)) -> dict:
    """Per-item summaries, each [N]: 'width_sum' (sum over classes of quantile-interval width), 'width_max', 'mi'
    (entropy of mean - mean entropy), 'au' (mean entropy of members), 'total'. QUANTILE width, not max-min."""
    lo, hi = np.quantile(probs, interval, axis=0)          # [N,K] each
    width = hi - lo
    mean = probs.mean(0)
    total = _entropy(mean)
    au = _entropy(probs).mean(0)
    return {"width_sum": width.sum(-1), "width_max": width.max(-1), "mi": np.maximum(total - au, 0.0),
            "au": au, "total": total, "mean_probs": mean}


def seed_only_ensemble(F_tr, y_tr, M, wd, device=None):
    """Same data, unit weights; members differ only by initialisation. For a strictly convex objective (wd > 0)
    this collapses to a point mass: seed variance is not data variance."""
    hs = [fit_bootstrap_heads(F_tr, y_tr, 1, 1.0, s, wd, poisson=False, device=device) for s in range(M)]
    return Heads(np.concatenate([h.W for h in hs]), np.concatenate([h.b for h in hs]), hs[0].mu, hs[0].sd)


def blr_posterior_var(F_tr, F_te, sigma2: float, tau2: float, form: str = "primal") -> np.ndarray:
    """Epistemic variance per test item, form in {'primal','dual'}; float64 on CPU. The two forms agree."""
    P = np.asarray(F_tr, dtype=np.float64)
    X = np.asarray(F_te, dtype=np.float64)
    lam = sigma2 / tau2
    if form == "primal":
        A = P.T @ P + lam * np.eye(P.shape[1])
        Z = np.linalg.solve(A, X.T)
        return sigma2 * np.einsum("ij,ji->i", X, Z)
    if form == "dual":
        Kmat = P @ P.T
        kx = X @ P.T
        sol = np.linalg.solve(Kmat + lam * np.eye(len(Kmat)), kx.T)
        return tau2 * ((X * X).sum(1) - (kx * sol.T).sum(1))
    raise ValueError(form)


def blr_var_rho(F_tr, F_te, rho: float) -> np.ndarray:
    """Scale-normalised BLR epistemic variance x^T (Sigma_hat + rho I)^-1 x with Sigma_hat = Phi^T Phi / n
    (equals n v(x) / sigma2 with lam = n rho). Features are used as given (centre them upstream)."""
    P = np.asarray(F_tr, dtype=np.float64)
    X = np.asarray(F_te, dtype=np.float64)
    S = P.T @ P / len(P)
    Z = np.linalg.solve(S + rho * np.eye(S.shape[0]), X.T)
    return np.einsum("ij,ji->i", X, Z)


def blr_onehot_fit(F_tr, y_tr, sigma2: float, tau2: float):
    """Ridge regression on one-hot targets: posterior mean W [d,K] and covariance factor A^-1 (shared over classes)."""
    P = np.asarray(F_tr, dtype=np.float64)
    K = int(y_tr.max()) + 1
    Y = np.eye(K)[y_tr]
    Ainv = np.linalg.inv(P.T @ P + sigma2 / tau2 * np.eye(P.shape[1]))
    return {"W": Ainv @ P.T @ Y, "cov": sigma2 * Ainv}


def laplace_last_layer(heads_or_weights, F_tr, y_tr, prior_prec: float, F_te=None, n_samples: int = 100,
                       seed: int = 0, chunk: int = 4096):
    """Second EU estimator: last-layer Laplace around a MAP softmax head (member 0 of `heads`), with the class-block-
    diagonal GGN  H_k = sum_i p_ik (1 - p_ik) x_i x_i^T + prior_prec I  (cross-class blocks dropped; stated in paper).
    Returns per-test-item {'logit_var' [N] (sum_k x^T H_k^-1 x), 'mi', 'width_sum', 'au'} via MC over logits."""
    h = heads_or_weights
    W, b = h.W[0], h.b[0]
    X = ((F_tr - h.mu) / h.sd).astype(np.float64)
    logits = X @ W + b
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    d, K = W.shape
    covs = []
    for k in range(K):
        Hk = (X * (p[:, k] * (1 - p[:, k]))[:, None]).T @ X + prior_prec * np.eye(d)
        covs.append(np.linalg.inv(Hk))
    if F_te is None:
        return covs
    Xt = ((F_te - h.mu) / h.sd).astype(np.float64)
    mean_logit = Xt @ W + b
    var = np.stack([np.einsum("ij,jk,ik->i", Xt, C, Xt) for C in covs], 1)       # [N,K]
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_samples,) + var.shape)
    L = mean_logit[None] + eps * np.sqrt(var)[None]
    L -= L.max(-1, keepdims=True)
    P = np.exp(L); P /= P.sum(-1, keepdims=True)
    out = eu_au_from_probs(P)
    out["logit_var"] = var.sum(1)
    return out


def learning_curve_reducibility(F_tr, y_tr, F_te, y_te_hard, fracs, n_seeds: int, wd: float, device=None) -> dict:
    """Per-item loss drop between small-fraction and full-data probes against HARD test labels.
    Return {'drop': [N], 'reliability': split-seed Spearman}. This is our own methodology (no published protocol)."""
    from scipy.stats import spearmanr
    stats = standardise_stats(F_tr)
    losses = {}
    for f in (min(fracs), max(fracs)):
        L = []
        for s in range(n_seeds):
            h = fit_bootstrap_heads(F_tr, y_tr, 1, f, 1000 + s, wd, poisson=False, device=device, stats=stats)
            pr = predict_probs(h, F_te)[0]
            L.append(-np.log(np.clip(pr[np.arange(len(pr)), y_te_hard], 1e-12, 1)))
        losses[f] = np.stack(L)
    drop = losses[min(fracs)] - losses[max(fracs)]                      # [S,N]
    half = n_seeds // 2
    rel = spearmanr(drop[:half].mean(0), drop[half:].mean(0)).correlation if n_seeds >= 2 else np.nan
    return {"drop": drop.mean(0), "reliability": float(rel)}


def regret_vs_human(mean_probs: np.ndarray, human_counts: np.ndarray) -> np.ndarray:
    """Per-item KL(p_human || p_model) (log-loss regret). Human labels are used for EVALUATION only."""
    ph = human_counts / human_counts.sum(1, keepdims=True)
    pm = np.clip(mean_probs, 1e-12, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ph > 0, ph * (np.log(ph) - np.log(pm)), 0.0).sum(1)


def choose_M(F_tr, y_tr, F_te, Ms=(10, 20, 50, 100), wd: float = 1e-3, frac: float = 1.0, device=None,
             key: str = "width_sum") -> dict:
    """Smallest M where Spearman(EU_M, EU_2M) >= 0.95 on held-out items (two independent ensembles of size M and 2M)."""
    from scipy.stats import spearmanr
    stats = standardise_stats(F_tr)
    curve = {}
    for M in Ms:
        a = eu_au_from_probs(predict_probs(fit_bootstrap_heads(F_tr, y_tr, M, frac, 11, wd, device=device,
                                                               stats=stats), F_te))[key]
        b = eu_au_from_probs(predict_probs(fit_bootstrap_heads(F_tr, y_tr, 2 * M, frac, 12, wd, device=device,
                                                               stats=stats), F_te))[key]
        curve[M] = float(spearmanr(a, b).correlation)
    ok = [M for M in Ms if curve[M] >= 0.95]
    return {"curve": curve, "M": min(ok) if ok else max(Ms)}
