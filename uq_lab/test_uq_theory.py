"""test_uq_theory.py

PART 1 (test_ref_*): self-contained reference checks of the theory claims. They run NOW with numpy/scipy only and need
none of your code. Run first:  pytest -k ref -q
PART 2 (test_impl_*): call YOUR implementations and compare to the references; skipped until the stub is implemented.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

RNG = np.random.default_rng(0)

# ---------------- reference implementations (independent of the stubs) ----------------
def center(X): return X - X.mean(0, keepdims=True)

def cka_ref(A, B):
    A, B = center(A), center(B)
    return np.linalg.norm(A.T @ B, "fro") ** 2 / (np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro"))

def v_primal_ref(Phi, X, sigma2, tau2):
    lam = sigma2 / tau2
    M = np.linalg.inv(Phi.T @ Phi + lam * np.eye(Phi.shape[1]))
    return sigma2 * np.einsum("ij,jk,ik->i", X, M, X)

def v_dual_ref(Phi, X, sigma2, tau2):
    lam = sigma2 / tau2
    K = Phi @ Phi.T
    kx = X @ Phi.T
    kxx = (X * X).sum(1)
    sol = np.linalg.solve(K + lam * np.eye(len(K)), kx.T)
    return tau2 * (kxx - (kx * sol.T).sum(1))

def leverage_ref(X, ridge=0.0):
    C = X.T @ X / len(X) + ridge * np.eye(X.shape[1])
    return np.einsum("ij,jk,ik->i", X, np.linalg.inv(C), X)

def deff_ref(eigs, rho): return float(np.sum(eigs / (eigs + rho)))

def toy(n=5000, k=9, m=200, var_big=10.0, var_small=0.1, share_tail=False, seed=0):
    r = np.random.default_rng(seed)
    S = r.normal(size=(n, k)) * np.sqrt(var_big)
    WA = r.normal(size=(n, m)) * np.sqrt(var_small)
    WB = WA.copy() if share_tail else r.normal(size=(n, m)) * np.sqrt(var_small)
    return np.hstack([S, WA]), np.hstack([S, WB])

# ---------------- PART 1: reference checks ----------------
def test_ref_blr_dual_equals_primal():
    Phi, X = RNG.normal(size=(40, 6)), RNG.normal(size=(5, 6))
    assert np.allclose(v_primal_ref(Phi, X, 0.7, 1.3), v_dual_ref(Phi, X, 0.7, 1.3))

def test_ref_scale_is_a_prior_change():
    """P2: v(c*Phi, c*x; lam) == v(Phi, x; lam/c^2) exactly; CKA unchanged by the scale."""
    Phi, X = RNG.normal(size=(60, 5)), RNG.normal(size=(4, 5))
    c, s2, t2 = 3.0, 0.5, 2.0
    lhs = v_primal_ref(c * Phi, c * X, s2, t2)
    rhs = v_primal_ref(Phi, X, s2, t2 * c**2)          # lam/c^2 <=> tau2 * c^2
    assert np.allclose(lhs, rhs)
    assert abs(cka_ref(Phi, c * Phi) - 1.0) < 1e-12

def test_ref_cka_orthogonal_and_scale_invariance():
    A = RNG.normal(size=(200, 7)); B = RNG.normal(size=(200, 7))
    Q, _ = np.linalg.qr(RNG.normal(size=(7, 7)))
    assert abs(cka_ref(A, B) - cka_ref(2.5 * A @ Q, B)) < 1e-10

def test_ref_resolvent_bound_P3():
    """Training-point posterior covariance difference is bounded by ||K-L||_op, constant 1, for any sigma2 > 0."""
    n = 30
    for s2 in (0.05, 0.5, 5.0):
        A0, B0 = RNG.normal(size=(n, 8)), RNG.normal(size=(n, 8))
        K, L = A0 @ A0.T, B0 @ B0.T
        post = lambda G: G - G @ np.linalg.solve(G + s2 * np.eye(n), G)
        lhs = np.linalg.norm(post(K) - post(L), 2)
        assert lhs <= np.linalg.norm(K - L, 2) + 1e-9

def test_ref_toy_high_cka_but_unrelated_eu_ranking():
    """Alignment is dominated by the 9 shared high-variance directions; EU-like leverage is dominated by the 400 unshared ones."""
    XA, XB = toy()
    assert cka_ref(XA, XB) > 0.99
    rho = spearmanr(leverage_ref(XA), leverage_ref(XB)).correlation
    assert rho < 0.2, rho
    XA2, XB2 = toy(share_tail=True)
    assert spearmanr(leverage_ref(XA2), leverage_ref(XB2)).correlation > 0.9

def test_ref_tail_reshape_keeps_cka_moves_deff():
    XA, _ = toy()
    ev, U = np.linalg.eigh(np.cov(XA.T)); ev, U = ev[::-1], U[:, ::-1]
    s = np.ones(len(ev)); s[9:] = 2.0                  # scale the 200 tail directions by 2
    XR = (center(XA) @ U * s) @ U.T
    assert cka_ref(XA, XR) > 0.98
    ev2 = np.sort(np.linalg.eigvalsh(np.cov(XR.T)))[::-1]
    assert deff_ref(ev2, 0.1) / deff_ref(ev, 0.1) > 1.3

def test_ref_bootstrap_variance_is_posterior_times_shrinkage():
    """Fixed-design ridge: Var_noise(w_hat) = s2 A^-1 G A^-1 vs posterior s2 A^-1; per-direction ratio g/(g+lam)."""
    Phi = RNG.normal(size=(80, 6)) * np.array([5, 3, 1, 0.3, 0.1, 0.05]); s2, lam = 0.4, 0.2
    G = Phi.T @ Phi; A = G + lam * np.eye(6); Ai = np.linalg.inv(A)
    boot, post = s2 * Ai @ G @ Ai, s2 * Ai
    g, V = np.linalg.eigh(G)
    assert np.allclose(V.T @ boot @ V, np.diag(g / (g + lam)) * (V.T @ post @ V))
    ys = RNG.normal(size=(20000, 80)) * np.sqrt(s2)     # Monte Carlo over noise only
    W = ys @ (Ai @ Phi.T).T
    assert np.allclose(np.cov(W.T), boot, rtol=0.1, atol=0.01 * np.abs(boot).max())

def test_ref_mean_train_variance_equals_deff():
    """mean_i v(x_i) = (s2/n) * d_eff(rho) with rho = s2/(n tau2), exactly, on the training points."""
    n, d, s2, t2 = 500, 12, 0.3, 2.0
    Phi = RNG.normal(size=(n, d)) * np.linspace(3, 0.1, d)
    lhs = v_primal_ref(Phi, Phi, s2, t2).mean()
    ev = np.linalg.eigvalsh(Phi.T @ Phi / n)
    assert np.isclose(lhs, s2 / n * deff_ref(ev, s2 / (n * t2)))

def test_ref_s_rho_primal_matches_direct():
    """S_rho via d x d matrices equals the N x N smoother cosine."""
    n, d, rho = 60, 5, 0.7
    A, B = RNG.normal(size=(n, d)), RNG.normal(size=(n, d)) @ RNG.normal(size=(d, d))
    H = lambda P: P @ np.linalg.inv(P.T @ P + rho * np.eye(P.shape[1])) @ P.T
    direct = np.sum(H(A) * H(B)) / (np.linalg.norm(H(A)) * np.linalg.norm(H(B)))
    Ma, Mb = (np.linalg.inv(P.T @ P + rho * np.eye(d)) for P in (A, B))
    num = np.trace(Ma @ A.T @ B @ Mb @ B.T @ A)
    na = np.sqrt(np.trace(Ma @ (A.T @ A) @ Ma @ (A.T @ A)))
    nb = np.sqrt(np.trace(Mb @ (B.T @ B) @ Mb @ (B.T @ B)))
    assert np.isclose(direct, num / (na * nb))

# ---------------- PART 2: your implementations (skipped until implemented) ----------------
def call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError:
        pytest.skip(f"{fn.__module__}.{fn.__name__} not implemented yet")

def test_impl_blr_primal_dual():
    import uq_heads as H
    Phi, X = RNG.normal(size=(40, 6)), RNG.normal(size=(5, 6))
    vp = call(H.blr_posterior_var, Phi, X, 0.7, 1.3, "primal")
    vd = call(H.blr_posterior_var, Phi, X, 0.7, 1.3, "dual")
    assert np.allclose(vp, vd) and np.allclose(vp, v_primal_ref(Phi, X, 0.7, 1.3))

def test_impl_cka_matches_reference_and_invariances():
    import uq_align as A
    X, Y = RNG.normal(size=(200, 7)), RNG.normal(size=(200, 7))
    assert np.isclose(call(A.linear_cka_primal, X, Y), cka_ref(X, Y))

def test_impl_d_eff_and_participation_ratio():
    import uq_spectrum as S
    ev = np.array([10.0, 1.0, 0.1])
    assert np.isclose(call(S.d_eff, ev, 0.5), deff_ref(ev, 0.5))
    assert np.isclose(call(S.participation_ratio, ev), ev.sum() ** 2 / (ev ** 2).sum())

def test_impl_rotate_and_scale_keeps_cka():
    import uq_spectrum as S
    X = RNG.normal(size=(300, 10))
    assert abs(cka_ref(X, call(S.rotate_and_scale, X, 3.0)) - 1) < 1e-8

def test_impl_tail_reshape():
    import uq_spectrum as S
    XA, _ = toy()
    XR = call(S.tail_reshape, XA, 0.95, 2.0)
    assert cka_ref(XA, XR) > 0.95

def test_impl_s_rho_matches_direct():
    import uq_align as A
    X, Y = RNG.normal(size=(60, 5)), RNG.normal(size=(60, 5))
    got = call(A.s_rho, X, Y, 0.7, False)
    Ma, Mb = (np.linalg.inv(P.T @ P + 0.7 * np.eye(5)) for P in (X, Y))
    H = lambda P, M: P @ M @ P.T
    ref = np.sum(H(X, Ma) * H(Y, Mb)) / (np.linalg.norm(H(X, Ma)) * np.linalg.norm(H(Y, Mb)))
    assert np.isclose(got, ref)

def test_impl_human_entropy_uniform():
    import uq_data as D
    counts = np.full((3, 10), 5)
    assert np.allclose(call(D.human_entropy, counts, "plugin"), np.log(10))

def test_impl_eu_au_shapes():
    import uq_heads as H
    probs = RNG.dirichlet(np.ones(10), size=(20, 7))   # [M=20, N=7, K=10]
    out = call(H.eu_au_from_probs, probs)
    for key in ("width_sum", "mi", "au", "total"):
        assert out[key].shape == (7,)
    assert np.all(out["mi"] >= -1e-9)
