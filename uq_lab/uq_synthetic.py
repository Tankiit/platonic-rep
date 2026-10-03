"""uq_synthetic.py -- numerical checks of the paper's theory on synthetic representations (numpy only).

S1  Theorem 1: Pearson corr of BLR epistemic variance across items == S_rho (population, Gaussian); CKA = rho->inf.
S2  Corollary 2 construction: shared k / private m directions -> CKA ~ 1 while EU corr ~ k/(k+m); and the converse.
S3  Corollary 3: identical features, different effective rho (data fraction / scale) -> EU agreement < 1.
Outputs: uq_runs/results/synthetic_*.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import uq_align as A
import uq_heads as H
from uq_config import Config

cfg = Config()
OUT = cfg.root / "results"
OUT.mkdir(parents=True, exist_ok=True)


def pop_index(LA, LB, LAB, MA, MB):
    """tr(MA SAB MB SBA) / sqrt(tr((MA SA)^2) tr((MB SB)^2)) for given covariances and quadratic-form matrices."""
    num = np.trace(MA @ LAB @ MB @ LAB.T)
    return num / np.sqrt(np.trace(MA @ LA @ MA @ LA) * np.trace(MB @ LB @ MB @ LB))


def sample_pair(rng, k, mA, mB, alpha, gauss=True, n=20000):
    """Latent z = [shared k | private A mA | private B mB]; each block power-law spectrum; random mixing per side."""
    def spec(m, off):
        return (np.arange(off + 1, off + m + 1) ** -alpha) if m else np.zeros(0)
    s_sh = spec(k, 0)
    sA = np.concatenate([s_sh * rng.uniform(0.5, 2.0, k), spec(mA, k // 2)])     # shared dims may differ in scale
    sB = np.concatenate([s_sh * rng.uniform(0.5, 2.0, k), spec(mB, k // 2)])
    D = k + mA + mB
    Z = rng.standard_normal((n, D)) if gauss else rng.laplace(size=(n, D)) / np.sqrt(2)
    ZA = np.concatenate([Z[:, :k], Z[:, k:k + mA]], 1) * np.sqrt(sA)
    ZB = np.concatenate([Z[:, :k], Z[:, k + mA:]], 1) * np.sqrt(sB)
    QA = np.linalg.qr(rng.standard_normal((k + mA, k + mA)))[0]
    QB = np.linalg.qr(rng.standard_normal((k + mB, k + mB)))[0]
    XA, XB = ZA @ QA.T, ZB @ QB.T
    XA /= np.sqrt(sA.sum() / len(sA))            # mean eigenvalue 1 on both sides so rho is comparable
    XB /= np.sqrt(sB.sum() / len(sB))
    return XA, XB


def s1_theorem(n_pairs=60, rhos=(0.01, 0.1, 1.0, 10.0), n_tr=20000, n_te=5000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pairs):
        k = int(rng.integers(2, 60)); mA = int(rng.integers(0, 250)); mB = int(rng.integers(0, 250))
        alpha = float(rng.uniform(0.3, 1.5)); gauss = p % 3 != 2
        XA, XB = sample_pair(rng, k, mA, mB, alpha, gauss, n_tr + n_te)
        XA -= XA[:n_tr].mean(0); XB -= XB[:n_tr].mean(0)
        trA, teA, trB, teB = XA[:n_tr], XA[n_tr:], XB[:n_tr], XB[n_tr:]
        cka = A.linear_cka_primal(trA, trB)
        for rho in rhos:
            vA, vB = H.blr_var_rho(trA, teA, rho), H.blr_var_rho(trB, teB, rho)
            rows.append(dict(pair=p, k=k, mA=mA, mB=mB, alpha=alpha, gaussian=gauss, rho=rho, cka=cka,
                             s_rho=A.s_rho_pop(trA, trB, rho), s_rho_0=A.s_rho_pop(trA, trB, 1e-6),
                             eu_pearson=pearsonr(vA, vB)[0], eu_spearman=spearmanr(vA, vB).correlation))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_s1_theorem.csv", index=False)
    return df


def s2_construction(ks=(9,), ms=(0, 10, 25, 50, 100, 200, 400), lam_s=10.0, lam_p=0.1, rho=0.01,
                    n_tr=20000, n_te=5000, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for m in ms:
            Z = rng.standard_normal((n_tr + n_te, k + 2 * m))
            XA = np.concatenate([Z[:, :k] * np.sqrt(lam_s), Z[:, k:k + m] * np.sqrt(lam_p)], 1)
            XB = np.concatenate([Z[:, :k] * np.sqrt(lam_s), Z[:, k + m:] * np.sqrt(lam_p)], 1)
            w = lambda l: l / (l + rho)
            th_cka = k * lam_s ** 2 / (k * lam_s ** 2 + m * lam_p ** 2)
            th_eu = k * w(lam_s) ** 2 / (k * w(lam_s) ** 2 + m * w(lam_p) ** 2)
            vA = H.blr_var_rho(XA[:n_tr], XA[n_tr:], rho); vB = H.blr_var_rho(XB[:n_tr], XB[n_tr:], rho)
            rows.append(dict(regime="high_cka", k=k, m=m, cka=A.linear_cka_primal(XA[:n_tr], XB[:n_tr]),
                             cka_theory=th_cka, eu_pearson=pearsonr(vA, vB)[0], eu_theory=th_eu,
                             eu_spearman=spearmanr(vA, vB).correlation))
    # converse: many low-variance shared dims, few high-variance private dims -> CKA ~ 0, EU corr ~ 1
    for kk in (50, 100, 200, 400):
        m, ls, lp = 2, 0.1, 10.0
        Z = rng.standard_normal((n_tr + n_te, kk + 2 * m))
        XA = np.concatenate([Z[:, :kk] * np.sqrt(ls), Z[:, kk:kk + m] * np.sqrt(lp)], 1)
        XB = np.concatenate([Z[:, :kk] * np.sqrt(ls), Z[:, kk + m:] * np.sqrt(lp)], 1)
        w = lambda l: l / (l + rho)
        vA = H.blr_var_rho(XA[:n_tr], XA[n_tr:], rho); vB = H.blr_var_rho(XB[:n_tr], XB[n_tr:], rho)
        rows.append(dict(regime="low_cka", k=kk, m=m, cka=A.linear_cka_primal(XA[:n_tr], XB[:n_tr]),
                         cka_theory=kk * ls ** 2 / (kk * ls ** 2 + m * lp ** 2), eu_pearson=pearsonr(vA, vB)[0],
                         eu_theory=kk * w(ls) ** 2 / (kk * w(ls) ** 2 + m * w(lp) ** 2),
                         eu_spearman=spearmanr(vA, vB).correlation))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_s2_construction.csv", index=False)
    return df


def s3_same_features(d=300, alpha=1.0, n_full=20000, fracs=(0.01, 0.05, 0.1, 0.25, 1.0), lam=200.0, seed=2):
    """Identical features (CKA = 1); heads see n = frac * n_full points with the SAME ridge lam, so rho = lam / n.
    Theory: corr(v_1, v_f) = sum w1 wf / sqrt(sum w1^2 sum wf^2), w = l/(l+rho)."""
    rng = np.random.default_rng(seed)
    ev = np.arange(1, d + 1) ** -alpha
    ev = ev / ev.mean()
    X = rng.standard_normal((n_full + 5000, d)) * np.sqrt(ev)
    tr, te = X[:n_full], X[n_full:]
    v_ref = H.blr_var_rho(tr, te, lam / n_full)
    rows = []
    for f in fracs:
        n = int(f * n_full)
        rho = lam / n
        v = H.blr_var_rho(tr[:n], te, rho)
        w1, wf = ev / (ev + lam / n_full), ev / (ev + rho)
        rows.append(dict(frac=f, n=n, rho=rho, cka=1.0, eu_pearson=pearsonr(v, v_ref)[0],
                         eu_spearman=spearmanr(v, v_ref).correlation,
                         theory=(w1 * wf).sum() / np.sqrt((w1 ** 2).sum() * (wf ** 2).sum()),
                         mean_eu_ratio=v.mean() / v_ref.mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_s3_same_features.csv", index=False)
    return df


if __name__ == "__main__":
    d1 = s1_theorem()
    for g, sub in d1.groupby("gaussian"):
        print(f"S1 gaussian={g}: |S_rho - EU pearson| mean={np.abs(sub.s_rho - sub.eu_pearson).mean():.4f} "
              f"max={np.abs(sub.s_rho - sub.eu_pearson).max():.4f}; corr(S_rho,EU)={pearsonr(sub.s_rho, sub.eu_pearson)[0]:.3f} "
              f"corr(CKA,EU)={pearsonr(sub.cka, sub.eu_pearson)[0]:.3f}; |CKA-EU| mean={np.abs(sub.cka - sub.eu_pearson).mean():.3f}")
    print(d1.groupby("rho")[["cka", "s_rho", "eu_pearson"]].apply(
        lambda s: pd.Series({"mae_srho": np.abs(s.s_rho - s.eu_pearson).mean(), "mae_cka": np.abs(s.cka - s.eu_pearson).mean()})))
    print(s2_construction().round(4).to_string())
    print(s3_same_features().round(4).to_string())
