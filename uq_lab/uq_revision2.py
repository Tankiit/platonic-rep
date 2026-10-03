"""uq_revision2.py -- round-2 post-review analyses (exploratory).

REV-11  leave-one-encoder-out (LOEO) sensitivity of index-vs-EU-agreement correlations (replaces the G=5 cluster bootstrap)
REV-16  Gaussianization test of the 4th-cumulant explanation: same joint covariance, Gaussian draws
REV-15  S_rho at a curvature-calibrated rho_eff = wd / mean_i p_i(1-p_i) of the MAP softmax head
REV-14  identical-features agreement at M=200 for all encoders, 3 draws, item-bootstrap intervals, wd pairs, residual
        variance fraction
Outputs: uq_runs/results/rev2_*.csv
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

import uq_align as A
import uq_experiments as X
import uq_heads as H
import uq_stats as St
from uq_config import Config, ENCODERS

cfg = Config()
R = cfg.root / "results"
ENCS = list(ENCODERS)
LI = len(cfg.rel_depths) - 1


def loeo():
    sr = pd.read_csv(R / "e_srho_vs_cka.csv")
    preds = ["cka", "mknn", "lin_pred", "s_rho_0", "s_rho"]
    targets = ["eu_width_sum_partial", "eu_blr", "eu_au"]
    rows = []
    for drop in [None] + ENCS:
        g = sr if drop is None else sr[(sr.encoder_a != drop) & (sr.encoder_b != drop)]
        for t in targets:
            for p in preds:
                rows.append(dict(dropped=drop or "none", target=t, index=p, rho=St.spearman(g[p], g[t]), n=len(g)))
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev2_loeo.csv", index=False)
    piv = df.pivot_table(index=["target", "index"], columns="dropped", values="rho")
    print(piv.round(3))
    # also the sweep at wd=1
    ws = pd.read_csv(R / "rev_wd_sweep.csv")
    rows = []
    for wd, gw in ws.groupby("wd"):
        for drop in [None] + ENCS:
            g = gw if drop is None else gw[(gw.encoder_a != drop) & (gw.encoder_b != drop)]
            for t in ("eu_blr", "eu_width_sum_partial"):
                rows.append(dict(wd=wd, dropped=drop or "none", target=t,
                                 diff_s0=St.spearman(g.s_rho, g[t]) - St.spearman(g.s_rho_0, g[t]),
                                 diff_cka=St.spearman(g.s_rho, g[t]) - St.spearman(g.cka, g[t])))
    pd.DataFrame(rows).to_csv(R / "rev2_loeo_sweep.csv", index=False)


def gaussianize(n_tr=20000, n_te=10000, seed=0):
    """For each of the 30 pairs: draw joint Gaussian features with the empirical joint covariance of (A,B) and compare
    the Pearson agreement of closed-form EU on (a) real features, (b) Gaussian surrogates, against S_rho."""
    rng = np.random.default_rng(seed)
    sub = np.random.default_rng(5).choice(50000, 10000, replace=False)
    rows = []
    for li, r in enumerate(cfg.rel_depths):
        for ea, eb in combinations(ENCS, 2):
            FtrA, FteA, _ = X.feats(cfg, ea, li); FtrB, FteB, _ = X.feats(cfg, eb, li)
            wa, wb = X.hyper(cfg)[f"{ea}|{li}"]["wd"], X.hyper(cfg)[f"{eb}|{li}"]["wd"]
            real = pearsonr(H.blr_var_rho(FtrA, FteA, wa), H.blr_var_rho(FtrB, FteB, wb))[0]
            J = np.hstack([FtrA, FtrB]); J = J - J.mean(0)
            C = J.T @ J / len(J)
            L = np.linalg.cholesky(C + 1e-8 * np.trace(C) / len(C) * np.eye(len(C)))
            Z = rng.standard_normal((n_tr + n_te, len(C))) @ L.T
            dA = FtrA.shape[1]
            gA, gB = Z[:, :dA], Z[:, dA:]
            gauss = pearsonr(H.blr_var_rho(gA[:n_tr], gA[n_tr:], wa), H.blr_var_rho(gB[:n_tr], gB[n_tr:], wb))[0]
            UA, UB = np.vstack([FtrA[sub], FteA]), np.vstack([FtrB[sub], FteB])
            s = X.s_rho_two(UA, UB, wa, wb)
            # excess kurtosis of the standardized leverage-direction projections (summary of non-Gaussianity)
            kA = float(np.mean(((FteA - FteA.mean(0)) / (FteA.std(0) + 1e-9)) ** 4) - 3)
            rows.append(dict(encoder_a=ea, encoder_b=eb, rel_depth=r, s_rho=s, real_pearson=real, gauss_pearson=gauss,
                             mean_excess_kurtosis_a=kA))
        print("depth", r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev2_gaussianize.csv", index=False)
    print("MAE real vs S", np.abs(df.real_pearson - df.s_rho).mean().round(3),
          "MAE gauss vs S", np.abs(df.gauss_pearson - df.s_rho).mean().round(3))


def rho_eff():
    """Curvature-calibrated prior strength for the MAP softmax head: rho_eff = wd / mean_i sum_k p_ik(1-p_ik)/K ...
    we use the average diagonal GGN scale h = mean_i mean_k p_ik(1-p_ik)."""
    ytr = X.labels()[0]
    hs = {}
    for enc in ENCS:
        for li in range(len(cfg.rel_depths)):
            Ftr, _, _ = X.feats(cfg, enc, li)
            wd = X.hyper(cfg)[f"{enc}|{li}"]["wd"]
            h = H.fit_bootstrap_heads(Ftr.astype(np.float32), ytr, 1, 1.0, 0, wd, poisson=False, device=X.DEV,
                                      stats=X.ident_stats(Ftr.shape[1]))
            p = H.predict_probs(h, Ftr.astype(np.float32))[0]
            hbar = float((p * (1 - p)).mean())
            hs[(enc, li)] = dict(wd=wd, hbar=hbar, rho_eff=wd / hbar)
    sr = pd.read_csv(R / "e_srho_vs_cka.csv")
    sub = np.random.default_rng(5).choice(50000, 10000, replace=False)
    vals = []
    for _, row in sr.iterrows():
        li = list(cfg.rel_depths).index(row.rel_depth)
        FtrA, FteA, _ = X.feats(cfg, row.encoder_a, li); FtrB, FteB, _ = X.feats(cfg, row.encoder_b, li)
        UA, UB = np.vstack([FtrA[sub], FteA]), np.vstack([FtrB[sub], FteB])
        vals.append(X.s_rho_two(UA, UB, hs[(row.encoder_a, li)]["rho_eff"], hs[(row.encoder_b, li)]["rho_eff"]))
    sr["s_rho_eff"] = vals
    sr.to_csv(R / "rev2_rho_eff_pairs.csv", index=False)
    pd.DataFrame([dict(encoder=k[0], li=k[1], **v) for k, v in hs.items()]).to_csv(R / "rev2_rho_eff.csv", index=False)
    for t in ("eu_width_sum_partial", "eu_width_sum", "eu_blr", "eu_au"):
        print(t, {p: round(St.spearman(sr[p], sr[t]), 3) for p in ("cka", "s_rho_0", "s_rho", "s_rho_eff")})
    print(pd.DataFrame(hs).T.describe().round(4))


def _partial(a, b, ca, cb, h):
    return St.partial_spearman(a, b, np.column_stack([ca, cb, h]))


def _resid_frac(a, ca, cb, h):
    """Fraction of rank variance of a left after regressing on the ranked controls."""
    Z = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in (ca, cb, h)])
    ra = rankdata(a)
    e = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    return float(e.var() / ra.var())


def identical_m200(M=200, n_draws=3, n_boot=200, seed=0, encs=None):
    ytr, _, _, hent = X.labels()
    rng = np.random.default_rng(seed)
    out = R / "rev2_identical_m200.csv"
    rows = pd.read_csv(out).to_dict("records") if (encs and out.exists()) else []
    for enc in (encs or ENCS):
        Ftr, Fte, _ = X.feats(cfg, enc, LI)
        wd = X.hyper(cfg)[f"{enc}|{LI}"]["wd"]
        fit = lambda seed_, idx=None, w=wd: X.fit_eu(Ftr, ytr, Fte, M, 1.0, seed_, w, idx)
        ref, ref2 = fit(0), fit(1)
        pairs = [("retest", 0, ref, ref, ref2, ref2)]
        for w, tag, s0 in ((wd / 10, "wd/10", 50), (wd * 10, "wdx10", 60)):
            pairs.append((tag, 0, ref, ref2, fit(s0, None, w), fit(s0 + 1, None, w)))
        for s in range(n_draws):
            r_ = np.random.default_rng(100 + s)
            sub = np.sort(r_.choice(len(Ftr), 5000, replace=False))
            perm = r_.permutation(len(Ftr))
            hA, hB = np.sort(perm[:25000]), np.sort(perm[25000:])
            pairs.append(("data_fraction_10pct", s, ref, ref2, fit(200 + s, sub), fit(300 + s, sub)))
            pairs.append(("disjoint_halves", s, fit(400 + s, hA), fit(500 + s, hA), fit(600 + s, hB), fit(700 + s, hB)))
        for name, s, x, x2, y, y2 in pairs:
            for metric in ("width_sum", "mi", "au"):
                cx, cx2 = x["mean_probs"].max(1), x2["mean_probs"].max(1)
                cy, cy2 = y["mean_probs"].max(1), y2["mean_probs"].max(1)
                obs = _partial(x[metric], y[metric], cx, cy, hent)
                row = dict(encoder=enc, pair=name, draw=s, metric=metric, M=M, raw=St.spearman(x[metric], y[metric]),
                           partial=obs, resid_frac=_resid_frac(x[metric], cx, cy, hent))
                if name != "retest":
                    rxx = _partial(x[metric], x2[metric], cx, cx2, hent)
                    ryy = _partial(y[metric], y2[metric], cy, cy2, hent)
                    corr = obs / np.sqrt(max(rxx * ryy, 1e-9))
                    bs = []
                    for _ in range(n_boot):
                        j = rng.integers(0, len(hent), len(hent))
                        o = _partial(x[metric][j], y[metric][j], cx[j], cy[j], hent[j])
                        a_ = _partial(x[metric][j], x2[metric][j], cx[j], cx2[j], hent[j])
                        b_ = _partial(y[metric][j], y2[metric][j], cy[j], cy2[j], hent[j])
                        bs.append(o / np.sqrt(max(a_ * b_, 1e-9)))
                    row.update(rel_x=rxx, rel_y=ryy, corrected=corr, corr_lo=float(np.quantile(bs, .025)),
                               corr_hi=float(np.quantile(bs, .975)))
                rows.append(row)
        print("done", enc, flush=True)
        pd.DataFrame(rows).to_csv(R / "rev2_identical_m200.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["pair", "metric"])[["raw", "partial", "corrected", "corr_lo", "corr_hi", "resid_frac"]].mean().round(3))


def identical_m200_mae():
    identical_m200(encs=["mae_b16"], seed=4)


if __name__ == "__main__":
    for step in (sys.argv[1:] or ["loeo", "gaussianize", "rho_eff", "identical_m200"]):
        print("=====", step, flush=True)
        globals()[step]()
        print("=====", step, "done", flush=True)
