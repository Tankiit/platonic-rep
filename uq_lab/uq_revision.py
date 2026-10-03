"""uq_revision.py -- POST-REVIEW analyses (round 1). Exploratory, NOT pre-registered; labelled as such in the paper.

R5   encoder-cluster bootstrap CIs for differences of Spearman correlations across the 30 encoder pairs
R7   reliability of (partial) EU agreement vs ensemble size M; subset redraws; reliability-corrected agreement
S22  sensitivity of partial agreement to the human-entropy covariate (plug-in, Dirichlet, two split halves)
S24  decision-level agreement: overlap of the top-10% EU (acquisition) sets
R3   weight-decay sweep moving d_eff/d below 1: does S_rho at the matched rho beat its rho->0 limit and CKA?
S14  S_rho on training-only inputs vs training U evaluation inputs
S23  item-level correlations among EU summaries, AU, confidence and the closed form
Outputs: uq_runs/results/rev_*.csv
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd

import uq_align as A
import uq_data as D
import uq_experiments as X
import uq_heads as H
import uq_spectrum as S
import uq_stats as St
from uq_config import Config, ENCODERS

cfg = Config()
R = cfg.root / "results"
ENCS = list(ENCODERS)
LI = len(cfg.rel_depths) - 1


def covariates():
    _, _, counts, hent = X.labels()
    rng = np.random.default_rng(11)
    half = np.stack([rng.multivariate_hypergeometric(c, c.sum() // 2) for c in counts])
    return {"plugin": [hent], "dirichlet": [D.human_entropy(counts, "dirichlet")],
            "split_halves": [D.human_entropy(half, "plugin"), D.human_entropy(counts - half, "plugin")]}


def partial(a, b, ca, cb, cov):
    return St.partial_spearman(a, b, np.column_stack([ca, cb] + cov))


def topk_overlap(a, b, q=0.10):
    k = int(q * len(a))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


# ---------------------------------------------------------------- R5
def cluster_bootstrap(B=2000, seed=0):
    sr = pd.read_csv(R / "e_srho_vs_cka.csv")
    preds = ["cka", "mknn", "lin_pred", "s_rho_0", "s_rho"]
    targets = ["eu_width_sum_partial", "eu_width_sum", "eu_blr", "eu_au"]
    rng = np.random.default_rng(seed)
    key = {tuple(sorted(p)): g for p, g in sr.groupby(["encoder_a", "encoder_b"])}

    def stat(df):
        return {(p, t): St.spearman(df[p], df[t]) for p in preds for t in targets}

    obs = stat(sr)
    boots = []
    for _ in range(B):
        samp = rng.choice(ENCS, len(ENCS), replace=True)
        parts = [key[tuple(sorted((samp[i], samp[j])))] for i, j in combinations(range(len(samp)), 2)
                 if samp[i] != samp[j]]
        if len(parts) < 3:
            continue
        boots.append(stat(pd.concat(parts)))
    rows = []
    for t in targets:
        for p in preds:
            v = np.array([b[(p, t)] for b in boots])
            rows.append(dict(target=t, index=p, kind="rho", est=obs[(p, t)], ci_lo=np.nanquantile(v, .025),
                             ci_hi=np.nanquantile(v, .975)))
        for p in ["cka", "mknn", "lin_pred", "s_rho_0"]:
            v = np.array([b[("s_rho", t)] - b[(p, t)] for b in boots])
            rows.append(dict(target=t, index=f"s_rho-{p}", kind="diff", est=obs[("s_rho", t)] - obs[(p, t)],
                             ci_lo=np.nanquantile(v, .025), ci_hi=np.nanquantile(v, .975),
                             p_le0=float(np.mean(v <= 0))))
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_cluster_bootstrap.csv", index=False)
    print(df.round(3).to_string())


# ---------------------------------------------------------------- R7, S22, S24
def reliability_redraws(M=50, n_draws=3):
    ytr, yte, counts, hent = X.labels()
    cov = covariates()
    rows = []
    for enc in ENCS:
        Ftr, Fte, _ = X.feats(cfg, enc, LI)
        wd = X.hyper(cfg)[f"{enc}|{LI}"]["wd"]
        fit = lambda frac, seed, idx=None: X.fit_eu(Ftr, ytr, Fte, M, frac, seed, wd, idx)
        ref, ref2 = fit(1.0, 0), fit(1.0, 1)
        pairs = []
        for s in range(n_draws):
            rng = np.random.default_rng(100 + s)
            sub = np.sort(rng.choice(len(Ftr), 5000, replace=False))
            perm = rng.permutation(len(Ftr))
            hA, hB = np.sort(perm[:25000]), np.sort(perm[25000:])
            f, f2 = fit(1.0, 200 + s, sub), fit(1.0, 300 + s, sub)
            a, a2, b, b2 = fit(1.0, 400 + s, hA), fit(1.0, 500 + s, hA), fit(1.0, 600 + s, hB), fit(1.0, 700 + s, hB)
            pairs += [("data_fraction_10pct", s, ref, ref2, f, f2), ("disjoint_halves", s, a, a2, b, b2)]
        pairs.append(("retest", 0, ref, ref, ref2, ref2))   # x=ref vs y=ref2 (fixed: previously compared ref with itself)
        for name, s, x, x2, y, y2 in pairs:
            for metric in ("width_sum", "mi", "au"):
                cx, cy = x["mean_probs"].max(1), y["mean_probs"].max(1)
                row = dict(encoder=enc, pair=name, draw=s, metric=metric, raw=St.spearman(x[metric], y[metric]),
                           top10_overlap=topk_overlap(x[metric], y[metric]))
                for cname, cv in cov.items():
                    row[f"partial_{cname}"] = partial(x[metric], y[metric], cx, cy, cv)
                if name != "retest":
                    rxx = partial(x[metric], x2[metric], cx, x2["mean_probs"].max(1), cov["plugin"])
                    ryy = partial(y[metric], y2[metric], cy, y2["mean_probs"].max(1), cov["plugin"])
                    row.update(rel_x=rxx, rel_y=ryy, partial_corrected=row["partial_plugin"] / np.sqrt(max(rxx * ryy, 1e-9)))
                rows.append(row)
        print("done", enc, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_reliability_redraws.csv", index=False)
    print(df.groupby(["pair", "metric"])[["raw", "partial_plugin", "partial_dirichlet", "partial_split_halves",
                                          "partial_corrected", "top10_overlap"]].mean().round(3))


def m_curve(enc="dinov2_b", Ms=(10, 25, 50, 100, 200)):
    ytr, _, _, hent = X.labels()
    Ftr, Fte, _ = X.feats(cfg, enc, LI)
    wd = X.hyper(cfg)[f"{enc}|{LI}"]["wd"]
    rows = []
    for M in Ms:
        a, b = X.fit_eu(Ftr, ytr, Fte, M, 1.0, 1000 + M, wd), X.fit_eu(Ftr, ytr, Fte, M, 1.0, 2000 + M, wd)
        for metric in ("width_sum", "mi", "au"):
            rows.append(dict(encoder=enc, M=M, metric=metric, raw=St.spearman(a[metric], b[metric]),
                             partial=partial(a[metric], b[metric], a["mean_probs"].max(1), b["mean_probs"].max(1), [hent])))
        print("M", M, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_m_curve.csv", index=False)
    print(df.round(3).to_string())


# ---------------------------------------------------------------- R3, S14, S24 (cross-encoder)
def wd_sweep(wds=(1e-3, 1e-2, 1e-1, 1.0), M=20):
    ytr, _, _, hent = X.labels()
    sub = np.random.default_rng(5).choice(50000, 10000, replace=False)
    rows, spec = [], []
    for wd in wds:
        eus = {}
        for enc in ENCS:
            for li in range(len(cfg.rel_depths)):
                Ftr, Fte, _ = X.feats(cfg, enc, li)
                o = X.fit_eu(Ftr, ytr, Fte, M, 1.0, 0, wd)
                eus[(enc, li)] = dict(width_sum=o["width_sum"], au=o["au"], conf=o["mean_probs"].max(1),
                                      blr=H.blr_var_rho(Ftr, Fte, wd))
                ev, _ = S.covariance_eigs(Ftr)
                spec.append(dict(wd=wd, encoder=enc, li=li, deff_frac=S.d_eff(ev, wd) / len(ev),
                                 acc=float((o["mean_probs"].argmax(1) == X.labels()[1]).mean())))
        for li, r in enumerate(cfg.rel_depths):
            for ea, eb in combinations(ENCS, 2):
                FtrA, FteA, _ = X.feats(cfg, ea, li); FtrB, FteB, _ = X.feats(cfg, eb, li)
                UA, UB = np.vstack([FtrA[sub], FteA]), np.vstack([FtrB[sub], FteB])
                ua, ub = eus[(ea, li)], eus[(eb, li)]
                row = dict(wd=wd, encoder_a=ea, encoder_b=eb, rel_depth=r, cka=A.linear_cka_primal(UA, UB),
                           s_rho=X.s_rho_two(UA, UB, wd, wd), s_rho_0=X.s_rho_two(UA, UB, 1e-6, 1e-6),
                           s_rho_trainonly=X.s_rho_two(FtrA[sub], FtrB[sub], wd, wd))
                for m in ("width_sum", "blr", "au"):
                    row[f"eu_{m}"] = St.spearman(ua[m], ub[m])
                    row[f"eu_{m}_partial"] = partial(ua[m], ub[m], ua["conf"], ub["conf"], [hent])
                row["top10_width"] = topk_overlap(ua["width_sum"], ub["width_sum"])
                rows.append(row)
        print("wd done", wd, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_wd_sweep.csv", index=False)
    pd.DataFrame(spec).to_csv(R / "rev_wd_sweep_spec.csv", index=False)
    summ = []
    for wd, g in df.groupby("wd"):
        for t in ("eu_width_sum_partial", "eu_blr", "eu_width_sum"):
            summ.append(dict(wd=wd, target=t, **{p: St.spearman(g[p], g[t]) for p in
                                                 ("cka", "s_rho_0", "s_rho", "s_rho_trainonly")}))
    print(pd.DataFrame(summ).round(3).to_string())
    print(pd.DataFrame(spec).groupby("wd")[["deff_frac", "acc"]].agg(["min", "max"]).round(3))


# ---------------------------------------------------------------- S23
def estimator_correlations():
    rows = []
    for enc in ENCS:
        u = X.eu_main(cfg, enc, LI)
        keys = ["width_sum", "mi", "au", "conf", "blr"]
        for a, b in combinations(keys, 2):
            rows.append(dict(encoder=enc, a=a, b=b, spearman=St.spearman(u[a], u[b])))
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_estimator_corr.csv", index=False)
    print(df.pivot_table(index=["a", "b"], columns="encoder", values="spearman").round(3))


def sweep_bootstrap(B=2000, seed=0):
    """Encoder-cluster bootstrap CIs for S_rho - S_rho0 and S_rho - CKA within each weight decay of the sweep."""
    ws = pd.read_csv(R / "rev_wd_sweep.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for wd, g in ws.groupby("wd"):
        key = {tuple(sorted(p)): gg for p, gg in g.groupby(["encoder_a", "encoder_b"])}
        for t in ("eu_blr", "eu_width_sum_partial"):
            obs = {p: St.spearman(g[p], g[t]) for p in ("s_rho", "s_rho_0", "cka")}
            bs = []
            for _ in range(B):
                samp = rng.choice(ENCS, len(ENCS), replace=True)
                parts = [key[tuple(sorted((samp[i], samp[j])))] for i, j in combinations(range(len(samp)), 2)
                         if samp[i] != samp[j]]
                if len(parts) < 3:
                    continue
                df = pd.concat(parts)
                bs.append({p: St.spearman(df[p], df[t]) for p in obs})
            for other in ("s_rho_0", "cka"):
                v = np.array([b["s_rho"] - b[other] for b in bs])
                rows.append(dict(wd=wd, target=t, diff=f"s_rho-{other}", est=obs["s_rho"] - obs[other],
                                 ci_lo=np.nanquantile(v, .025), ci_hi=np.nanquantile(v, .975), p_le0=float(np.mean(v <= 0))))
    df = pd.DataFrame(rows)
    df.to_csv(R / "rev_sweep_bootstrap.csv", index=False)
    print(df.round(3).to_string())


def fix_retest(M=50):
    """Recompute only the re-test rows of rev_reliability_redraws.csv (same seeds 0/1 as reliability_redraws)."""
    ytr, _, _, _ = X.labels()
    cov = covariates()
    df = pd.read_csv(R / "rev_reliability_redraws.csv")
    df = df[df.pair != "retest"]
    rows = []
    for enc in ENCS:
        Ftr, Fte, _ = X.feats(cfg, enc, LI)
        wd = X.hyper(cfg)[f"{enc}|{LI}"]["wd"]
        x, y = X.fit_eu(Ftr, ytr, Fte, M, 1.0, 0, wd), X.fit_eu(Ftr, ytr, Fte, M, 1.0, 1, wd)
        for metric in ("width_sum", "mi", "au"):
            cx, cy = x["mean_probs"].max(1), y["mean_probs"].max(1)
            row = dict(encoder=enc, pair="retest", draw=0, metric=metric, raw=St.spearman(x[metric], y[metric]),
                       top10_overlap=topk_overlap(x[metric], y[metric]))
            for cname, cv in cov.items():
                row[f"partial_{cname}"] = partial(x[metric], y[metric], cx, cy, cv)
            rows.append(row)
        print("retest", enc, flush=True)
    pd.concat([df, pd.DataFrame(rows)]).to_csv(R / "rev_reliability_redraws.csv", index=False)


if __name__ == "__main__":
    for step in (sys.argv[1:] or ["cluster_bootstrap", "estimator_correlations", "m_curve", "reliability_redraws",
                                  "wd_sweep"]):
        print("=====", step, flush=True)
        globals()[step]()
        print("=====", step, "done", flush=True)
