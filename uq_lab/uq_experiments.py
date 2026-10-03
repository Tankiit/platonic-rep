"""uq_experiments.py -- one function per experiment.

Results: a tidy CSV per experiment (cfg.result_path(name)); EU vectors cached in <root>/eu/*.npz.
Feature convention (decided once): centre with the TRAIN mean and divide by ONE scalar (sqrt of the mean per-dimension
TRAIN variance). No per-dimension standardisation: that would undo the transformations of E2b and break the link to
the theory, which is stated in raw feature coordinates. Weight decay is chosen per (encoder, depth) by validation NLL on
a held-out 10k split of the CIFAR-10 TRAIN set with HARD labels. Human soft labels are used only for evaluation.
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
import torch

import uq_align as A
import uq_data as D
import uq_features as Fe
import uq_heads as H
import uq_spectrum as S
import uq_stats as St
from uq_config import ENCODERS, PREREG

DEV = torch.device("cpu")          # CPU was faster than MPS for these L-BFGS fits (timed); float32
M_DEFAULT = 50
RHO_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


# ------------------------------------------------------------------ shared plumbing
_cache = {}


def labels():
    if "y" not in _cache:
        _, ytr = D.load_cifar10(train=True)
        _, yte = D.load_cifar10(train=False)
        counts = D.load_cifar10h(test_labels=yte)
        _cache["y"] = (ytr, yte, counts, D.human_entropy(counts, "plugin"))
    return _cache["y"]


def feats(cfg, enc, layer):
    """Centred, scalar-normalised (train stats) features: (Ftr [50000,d], Fte [10000,d], s0)."""
    key = (enc, layer)
    if key not in _cache:
        Ftr = Fe.load_features(cfg, enc, "cifar10", "train", layer).astype(np.float64)
        Fte = Fe.load_features(cfg, enc, "cifar10", "test", layer).astype(np.float64)
        mu = Ftr.mean(0)
        s0 = np.sqrt(((Ftr - mu) ** 2).mean())
        _cache[key] = ((Ftr - mu) / s0, (Fte - mu) / s0, s0)
    return _cache[key]


def ident_stats(d):
    return (np.zeros(d, np.float32), np.ones(d, np.float32))


def fit_eu(Ftr, ytr, Fte, M, frac, seed, wd, subset_idx=None):
    h = H.fit_bootstrap_heads(Ftr.astype(np.float32), ytr, M, frac, seed, wd, device=DEV,
                              stats=ident_stats(Ftr.shape[1]), subset_idx=subset_idx)
    out = H.eu_au_from_probs(H.predict_probs(h, Fte.astype(np.float32)))
    out["acc_heads"] = h
    return out


def select_wd(Ftr, ytr, grid=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
    """Validation NLL on a fixed held-out 10k of the TRAIN set (hard labels)."""
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(Ftr))
    va, tr = idx[:10000], idx[10000:]
    nll = {}
    for wd in grid:
        h = H.fit_bootstrap_heads(Ftr[tr].astype(np.float32), ytr[tr], 1, 1.0, 0, wd, poisson=False, device=DEV,
                                  stats=ident_stats(Ftr.shape[1]))
        p = H.predict_probs(h, Ftr[va].astype(np.float32))[0]
        nll[wd] = float(-np.log(np.clip(p[np.arange(len(va)), ytr[va]], 1e-12, 1)).mean())
    return min(nll, key=nll.get), nll


def hyper(cfg):
    """wd per (encoder, layer); cached to json."""
    path = cfg.root / "results" / "hyper.json"
    if path.exists():
        return json.loads(path.read_text())
    ytr = labels()[0]
    out = {}
    for enc in ENCODERS:
        for li, r in enumerate(cfg.rel_depths):
            Ftr, _, _ = feats(cfg, enc, li)
            wd, nll = select_wd(Ftr, ytr)
            out[f"{enc}|{li}"] = {"wd": wd, "nll": nll}
            print("hyper", enc, r, wd, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def eu_main(cfg, enc, li, M=M_DEFAULT):
    """Main-config ensemble (all train data, selected wd) for (encoder, layer); cached."""
    path = cfg.root / "eu" / f"main_{enc}_{li}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    ytr, yte, _, _ = labels()
    Ftr, Fte, _ = feats(cfg, enc, li)
    wd = hyper(cfg)[f"{enc}|{li}"]["wd"]
    o = fit_eu(Ftr, ytr, Fte, M, 1.0, 0, wd)
    blr = H.blr_var_rho(Ftr, Fte, wd)
    res = dict(width_sum=o["width_sum"], mi=o["mi"], au=o["au"], conf=o["mean_probs"].max(1),
               acc=np.array((o["mean_probs"].argmax(1) == yte).mean()), blr=blr, wd=np.array(wd))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return res


def agreement_row(a, b, conf_a, conf_b, hent):
    return dict(spearman=St.spearman(a, b), partial=St.partial_spearman(a, b, np.column_stack([conf_a, conf_b, hent])),
                stratified=St.stratified_agreement(a, b, hent)["mean"])


def write(cfg, name, rows):
    df = pd.DataFrame(rows)
    p = cfg.result_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(df.round(4).to_string())
    return df


# ------------------------------------------------------------------ experiments
def e0_gate(cfg):
    """Reliability ceiling of human AU, estimator robustness, and a known-AU positive control.
    Positive control (deviation from the plan: DCIC 'Synthetic' is not used): on frozen features of each encoder,
    draw training labels from a known well-specified softmax teacher; true AU = teacher entropy on test items.
    GATE: Spearman(model AU, true AU) >= 0.5."""
    _, yte, counts, hent = labels()
    ytr = labels()[0]
    rows = []
    c = D.split_half_ceiling(counts, n_rep=50)
    est = {m: St.spearman(hent, D.human_entropy(counts, m)) for m in ("miller_madow", "dirichlet")}
    rows.append(dict(check="cifar10h_split_half", value=c["ceiling"], ci_lo=c["ci"][0], ci_hi=c["ci"][1],
                     r_half=c["r_half"], **{f"rho_plugin_{k}": v for k, v in est.items()}))
    rng = np.random.default_rng(0)
    for enc in ENCODERS:
        Ftr, Fte, _ = feats(cfg, enc, len(cfg.rel_depths) - 1)
        wd = hyper(cfg)[f"{enc}|{len(cfg.rel_depths) - 1}"]["wd"]
        teacher = H.fit_bootstrap_heads(Ftr.astype(np.float32), ytr, 1, 1.0, 0, wd, poisson=False, device=DEV,
                                        stats=ident_stats(Ftr.shape[1]))
        for T in (1.0, 3.0):
            teacher_T = H.Heads(teacher.W / T, teacher.b / T, teacher.mu, teacher.sd)
            ptr = H.predict_probs(teacher_T, Ftr.astype(np.float32))[0]
            pte = H.predict_probs(teacher_T, Fte.astype(np.float32))[0]
            ysyn = np.array([rng.choice(10, p=p / p.sum()) for p in ptr])
            o = fit_eu(Ftr, ysyn, Fte, 20, 1.0, 1, wd)
            true_au = -(pte * np.log(np.clip(pte, 1e-12, 1))).sum(1)
            r = St.spearman(o["au"], true_au)
            rows.append(dict(check=f"known_au_{enc}_T{T}", value=r, gate=PREREG["E0_gate"]["rule"], passed=r >= 0.5))
    return write(cfg, "e0_gate", rows)


def e2_constructed_pairs(cfg, M=M_DEFAULT):
    """HEADLINE. Identical frozen features (alignment = 1 by construction); heads differ in data fraction, disjoint
    halves, weight decay. Retest pair (same config, new Poisson weights) gives the EU-agreement noise floor."""
    ytr, yte, counts, hent = labels()
    li = len(cfg.rel_depths) - 1
    c = D.split_half_ceiling(counts, n_rep=20)["ceiling"]
    rows = []
    for enc in ENCODERS:
        Ftr, Fte, _ = feats(cfg, enc, li)
        wd = hyper(cfg)[f"{enc}|{li}"]["wd"]
        perm = np.random.default_rng(7).permutation(len(Ftr))
        halfA, halfB = np.sort(perm[:25000]), np.sort(perm[25000:])
        cfgs = {"ref": dict(frac=1.0, seed=0, wd=wd), "retest": dict(frac=1.0, seed=1, wd=wd),
                "frac0.1": dict(frac=0.1, seed=2, wd=wd), "frac0.25": dict(frac=0.25, seed=3, wd=wd),
                "halfA": dict(frac=None, seed=4, wd=wd, idx=halfA), "halfB": dict(frac=None, seed=5, wd=wd, idx=halfB),
                "wd/10": dict(frac=1.0, seed=6, wd=wd / 10), "wdx10": dict(frac=1.0, seed=8, wd=wd * 10)}
        eu = {k: fit_eu(Ftr, ytr, Fte, M, v["frac"] or 1.0, v["seed"], v["wd"], v.get("idx")) for k, v in cfgs.items()}
        for k, o in eu.items():
            rows.append(dict(encoder=enc, pair=f"{k}|human", factor="au_vs_human", metric="au",
                             value=St.spearman(o["au"], hent), ceiling=c,
                             value_norm=St.ceiling_normalise(St.spearman(o["au"], hent), c),
                             acc=float((o["mean_probs"].argmax(1) == yte).mean())))
        pairs = [("ref", "retest", "noise_floor"), ("ref", "frac0.1", "data_fraction"), ("ref", "frac0.25", "data_fraction"),
                 ("halfA", "halfB", "disjoint_data"), ("ref", "wd/10", "weight_decay"), ("ref", "wdx10", "weight_decay")]
        for a, b, factor in pairs:
            ca, cb = eu[a]["mean_probs"].max(1), eu[b]["mean_probs"].max(1)
            for metric in ("width_sum", "mi", "au"):
                r = agreement_row(eu[a][metric], eu[b][metric], ca, cb, hent)
                rows.append(dict(encoder=enc, pair=f"{a}|{b}", factor=factor, metric=metric, value=r["spearman"],
                                 partial=r["partial"], stratified=r["stratified"]))
        print("done", enc, flush=True)
    return write(cfg, "e2_constructed_pairs", rows)


def _apply(Ftr, Fte, T):
    return Ftr @ T.T, Fte @ T.T


def e2b_invariant_transforms(cfg, encs=("dinov2_b", "clip_b16"), M=M_DEFAULT):
    """Scale dial c (Prop. 1) and tail dial s (Prop. 3) applied to the SAME features; CKA/mKNN/linear predictivity
    vs the original next to EU agreement and mean EU. Fixed wd, and wd re-tuned by validation NLL (control).
    EU from bootstrap ensembles AND last-layer Laplace AND BLR."""
    ytr, yte, _, hent = labels()
    li = len(cfg.rel_depths) - 1
    rows = []
    for enc in encs:
        Ftr, Fte, _ = feats(cfg, enc, li)
        d = Ftr.shape[1]
        wd = hyper(cfg)[f"{enc}|{li}"]["wd"]
        ev, U = S.covariance_eigs(Ftr)
        k95 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        Q = S.random_orthogonal(d, 0)
        transforms = {f"scale_c={c}": c * Q for c in (0.1, 0.3, 1.0, 3.0, 10.0)}
        for s in (0.5, 2.0, 4.0):
            sc = np.ones(d); sc[k95:] = s
            transforms[f"tail_s={s}"] = (U * sc) @ U.T
        ref = None
        for name, T in transforms.items():
            Gtr, Gte = _apply(Ftr, Fte, T)
            for mode in ("fixed_wd", "retuned_wd"):
                w = wd if mode == "fixed_wd" else select_wd(Gtr, ytr, grid=tuple(wd * 10.0 ** np.arange(-4, 5)))[0]
                o = fit_eu(Gtr, ytr, Gte, M, 1.0, 0, w)
                lap = H.laplace_last_layer(o["acc_heads"], Gtr.astype(np.float32), ytr, prior_prec=w * len(Gtr),
                                           F_te=Gte.astype(np.float32))
                blr = H.blr_var_rho(Gtr, Gte, w)
                cur = dict(width_sum=o["width_sum"], mi=o["mi"], laplace_mi=lap["mi"], laplace_var=lap["logit_var"],
                           blr=blr, au=o["au"])
                if name == "scale_c=1.0" and mode == "fixed_wd":
                    ref = cur
                rows.append(dict(encoder=enc, transform=name, mode=mode, wd=w, k95=k95,
                                 cka=A.linear_cka_primal(Fte, Gte), mknn=A.global_mknn(Fte, Gte, 10),
                                 lin_pred=A.linear_predictivity(Fte, Gte),
                                 acc=float((o["mean_probs"].argmax(1) == yte).mean()), _cur=cur))
        for r in rows:
            if r["encoder"] != enc or "_cur" not in r:
                continue
            cur = r.pop("_cur")
            for m in cur:
                r[f"rho_{m}"] = St.spearman(cur[m], ref[m])
                r[f"meanratio_{m}"] = float(np.mean(cur[m]) / np.mean(ref[m]))
        print("done", enc, flush=True)
    return write(cfg, "e2b_invariant_transforms", rows)


def s_rho_two(FA, FB, rhoA, rhoB):
    """Population EU-agreement index with each side's own rho (Theorem 1 with M_A=(S_A+rho_A)^-1, M_B=(S_B+rho_B)^-1)."""
    a = FA - FA.mean(0); b = FB - FB.mean(0); n = len(a)
    Sa, Sb, Sab = a.T @ a / n, b.T @ b / n, a.T @ b / n
    Ma = np.linalg.inv(Sa + rhoA * np.eye(len(Sa))); Mb = np.linalg.inv(Sb + rhoB * np.eye(len(Sb)))
    return float(np.trace(Ma @ Sab @ Mb @ Sab.T) / np.sqrt(np.trace(Ma @ Sa @ Ma @ Sa) * np.trace(Mb @ Sb @ Mb @ Sb)))


def e_srho_vs_cka(cfg):
    """Across encoder pairs (all 10 pairs x 3 matched relative depths): does S_rho at the heads' own rho predict EU
    agreement better than CKA, mKNN, linear predictivity? Includes the E1 atlas columns and the
    Theorem-1 check on real (non-Gaussian) features (BLR Pearson agreement vs S_rho)."""
    from scipy.stats import pearsonr
    _, _, _, hent = labels()
    rows = []
    ytr = labels()[0]
    sub = np.random.default_rng(5).choice(50000, 10000, replace=False)
    for li, r in enumerate(cfg.rel_depths):
        eus = {enc: eu_main(cfg, enc, li) for enc in ENCODERS}
        for ea, eb in combinations(ENCODERS, 2):
            FtrA, FteA, _ = feats(cfg, ea, li); FtrB, FteB, _ = feats(cfg, eb, li)
            UA, UB = np.vstack([FtrA[sub], FteA]), np.vstack([FtrB[sub], FteB])      # train UNION eval (Prop. 4)
            ua, ub = eus[ea], eus[eb]
            row = dict(encoder_a=ea, encoder_b=eb, rel_depth=r, cka=A.linear_cka_primal(UA, UB),
                       mknn=A.global_mknn(FteA, FteB, 10), lin_pred=0.5 * (A.linear_predictivity(FteA, FteB) +
                                                                          A.linear_predictivity(FteB, FteA)),
                       s_rho=s_rho_two(UA, UB, float(ua["wd"]), float(ub["wd"])),
                       s_rho_0=s_rho_two(UA, UB, 1e-6, 1e-6),
                       **{f"s_rho_grid_{g:g}": s_rho_two(UA, UB, g, g) for g in RHO_GRID},   # exploratory
                       blr_pearson=pearsonr(ua["blr"], ub["blr"])[0])
            for m in ("width_sum", "mi", "blr", "au"):
                ag = agreement_row(ua[m], ub[m], ua["conf"], ub["conf"], hent)
                row[f"eu_{m}"] = ag["spearman"]; row[f"eu_{m}_partial"] = ag["partial"]
            rows.append(row)
        print("done depth", r, flush=True)
    return write(cfg, "e_srho_vs_cka", rows)


def e_spec(cfg):
    """Label-free prediction: d_eff(rho) of each encoder's TRAIN spectrum vs mean EU on test (BLR and bootstrap);
    power-law fit on the middle of the spectrum."""
    rows = []
    for enc in ENCODERS:
        for li, r in enumerate(cfg.rel_depths):
            Ftr, Fte, _ = feats(cfg, enc, li)
            ev, _ = S.covariance_eigs(Ftr)
            u = eu_main(cfg, enc, li)
            wd = float(u["wd"])
            d = len(ev)
            pl = S.fit_power_law(ev, int(0.05 * d), int(0.5 * d))
            rows.append(dict(encoder=enc, rel_depth=r, d=d, wd=wd, d_eff=S.d_eff(ev, wd), d_eff_fixed=S.d_eff(ev, 1e-3),
                             **{f"d_eff_grid_{g:g}": S.d_eff(ev, g) for g in RHO_GRID},                 # exploratory
                             pr=S.participation_ratio(ev), alpha=pl["alpha"], alpha_r2=pl["r2"], alpha_good=pl["good"],
                             mean_blr=float(u["blr"].mean()), mean_width=float(u["width_sum"].mean()),
                             mean_mi=float(u["mi"].mean()), acc=float(u["acc"])))
    return write(cfg, "e_spec", rows)


def e4_swap_design(cfg, M=M_DEFAULT):
    """2 encoders x 2 head configs x 2 data subsets; Shapley shares of EU disagreement 1 - Spearman(EU_000, EU_cell).
    Encoders: the highest-CKA pair at the final depth (from e_srho_vs_cka)."""
    ytr, _, _, _ = labels()
    li = len(cfg.rel_depths) - 1
    at = pd.read_csv(cfg.result_path("e_srho_vs_cka"))
    at = at[at.rel_depth == cfg.rel_depths[-1]].sort_values("cka", ascending=False).iloc[0]
    encs = (at.encoder_a, at.encoder_b)
    perm = np.random.default_rng(7).permutation(50000)
    halves = (np.sort(perm[:25000]), np.sort(perm[25000:]))
    rows, out = [], {}
    for metric in ("width_sum", "mi"):
        out[metric] = {}
    base = None
    for ri in (0, 1):
        Ftr, Fte, _ = feats(cfg, encs[ri], li)
        wd0 = hyper(cfg)[f"{encs[0]}|{li}"]["wd"]
        for hi in (0, 1):
            for di in (0, 1):
                o = fit_eu(Ftr, ytr, Fte, M, 1.0, 10 + 4 * ri + 2 * hi + di, wd0 * (10.0 if hi else 1.0), halves[di])
                for metric in out:
                    out[metric][(ri, hi, di)] = o[metric]
                if (ri, hi, di) == (0, 0, 0):
                    base = fit_eu(Ftr, ytr, Fte, M, 1.0, 99, wd0, halves[0])
    for metric, cells in out.items():
        vals = {k: 1 - St.spearman(cells[(0, 0, 0)], v) for k, v in cells.items()}
        floor = 1 - St.spearman(cells[(0, 0, 0)], base[metric])
        sh = St.shapley_three_source(vals)
        for k, v in sh["shares"].items():
            rows.append(dict(encoder_a=encs[0], encoder_b=encs[1], metric=metric, source=k, share=v,
                             phi=sh["phi"][k], total=sh["total"], noise_floor=floor, cka=at.cka))
        for k, v in vals.items():
            rows.append(dict(encoder_a=encs[0], encoder_b=encs[1], metric=metric, source=f"cell{k}", share=np.nan,
                             phi=v, total=sh["total"], noise_floor=floor, cka=at.cka))
    return write(cfg, "e4_swap_design", rows)


def e_rho_dial(cfg, enc="dinov2_b"):
    """Spearman matrix of per-item BLR EU across a rho grid (Mahalanobis-like at small rho, norm-like at large)."""
    li = len(cfg.rel_depths) - 1
    Ftr, Fte, _ = feats(cfg, enc, li)
    grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    v = {r: H.blr_var_rho(Ftr, Fte, r) for r in grid}
    norm = (Fte ** 2).sum(1)
    lev = H.blr_var_rho(Ftr, Fte, 1e-8)
    rows = [dict(encoder=enc, rho_a=a, rho_b=b, value=St.spearman(v[a], v[b])) for a in grid for b in grid]
    rows += [dict(encoder=enc, rho_a=a, rho_b="norm", value=St.spearman(v[a], norm)) for a in grid]
    rows += [dict(encoder=enc, rho_a=a, rho_b="leverage", value=St.spearman(v[a], lev)) for a in grid]
    return write(cfg, "e_rho_dial", rows)


def e1_alignment_atlas(cfg):
    """Folded into e_srho_vs_cka (same pairs, same columns)."""
    return e_srho_vs_cka(cfg)


def e_au_function_agreement(cfg):
    raise NotImplementedError("not run for the submission")


def h8_class_residual(cfg):
    raise NotImplementedError("not run for the submission (share_W gate not evaluated)")


def e3_two_factor_grid(cfg):
    raise NotImplementedError("not run for the submission")


def e6_robustness(cfg):
    raise NotImplementedError("not run for the submission")
