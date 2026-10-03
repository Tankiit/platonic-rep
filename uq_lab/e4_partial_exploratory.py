"""EXPLORATORY (not pre-registered): E4 swap design with partial-Spearman disagreement (controls: both cells' confidence
and human entropy). Same seeds/cells as uq_experiments.e4_swap_design."""
import numpy as np, pandas as pd
import uq_experiments as X, uq_stats as St
from uq_config import Config
cfg = Config(); ytr, _, _, hent = X.labels(); li = len(cfg.rel_depths) - 1
encs = ("dinov2_b", "vit_sup_b16")
perm = np.random.default_rng(7).permutation(50000); halves = (np.sort(perm[:25000]), np.sort(perm[25000:]))
wd0 = X.hyper(cfg)[f"{encs[0]}|{li}"]["wd"]; cells = {}
for ri in (0, 1):
    Ftr, Fte, _ = X.feats(cfg, encs[ri], li)
    for hi in (0, 1):
        for di in (0, 1):
            o = X.fit_eu(Ftr, ytr, Fte, 50, 1.0, 10 + 4 * ri + 2 * hi + di, wd0 * (10.0 if hi else 1.0), halves[di])
            cells[(ri, hi, di)] = (o["width_sum"], o["mean_probs"].max(1))
    if ri == 0:
        Ftr0, Fte0 = Ftr, Fte
b = X.fit_eu(Ftr0, ytr, Fte0, 50, 1.0, 99, wd0, halves[0])
ref = cells[(0, 0, 0)]
pd_ = lambda c: 1 - St.partial_spearman(ref[0], c[0], np.column_stack([ref[1], c[1], hent]))
vals = {k: (0.0 if k == (0, 0, 0) else pd_(v)) for k, v in cells.items()}
floor = pd_((b["width_sum"], b["mean_probs"].max(1)))
sh = St.shapley_three_source(vals)
rows = [dict(source=k, share=v, phi=sh["phi"][k], total=sh["total"], noise_floor=floor) for k, v in sh["shares"].items()]
rows += [dict(source=f"cell{k}", share=np.nan, phi=v, total=sh["total"], noise_floor=floor) for k, v in vals.items()]
df = pd.DataFrame(rows); df.to_csv(cfg.result_path("e4_partial_exploratory"), index=False); print(df.round(3).to_string())
