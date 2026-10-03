"""run_all.py -- run the submitted experiment sequence in order (README run order), logging to stdout.
Usage: python run_all.py [choose_M] [main]"""
import json
import sys
import time

import numpy as np

import uq_experiments as X
import uq_heads as H
from uq_config import Config

cfg = Config()


def choose_M():
    ytr = X.labels()[0]
    Ftr, Fte, _ = X.feats(cfg, "dinov2_b", len(cfg.rel_depths) - 1)
    wd, _ = X.select_wd(Ftr, ytr)
    res = H.choose_M(Ftr.astype(np.float32), ytr, Fte.astype(np.float32), Ms=(10, 25, 50, 100), wd=wd, device=X.DEV)
    res["wd"] = wd
    (cfg.root / "results").mkdir(parents=True, exist_ok=True)
    (cfg.root / "results" / "choose_M.json").write_text(json.dumps(res, indent=1))
    print("choose_M", res, flush=True)


def main():
    for name in ("hyper", "e0_gate", "e2_constructed_pairs", "e_srho_vs_cka", "e_spec", "e2b_invariant_transforms",
                 "e4_swap_design", "e_rho_dial"):
        t = time.time()
        print(f"===== {name}", flush=True)
        getattr(X, name)(cfg)
        print(f"===== {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["choose_M", "main"]
    for s in steps:
        globals()[s]()
