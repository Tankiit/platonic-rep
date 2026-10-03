"""uq_run.py -- CLI dispatcher. Example:  python uq_run.py e0_gate   |   python uq_run.py e2_constructed_pairs --seeds 0 1 2"""
import argparse
import uq_experiments as X
from uq_config import Config

REGISTRY = {name: getattr(X, name) for name in dir(X) if name.startswith(("e0_", "e1_", "e2", "e3_", "e4_", "e6_", "e_", "h8_"))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=sorted(REGISTRY))
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--root", default=None)
    a = p.parse_args()
    cfg = Config()
    if a.seeds is not None:
        cfg.seeds = tuple(a.seeds)
    if a.root:
        from pathlib import Path
        cfg.root = Path(a.root)
    try:
        REGISTRY[a.experiment](cfg)
    except NotImplementedError as e:
        raise SystemExit(f"{a.experiment} is still a stub: fill the TODOs in uq_experiments.py and the modules it calls. ({e})")

if __name__ == "__main__":
    main()
