"""Extract CIFAR-10 caches for uq_blindspot.py: a seeded train subset and the FULL test split
(full test is required for CIFAR-10H / B5). Resumable; one encoder at a time."""
import argparse
from pathlib import Path

import numpy as np

from uq_config import BLINDSPOT_ENCODERS, Config
from uq_data import load_cifar10
from uq_features import extract_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('uq_runs_blindspot'))
    p.add_argument('--encoders', nargs='+', choices=list(BLINDSPOT_ENCODERS), default=list(BLINDSPOT_ENCODERS))
    p.add_argument('--n-train', type=int, default=10000)
    p.add_argument('--n-test', type=int, help='Smoke tests only; B5 needs the full test split')
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    cfg = Config(root=args.root, batch_size=args.batch_size)
    cfg.root.mkdir(parents=True, exist_ok=True)
    for split, n in (('test', args.n_test), ('train', args.n_train)):
        images, labels = load_cifar10(train=split == 'train')
        idx = np.arange(len(images))
        if n is not None and n < len(images):
            idx = np.sort(np.random.default_rng(2026).choice(len(images), n, replace=False))
        images, labels = images[idx], labels[idx]
        for name, arr in (('labels', labels), ('index', idx)):
            path = cfg.root / f'cifar10_{split}_{name}.npy'
            if path.exists() and not np.array_equal(np.load(path), arr):
                raise ValueError(f'{path} differs from this request: choose a new --root')
            np.save(path, arr)
        for enc in args.encoders:
            extract_split(cfg, enc, 'cifar10', split, images=images)


if __name__ == '__main__':
    main()
