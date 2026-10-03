"""Extract CIFAR-10 caches and training labels for uq_eu_specific.py."""
import argparse
import json
from pathlib import Path
import numpy as np
from uq_config import Config, ENCODERS
from uq_data import load_cifar10
from uq_features import extract_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('uq_runs'))
    p.add_argument('--encoders',nargs='+',choices=list(ENCODERS),default=list(ENCODERS))
    p.add_argument('--limit-train',type=int)
    p.add_argument('--limit-test',type=int)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    cfg=Config(root=args.root,batch_size=args.batch_size)
    for split in ('train','test'):
        images, labels=load_cifar10(str(args.data_root),train=split=='train')
        limit=args.limit_train if split=='train' else args.limit_test
        if limit is not None:
            if limit<100 or limit>len(images):
                p.error('Limits must be between 100 and the split size')
            idx=np.sort(np.random.default_rng(2026).choice(len(images),limit,replace=False))
            images,labels=images[idx],labels[idx]
        cfg.root.mkdir(parents=True,exist_ok=True)
        label_path=cfg.root/f'cifar10_{split}_labels.npy'
        if label_path.exists() and not np.array_equal(np.load(label_path),labels):
            raise ValueError('Existing cache labels differ: choose a new output root')
        np.save(label_path,labels)
        for enc in args.encoders:
            # Preserve, then regenerate caches with the wrong CLIP activation.
            cache=cfg.feature_path('cifar10',split,enc).with_suffix('.npy')
            meta=cache.with_suffix('.json')
            if enc=='clip_b16' and meta.exists():
                if not json.loads(meta.read_text()).get('force_quick_gelu',False):
                    backup=cfg.root/'invalid_activation'
                    backup.mkdir(exist_ok=True)
                    for path in (cache,meta):
                        if path.exists():
                            dest=backup/path.name
                            if dest.exists():
                                raise ValueError(f'Backup already exists: {dest}')
                            path.rename(dest)
            extract_split(cfg,enc,'cifar10',split,images=images)


if __name__=='__main__':
    main()
