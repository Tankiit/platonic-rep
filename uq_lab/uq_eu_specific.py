"""Exploratory EU isolation benchmark; run independently of the frozen PREREG.
Exact heteroscedastic Bayesian linear regression (linear-kernel GP).
AU is prescribed observation variance, EU is latent posterior variance.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist

import uq_align as A


def rank_corr(x, y):
    if len(x)<2 or len(y)<2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(spearmanr(x, y).statistic)


def posterior_variance(train, query, noise, prior=1.0):
    """Actual latent variance, with FIXED prior across sample sizes (no n rescaling)."""
    train, query = np.asarray(train, float), np.asarray(query, float)
    noise = np.broadcast_to(np.asarray(noise, float), (len(train),))
    if not np.isfinite(prior) or prior <= 0 or np.any(noise <= 0) or not np.all(np.isfinite(noise)):
        raise ValueError('Prior and observation variances must be finite and positive')
    if len(train)<train.shape[1]:
        kernel=prior*train@train.T+np.diag(noise)
        cross=prior*query@train.T
        solved=cho_solve(cho_factor(kernel),cross.T)
        return np.maximum(prior*np.sum(query*query,axis=1)-np.einsum('ij,ji->i',cross,solved),0)
    precision = (train.T / noise) @ train + np.eye(train.shape[1]) / prior
    solved = cho_solve(cho_factor(precision), query.T)
    return np.maximum(np.einsum('ij,ji->i', query, solved), 0)


def gp_variance(train, query, noise, prior=1.0, lengthscale=1.0):
    """Exact RBF GP latent variance; hyperparameters frozen across training sizes."""
    if prior <= 0 or lengthscale <= 0:
        raise ValueError('Positive kernel parameters required')
    noise=np.broadcast_to(np.asarray(noise,float),(len(train),))
    if np.any(noise<=0):
        raise ValueError('Positive noise required')
    kernel=prior*np.exp(-cdist(train,train,'sqeuclidean')/(2*lengthscale**2))
    cross=prior*np.exp(-cdist(query,train,'sqeuclidean')/(2*lengthscale**2))
    factor=cho_factor(kernel+np.diag(noise))
    return np.maximum(prior-np.einsum('ij,ji->i',cross,cho_solve(factor,cross.T)),0)


def acquisition_overlap(x, y, frac=0.1):
    k = max(1, int(np.ceil(len(x) * frac)))
    return len(set(np.argsort(x, kind='stable')[-k:]) & set(np.argsort(y, kind='stable')[-k:])) / k


def metrics(a, b, seed):
    return dict(cka=A.linear_cka_primal(a, b),
                mknn=A.global_mknn(a, b, min(10, len(a)-1)),
                predictivity=(A.linear_predictivity(a, b, seed=seed) +
                              A.linear_predictivity(b, a, seed=seed))/2,
                cca_type=A.s_rho_pop(a, b, 1e-6),
                smoother_001=A.s_rho_pop(a, b, .01),
                smoother_1=A.s_rho_pop(a, b, 1.0))


def evaluate(views, train_group, query_group, coverage_group, seed, priors=(.1, 1., 10.), fractions=(.1, .25, 1.), head='linear', max_train=1024):
    rng = np.random.default_rng(seed)
    # Independent alignment partition: no posterior/acquisition query inputs used.
    n = len(train_group)
    idx = rng.permutation(n)
    align_idx, pool = idx[:min(n//4,1024)], idx[n//4:]
    pool=pool[:max_train]
    order = rng.permutation(pool)
    # Known noise is driven by metadata, never predicted confidence or human labels.
    tr_noise = np.where(train_group, 1., .1)
    au = np.where(query_group, 1., .1)
    normalized={}
    lengthscales={}
    for name,(tr,te) in views.items():
        mu=tr[pool].mean(0)
        scale=float(np.sqrt(np.mean((tr[pool]-mu)**2)))
        if not np.isfinite(scale) or scale<=0:
            raise ValueError(f'Degenerate features: {name}')
        normalized[name]=((tr-mu)/scale,(te-mu)/scale)
        if head=='rbf':
            lengthscales[name]=max(float(np.median(pdist(normalized[name][0][pool[:256]]))),1e-8)
    pair_metrics = {pair: metrics(normalized[pair[0]][0][align_idx], normalized[pair[1]][0][align_idx], seed)
                    for pair in combinations(views, 2)}
    rows, checks = [], []
    for prior in priors:
        previous = {}
        for frac in fractions:
            subset = order[:max(2, int(len(order)*frac))]
            eus = {}
            for name, (x,q) in normalized.items():
                if head=='linear':
                    eu = posterior_variance(x[subset], q, tr_noise[subset], prior)
                else:
                    lengthscale=lengthscales[name]
                    eu = gp_variance(x[subset],q,tr_noise[subset],prior,lengthscale)
                eus[name] = eu
                max_increase = float(np.max(eu-previous[name])) if name in previous else 0.
                checks.append(dict(seed=seed, view=name, prior=prior, fraction=frac,
                                   mean_eu=float(eu.mean()), mean_au=float(au.mean()),
                                   eu_au_rank=rank_corr(eu, au),
                                   mean_eu_id=float(eu[~coverage_group].mean()),
                                   mean_eu_heldout=float(eu[coverage_group].mean()),
                                   max_increase=max_increase,
                                   monotonic_pass=max_increase < 1e-8))
                previous[name] = eu
            for (a,b), ms in pair_metrics.items():
                rows.append(dict(seed=seed, a=a, b=b, prior=prior, fraction=frac,
                                 n_alignment=len(align_idx), n_train=len(subset),
                                 dim_a=normalized[a][0].shape[1], dim_b=normalized[b][0].shape[1],
                                 eu_agreement=rank_corr(eus[a], eus[b]),
                                 acquisition_overlap=acquisition_overlap(eus[a], eus[b]),
                                 # Stratification prevents a mixed ID/OOD ranking driving agreement.
                                 eu_group0=rank_corr(eus[a][~query_group], eus[b][~query_group]),
                                 eu_group1=rank_corr(eus[a][query_group], eus[b][query_group]),
                                 eu_id=rank_corr(eus[a][~coverage_group], eus[b][~coverage_group]),
                                 eu_heldout=rank_corr(eus[a][coverage_group], eus[b][coverage_group]),
                                 **ms))
    return rows, checks


def synthetic(seed, n=800, nq=400, d=12):
    rng = np.random.default_rng(seed)
    tr, te = rng.normal(size=(n,d)), rng.normal(size=(nq,d))
    # Training has little coverage in the final coordinates; OOD queries explore them.
    tr[:, -3:] *= .05
    group = np.arange(nq) >= nq//2
    te[group, -3:] *= 4
    # Noise metadata deliberately independent of coverage/OOD status.
    ngtr, ngte = rng.random(n)>.5, rng.random(nq)>.5
    views = {'base': (tr,te)}
    for i, strength in enumerate((.1,.3,1.,3.)):
        q, _ = np.linalg.qr(rng.normal(size=(d,d)))
        transform = q @ np.diag(np.geomspace(strength,1/strength,d))
        views[f'transform_{i}'] = (tr@transform, te@transform)
    # group used for stratification is noise group; OOD diagnostics stored separately.
    return views, ngtr, ngte, group


def real_features(root, labels, heldout, layers=None):
    ytr = np.load(labels)
    views = {}
    for path in sorted((root/'features').glob('cifar10_train_*.npy')):
        enc = path.stem.removeprefix('cifar10_train_')
        test = root/'features'/f'cifar10_test_{enc}.npy'
        if not test.exists():
            continue
        tr = np.load(path, mmap_mode='r')
        meta_path=path.with_suffix('.json')
        if not meta_path.exists():
            raise ValueError(f'Missing extraction metadata: {meta_path}')
        meta=json.loads(meta_path.read_text())
        if meta.get('done')!=len(tr) or meta.get('N')!=len(tr):
            raise ValueError(f'Incomplete cache: {path}')
        if len(ytr) != len(tr):
            raise ValueError('Training labels must follow feature-cache row order')
        # Query disjoint from training, all from TRAIN caches: class withholding arm.
        query = np.arange(len(tr)) % 5 == 0
        fit = ~query & (ytr != heldout)
        for li in (layers if layers is not None else range(tr.shape[1])):
            if li<0 or li>=tr.shape[1]:
                raise ValueError(f"Invalid layer {li} for {enc}")
            dims=meta.get('tap_dims',[tr.shape[2]]*tr.shape[1])[li]
            views[f'{enc}|{li}'] = (np.asarray(tr[fit,li,:dims],float), np.asarray(tr[query,li,:dims],float))
    if not np.any(ytr==heldout) or not np.any(ytr!=heldout):
        raise ValueError('Held-out class must exist alongside other classes')
    if len(views)<2:
        raise ValueError('Need at least two complete train/test encoder caches; cloned JSON sidecars are insufficient')
    # AU prescribed by parity of class ID, not inferred from human disagreement.
    return views, (ytr[fit]%2).astype(bool), (ytr[query]%2).astype(bool), ytr[query]==heldout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-train',type=int,default=1024)
    p.add_argument('--head',choices=('linear','rbf'),default='linear')
    p.add_argument('--mode', choices=('synthetic','features'), default='synthetic')
    p.add_argument('--root', type=Path, default=Path('uq_runs'))
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--labels', type=Path)
    p.add_argument('--heldout', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    p.add_argument('--output', type=Path, default=Path('uq_runs/eu_specific'))
    args=p.parse_args()
    if args.max_train<20:
        p.error('--max-train must be at least 20')
    if args.mode=='features' and args.labels is None:
        p.error('--labels is required in features mode')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.labels is not None:
        manifest['label_sha256']=hashlib.sha256(args.labels.read_bytes()).hexdigest()
        feature_hashes={}
        for path in sorted((args.root/'features').glob('cifar10_train_*.npy')):
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            feature_hashes[path.name]=digest.hexdigest()
        manifest['feature_sha256']=feature_hashes
        manifest['cache_metadata']={p.name:json.loads(p.read_text())
            for p in sorted((args.root/'features').glob('cifar10_train_*.json'))}
    manifest['alignment_source_sha256']=hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest()
    manifest['numpy_version']=np.__version__
    manifest.update(exploratory=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    interpretation='Exact EU under a prescribed Gaussian likelihood; not classification EU')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, checks=[],[]
    for seed in args.seeds:
        data=synthetic(seed) if args.mode=='synthetic' else real_features(args.root,args.labels,args.heldout,args.layers)
        for noise_condition in ('heteroscedastic','homoscedastic'):
            views, tg, qg, coverage = data
            if noise_condition=='homoscedastic':
                tg, qg = np.zeros_like(tg), np.zeros_like(qg)
            r,c=evaluate(views,tg,qg,coverage,seed=seed,head=args.head,max_train=args.max_train)
            for row in r+c:
                row['noise_condition']=noise_condition
                row['head']=args.head
            rows.extend(r); checks.extend(c)
        print(f'seed {seed} complete', flush=True)
    pairs, diagnostics=pd.DataFrame(rows),pd.DataFrame(checks)
    pairs.to_csv(args.output/'pairs.csv',index=False)
    diagnostics.to_csv(args.output/'diagnostics.csv',index=False)
    summary=[]
    for keys, frame in pairs.groupby(['noise_condition','seed','prior','fraction']):
        for metric in ('cka','mknn','predictivity','cca_type','smoother_001','smoother_1'):
            for target in ('eu_agreement','acquisition_overlap','eu_group0','eu_group1','eu_id','eu_heldout'):
                summary.append(dict(noise_condition=keys[0],seed=keys[1],prior=keys[2],fraction=keys[3],metric=metric,
                                    target=target,metric_span=float(np.ptp(frame[metric])),
                                    correlation=rank_corr(frame[metric],frame[target])))
    pd.DataFrame(summary).to_csv(args.output/'metric_prediction.csv',index=False)
    omissions=[]
    for keys, frame in pairs.groupby(['noise_condition','seed','prior','fraction']):
        encoders=sorted(set(frame.a.str.split('|').str[0]) | set(frame.b.str.split('|').str[0]))
        for omitted in encoders:
            keep=frame[(frame.a.str.split('|').str[0]!=omitted) &
                       (frame.b.str.split('|').str[0]!=omitted)]
            for metric in ('cka','mknn','predictivity','cca_type','smoother_001','smoother_1'):
                for target in ('eu_agreement','acquisition_overlap','eu_id','eu_heldout'):
                    omissions.append(dict(noise_condition=keys[0],seed=keys[1],prior=keys[2],
                        fraction=keys[3],omitted=omitted,metric=metric,target=target,
                        metric_span=float(np.ptp(keep[metric])) if len(keep) else float('nan'),
                        correlation=rank_corr(keep[metric],keep[target])))
    pd.DataFrame(omissions).to_csv(args.output/'leave_one_encoder_out.csv',index=False)
    if not diagnostics.monotonic_pass.all():
        raise RuntimeError('Posterior variance monotonicity failed')
    print(f'{len(pairs)} pairs; all variance-contraction checks passed; output: {args.output}')


if __name__=='__main__':
    main()
