"""Blind-spot follow-up (B0-B7); exploratory, run independently of the frozen PREREG.
See BLINDSPOT_PLAN.md. A factor level is a blind spot when alignment S stays flat (|dS| <= delta_S)
while EU agreement R moves (|dR| > delta_R); a false alarm in the reverse case.
B0 freezes delta_S / delta_R to <root>/thresholds.json; B1-B7 refuse to run without it.

EU (default) is label-free Gaussian EU x^T (Sigma_hat + rho I)^-1 x, rho = RHO_REL * mean eigenvalue, on features
centred and scalar-normalised with frozen train-subset statistics. R = Spearman of EU over the same queries.
Alignment inputs, head-training inputs and queries are disjoint: alignment and heads come from disjoint
partitions of the train subset, queries from the test split.

Usage: python uq_blindspot.py b0 [b1 ... b7 | all] [--root uq_runs_blindspot]
"""
import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import uq_align as A
import uq_data as D
import uq_heads as H
import uq_spectrum as SP
from uq_config import BLINDSPOT_ENCODERS, FAMILIES

ROOT = Path('uq_runs_blindspot')
NATURAL = tuple(e for e in BLINDSPOT_ENCODERS if FAMILIES[e] != 'random')
FINAL = 2                       # tap index of relative depth 1.0
LAYERS = (0, 1, 2)
N_ALIGN = 8192                  # >= 4d for the widest view (ResNet-50 final, d = 2048)
N_HEAD = 4096
N_QUERY = 2000
RHO_REL = 1e-3
SEEDS = (0, 1, 2, 3, 4)
K_NN = 10
WD = 1e-3                       # selected for most views in uq_runs/results/hyper.json
M_HEADS = 20
ALIGN_MEASURES = ('cka', 'mknn', 'predictivity', 'cca_type')
ALL_MEASURES = ALIGN_MEASURES + ('smoother_001', 'smoother_1')

# Pre-specified decision constants (BLINDSPOT_PLAN.md); fixed before any real-data result.
RHO_GRID = tuple(10.0 ** np.arange(-6, 3))
B1_STABLE = 0.95                # rankings "barely move" if every pair >= 5 orders apart has Spearman > this
TAIL_S = (1.0, 1.5, 2.0, 3.0)
TAIL_ENERGY = 0.95
GL_KAPPA = 10.0
FRACTIONS = (1.0, 0.5, 0.25, 0.1)
B4_N_SET = 4096
B4_TRIPLES = 3
B4_GATE = 0.1                   # held-out predictability must change by >= this across alignment sets
B6_SHARE_W_GATE = 0.5
B7_ORDER_FLAG = 0.5


# ---------------------------------------------------------------- utilities

def rank_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(spearmanr(x, y).statistic)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def write(root, name, rows):
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f'{name}.csv', index=False)
    print(f'[{name}] {len(frame)} rows -> {out / name}.csv', flush=True)
    return frame


def write_json(root, name, obj):
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(obj, indent=1, default=float))
    print(f'[{name}] -> {out / name}.json', flush=True)


# ---------------------------------------------------------------- data

class Data:
    """Frozen-normalised features per view (enc, layer), loaded lazily as float32."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.ytr = np.load(self.root / 'cifar10_train_labels.npy')
        self.yte = np.load(self.root / 'cifar10_test_labels.npy')
        self.tr_index = np.load(self.root / 'cifar10_train_index.npy')
        self.te_index = np.load(self.root / 'cifar10_test_index.npy')
        self._cache = {}

    def encoders(self):
        return [e for e in BLINDSPOT_ENCODERS if (self.root / 'features' / f'cifar10_train_{e}.json').exists()
                and self._complete(e)]

    def _complete(self, enc):
        for split, n in (('train', len(self.ytr)), ('test', len(self.yte))):
            meta = self.root / 'features' / f'cifar10_{split}_{enc}.json'
            if not meta.exists() or json.loads(meta.read_text()).get('done') != n:
                return False
        return True

    def view(self, enc, layer):
        key = (enc, layer)
        if key not in self._cache:
            if enc == 'colorhist':
                tr, te = colour_histograms(self)
            else:
                out = []
                for split in ('train', 'test'):
                    path = self.root / 'features' / f'cifar10_{split}_{enc}.npy'
                    meta = json.loads(path.with_suffix('.json').read_text())
                    arr = np.load(path, mmap_mode='r')
                    out.append(np.asarray(arr[:, layer, :meta['tap_dims'][layer]], np.float32))
                tr, te = out
            mu = tr.mean(0, dtype=np.float64)
            scale = float(np.sqrt(np.mean((tr - mu) ** 2)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'Degenerate features: {key}')
            self._cache[key] = (((tr - mu) / scale).astype(np.float32), ((te - mu) / scale).astype(np.float32))
        return self._cache[key]

    def drop(self, keep=()):
        for k in list(self._cache):
            if k not in keep:
                del self._cache[k]

    def split(self, seed):
        """Disjoint train partitions: alignment (N_ALIGN) and head pool (rest); queries from test."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.ytr))
        align, pool = np.sort(perm[:N_ALIGN]), perm[N_ALIGN:]
        head = np.sort(pool[:N_HEAD])
        query = np.sort(rng.choice(len(self.yte), N_QUERY, replace=False))
        return dict(align=align, pool=pool, head=head, query=query)


def colour_histograms(data, bins=4):
    """Trivial baseline: joint 4x4x4 RGB histogram of the raw 32x32 image (64-d)."""
    out = []
    for train, index in ((True, data.tr_index), (False, data.te_index)):
        images, _ = D.load_cifar10(train=train)
        q = (images[index].astype(np.int64) * bins // 256).reshape(len(index), -1, 3)
        code = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
        hist = np.stack([np.bincount(c, minlength=bins ** 3) for c in code]).astype(np.float32)
        out.append(hist / hist.sum(1, keepdims=True))
    return out


# ---------------------------------------------------------------- EU

def eu_operator(Ftrain, rho_rel=RHO_REL):
    """Return M = (Sigma_hat + rho I)^-1 (float64) with Sigma_hat = X^T X / n (features already centred)."""
    X = np.asarray(Ftrain, np.float64)
    S = X.T @ X / len(X)
    rho = rho_rel * np.trace(S) / S.shape[0]
    return np.linalg.inv(S + rho * np.eye(S.shape[0]))


def eu(Ftrain, Fq, rho_rel=RHO_REL, M=None):
    M = eu_operator(Ftrain, rho_rel) if M is None else M
    Q = np.asarray(Fq, np.float64)
    return np.einsum('ij,jk,ik->i', Q, M, Q)


def norm_score(Fq):
    return np.sum(np.asarray(Fq, np.float64) ** 2, axis=1)


def logistic_heads(Ftrain, ytrain, Fq, seed, M=M_HEADS):
    """Bootstrap logistic EU/AU (B5, B7, AU in B2). Features used as given (already normalised)."""
    d = Ftrain.shape[1]
    heads = H.fit_bootstrap_heads(np.asarray(Ftrain, np.float32), ytrain, M, 1.0, seed, WD,
                                  stats=(np.zeros(d, np.float32), np.ones(d, np.float32)),
                                  subset_idx=np.arange(len(Ftrain)), K=10)
    out = H.eu_au_from_probs(H.predict_probs(heads, np.asarray(Fq, np.float32)))
    return dict(mi=out['mi'], width=out['width_sum'], au=out['au'], mean_probs=out['mean_probs'])


# ---------------------------------------------------------------- alignment (cached; equals E.metrics)

class ViewStats:
    """Per-view caches on one alignment set so that pair metrics cost only cross-products."""

    def __init__(self, F, seed, rho_pops=(1e-6, .01, 1.0), ridge=1e-3, test_frac=0.3):
        X = np.asarray(F, np.float64)
        n = len(X)
        C = X - X.mean(0)
        self.G = C.T @ C
        self.C = C.astype(np.float32)          # float32 storage; products are upcast (memory for 30 views)
        eye = np.eye(self.G.shape[0])
        self.sm = {}
        for rp in rho_pops:
            Mi = np.linalg.inv(self.G + rp * n * eye)
            MG = Mi @ self.G
            self.sm[rp] = (Mi, float(np.sqrt(np.trace(MG @ MG))))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        te, tr = idx[:int(test_frac * n)], idx[int(test_frac * n):]
        mtr = X[tr].mean(0)
        tr_c, te_c = X[tr] - mtr, X[te] - mtr
        self.te_var = float(((X[te] - X[te].mean(0)) ** 2).sum())
        Gt = tr_c.T @ tr_c
        self.tr_c, self.te_c = tr_c.astype(np.float32), te_c.astype(np.float32)
        self.P = np.linalg.inv(Gt + ridge * np.trace(Gt) / Gt.shape[0] * eye)
        self.knn = A._knn(X, K_NN, 'cosine')
        self.gnorm = float(np.linalg.norm(self.G))


def _predict(a, b):
    W = a.P @ (a.tr_c.T.astype(np.float64) @ b.tr_c.astype(np.float64))
    res = b.te_c.astype(np.float64) - a.te_c.astype(np.float64) @ W
    return 1 - float((res ** 2).sum()) / b.te_var


def pair_metrics(a, b):
    Gab = a.C.T.astype(np.float64) @ b.C.astype(np.float64)
    out = dict(cka=float(np.linalg.norm(Gab) ** 2 / (a.gnorm * b.gnorm)),
               mknn=float(np.mean([len(set(x) & set(y)) / K_NN for x, y in zip(a.knn, b.knn)])),
               predictivity=(_predict(a, b) + _predict(b, a)) / 2)
    for rp, name in ((1e-6, 'cca_type'), (.01, 'smoother_001'), (1.0, 'smoother_1')):
        (Ma, na), (Mb, nb) = a.sm[rp], b.sm[rp]
        out[name] = float(np.trace(Ma @ Gab @ Mb @ Gab.T) / (na * nb))
    return out


# ---------------------------------------------------------------- thresholds and verdicts

def load_thresholds(root):
    path = Path(root) / 'thresholds.json'
    if not path.exists():
        raise RuntimeError('Run b0 first: thresholds are frozen there (BLINDSPOT_PLAN.md)')
    obj = json.loads(path.read_text())
    payload = {k: v for k, v in obj.items() if k != 'sha256'}
    if sha(payload) != obj['sha256']:
        raise RuntimeError('thresholds.json was modified after freezing')
    return obj


def verdict(S_curve, R_curve, delta_S, delta_R):
    """Per-level 'blind' / 'false alarm' / 'consistent', relative to the first (reference) level."""
    if delta_S is None or delta_R is None:
        raise ValueError('delta_S and delta_R must be pre-registered (see BLINDSPOT_PLAN.md)')
    S, R = np.asarray(S_curve, float), np.asarray(R_curve, float)
    if S.shape != R.shape or S.ndim != 1:
        raise ValueError('S_curve and R_curve must be 1-D and the same length')
    s_moves = np.abs(S - S[0]) > delta_S
    r_moves = np.abs(R - R[0]) > delta_R
    return ['blind' if r and not s else 'false alarm' if s and not r else 'consistent'
            for s, r in zip(s_moves, r_moves)]


def pairwise_verdicts(S, R, delta_S, delta_R):
    """Apply the definition with 'which pair' as the factor: counts over all pairs of pair-records."""
    S, R = np.asarray(S, float), np.asarray(R, float)
    i, j = np.triu_indices(len(S), 1)
    ds, dr = np.abs(S[i] - S[j]) > delta_S, np.abs(R[i] - R[j]) > delta_R
    n = len(i)
    return dict(n=int(n), blind=float(np.mean(dr & ~ds)), false_alarm=float(np.mean(ds & ~dr)),
                both=float(np.mean(ds & dr)), neither=float(np.mean(~ds & ~dr)))


# ---------------------------------------------------------------- B0

def toy_check():
    """Known answer: 9 shared high-variance directions + 200 unshared tail directions (test_uq_theory.toy)."""
    from test_uq_theory import toy
    out = {}
    for share_tail in (False, True):
        XA, XB = toy(share_tail=share_tail)
        n_tr = 4000
        ms = pair_metrics(ViewStats(XA[:2000], 0), ViewStats(XB[:2000], 0))
        r = rank_corr(eu(XA[:n_tr] - XA[:n_tr].mean(0), XA[n_tr:] - XA[:n_tr].mean(0), 1e-6),
                      eu(XB[:n_tr] - XB[:n_tr].mean(0), XB[n_tr:] - XB[:n_tr].mean(0), 1e-6))
        out['shared_tail' if share_tail else 'unshared_tail'] = dict(R_eu=r, **ms)
    u, s = out['unshared_tail'], out['shared_tail']
    out['pass'] = bool(u['cka'] > 0.99 and abs(u['R_eu']) < 0.2 and s['R_eu'] > 0.9)
    out['expected'] = 'unshared: CKA ~ 0.998, R_eu ~ 0.03; shared: R_eu > 0.9'
    return out


def gates(cfg):
    """B0: toy, self-reshape control, n_align/saturation asserts, EU seed ceiling, S noise, trivial baselines.
    Freezes thresholds.json. Raises if the toy or the reshape control fails."""
    root = cfg['root']
    data = Data(root)
    encs = [e for e in data.encoders() if e in NATURAL]
    if len(encs) < 2:
        raise RuntimeError(f'Need complete caches; found {encs}')
    report = dict(encoders=encs, toy=toy_check())
    print('toy', report['toy'], flush=True)
    if not report['toy']['pass']:
        write_json(root, 'b0_gates', report)
        raise RuntimeError('B0 toy known-answer failed: stop and debug')

    # Self-reshape control on a real view.
    sp = data.split(0)
    F_tr, F_te = data.view(encs[0], FINAL)
    Fa = F_tr[sp['align']]
    self_ms = pair_metrics(ViewStats(Fa, 0), ViewStats(Fa, 0))
    T = tail_transform(F_tr[sp['head']], 2.0)
    reshape_ms = pair_metrics(ViewStats(Fa, 0), ViewStats(Fa @ T, 0))
    r_self = rank_corr(eu(F_tr[sp['head']], F_te[sp['query']]), eu(F_tr[sp['head']], F_te[sp['query']]))
    ev = SP.covariance_eigs(F_tr[sp['head']])[0]
    ev2 = SP.covariance_eigs(F_tr[sp['head']] @ T)[0]
    rho = RHO_REL * ev.mean()
    report['self_reshape'] = dict(encoder=encs[0], self_metrics=self_ms, self_R=r_self, reshape_metrics=reshape_ms,
                                  deff_ratio=float(SP.d_eff(ev2, rho) / SP.d_eff(ev, rho)))
    sr = report['self_reshape']
    # 1e-3: predictivity carries a ridge term, so self-predictivity is slightly below 1.
    sr['pass'] = bool(all(abs(self_ms[m] - 1) < 1e-3 for m in ALIGN_MEASURES) and abs(r_self - 1) < 1e-9
                      and reshape_ms['cka'] > 0.95)
    print('self_reshape', sr, flush=True)
    if not sr['pass']:
        write_json(root, 'b0_gates', report)
        raise RuntimeError('B0 self/reshape control failed: stop and debug')

    # EU self-agreement ceiling: two disjoint equal-size head draws of the same encoder.
    ceiling = []
    for enc in encs:
        F_tr, F_te = data.view(enc, FINAL)
        assert N_ALIGN >= 4 * F_tr.shape[1], f'n_align < 4d for {enc}'
        for seed in SEEDS:
            sp = data.split(seed)
            h2 = np.sort(sp['pool'][N_HEAD:2 * N_HEAD])
            ceiling.append(dict(encoder=enc, seed=seed, d=F_tr.shape[1],
                                R_self=rank_corr(eu(F_tr[sp['head']], F_te[sp['query']]),
                                                 eu(F_tr[h2], F_te[sp['query']]))))
        data.drop()
    ceiling = pd.DataFrame(ceiling)
    write(root, 'b0_eu_ceiling', ceiling)

    # S noise over independent alignment draws; saturation asserts on every natural pair.
    stats = {}
    for seed in SEEDS:
        sp = data.split(seed)
        for enc in encs:
            stats[(enc, seed)] = ViewStats(data.view(enc, FINAL)[0][sp['align']], seed)
            data.drop()
        print(f'b0 alignment stats seed {seed}', flush=True)
    snoise = []
    for a, b in combinations(encs, 2):
        for seed in SEEDS:
            snoise.append(dict(a=a, b=b, seed=seed, **pair_metrics(stats[(a, seed)], stats[(b, seed)])))
    snoise = pd.DataFrame(snoise)
    write(root, 'b0_alignment_noise', snoise)
    report['saturation'] = dict(max_cca_type=float(snoise.cca_type.max()),
                                min_predictivity=float(snoise.predictivity.min()))
    assert snoise.cca_type.max() < 1, 'cca_type saturated: n_align too small relative to d'
    assert snoise.predictivity.min() > 0, 'non-positive predictivity R^2'

    # Trivial baselines vs each natural encoder (seed 0): random-init net, colour histogram, norm-only score.
    sp = data.split(0)
    base_rows = []
    baselines = [('vit_rand_b16', FINAL)] if 'vit_rand_b16' in data.encoders() else []
    baselines.append(('colorhist', 0))
    bstats = {}
    beu = {}
    for key in baselines:
        tr, te = data.view(*key)
        bstats[key] = ViewStats(tr[sp['align']], 0)
        beu[key] = eu(tr[sp['head']], te[sp['query']])
    for enc in encs:
        tr, te = data.view(enc, FINAL)
        st, e = stats[(enc, 0)], eu(tr[sp['head']], te[sp['query']])
        base_rows.append(dict(encoder=enc, baseline='norm_only', R_eu=rank_corr(e, norm_score(te[sp['query']]))))
        for key in baselines:
            base_rows.append(dict(encoder=enc, baseline=key[0], R_eu=rank_corr(e, beu[key]),
                                  **pair_metrics(st, bstats[key])))
        data.drop(keep=baselines)
    write(root, 'b0_baselines', base_rows)

    # Freeze thresholds.
    per_enc_sd = ceiling.groupby('encoder').R_self.std(ddof=1)
    per_pair_sd = snoise.groupby(['a', 'b'])[list(ALL_MEASURES)].std(ddof=1)
    payload = dict(
        rule='delta_R = max(0.02, 2 * max_enc SD_seed(R_self)); delta_S[m] = max(0.005, 2 * max_pair SD_seed(S_m))',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        delta_R=float(max(0.02, 2 * per_enc_sd.max())),
        delta_S={m: float(max(0.005, 2 * per_pair_sd[m].max())) for m in ALL_MEASURES},
        R_self_mean={k: float(v) for k, v in ceiling.groupby('encoder').R_self.mean().items()},
        rho_rel=RHO_REL, n_align=N_ALIGN, n_head=N_HEAD, n_query=N_QUERY)
    payload['sha256'] = sha(payload)
    path = Path(root) / 'thresholds.json'
    if path.exists():
        raise RuntimeError(f'{path} already frozen; delete it deliberately to re-run B0')
    path.write_text(json.dumps(payload, indent=1))
    report['thresholds'] = payload
    report['pass'] = True
    write_json(root, 'b0_gates', report)
    return report


def tail_transform(F_fit, s, energy=TAIL_ENERGY):
    """Linear map scaling eigendirections outside the top-`energy` set by s (eigenbasis fit on F_fit)."""
    ev, U = SP.covariance_eigs(F_fit, center=False)
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), energy) + 1)
    scale = np.ones_like(ev)
    scale[k:] = s
    return ((U * scale) @ U.T).astype(np.float32)


# ---------------------------------------------------------------- B1

def rho_dial(cfg):
    """B1: label-free dependence of EU ranking on rho (relative to the mean eigenvalue)."""
    root = cfg['root']
    load_thresholds(root)
    data = Data(root)
    encs = data.encoders()
    sp = data.split(0)
    rows, deff_rows, eus = [], [], {}
    for enc in encs:
        tr, te = data.view(enc, FINAL)
        X, Q = tr[sp['head']], te[sp['query']]
        ev = SP.covariance_eigs(X, center=False)[0]
        mean_ev = ev.mean()
        v = {r: eu(X, Q, r) for r in RHO_GRID}
        maha = eu(X, Q, 1e-10)
        nrm = norm_score(Q)
        for a in RHO_GRID:
            for b in RHO_GRID:
                rows.append(dict(encoder=enc, rho_a=a, rho_b=b, spearman=rank_corr(v[a], v[b])))
            rows.append(dict(encoder=enc, rho_a=a, rho_b='mahalanobis', spearman=rank_corr(v[a], maha)))
            rows.append(dict(encoder=enc, rho_a=a, rho_b='norm', spearman=rank_corr(v[a], nrm)))
            deff_rows.append(dict(encoder=enc, rho_rel=a, d=X.shape[1], n=len(X),
                                  d_eff=float(SP.d_eff(ev, a * mean_ev))))
        eus[enc] = v
        data.drop()
    rows, deff_rows = write(root, 'b1_rho_matrix', rows), write(root, 'b1_deff', deff_rows)
    # Pair-level: how much does cross-encoder EU agreement move with rho alone?
    pr = []
    nat = [e for e in encs if e in NATURAL]
    for a, b in combinations(nat, 2):
        for r in RHO_GRID:
            pr.append(dict(a=a, b=b, rho_rel=r, R_eu=rank_corr(eus[a][r], eus[b][r])))
    pr = write(root, 'b1_pair_agreement', pr)
    far = rows[(rows.rho_b != 'mahalanobis') & (rows.rho_b != 'norm')].copy()
    far['orders'] = np.abs(np.log10(far.rho_a.astype(float)) - np.log10(far.rho_b.astype(float)))
    per = far[far.orders >= 5].groupby('encoder').spearman.min()
    span = pr.groupby(['a', 'b']).R_eu.agg(np.ptp)
    summary = dict(min_spearman_5_orders={k: float(v) for k, v in per.items()},
                   rankings_barely_move={k: bool(v > B1_STABLE) for k, v in per.items()},
                   pair_R_span_over_rho=dict(median=float(span.median()), max=float(span.max())),
                   kill_criterion_met=bool((per > B1_STABLE).all()))
    write_json(root, 'b1_summary', summary)
    return summary


# ---------------------------------------------------------------- B2

def _arms_metrics(Fa_align, Fb_align, seed):
    return pair_metrics(ViewStats(Fa_align, seed), ViewStats(Fb_align, seed))


def intervene(cfg, encs=None, with_heads=True):
    """B2: per factor and level, S (all measures) and R_EU, R_AU (+ logistic MI/width agreement for B7)."""
    root = cfg['root']
    th = load_thresholds(root)
    data = Data(root)
    encs = encs or [e for e in data.encoders() if e in NATURAL]
    rows = []
    for enc in encs:
        tr, te = data.view(enc, FINAL)
        d = tr.shape[1]
        for seed in SEEDS:
            sp = data.split(seed)
            rng = np.random.default_rng(1000 + seed)
            Xh, Q, Al = tr[sp['head']], te[sp['query']], tr[sp['align']]
            yh = data.ytr[sp['head']]
            ref_eu = eu(Xh, Q)
            ref_h = logistic_heads(Xh, yh, Q, seed) if with_heads else None
            ref_stats = ViewStats(Al, seed)

            def record(factor, level, eu_b, heads_b, metrics, deff_b=None):
                row = dict(encoder=enc, seed=seed, factor=factor, level=level, R_eu=rank_corr(ref_eu, eu_b),
                           deff_b=deff_b, **metrics)
                if heads_b is not None:
                    row.update(R_au=rank_corr(ref_h['au'], heads_b['au']),
                               R_mi=rank_corr(ref_h['mi'], heads_b['mi']),
                               R_width=rank_corr(ref_h['width'], heads_b['width']))
                rows.append(row)

            identical = pair_metrics(ref_stats, ref_stats)
            # Data factor: identical features, head data changes.
            for frac in FRACTIONS:
                sub = sp['head'][np.sort(rng.choice(N_HEAD, max(2, int(frac * N_HEAD)), replace=False))]
                record('data', f'frac_{frac}', eu(tr[sub], Q),
                       logistic_heads(tr[sub], data.ytr[sub], Q, seed) if with_heads else None, identical)
            disjoint = np.sort(sp['pool'][N_HEAD:2 * N_HEAD])
            record('data', 'disjoint', eu(tr[disjoint], Q),
                   logistic_heads(tr[disjoint], data.ytr[disjoint], Q, seed) if with_heads else None, identical)
            # Tail reshaping dial.
            ev = SP.covariance_eigs(Xh, center=False)[0]
            for s in TAIL_S:
                T = tail_transform(Xh, s)
                Xb, Qb = rescale(Xh @ T, Q @ T, Xh)
                evb = SP.covariance_eigs(Xb, center=False)[0]
                record('tail', f's_{s}', eu(Xb, Qb),
                       logistic_heads(Xb, yh, Qb, seed) if with_heads else None,
                       pair_metrics(ref_stats, ViewStats(Al @ T, seed)),
                       float(SP.d_eff(evb, RHO_REL * evb.mean()) / SP.d_eff(ev, RHO_REL * ev.mean())))
            # One non-orthogonal map and a pure rotation control (reference level = identity).
            q1, q2 = SP.random_orthogonal(d, 7 + seed), SP.random_orthogonal(d, 70 + seed)
            gl = (q1 * np.geomspace(np.sqrt(GL_KAPPA), 1 / np.sqrt(GL_KAPPA), d)) @ q2.T
            for factor, Tm in (('linear', gl), ('rotation', q1)):
                Tm = Tm.astype(np.float32)
                record(factor, 'identity', ref_eu, ref_h, identical)
                Xb, Qb = rescale(Xh @ Tm, Q @ Tm, Xh)
                record(factor, 'mapped', eu(Xb, Qb), logistic_heads(Xb, yh, Qb, seed) if with_heads else None,
                       pair_metrics(ref_stats, ViewStats(Al @ Tm, seed)))
            print(f'b2 {enc} seed {seed}', flush=True)
        data.drop()
    rows = write(root, 'b2_interventions', rows)
    return summarise_b2(root, rows, th)


def rescale(X, Q, X_ref):
    """Match the transformed arm's mean square to the reference arm (scalar only; centring is shared, so a
    level equal to the identity reproduces the reference exactly)."""
    c = float(np.sqrt(np.mean(np.asarray(X_ref, np.float64) ** 2) / np.mean(np.asarray(X, np.float64) ** 2)))
    return (X * c).astype(np.float32), (Q * c).astype(np.float32)


def summarise_b2(root, rows, th, targets=('R_eu', 'R_au', 'R_mi', 'R_width')):
    out = []
    levels = {'data': ['frac_1.0', 'frac_0.5', 'frac_0.25', 'frac_0.1', 'disjoint'],
              'tail': [f's_{s}' for s in TAIL_S], 'linear': ['identity', 'mapped'],
              'rotation': ['identity', 'mapped']}
    means = rows.groupby(['encoder', 'factor', 'level']).mean(numeric_only=True)
    for (enc, factor), _ in rows.groupby(['encoder', 'factor']):
        lv = levels[factor]
        curve = means.loc[(enc, factor)].reindex(lv)
        for target in targets:
            if target not in curve or curve[target].isna().all():
                continue
            for m in ALIGN_MEASURES:
                v = verdict(curve[m].values, curve[target].values, th['delta_S'][m], th['delta_R'])
                for level, vv, s_val, r_val in zip(lv, v, curve[m].values, curve[target].values):
                    out.append(dict(encoder=enc, factor=factor, level=level, target=target, measure=m,
                                    S=s_val, R=r_val, verdict=vv))
    out = write(root, 'b2_verdicts', out)
    tab = out[out.level.isin(['disjoint', 's_3.0', 'mapped'])].groupby(
        ['factor', 'target', 'measure']).verdict.value_counts().unstack(fill_value=0)
    write(root, 'b2_verdict_counts', tab.reset_index())
    return tab


# ---------------------------------------------------------------- B3

def natural_pairs(cfg, seed=0, n_perm=2000):
    """B3: all natural encoders x 3 depths; QAP permutation over encoder labels, additive encoder effects,
    within-family strata, norm-score baseline, pairwise blind-spot verdicts."""
    root = cfg['root']
    th = load_thresholds(root)
    data = Data(root)
    encs = [e for e in data.encoders() if e in NATURAL]
    sp = data.split(seed)
    views = [(e, l) for e in encs for l in LAYERS]
    stats, eus, norms = {}, {}, {}
    for v in views:
        tr, te = data.view(*v)
        stats[v] = ViewStats(tr[sp['align']], seed)
        eus[v] = eu(tr[sp['head']], te[sp['query']])
        norms[v] = norm_score(te[sp['query']])
        data.drop()
        print(f'b3 view {v}', flush=True)
    rows = []
    for a, b in combinations(views, 2):
        rows.append(dict(a=f'{a[0]}|{a[1]}', b=f'{b[0]}|{b[1]}', enc_a=a[0], enc_b=b[0], layer_a=a[1],
                         layer_b=b[1], same_encoder=a[0] == b[0],
                         same_family=FAMILIES[a[0]] == FAMILIES[b[0]],
                         R_eu=rank_corr(eus[a], eus[b]), R_norm=rank_corr(norms[a], norms[b]),
                         **pair_metrics(stats[a], stats[b])))
    pairs = write(root, 'b3_pairs', rows)
    np.savez(root / 'results' / 'b3_eu.npz', **{f'{k[0]}|{k[1]}': v for k, v in eus.items()})
    return summarise_b3(root, pairs, th, encs, n_perm, seed)


def _design(pairs, encs):
    cols = [np.ones(len(pairs))]
    for e in encs[1:]:
        cols.append((pairs.enc_a == e).astype(float) + (pairs.enc_b == e).astype(float))
    for l in LAYERS[1:]:
        cols.append((pairs.layer_a == l).astype(float) + (pairs.layer_b == l).astype(float))
    cols.append(pairs.same_encoder.astype(float))
    return np.column_stack(cols)


def _residual(y, Z):
    y = rankdata(y)
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def qap(pairs, x_col, y_col, encs, n_perm, seed, resid=False):
    """Mantel/QAP: permute encoder labels (depth kept) in the y matrix; two-sided p for Spearman/residual corr."""
    key = {(r.enc_a, r.layer_a, r.enc_b, r.layer_b): getattr(r, y_col) for r in pairs.itertuples()}
    key.update({(eb, lb, ea, la): v for (ea, la, eb, lb), v in list(key.items())})
    Z = _design(pairs, encs) if resid else None

    def stat(y):
        x = pairs[x_col].values
        if resid:
            ex, ey = _residual(x, Z), _residual(y, Z)
            return float(ex @ ey / np.sqrt((ex @ ex) * (ey @ ey)))
        return rank_corr(x, y)

    obs = stat(pairs[y_col].values)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        pi = dict(zip(encs, rng.permutation(encs)))
        y = np.array([key[(pi[r.enc_a], r.layer_a, pi[r.enc_b], r.layer_b)] for r in pairs.itertuples()])
        null.append(stat(y))
    null = np.asarray(null)
    return dict(obs=obs, p=float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)),
                null_mean=float(null.mean()), null_sd=float(null.std()))


def summarise_b3(root, pairs, th, encs, n_perm, seed):
    out = dict(n_encoders=len(encs), n_pairs=len(pairs), effective_n='number of encoders', measures={})
    for m in ALIGN_MEASURES + ('R_norm',):
        strata = {}
        for name, mask in (('cross_family', ~pairs.same_family), ('same_family_cross_encoder',
                           pairs.same_family & ~pairs.same_encoder), ('same_encoder', pairs.same_encoder)):
            sub = pairs[mask]
            strata[name] = dict(n=int(len(sub)), spearman=rank_corr(sub[m], sub.R_eu))
        out['measures'][m] = dict(
            spearman=rank_corr(pairs[m], pairs.R_eu),
            qap=qap(pairs, m, 'R_eu', encs, n_perm, seed),
            qap_residual_encoder_effects=qap(pairs, m, 'R_eu', encs, n_perm, seed, resid=True),
            strata=strata)
        if m != 'R_norm':
            Z = np.column_stack([np.ones(len(pairs)), rankdata(pairs.R_norm)])
            out['measures'][m]['partial_given_norm'] = float(
                (lambda a, b: a @ b / np.sqrt((a @ a) * (b @ b)))(_residual(pairs[m], Z), _residual(pairs.R_eu, Z)))
            out['measures'][m]['pairwise_verdicts'] = pairwise_verdicts(pairs[m], pairs.R_eu,
                                                                        th['delta_S'][m], th['delta_R'])
    write_json(root, 'b3_summary', out)
    return out


# ---------------------------------------------------------------- B4

def coverage(cfg):
    """B4: withhold 3 classes from head training; alignment on seen-only, seen+held-out, random sets;
    targets: EU agreement on held-out-class queries vs seen-class queries."""
    root = cfg['root']
    th = load_thresholds(root)
    data = Data(root)
    encs = [e for e in data.encoders() if e in NATURAL]
    rng = np.random.default_rng(4)
    triples = [tuple(sorted(rng.choice(10, 3, replace=False))) for _ in range(B4_TRIPLES)]
    rows = []
    for t_i, held in enumerate(triples):
        sp = data.split(t_i)
        held_tr, held_te = np.isin(data.ytr, held), np.isin(data.yte, held)
        head = np.sort(sp['pool'][~held_tr[sp['pool']]][:N_HEAD])
        r2 = np.random.default_rng(40 + t_i)
        q_seen = np.sort(r2.choice(np.flatnonzero(~held_te), 1000, replace=False))
        q_held = np.flatnonzero(held_te)
        al_seen_pool, al_held_pool = sp['align'][~held_tr[sp['align']]], sp['align'][held_tr[sp['align']]]
        sets = dict(seen=np.sort(r2.choice(al_seen_pool, B4_N_SET, replace=False)),
                    seen_heldout=np.sort(np.concatenate([r2.choice(al_seen_pool, B4_N_SET // 2, replace=False),
                                                         r2.choice(al_held_pool, B4_N_SET // 2, replace=False)])),
                    random=np.sort(r2.choice(sp['align'], B4_N_SET, replace=False)))
        stats, eu_s, eu_h = {}, {}, {}
        for enc in encs:
            tr, te = data.view(enc, FINAL)
            M = eu_operator(tr[head])
            eu_s[enc], eu_h[enc] = eu(None, te[q_seen], M=M), eu(None, te[q_held], M=M)
            stats[enc] = {k: ViewStats(tr[idx], t_i) for k, idx in sets.items()}
            data.drop()
        for a, b in combinations(encs, 2):
            base = dict(triple=str(held), a=a, b=b, R_seen=rank_corr(eu_s[a], eu_s[b]),
                        R_heldout=rank_corr(eu_h[a], eu_h[b]))
            for k in sets:
                rows.append(dict(base, alignment_set=k, **pair_metrics(stats[a][k], stats[b][k])))
        print(f'b4 triple {held}', flush=True)
    rows = write(root, 'b4_coverage', rows)
    summary = {}
    for (triple, k), f in rows.groupby(['triple', 'alignment_set']):
        for m in ALIGN_MEASURES:
            summary.setdefault(triple, {}).setdefault(m, {})[k] = dict(
                predict_heldout=rank_corr(f[m], f.R_heldout), predict_seen=rank_corr(f[m], f.R_seen))
    gate = {}
    for triple, per_m in summary.items():
        for m, per_set in per_m.items():
            vals = [per_set[k]['predict_heldout'] for k in per_set]
            gate.setdefault(m, []).append(float(np.nanmax(vals) - np.nanmin(vals)))
    one = rows[rows.alignment_set == 'random'].drop_duplicates(['triple', 'a', 'b'])
    region = [dict(triple=r.triple, a=r.a, b=r.b, verdict=verdict([0, 0], [r.R_seen, r.R_heldout],
                                                                   0, th['delta_R'])[1]) for r in one.itertuples()]
    region = pd.DataFrame(region)
    out = dict(triples=[list(map(int, t)) for t in triples], prediction=summary,
               heldout_predictability_change={m: dict(per_triple=v, mean=float(np.mean(v))) for m, v in gate.items()},
               gate_kept={m: bool(np.mean(v) >= B4_GATE) for m, v in gate.items()},
               region_verdicts=region.verdict.value_counts().to_dict(),
               mean_R_seen=float(one.R_seen.mean()), mean_R_heldout=float(one.R_heldout.mean()))
    write_json(root, 'b4_summary', out)
    return out


# ---------------------------------------------------------------- B5

def au_human(cfg, seed=0):
    """B5: CIFAR-10H on the full test split. Human entropy + split-half ceiling, logistic-probe AU,
    agreement vs alignment with function agreement controlled. Heads never see human labels."""
    root = cfg['root']
    load_thresholds(root)
    data = Data(root)
    if len(data.yte) != 10000 or not np.array_equal(data.te_index, np.arange(10000)):
        raise RuntimeError('B5 needs the full CIFAR-10 test split in original order')
    counts = D.load_cifar10h(test_labels=data.yte)
    h_ent = D.human_entropy(counts, 'plugin')
    ceil = D.split_half_ceiling(counts, n_rep=20)
    encs = [e for e in data.encoders() if e in NATURAL]
    sp = data.split(seed)
    heads, stats, geu = {}, {}, {}
    per_enc = []
    for enc in encs:
        tr, te = data.view(enc, FINAL)
        heads[enc] = logistic_heads(tr[sp['head']], data.ytr[sp['head']], te, seed)
        geu[enc] = eu(tr[sp['head']], te)
        stats[enc] = ViewStats(tr[sp['align']], seed)
        acc = float(np.mean(heads[enc]['mean_probs'].argmax(1) == data.yte))
        per_enc.append(dict(encoder=enc, accuracy=acc, au_vs_human=rank_corr(heads[enc]['au'], h_ent),
                            au_vs_human_ceiling_norm=rank_corr(heads[enc]['au'], h_ent) / ceil['ceiling'],
                            mi_vs_human=rank_corr(heads[enc]['mi'], h_ent)))
        data.drop()
        print(f'b5 {enc}', flush=True)
    write(root, 'b5_encoders', per_enc)
    np.savez(root / 'results' / 'b5_heads.npz', **{f'{e}|{k}': v for e in encs for k, v in heads[e].items()})
    rows = []
    for a, b in combinations(encs, 2):
        pa, pb = heads[a]['mean_probs'], heads[b]['mean_probs']
        rows.append(dict(a=a, b=b, R_au=rank_corr(heads[a]['au'], heads[b]['au']),
                         R_mi=rank_corr(heads[a]['mi'], heads[b]['mi']),
                         R_width=rank_corr(heads[a]['width'], heads[b]['width']),
                         R_eu_gauss=rank_corr(geu[a], geu[b]),
                         function_agreement=float(np.mean(pa.argmax(1) == pb.argmax(1))),
                         function_tv=float(1 - 0.5 * np.abs(pa - pb).sum(1).mean()),
                         **pair_metrics(stats[a], stats[b])))
    pairs = write(root, 'b5_pairs', rows)
    out = dict(human_ceiling=ceil, encoders=per_enc, contrast={})
    Z = np.column_stack([np.ones(len(pairs)), rankdata(pairs.function_tv)])
    pc = lambda x, y: float((lambda a, b: a @ b / np.sqrt((a @ a) * (b @ b)))(_residual(x, Z), _residual(y, Z)))
    for m in ALIGN_MEASURES:
        out['contrast'][m] = {t: dict(spearman=rank_corr(pairs[m], pairs[t]),
                                      partial_given_function=pc(pairs[m], pairs[t]))
                              for t in ('R_au', 'R_mi', 'R_eu_gauss')}
    write_json(root, 'b5_summary', out)
    return out


# ---------------------------------------------------------------- B6

def mechanism(cfg, seed=0):
    """B6: rank-share curves (CKA weight lambda^2 vs EU weight lambda/(lambda+rho)), class/residual split with the
    share_W gate, nuisance floors from B0 baselines."""
    root = cfg['root']
    th = load_thresholds(root)
    data = Data(root)
    encs = [e for e in data.encoders() if e in NATURAL]
    sp = data.split(seed)
    curves, splits, stats_B, stats_W, eus = [], [], {}, {}, {}
    for enc in encs:
        tr, te = data.view(enc, FINAL)
        X, Q = tr[sp['head']], te[sp['query']]
        ev = SP.covariance_eigs(X, center=False)[0]
        rho = RHO_REL * ev.mean()
        cka_w, eu_w = np.cumsum(ev ** 2) / np.sum(ev ** 2), np.cumsum(ev / (ev + rho)) / SP.d_eff(ev, rho)
        for frac in (0.5, 0.9, 0.99):
            curves.append(dict(encoder=enc, share=frac, d=len(ev),
                               k_cka=int(np.searchsorted(cka_w, frac) + 1), k_eu=int(np.searchsorted(eu_w, frac) + 1)))
        M = eu_operator(X)
        QB, QW = SP.class_residual_split(X, data.ytr[sp['head']], Q)
        eB = np.einsum('ij,jk,ik->i', QB, M, QB)
        eW = np.einsum('ij,jk,ik->i', QW, M, QW)
        share_W = float(np.var(eW) / (np.var(eB) + np.var(eW)))
        eus[enc] = eu(None, Q, M=M)
        splits.append(dict(encoder=enc, share_W=share_W, R_full_vs_W=rank_corr(eus[enc], eW),
                           R_full_vs_B=rank_corr(eus[enc], eB)))
        AB, AW = SP.class_residual_split(X, data.ytr[sp['head']], tr[sp['align']])
        stats_B[enc], stats_W[enc] = ViewStats(AB, seed), ViewStats(AW, seed)
        data.drop()
        print(f'b6 {enc}', flush=True)
    write(root, 'b6_rank_share', curves)
    splits = write(root, 'b6_share_W', splits)
    gate = bool(splits.share_W.median() >= B6_SHARE_W_GATE)
    out = dict(share_W_median=float(splits.share_W.median()), share_W_gate_passed=gate)
    if gate:
        rows = []
        for a, b in combinations(encs, 2):
            mB, mW = pair_metrics(stats_B[a], stats_B[b]), pair_metrics(stats_W[a], stats_W[b])
            rows.append(dict(a=a, b=b, R_eu=rank_corr(eus[a], eus[b]),
                             **{f'{m}_class': mB[m] for m in ALIGN_MEASURES},
                             **{f'{m}_resid': mW[m] for m in ALIGN_MEASURES}))
        rows = write(root, 'b6_class_residual_pairs', rows)
        out['class_vs_residual'] = {m: dict(class_predicts_R=rank_corr(rows[f'{m}_class'], rows.R_eu),
                                            residual_predicts_R=rank_corr(rows[f'{m}_resid'], rows.R_eu),
                                            mean_class=float(rows[f'{m}_class'].mean()),
                                            mean_resid=float(rows[f'{m}_resid'].mean()))
                                    for m in ALIGN_MEASURES}
    base = root / 'results' / 'b0_baselines.csv'
    if base.exists():
        b = pd.read_csv(base)
        out['nuisance_floors'] = {k: {c: float(f[c].max()) for c in ('R_eu',) + ALIGN_MEASURES if c in f and f[c].notna().any()}
                                  for k, f in b.groupby('baseline')}
    write_json(root, 'b6_summary', out)
    return out


# ---------------------------------------------------------------- B7

def head_check(cfg):
    """B7: bootstrap logistic EU (MI, width) vs Gaussian EU. B2 already stores R_mi/R_width per level;
    the B3 rerun uses the B5 heads (final layer, same seed-0 head draw)."""
    root = cfg['root']
    th = load_thresholds(root)
    res = root / 'results'
    out = {}
    b5 = res / 'b5_pairs.csv'
    if not b5.exists():
        raise RuntimeError('Run b5 first (B7 reuses its logistic heads)')
    p = pd.read_csv(b5)
    out['pair_order'] = {t: rank_corr(p.R_eu_gauss, p[t]) for t in ('R_mi', 'R_width')}
    out['pair_order_flag'] = {t: bool(v < B7_ORDER_FLAG) for t, v in out['pair_order'].items()}
    out['b3_final_layer'] = {t: {m: dict(spearman=rank_corr(p[m], p[t]),
                                         verdicts=pairwise_verdicts(p[m], p[t], th['delta_S'][m], th['delta_R']))
                                 for m in ALIGN_MEASURES} for t in ('R_eu_gauss', 'R_mi', 'R_width')}
    b2 = res / 'b2_verdicts.csv'
    if b2.exists():
        v = pd.read_csv(b2)
        v = v[v.level.isin(['disjoint', 's_3.0', 'mapped'])]
        out['b2_verdicts_by_target'] = {f'{k[0]}|{k[1]}|{k[2]}': n for k, n in
                                        v.groupby(['factor', 'target', 'verdict']).size().items()}
    write_json(root, 'b7_summary', out)
    return out


EXPERIMENTS = dict(b0=gates, b1=rho_dial, b2=intervene, b3=natural_pairs, b4=coverage, b5=au_human,
                   b6=mechanism, b7=head_check)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiments', nargs='+', choices=list(EXPERIMENTS) + ['all'])
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    todo = list(EXPERIMENTS) if 'all' in args.experiments else args.experiments
    cfg = dict(root=args.root)
    for name in todo:
        t0 = time.time()
        print(f'=== {name} ===', flush=True)
        EXPERIMENTS[name](cfg)
        print(f'=== {name} done in {time.time() - t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
