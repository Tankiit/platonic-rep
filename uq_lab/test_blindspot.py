"""Unit checks for uq_blindspot.py (pytest -q test_blindspot.py)."""
import numpy as np
import pandas as pd

import uq_blindspot as B
import uq_eu_specific as E
import uq_heads as H
import uq_spectrum as SP

RNG = np.random.default_rng(0)


def _pair(n=600, d=20):
    Z = RNG.normal(size=(n, 8))
    a = Z @ RNG.normal(size=(8, d)) + 0.5 * RNG.normal(size=(n, d))
    b = Z @ RNG.normal(size=(8, d + 5)) + 0.5 * RNG.normal(size=(n, d + 5))
    return a, b


def test_cached_metrics_equal_reference():
    a, b = _pair()
    ref = E.metrics(a, b, 3)
    got = B.pair_metrics(B.ViewStats(a, 3), B.ViewStats(b, 3))
    for k in ref:
        assert abs(ref[k] - got[k]) < 1e-5, (k, ref[k], got[k])


def test_eu_matches_blr_var_rho():
    X, Q = RNG.normal(size=(300, 12)), RNG.normal(size=(50, 12))
    S = X.T @ X / len(X)
    rho = 1e-2 * np.trace(S) / 12
    assert np.allclose(B.eu(X, Q, 1e-2), H.blr_var_rho(X, Q, rho))


def test_eu_rotation_invariant_and_scale_free():
    X, Q = RNG.normal(size=(300, 12)), RNG.normal(size=(50, 12))
    R = SP.random_orthogonal(12, 1)
    assert np.allclose(B.eu(X, Q), B.eu(X @ R, Q @ R))
    assert np.allclose(B.eu(X, Q), B.eu(3 * X, 3 * Q) * 1.0)


def test_tail_identity_and_rescale_identity():
    X = RNG.normal(size=(200, 10)) * np.geomspace(5, .1, 10)
    T = B.tail_transform(X, 1.0)
    assert np.allclose(T, np.eye(10), atol=1e-5)
    Xb, Qb = B.rescale(X @ T, X[:5] @ T, X)
    assert np.allclose(Xb, X, atol=1e-4)


def test_verdict_definition():
    assert B.verdict([.9, .9, .5, .5], [.8, .3, .3, .8], .05, .1) == ['consistent', 'blind', 'consistent',
                                                                     'false alarm']
    v = B.pairwise_verdicts([0, 0, 1], [0, 1, 1], .5, .5)
    assert abs(v['blind'] + v['false_alarm'] + v['both'] + v['neither'] - 1) < 1e-12


def test_toy_known_answer():
    assert B.toy_check()['pass']


def test_qap_null_on_unrelated_matrices():
    encs = [f'e{i}' for i in range(6)]
    rows = []
    for i in range(6):
        for j in range(i + 1, 6):
            for la in B.LAYERS[:1]:
                rows.append(dict(enc_a=encs[i], enc_b=encs[j], layer_a=la, layer_b=la, same_encoder=False,
                                 x=RNG.normal(), y=RNG.normal()))
    out = B.qap(pd.DataFrame(rows), 'x', 'y', encs, 200, 0)
    assert 0 < out['p'] <= 1
