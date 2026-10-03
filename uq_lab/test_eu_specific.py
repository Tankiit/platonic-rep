import numpy as np
from uq_eu_specific import posterior_variance, acquisition_overlap


def test_heteroscedastic_matches_dual_gp():
    rng=np.random.default_rng(10)
    x,q=rng.normal(size=(20,7)),rng.normal(size=(11,7))
    noise=np.geomspace(.1,2,20)
    prior=.7
    cross=prior*q@x.T
    expected=prior*np.sum(q*q,axis=1)-np.einsum('ij,ji->i',cross,
        np.linalg.solve(prior*x@x.T+np.diag(noise),cross.T))
    np.testing.assert_allclose(posterior_variance(x,q,noise,prior),expected,atol=1e-10)


def test_variance_contracts_and_rotation_invariance():
    rng=np.random.default_rng(1)
    x,q=rng.normal(size=(30,6)),rng.normal(size=(40,6))
    rotation,_=np.linalg.qr(rng.normal(size=(6,6)))
    small=posterior_variance(x[:10],q,.2)
    full=posterior_variance(x,q,.2)
    assert np.all(full<=small+1e-12)
    np.testing.assert_allclose(full,posterior_variance(x@rotation,q@rotation,.2),atol=1e-12)
    assert acquisition_overlap(full,full)==1


def test_rbf_single_observation_and_contraction():
    from uq_eu_specific import gp_variance
    x=np.array([[0.],[1.]])
    q=np.array([[0.],[.5],[10.]])
    got=gp_variance(x[:1],q,.2,prior=.7,lengthscale=1.)
    cross=.7*np.exp(-q[:,0]**2/2)
    np.testing.assert_allclose(got,.7-cross**2/(.7+.2),atol=1e-12)
    assert np.all(gp_variance(x,q,.2,prior=.7)<=got+1e-12)


def test_benchmark_smoke_and_constant_au():
    from uq_eu_specific import synthetic, evaluate
    views,tg,qg,coverage=synthetic(3,n=100,nq=40,d=6)
    for head in ('linear','rbf'):
        rows,checks=evaluate(views,np.zeros_like(tg),np.zeros_like(qg),coverage,seed=3,
                             priors=(1.,),fractions=(.25,1.),head=head)
        assert len(rows)==20
        assert all(c['monotonic_pass'] for c in checks)
        assert all(np.isnan(c['eu_au_rank']) for c in checks)
        assert all(0<=r['acquisition_overlap']<=1 for r in rows)


def test_wide_features_match_primal_covariance():
    rng=np.random.default_rng(11)
    x,q=rng.normal(size=(8,20)),rng.normal(size=(5,20))
    noise=np.geomspace(.1,2,8)
    cov=np.linalg.inv((x.T/noise)@x+np.eye(20)/.7)
    expected=np.einsum('ij,jk,ik->i',q,cov,q)
    np.testing.assert_allclose(posterior_variance(x,q,noise,.7),expected,atol=1e-10)
