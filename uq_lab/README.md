# uq_lab: experiments for "Alignment Does Not Identify Epistemic Uncertainty"

Implemented experiment suite with logged submission and revision results. Some optional experiments remain unimplemented. Conventions: plain imports,
Mac-first (MPS; float64 CPU for linear algebra), no multiprocessing, established libraries (torch, timm, open_clip,
scikit-learn, scipy). Only `factory.py` of your existing code imports `tinker`; the text/Tinker arm is banked, not in this package.

## EU-specific follow-up

See [EU_SPECIFIC.md](EU_SPECIFIC.md) for exact linear/RBF GP experiments that separate latent posterior variance from prescribed observation noise, compare six alignment indices, and test class-withholding and acquisition-set agreement. Use `prepare_eu_features.py` to extract caches and `run_eu_image_suite.py` for all ten withheld classes. These analyses are exploratory and do not modify the frozen original PREREG.

## Blind-spot follow-up

See [BLINDSPOT_PLAN.md](BLINDSPOT_PLAN.md) for the B0–B7 plan built on one operational blind-spot definition, and `uq_blindspot.py` for its skeleton. Exploratory; does not modify the frozen PREREG.

## Original design run order (historical)
1. `pytest -k ref -q`  -> the 9 reference tests pass with no code of yours; they check the theory (BLR dual=primal, scale = prior change,
   resolvent bound, toy CKA-vs-EU, tail reshaping, bootstrap-vs-posterior shrinkage, d_eff identity, S_rho primal trick).
2. Fill `uq_data.py` + `uq_features.py`; run `benchmark_throughput` for each encoder; fix the encoder list (verify timm/open_clip names).
3. `python uq_run.py e0_gate` -> reliability ceilings + Synthetic positive control. STOP if it fails.
4. Fill `uq_heads.py`, `uq_spectrum.py`, `uq_align.py`, `uq_stats.py` (the `test_impl_*` tests tell you when each is right).
5. `e2_constructed_pairs`, `e2b_invariant_transforms`, then `e4_swap_design`, `e_spec`, `h8_class_residual` (do its share_W gate first).

## Files
| file | role |
|---|---|
| uq_config.py | encoders, grids, thresholds, PREREG predictions/falsifiers (fill the TODO numbers before seeing results) |
| uq_data.py | CIFAR-10(H), DCIC loaders, human entropy, split-half ceiling |
| uq_features.py | per-layer extraction, resumable fp16 cache |
| uq_spectrum.py | eigenspectrum, d_eff(rho), power-law fit, transforms, class/residual split, NC1 |
| uq_heads.py | Poisson-bootstrap heads, BLR primal/dual, Laplace, EU/AU summaries, reducibility, regret |
| uq_align.py | CKA, linear predictivity, per-item mKNN, S_rho, permutation nulls |
| uq_stats.py | Spearman / partial / stratified agreement, bootstrap CIs, incremental AUROC, Shapley |
| uq_experiments.py | E0, E1, E2, E2b, E3, E4, E6, E_spec, E_rho_dial, AU-function control, S_rho vs CKA, H8 |
| uq_run.py | CLI |
| test_uq_theory.py | reference tests (run now) + implementation tests (skip until implemented) |

## Original experiment conventions
- Human soft labels are for evaluation only, never to fit or tune anything that yields EU.
- EU width = quantile interval, not max-min. Report the M-stability curve.
- Calibrate alignment with permutation nulls; per-item metrics need per-item nulls; calibrate the MAX if you select layers.
- Compare layers by relative depth. Decide pooling once.
- Report the reliability ceiling next to every AU number.
- Do not reuse the old S_optim, the 30-sample feature files, or the participation-ratio "effective rank".

## Verify before relying
timm/open_clip identifiers, licences (CIFAR-10H CC BY-NC-SA 4.0; DCIC per-dataset), CIFAR-10H file names and row order,
whether the AISTATS full paper deadline (6 Oct 2026) still holds.
