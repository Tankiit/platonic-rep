# Blind-spot follow-up plan (exploratory; B0–B7)

The frozen original PREREG (`PREREG_FROZEN.txt`) is unchanged. This plan organises the remaining
experiments around one operational definition, so each experiment returns a yes/no on it.
Skeleton: `uq_blindspot.py`.

## Operational definition

For a factor f (a transformation, a data condition, or a region of input space) with levels, compute
alignment S(level) and EU agreement R(level).

- **Blind spot**: S stays flat (|ΔS| ≤ δ_S) while R moves by more than its noise (|ΔR| > δ_R).
- **False alarm**: S moves (|ΔS| > δ_S) while R stays flat (|ΔR| ≤ δ_R).
- **Consistent**: both move, or neither moves.

δ_S and δ_R are pre-registered before running. The noise ceiling for δ_R is the seed-to-seed EU
agreement of the same encoder.

**Pre-registered threshold rule** (committed 2026-10-04, before any real-data result; default chosen
because open question 2 was unanswered, so the user may override it before B1 runs). B0 computes the values and freezes them
to `uq_runs_blindspot/thresholds.json` (with SHA-256). B1–B7 refuse to run without that file.

- **δ_R** = 2 × the largest per-encoder SD of R across 5 replicates (final layer). Each replicate is
  the EU agreement between two independent equal-size head-data draws of the same encoder, with floor 0.02.
  This is the noise in an R estimate. The level of the seed-to-seed ceiling R_self is reported separately.
  1 − R_self is *not* used as δ_R, because the B2 disjoint-subset level would then sit on the threshold by construction.
- **δ_S** (per measure) = 2 × the largest per-pair SD of S across 5 independent alignment-set draws
  (all natural pairs, final layer, n_align fixed), with floor 0.005.
- EU is the label-free Gaussian EU x^T(Σ̂ + ρI)^{-1}x on centred, scalar-normalised features, with
  ρ = ρ_rel · mean eigenvalue, ρ_rel = 1e-3 (≈ prior 1 at n = 1024 in the EU_SPECIFIC pilot). B1 reports how
  much the ranking depends on ρ_rel. R is the Spearman correlation of EU over the same query items.
- A verdict is reported per alignment measure (CKA, mKNN, predictivity, cca_type).

| threshold | value | source | committed at |
|---|---|---|---|
| δ_S | from B0 | `thresholds.json` | written by B0 |
| δ_R | from B0 | `thresholds.json` | written by B0 |

## Experiments, in run order

| ID | What | Answers | Gate / kill criterion |
|---|---|---|---|
| **B0** validity | Toy known-answer through the runner (expect CKA ≈ 0.998, EU rank correlation ≈ 0.03). Self-reshape control. n_align ≥ 4d (3,072+ for d = 768), with asserts that `cca_type` < 1 and predictivity R² > 0. EU self-agreement ceiling across seeds. Trivial baselines: random-init network, colour histogram, norm-only score. | Is the pipeline measuring what it claims? | Stop if the toy or reshape control fails |
| **B1** ρ-dial | Label-free: Spearman of EU rankings across ρ from 1e-6 to 1e2 (relative to the mean eigenvalue) for each encoder, plus d_eff(ρ) and the Mahalanobis and norm rankings as endpoints. | Does EU ranking depend on regularisation at all? The pilot's priors span only two orders and shifted agreement by 0.04–0.12 | If rankings barely move over 5+ orders, EU ≈ one fixed leverage score, and the prior-dependence argument has little bite in practice |
| **B2** interventions | On real features: same features with heads on disjoint data subsets and fractions; tail-reshape dial s ∈ {1, 1.5, 2, 3}; one non-orthogonal map; a pure rotation as a control. Report all four alignment measures and EU/AU agreement. | Can alignment stay flat while EU agreement drops? Direct blind-spot evidence | Weakened if EU agreement stays high under tail reshaping. Check d_eff and ρ regime before concluding |
| **B3** natural pairs | 8+ encoders × 3 relative depths. Permutation tests over encoder labels (Mantel/QAP-style), additive encoder effects, within-family strata. Compare to the norm-score agreement baseline. | Do naturally occurring pairs show the effect, and does it survive removing encoder identity? | Effective sample size is the number of encoders. Pilot residual correlation was 0.58 on 5 df |
| **B4** coverage | Withhold 3 classes (300+ held-out queries). Alignment on seen-only, seen + held-out unlabeled, and random subsets, against held-out-only EU agreement. | The sparse-region blind spot, and Prop C (alignment must be measured on the right set) | Dropped if held-out predictability doesn't change with the alignment set |
| **B5** AU | CIFAR-10H test images (needs full test caches, not the 1,000-image pilot). Human entropy, split-half ceiling, model AU from logistic probes. Agreement vs alignment, with function agreement controlled. | The EU-vs-AU contrast the abstract promises | If it can't run, restate the AU claim as theory only |
| **B6** mechanism | Rank-share curves (CKA vs EU), class/residual split with the `share_W` gate, nuisance floors. | Why the blind spot appears | Drop the class-residual story if `share_W` is low |
| **B7** head check | Bootstrap logistic EU (width, MI) vs the Gaussian regression EU; rerun B2/B3 with it. | Is the result specific to a label-free leverage-type EU? | Flag if the ordering changes |

**Priority:** B0, B1, B2 are necessary. B4 and B6 are cheap on cached features. B3 needs new
encoders; B5 and B7 only if time remains. The paper states which ones ran.

## Outcome → wording (fix before seeing results)

| Outcome | Claim |
|---|---|
| B2 shows blind spots; B3/B4 show tracking where data are dense but not in sparse regions | "Alignment predicts EU where data are dense and fails where decisions are made" (strongest) |
| B2 shows blind spots; natural pairs sit near the identifiable case | "Blind spots are constructible, not typical": diagnostic + theory framing |
| B2 fails to produce a blind spot | Thesis weakened. Check the regime (d_eff, ρ, n/d) before concluding |

The registered title is stronger than the middle outcome, so the outcome → title mapping must be
decided before results are seen.

## Open questions

1. How many new encoders can be extracted before Sunday? B3 needs at least eight for any pair-level statement.
2. Set δ_S and δ_R from the seed-to-seed ceiling, or use values chosen in advance?

## Implementation and fixed decision constants (committed 2026-10-04, before any real-data result)

Data: `prepare_blindspot_features.py` → `uq_runs_blindspot/`. It uses 11 encoders: 10 natural ones (DINOv2-B, CLIP-B/16,
SigLIP-B/16, ViT-B/16 sup, DeiT-B, Swin-B, ConvNeXt-S, ResNet-50, Mixer-B/16, MAE-B/16) plus a random-init ViT-B/16
baseline, each tapped at 3 relative depths. There are 20,000 train images (seed 2026) and the full 10,000-image test split.
Per seed, the train subset is split into an alignment set (8,192 = 4 × the widest d) and a disjoint head pool
(head draw 4,096). Queries are 2,000 test images (all 10,000 in B5). Families for the strata are in `uq_config.FAMILIES`.

| constant | value |
|---|---|
| ρ_rel (EU) | 1e-3 × mean eigenvalue; B1 grid 1e-6 … 1e2 |
| B1 "rankings barely move" | every ρ pair ≥ 5 orders apart has Spearman > 0.95 |
| B2 levels | fractions 1, .5, .25, .1 and a disjoint equal-size draw; tail s ∈ {1, 1.5, 2, 3} beyond 95% energy; GL map with κ = 10; random rotation |
| B2 verdict levels | disjoint, s = 3, mapped (vs the first level of each factor), on seed-mean curves (5 seeds) |
| B3 | 10 encoders × 3 depths = 30 views, 435 pairs; QAP permuting encoder labels (2,000 perms); additive encoder + depth + same-encoder effects |
| B4 | 3 random class triples (seed 4); alignment sets of 4,096: seen, ½ seen + ½ held-out, random; gate: held-out predictability changes by ≥ 0.1 |
| B5 | logistic bootstrap heads (M = 20, wd = 1e-3, hard train labels only), plug-in human entropy, 20-rep split-half ceiling; function agreement = 1 − mean TV distance |
| B6 | share_W = Var(x_Wᵀ M x_W) / (Var(x_Bᵀ M x_B) + Var(x_Wᵀ M x_W)); gate: median ≥ 0.5 |
| B7 | flag if Spearman(Gaussian R, logistic R) across pairs < 0.5 |

Run: `python uq_blindspot.py all` (outputs in `uq_runs_blindspot/results/`). Tests: `pytest -q test_blindspot.py`.

## Status

| ID | Status |
|---|---|
| B0–B7 | real-data run complete 2026-10-04 (exit 0); outputs in `uq_runs_blindspot/results/`; thresholds frozen in `uq_runs_blindspot/thresholds.json` |
