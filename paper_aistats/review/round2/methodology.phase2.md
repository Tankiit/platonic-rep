contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: warn
trigger: "Supporting analyses lack intervals, seeds, or ablations"

The headline empirical conclusions are hedged appropriately and come with controls (re-test floors, partial agreement, encoder-cluster intervals). But the interval procedure used for the cross-encoder comparison is degenerate at five encoders, the reliability-corrected agreements behind a "robust" finding have no intervals, and the human-reliability ceiling is misapplied. All of this is repairable without a new design, so the score is warn, not block.

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: warn
trigger: "Isolated overstatements, unaddressed counterexamples, or loosely connected secondary claims appear"

The core thesis (CKA is neither necessary nor sufficient for EU agreement, and is blind to scale-as-prior) follows from correct proofs and from the identical-features experiments. Several secondary claims go beyond what was tested: matched-rho predictions are evaluated at a rho the paper itself says is not the softmax estimators' effective prior, a Lemma 7 "prediction" is close to an identity, and one Table 1 verdict is credited although its falsifier could not fire. Modest rewording repairs these.

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I reviewed this as a statistician working on Bayesian linear models and resampling. I re-derived every result in Appendix A and audited the estimators, the agreement measures and the interval procedures in Sections 4.2 to 4.4 and Appendices B to D.

The theory is correct as stated. I checked the Isserlis contraction and the variance normalisation in Theorem 2, both limits in Proposition 3 (using the degree-one homogeneity of S in each M), the constants in the Corollary 4 proof (for example, m at least 4k/epsilon gives corr at most epsilon/(1+epsilon)), the change of variables in Proposition 5, the identity 1 - CKA^2 = eta(s^2-1)^2/((1+eta)(1+s^4 eta)) in Proposition 6, the trace identity in Lemma 7, the sandwich-covariance weights w^4 in Lemma 9, and the resolvent factorisation in Proposition 10. The sample form of S_rho as a Frobenius cosine of ridge smoothers also checks out. Recomputing the Figure 1b setting (k = 9, m = 200, lambda_s = 10, lambda_p = 0.1, rho = 0.01) gives a closed-form CKA of 0.9978 and an EU correlation of 0.0515, consistent with the reported 0.998 and the empirical 0.06.

The empirical part is unusually candid. My concerns are about whether the uncertainty statements attached to it are valid, and whether the measured "EU agreement" is the quantity the theory describes.

### S1: Closed-form results are correct and assumptions are stated
**Evidence Anchor**: equation: Appendix A, Proof of Theorem 2 (Isserlis contraction giving Cov = 2 tr(P Sigma_AB Q Sigma_BA))

Every proof I re-derived holds. The paper also says where an extension fails, for example the constant-one resolvent bound for unobserved points, with a numerical counterexample. Proposition 3(ii) correctly notes that the rho to 0 limit is a mean squared canonical correlation only when d_A = d_B.

### S2: Pre-registered predictions reported with failures
**Evidence Anchor**: table: Table 1 (all eight rows, including E2-b, E4, S and Spec verdicts)

Four of the eight predictions are recorded as not supported or only partly supported. Thresholds operationalised post hoc are flagged (double dagger), and Appendix B says the hash certifies content, not time. This is a model of transparent reporting.

### S3: Estimator noise floor quantified and shown to be Monte Carlo
**Evidence Anchor**: figure: Figure 3a (re-test partial agreement versus ensemble size M, DINOv2 final layer)

The M sweep (0.45 at M = 10, 0.60 at M = 50, 0.82 at M = 200) correctly identifies the re-test floor as Monte Carlo limited rather than intrinsic. That is the right diagnostic before disattenuating.

### S4: Synthetic verification includes non-Gaussian latents and the rho dependence
**Evidence Anchor**: text: Section 4.1, "pirical EU correlation with mean absolute error 0.010" and "one third with Laplace instead of Gaussian latents)"

The synthetic check covers 60 pairs, four values of rho and some non-Gaussian latents, and it reproduces the predicted shrinkage of CKA's error as rho grows.

### S5: Covariate-estimator sensitivity analysis
**Evidence Anchor**: text: Section 4.2, "partial agreements by at most 0.001"

Swapping the plug-in human entropy for the Dirichlet-posterior estimator or for split-half entropies addresses the concern that a noisy covariate drives the partial agreement. The training-only S_rho variant (a change of at most 0.03) similarly addresses the train-plus-test design choice.

### W1: The encoder-cluster percentile bootstrap is degenerate with five clusters, so the reported intervals and the "0.09 of resamples" figure are not valid uncertainty statements
**Severity**: Major
**Evidence Anchor**: text: Appendix C, "keep every pair of distinct sampled encoders (with multiplicity," and "pairs; intervals are 2.5–97.5% percentiles."
**Confidence**: 5 — direct enumeration of the resampling distribution

When five encoders are resampled with replacement, the number of distinct encoders per resample is 1, 2, 3, 4 or 5 with probabilities 0.0016, 0.096, 0.48, 0.384 and 0.0384 (enumerated over all 5^5 draws). So about 9.8% of resamples contain at most one distinct encoder pair, which gives at most three distinct (pair, depth) points with duplicates, and the Spearman correlations on them take only a few lattice values. Another 48% contain only three distinct pairs (nine points). The near-universal upper limit of 1.00 in Table 5, the repeated +0.50 upper limits and the +0.00 lower limits for the differences are what this lattice produces, not features of the sampling distribution. The fraction of resamples at or below zero (0.09) is roughly the same as the share of resamples with at most two distinct encoders, which suggests ties in degenerate resamples drive it. The paper also does not say how resamples with no valid pair are handled. More generally, percentile cluster bootstraps with G = 5 clusters are known to under-cover badly, so reading 0.09 as a one-sided p-value ("would not survive a test at the 5% level") is not licensed. The paper already declines to claim significance, so the core survives. But the abstract quotes "(0.00 to 0.50)" as if it were a calibrated interval, and the weight-decay sweep intervals that "touch zero" have the same defect.

Requested remedy: report the leave-one-encoder-out jackknife and an exact enumeration over all 126 encoder multisets (or exclude resamples with fewer than three distinct encoders, and state that rule). Alternatively, use an exact permutation or randomisation test on the paired difference S_rho - CKA over the 10 encoder pairs. Present the intervals as descriptive, not inferential, wherever G = 5 is the unit.

### W2: The partial-agreement measure lives in a residual that holds well under 1% of rank variance, and the reliability-corrected values behind a "robust" claim have no intervals
**Severity**: Major
**Evidence Anchor**: text: Section 4.2 and Appendix B, "probes the bootstrap width has rank correlation -0.999" and "M = 50: 0.998, M = 100: 0.999. We use M = 50."
**Confidence**: 4 — standard variance-decomposition argument on reported values

With rank correlations between width and confidence of -0.997 to -0.999, residualising on confidence leaves 1 - r^2, about 0.2% to 0.6%, of the rank variance. The raw M-stability of 0.998 at M = 50 means Monte Carlo noise takes up a comparable share, which explains why the partial re-test reliability is only about 0.56 although raw stability is 0.998. Two consequences follow.

First, construct validity. The residual mostly reflects non-max class structure (summed quantile widths over the competing classes), not posterior variance in the paper's sense. The near-identical AU partial results (Table 3) are consistent with that reading. The paper concedes the empirical part "cannot separate EU-specific from general head agreement". Even so, the Discussion calls "even identical features leave the residual EU ranking dependent on the training data beyond estimator noise" robust.

Second, inference. The corrected agreements (0.48, 0.68) are ratio estimators, each built from one replicate pair per head, and they come with no interval over subset draws or items. The 0.26 to 0.66 range over encoders is not a confidence statement. The wd/10 and wd x10 rows of Table 2 (partial 0.24 and 0.49) get no reliability correction at all, although less-regularised heads plausibly have lower re-test reliability. And M = 50 is justified by raw stability while the analysis depends on partial reliability.

Requested remedy: run the identical-features analyses at M at least 200 for all encoders, bootstrap over items and subset draws to put intervals on corrected agreements, apply the correction to the weight-decay rows, and report the residual variance fraction alongside each partial agreement.

### W3: The matched-rho predictions are tested at a rho the paper itself says is not the softmax estimators' effective prior, and the closed-form EU barely tracks the estimators being predicted
**Severity**: Major
**Evidence Anchor**: text: Section 4.3 and Section 4.2, "prior strength is set by the curvature of the likelihood," and "u(x), which ignores labels, correlates with it at only"
**Confidence**: 4 — follows from the paper's stated estimator definitions

S_rho "at the heads' own rho" is evaluated at rho = wd (Appendix B). Section 4.3 states that the effective prior strength of the softmax (bootstrap, Laplace) estimators is set by likelihood curvature, which for confident heads with test accuracy 0.90 to 0.98 makes the effective rho much larger than wd. The closed-form u(x), the theory's object, correlates with bootstrap width at only -0.08 to 0.31. So the cross-encoder test with partial bootstrap agreement (Table 5) and the weight-decay sweep (Figure 3b, dashed) compare an index tuned to one estimator against agreement of another.

The closed-form column is close to a check of Theorem 2 on real features, because closed-form agreement is itself label-free and representation-determined. Its level error of 0.34 is attributed to fourth cumulants without a test, even though the synthetic runs with Laplace latents reached a mean absolute error of 0.010. The empirical fourth-moment form of Cov(a'Pa, b'Qb) can be computed directly from the cached features to confirm or reject that attribution.

Requested remedy: evaluate S_rho at a curvature-calibrated rho (for example, the Laplace GGN prior-to-curvature ratio), or reframe the bootstrap analyses as testing CCA-type versus CKA-type weighting only. Also test the cumulant explanation directly.

### W4: Fractions of the human-reliability ceiling divide by the reliability, not by its square root
**Severity**: Minor
**Evidence Anchor**: text: Appendix D, Table 6, "0.36 (0.52 of ceiling); test acc. 0.98"
**Confidence**: 5 — classical test theory attenuation bound

With full-length reliability r_xx = 0.70, the largest correlation any error-free model quantity can have with the observed human entropy is sqrt(0.70) = 0.837, not 0.70. The reported fractions (0.36/0.70 = 0.51, reported as 0.52, and the range 0.48 to 0.61) are therefore inflated by a factor of about 1.20. The correct range is about 0.41 to 0.51. Table 3's disattenuation correctly uses a square root, so the two analyses are inconsistent with each other.

### W5: The "95% CI" on the split-half ceiling reflects split randomness, not sampling uncertainty
**Severity**: Minor
**Evidence Anchor**: text: Section 4, Reliability and positive control, "(Spearman–Brown corrected, 95% CI [0.69, 0.71])"
**Confidence**: 4 — follows from the Appendix B procedure (50 random vote splits)

Percentiles over 50 random partitions of the same votes measure Monte Carlo variability of the split. They do not measure uncertainty over images or annotators. Either relabel the range, or bootstrap over images (and annotators) to obtain a sampling interval.

### W6: Headline index ordering depends on a post hoc choice among five agreement columns
**Severity**: Minor
**Evidence Anchor**: table: Table 7 (Appendix D), rows linear predictivity and mutual k-NN across width, width partial, MI, BLR, AU
**Confidence**: 4 — direct reading of reported values

The abstract's "CCA-type indices and linear predictivity ... at Spearman approximately 0.83 against 0.64 for CKA" uses the width-partial column, and the operationalisation of the S falsifier was fixed post hoc. In the other four columns, linear predictivity (0.41 to 0.56) is at or below CKA. Mutual k-NN, which shares CKA's scale invariance, reaches 0.85 to 0.90, close to S_rho. Report all columns in the main text, or name the pre-registered column, and treat the multiplicity across indices and columns explicitly.

### W7: The Lemma 7 "label-free prediction" for the closed form is close to an identity
**Severity**: Minor
**Evidence Anchor**: text: Section 4.4, Label-free prediction, "rank-correlates with mean closed-form EU at 0.90, as"
**Confidence**: 4 — algebra of the trace identity

The mean of u(x) over training points equals d_eff(rho) exactly. Over test points it equals tr((hat Sigma + rho I)^-1 Sigma_test). So a high rank correlation is guaranteed up to train/test shift and is not independent support for the theory. The informative half (bootstrap width, -0.19) failed. Table 1's "closed form only" verdict should say this. The paper should also explain why the value is 0.90 rather than about 1, for example whether d_eff uses the 10k subset.

### W8: Table 1 credits E2-a as "supported" although its falsifier could not fire
**Severity**: Minor
**Evidence Anchor**: text: Table 1 caption, "Uninformative in hindsight: the re-test partial agreement at M = 50 is itself ≈0.56,"
**Confidence**: 5 — stated by the paper

A prediction whose falsifier was unreachable is untested, not supported. The supporting evidence comes from the exploratory reliability-corrected redraws (Table 3), so the verdict should read "untested (exploratory support)". Similarly, E2b-tail's 6/6 counts six dependent settings (two encoders by three values of s) with no pre-specified magnitude threshold.

### W9: Re-tuned weight decays sit on both grid boundaries
**Severity**: Minor
**Evidence Anchor**: text: Section 4.3, "as c2 (from 10−5 at c = 0.1 to 10−1 at c = 10), which"
**Confidence**: 4 — the grid {1e-5, ..., 1e-1} is given in Appendix B

The selected values at c = 0.1 and c = 10 are the grid endpoints, so the optimum may lie outside the grid. The "scales as c^2" conclusion and the 0.086 maximum deviation depend on that truncation. Extend the grid by at least one decade on each side.

### W10: Baseline index definitions and the S_rho input set are underspecified
**Severity**: Minor
**Evidence Anchor**: absence: Section 4.4 and Appendix B — expected a definition of symmetric linear predictivity (regulariser, held-out protocol, symmetrisation) and of the input set and sample size used for mutual k-NN and S_rho; checked Section 2, Section 4.4, Table 5 caption, Appendix B, Appendix C
**Confidence**: 3 — supplementary code not inspected

Appendix C mentions a "10k training subset" for training-only S_rho, which implies the main S_rho is computed on 10k training plus 10k test inputs, not the full 40k. This should be stated where S_rho is defined. The parameters of the converse Corollary 4 construction (CKA 0.003, EU correlation 0.96) are not given.

### W11: Figure 1b caption approximation is loose at the stated parameters
**Severity**: Minor
**Evidence Anchor**: figure: Figure 1b caption (k = 9, lambda_s = 10, lambda_p = 0.1, rho = 0.01)
**Confidence**: 5 — direct recomputation

At these values w_p = 0.909, so k w_s^2/(k w_s^2 + m w_p^2) = 0.052 at m = 200, whereas k/(k + m) = 0.043. The approximation requires w_p close to 1. Drop it, or state the condition.

### W12: The E4 Shapley shares come from one encoder pair with no uncertainty and use a confidence-dominated outcome
**Severity**: Minor
**Evidence Anchor**: text: Section 4.4, Swap design, "the encoder accounts for 93%"
**Confidence**: 3 — design inferred from the text

The raw EU disagreement is, by W2's argument, mostly confidence disagreement. So a 93% encoder share may reflect accuracy differences between DINOv2 and supervised ViT. Report replicate-based intervals for the shares and give the partial-disagreement variant (61/30/10) equal prominence.

### W13: Pre-registration timing is self-certified
**Severity**: Minor
**Evidence Anchor**: text: Appendix B, Pre-registration, "so the hash certiﬁes content, not time"
**Confidence**: 3 — disclosed by the authors

The disclosure is honest. Even so, a remotely timestamped commit or a registry deposit would let readers verify that the predictions came before the data. This matters more here because some analyses were added after an internal review round.

### Questions for Authors

How are resamples with fewer than two distinct encoders handled in the cluster bootstrap, and how do the Table 5 intervals change if they are excluded? Do the corrected identical-features agreements stay below one when intervals over subset draws and items are added at M = 200? What curvature-calibrated rho does the Laplace GGN imply for each head, and does S_rho at that rho change the Table 5 ordering for partial bootstrap agreement?

## Arithmetic Receipts

no_recomputable_statistics: Checked all sections, tables and appendices; the manuscript reports Spearman, partial and stratified correlations, percentile bootstrap intervals, ratios and Shapley shares, but no t, z, F or chi-square statistic with df and p, no df-implied N, and no mean or SD of discrete-scale data with a stated N, so none of p_from_test_statistic, grim, grimmer or n_from_df applies (non-bounded checks of Corollary 4 values and ceiling fractions are reported in Review Body prose).
