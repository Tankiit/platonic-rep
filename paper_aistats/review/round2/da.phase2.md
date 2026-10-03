contract_role: da
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: warn
trigger: "Secondary claims overreach the evidence"

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

**Calibration status.** `NOT_CALIBRATED`. Criteria binding was unavailable for this run, so no venue-alignment claim is made.

**What the paper does well.** The mathematical core is clean and checkable. The Isserlis-based identity (Theorem 2), the limits in Proposition 3 and the Corollary 4 construction are correct as stated under their Gaussian, linear-head assumptions, and the synthetic check (MAE 0.010) confirms them. The paper is unusually candid. It reports failed pre-registered predictions (E2-b, E4, Spec), labels post hoc operationalisations, and states that the S-versus-CKA difference interval includes zero. The abstract has been narrowed to "consistent with this but limited". Because of this hedging, my challenge is aimed at the framing claims that remain in the Introduction and Discussion, not at the hedged abstract.

**Strongest counter-argument.** The paper says CKA fails as an EU-agreement index because it "discards exactly the two pieces of information that matter": feature scale, and spectral weighting above the head's prior strength. A skeptic can argue that the paper's own real-data results show neither piece doing measurable work in the regime the experiments actually test.

Take scale first. The best-performing cross-encoder index is S_ρ→0, the mean squared canonical correlation. It is invariant to every invertible linear map, including scale, and it ties S_ρ exactly. Mutual k-NN shares CKA's invariance to cQ and still reaches 0.90 on closed-form agreement. The large scale effect appears only when features are rescaled at a frozen weight decay, and it disappears once weight decay is re-tuned.

Now spectral weighting. At the selected weight decays, d_eff/d > 0.98, so EU reduces to leverage. Under tail reshaping the closed-form EU barely moves. The weight-decay sweep where ρ should matter has intervals that touch zero, and some heads underfit there.

A more parsimonious account fits all of this. The indices that do well are whitened, linear-decodability-type measures (CCA, linear predictivity), and they track general similarity of linear readouts. They predict AU agreement as well as EU agreement (0.92). The measured softmax "EU" is almost a deterministic function of confidence and is nearly uncorrelated with the closed-form u(x) that Theorem 2 describes. On this reading, the identity is a correct theorem about a quantity the experiments do not isolate. The empirical section then restates Harvey et al.'s view (CKA and CCA as average alignment of linear decoders) and Ding et al.'s view (CKA is tail-insensitive), not anything specific to EU. The paper still has value: an exact Gaussian identity and a clear warning about scale and prior confounding. But the claim that it identifies "which information an index needs" holds only inside the model, not on the data.

**Further observations (not table items).** The level of Theorem 2 on real features is off by 0.34. The paper attributes this to fourth cumulants, but it gives no diagnostic, so "holds ordinally" is the only supported statement. The E2-a falsifier is admitted to have been unable to fire, so a verdict of "supported" in Table 1 overstates what was tested. "Uninformative" would be the accurate verdict. The novelty of the CKA-tail argument is explicitly a special case of Ding et al. (2021), and the bootstrap lemma is textbook. These are contribution-scope points for the EIC, not coherence defects.

#### CRITICAL
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|

#### MAJOR
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| M1 | D3 | Internal tension with the headline answer. The Introduction says CKA fails because it discards scale and tail weighting. But the cross-encoder winner, S_ρ→0 (CCA), is itself invariant to scale and to every invertible linear map, and it ties S_ρ on every column. Mutual k-NN, which shares CKA's cQ invariance, reaches 0.90 on closed-form agreement. In the real-data comparison, scale information contributes nothing measurable. Either narrow the claim to the Gaussian model or show a regime where scale-sensitive indices win. | table: Table 5, rows Sρ→0, Sρ (head's ρ) and mutual k-NN, closed-form and width (partial) columns | 4 (direct reading of reported table against stated thesis) | n/a (severity not norm-based) | n/a |
| M2 | D3 | Theory–measurement disconnect. Theorem 2 concerns the label-free posterior variance u(x). The cross-encoder test uses partial bootstrap width, which has rank correlation -0.999 with confidence and only -0.08 to 0.31 with u(x). The same indices predict AU agreement equally well (0.92). The rival explanation is general linear-readout similarity, not EU, so the ordering 0.83 vs 0.64 is not evidence for the paper's EU mechanism. The Discussion partly concedes this, but the abstract still offers the ordering as evidence "consistent with" the theory. | text: §4.2 "u(x), which ignores labels, correlates with it at only" | 4 (numbers reported in the paper itself) | n/a (severity not norm-based) | n/a |
| M3 | D3 | The scale finding is called "robust" and headlined in the abstract ("three orders of magnitude"), but it is produced by an artificial manipulation: rescaling features while freezing weight decay. It follows definitionally from Proposition 5 (rescaling equals a prior change), and it vanishes under the paper's own standard tuning procedure (max deviation 0.086). As a failure mode of CKA in use, it is overstated. The abstract should present it as a demonstration of an identity, not as an empirical finding of comparable weight. | text: §4.3 "it vanishes when the weight decay is re-tuned" | 4 (explicit in the paper's own E2b results) | n/a (severity not norm-based) | n/a |
| M4 | D3 | The second mechanism (EU counts every direction above ρ, unlike CKA) is not operative for the theory's own estimator at the selected weight decays. There d_eff/d > 0.98, so closed-form EU reduces to leverage and barely moves under tail reshaping. The softmax estimators do move, but the paper attributes that to likelihood curvature, which lies outside the theory. The one test where ρ should matter (the common-ρ sweep) gives intervals touching zero and has underfitting heads. Saying Theorem 2 "explains" the observations therefore overreaches. "Is compatible with" is supported. | text: §4.3 "The closed-form EU barely moves (mean ratio within" | 4 (paper's own reported regime and results) | n/a (severity not norm-based) | n/a |
| M5 | D3 | The claim that heads on identical features "disagree beyond estimator noise" (called robust in the Discussion) rests on dividing partial correlations by M = 50 re-test reliabilities. Those reliabilities are themselves strongly M-dependent (0.60 at M = 50, rising to 0.82 at M = 200), and attenuation correction of residualised rank correlations is not shown to be unbiased. Corrected AU degrades just as much (0.52 vs 0.48), so the result is that heads trained on different data differ in general. It is not EU-specific, and it is close to expected by construction. Report agreement at large M directly, or narrow the claim. | text: §4.2 "at M = 10 to 0.60 at M = 50 and 0.82 at M = 200" and "0.52 vs. 0.48 for EU" | 3 (inference about correction bias is reasoned, not demonstrated) | n/a (severity not norm-based) | n/a |

