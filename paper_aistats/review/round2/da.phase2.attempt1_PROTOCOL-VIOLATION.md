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

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D3 argumentative_coherence | contract D3 and Phase 1 scoring plan | PARTLY_MEETS | table: Table 5 | The narrowed core thesis (high CKA does not imply EU agreement, with an exact mechanism for linear-Gaussian heads) follows from Theorem 2, Corollary 4, Proposition 5 and the identical-feature experiments. Several framing and secondary claims go further than that evidence. | Only five encoders; I did not re-derive the empirical numbers, only the proofs as printed | yes: it sets the D3 warn |

What the paper does well: the identities are correct as printed. I checked the Isserlis step in Theorem 2, both limits in Proposition 3, the scale-to-prior substitution in Proposition 5, the CKA bound in Proposition 6, the resolvent bound in Proposition 10, and the w^4 weighting that follows from Lemma 9. The manuscript also reports its own failed predictions and post hoc operationalisations openly. Nothing in this round depends on a broken proof, so I find no CRITICAL defect. Every finding below concerns the gap between what the theory licenses and what the framing and the experiments claim.

### Strongest Counter-Argument

A sceptic would grant that the theorem is correct and then deny that it carries the paper. Theorem 2 is a single Isserlis computation about the correlation of two Gaussian quadratic forms. The quantity it covers, the closed-form u(x), ignores labels and is a ridge leverage score, so it is a property of the representation by construction. That a ridge-CCA index predicts the correlation of two ridge leverages is close to definitional. The question the title asks is about heads that people actually use, and here the paper's own evidence separates the two objects. The closed form correlates with bootstrap width at only -0.08 to 0.31. Bootstrap width correlates with max-probability at -0.999. Once confidence is partialled out, the residual "partial EU agreement" has a re-test reliability of 0.56 at M = 50. All the headline cross-encoder numbers rest on that residual. On it, Sρ ties with linear predictivity and with its own ρ → 0 limit, every index predicts AU agreement as well as it predicts EU agreement (0.92), and the interval for Sρ minus CKA reaches zero.

A simpler account fits these data. Whitened indices track how similar two heads' functions are, so their heads share confusion structure. CKA tracks that structure less well because it is dominated by the top of the spectrum. Nothing EU-specific is needed. Two further observations point the same way. The paper's own tail intervention left the closed form unchanged, so the softmax effects are explained after the fact by likelihood curvature, which the theory does not model. And the scale "reason" applies equally to CCA, linear predictivity and mutual k-NN, which are the alternatives the paper recommends. What survives is a correct, modest theorem plus the point that scale-invariant indices cannot see a mismatch between prior and scale. That is a narrower contribution than "CKA discards exactly the two pieces of information that matter."

### Additional notes (Minor, prose only)

- Table 1 marks E2-a as "supported" even though its own footnote says the falsifier could not have fired. That verdict should read "uninformative". Falsifier S and falsifier E2-b were also operationalised post hoc. The pre-registration hash certifies content, not timing. Contribution 3 should therefore present the pre-registration as weaker evidence than it currently does.
- Recommendation (i), "Do not use CKA to transfer or validate EU", is categorical. The empirical evidence is weaker than that. CKA's rank correlation with agreement is 0.64 to 0.76, the Sρ minus CKA interval touches zero, and E4 shows that the encoder accounts for 93% of disagreement on the highest-CKA pair. Corollary 4 justifies "CKA is not sufficient". It does not justify "do not use".
- No real pair of different encoders reaches high CKA. The maximum is 0.837, so the abstract's "high CKA" claim on real data rests only on identical features and constructed transformations.
- The scope is in-distribution CIFAR-10H only. The introduction motivates the question with shared blind spots, and those matter most out of distribution, which the experiments do not test.

#### CRITICAL
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|

#### MAJOR
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| M1 | D3 logic chain | The thesis is framed as two exact reasons why CKA fails. One of them, blindness to feature scale (Prop. 5), applies equally to the indices the paper recommends: CCA-type indices, the ρ → 0 limit, linear predictivity and cosine mutual k-NN. All of these are invariant to c times Q, and the leverage is invariant to every invertible linear map, as Section 4.3 itself states. Scale is therefore an argument against any representation-only index, not against CKA specifically. Only the spectral-weighting reason separates CKA from the alternatives. The framing and the recommendations need restructuring around that single reason. | text: §1 "Our answer is that CKA discards exactly the two" | 5: direct reading of Props. 3, 5 and §4.3 | not norm-based | not norm-based |
| M2 | D3 theory-to-evidence gap | The theory covers the label-free closed-form posterior variance. The headline empirical comparison (0.83 vs 0.64) uses partial bootstrap width of softmax heads, which is nearly unrelated to the closed form on these features. The empirical ordering of indices therefore does not test Theorem 2. It tests a different quantity that the theory does not cover. The closed-form column in Table 5 is the only direct test, and there the Sρ minus CKA interval is -0.15 to +0.50. | text: §4.2 "which ignores labels, correlates with it at only" | 4: numbers internal to the paper; the link from the softmax summaries to u(x) is not modelled | not norm-based | not norm-based |
| M3 | D3 rival explanation | Partial agreement residualises ranks on confidence, but EU width is almost a monotone function of confidence. Its identity as an EU signal is therefore unestablished: the residual is dominated by Monte Carlo noise at M = 50 (re-test 0.56, rising to 0.82 at M = 200) and probably by multi-class confusion structure. The cross-encoder index ranking and the claim that identical features disagree "beyond estimator noise" both rest on this residual. The reliability correction assumes errors are independent across heads, and that assumption is not checked. The paper concedes that EU and AU cannot be separated, but it still calls the identical-feature result robust. | text: §4.2 "probes the bootstrap width has rank correlation -0.999" | 4: inference from the reported collinearity and the M-dependence of reliability | not norm-based | not norm-based |
| M4 | D3 mechanism attribution | The tail-reshaping intervention was meant to show on real features that EU counts low-variance directions. It left unchanged the only estimator the theory covers. The softmax estimators did move, and that movement is attributed post hoc to likelihood curvature, which the theory does not include. The Lemma 9 check (6 of 6) extrapolates a residual-resampling result to Poisson-bootstrapped softmax heads. The real-data support for the tail mechanism is therefore indirect and explained after the fact. | text: §4.3 "The closed-form EU barely moves (mean ratio within" | 4: the paper's own Table 4 and the prose in §4.3 | not norm-based | not norm-based |
| M5 | D3 core novelty claim | The paper says its addition to the ridge-CCA family is that the head's own prior strength is the right regularisation level. At the validated weight decays, Sρ is indistinguishable from its ρ → 0 limit and from linear predictivity. In the common-weight-decay sweep, the matched-ρ advantage appears only at ρ values where some heads underfit (accuracy 0.50 to 0.97), and the intervals touch zero. The distinctive content of the index therefore has no empirical support at realistic regularisation. | text: §4.4 "we cannot attribute any advantage to the head’s prior" | 4: Table 5, Fig. 3b and the §4.4 text | not norm-based | not norm-based |

