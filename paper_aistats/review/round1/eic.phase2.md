contract_role: eic
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: warn
trigger: "Localised clarity problems such as undefined notation, uncaptioned or hard-to-read figures, redundant or misordered sections, or minor departures from venue conventions that need revision but leave the work assessable"

### D6: venue_fit_and_contribution
score: block
trigger: "The claimed contribution is substantially overstated relative to what is delivered, or novelty over closely related prior work is not established, but a narrower, honestly scoped contribution could be salvaged through revision"
block_class: repairable

## Review Body

Reviewer identity: senior area chair for statistical machine learning, working in Bayesian deep learning and uncertainty quantification (Card #1). I take a journal-fit view: originality, significance to a statistics-and-ML conference readership, whether the framing matches the evidence, and presentation. I do not judge proof-level correctness or estimator choices beyond what affects the paper's standing.

Overall view. The topic suits a statistics-and-ML conference, and the paper mixes theory and experiment in a way that readership expects. The pre-registration, and the decision to report failed predictions alongside successful ones, set a standard the field rarely meets. Two problems at the contribution level stop me from scoring D6 above block, and both can be fixed. First, the title and abstract make the claim specific to epistemic uncertainty, but the paper's own results show the non-identification is not EU-specific: AU agreement degrades just as much, and the proposed index predicts AU agreement as well as EU agreement. Second, the novelty of the ridge-regularised index is argued only against CKA. The paper does not compare it with existing prediction-based, ridge-regularised representation distances, and in the paper's own supplement the index does no better than linear predictivity or the canonical-correlation limit on the headline metric. A narrower and still useful paper can be salvaged: alignment metrics do not determine the uncertainty of regularised heads, plus an exact Gaussian identity explaining why. That is why the block is repairable rather than fatal. D5 is a warn: the paper can be assessed, but several numbers in the text disagree with the tables they describe, experiment labels are inconsistent, and a pre-submission note was left in the AI-use statement.

### S1: Pre-registration with falsified predictions reported in the main text
Evidence Anchor: table: Table 1 (pre-registered predictions and outcomes, rows E2b and E4 marked falsified)
The paper freezes its predictions and falsifiers before any real-data result. It shows two triggered falsifiers (E2b, E4) and a partly failed prediction in the main text rather than hiding them, and the Discussion narrows its claims to match. This is uncommon and valuable for a statistics venue, and it is the paper's strongest asset.

### S2: A closed-form mechanism that makes the non-identification argument concrete
Evidence Anchor: figure: Figure 1 (a: empirical EU correlation against S_rho and CKA over 60 synthetic pairs; b: Corollary 4 construction with closed-form lines)
Corollary 4 gives a short construction showing CKA is neither necessary nor sufficient for EU agreement. The synthetic check (mean absolute error 0.010 for S_rho against 0.37 for CKA) shows the identity works as a calculation tool and not only as a formal statement. The account of spectral weighting (CKA weights directions by squared eigenvalue, EU weights by w_i(rho) squared) is a clear insight that readers will take away.

### S3: The identical-features design separates the prior and data terms from the representation
Evidence Anchor: table: Table 2 (E2: heads on identical frozen features, raw and partial agreement against a re-test floor)
Comparing heads on literally the same features, against an explicit re-test floor and with confidence and human-entropy controls, is a clean design. It establishes empirically that perfect alignment leaves the residual uncertainty ranking undetermined.

### S4: Limitations stated candidly and in proportion
Evidence Anchor: text: §5 Limitations "on real features only" and "Planned analyses that were not run"
The Limitations paragraph says the theory holds only ordinally on real features, names the approximations behind the softmax estimators, and lists planned analyses that were not run. This makes the paper easier to calibrate for readers.

### S5: Scope fits a statistics-and-ML readership
Evidence Anchor: text: Abstract "For Bayesian linear heads"
The core object is the posterior variance of Bayesian linear heads on Gaussian features, treated through moment identities, effective dimension and bootstrap-versus-posterior relations. This sits well within a statistics-and-ML conference's interest in uncertainty quantification. (No venue-criteria claim is made: criteria binding is unavailable.)

### W1: The title and abstract frame the result as EU-specific, but the paper's own results show it is not
**Severity**: Major
**Evidence Anchor**: text: §4.4 "First, Sρ predicts AU agreement as well" and §4.2 "therefore cannot claim the pre-registered contrast that"
**Confidence**: 4 (the evidence is the paper's own reported outcomes, which I read directly)

The title ("Alignment Does Not Identify Epistemic Uncertainty") and the abstract's opening claim make epistemic uncertainty the subject. The evidence does not support that specificity. (a) Falsifier E2b triggered: on identical features, AU agreement degrades about as much as EU agreement (raw drop 0.10 vs 0.10; partial retention 53% vs 46%). (b) S_rho predicts AU agreement across encoder pairs as well as it predicts EU agreement (0.92), and the paper itself concludes that it "indexes shared head behaviour rather than EU specifically". (c) E4 triggered: between the most aligned encoders tested, the representation accounts for 93% of the Shapley share of EU disagreement. The Discussion says the EU-specific mechanisms "stand" in theory, but the empirical programme does not separate EU from head-level uncertainty in general. The authors handle this honestly in the body, but the title and headline still promise an EU-specific result that the body withdraws. Remedy: reframe around head-level (prior- and data-dependent) uncertainty, or add an experiment where EU and AU agreement actually diverge. As written, the framing overstates the contribution, and this is the main driver of my D6 block.

### W2: Novelty of S_rho over existing ridge-regularised, prediction-based representation comparisons is not established
**Severity**: Major
**Evidence Anchor**: absence: §2 Setup and Related Work, §3 paragraph after Definition 1, and References — expected positioning against prediction-based ridge-regression distances between representations that interpolate between CCA-like and CKA-like limits (e.g. GULP, Boix-Adserà et al., NeurIPS 2022) and against recent surveys of representational similarity measures; checked the full reference list, the related-work paragraph in §2, and the novelty sentence following Definition 1
**Confidence**: 3 (I am confident this literature exists and is close; whether it already contains the S_rho limits exactly is for the domain reviewer to confirm)

The paper places S_rho only against classical ridge-CCA (Vinod 1976; Bach and Jordan 2002) and says that "what is new here is its exact link to EU". A family of representation distances is defined through regularised linear-regression predictors with a ridge parameter and has known limits relating to CCA and CKA. That family is the closest prior work to S_rho as an index, and it is missing. If the index itself is not new, the contribution reduces to the EU link (Theorem 2), which makes W3 more consequential. The authors need to state exactly what S_rho adds over these measures.

### W3: The central identity follows from Isserlis' theorem in one step, so its significance rests on interpretation
**Severity**: Minor
**Evidence Anchor**: text: Appendix A, Proof of Theorem 2 "Isserlis’ theorem (Isserlis, 1918) gives"
**Confidence**: 4 (the proof is four lines and its structure is clear without line-by-line checking)

Theorem 2 is the covariance of two Gaussian quadratic forms, a textbook moment identity, specialised to M = (Sigma + rho I)^{-1}. This is not a flaw: simple identities with sharp interpretations are welcome at a statistics venue. But the paper calls it "an exact identity" as its first contribution without saying that it is elementary, and it holds only ordinally on real features (level off by 0.34 on average). The contribution statement should present the identity as an interpretive lens, not a technical advance, so that reviewers do not discount it for appearing to claim more.

### W4: In the paper's own supplement, the headline index comparison does not single out S_rho at the head's own rho
**Severity**: Major
**Evidence Anchor**: table: Table 5 (supplement), rows "linear predictivity", "Sρ→0" and "Sρ (head's ρ)", columns "width partial" and "width"
**Confidence**: 4 (read directly from Table 5 and the §4.4 text)

In §4.4 the text reports partial-width agreement of 0.83 for S_rho against 0.64 for CKA and 0.72 for mutual k-NN. Table 5 shows that linear predictivity also reaches 0.83 on that same column, a tie the text does not mention. On raw width, mutual k-NN (0.85) is essentially level with S_rho (0.87). The canonical-correlation limit S_rho→0 matches or exceeds S_rho at the head's own rho in every column. The paper also says that all heads sit in the small-rho regime (d_eff/d > 0.98), so evaluating at "the head's own rho" was never tested where it differs from mean squared canonical correlation, a quantity already known. Recommendation (i) in §5 ("S_rho at the head's own rho ... is the better-founded choice") is therefore not empirically supported over existing CCA-type indices. Remedy: report all indices in the main text, or test regimes (stronger priors, smaller n) where the head's rho matters.

### W5: The abstract's headline ordering result carries no uncertainty estimate and rests on 30 dependent pairs
**Severity**: Major
**Evidence Anchor**: text: §4.4 "EU agreement at Spearman 0.83, against 0.64 for CKA"
**Confidence**: 3 (a judgement about how the headline will be received; the methodology seat owns the statistical detail)

The across-encoder comparison is the only quantitative evidence in the abstract that S_rho beats CKA. It compares two Spearman coefficients over 30 pairs that are not independent: 10 encoder pairs at three depths, so each encoder appears in many pairs. No confidence interval, permutation test or pair-level bootstrap is reported for the difference. Checklist item 3(c) points only to re-test floors for seed variability, which do not address this between-index comparison. Without such an interval, a statistics-venue reader cannot tell whether 0.83 vs 0.64 is a real difference. If the difference is not significant, it should not be in the abstract.

### W6: Numbers in the §4.3 text contradict Table 3
**Severity**: Major
**Evidence Anchor**: text: §4.3 "Laplace MI rank agreement down to 0.72" and "bootstrap width down to 0.89"
**Confidence**: 5 (direct comparison of text and table values)

The §4.3 prose does not match Table 3, which it summarises. The text gives Laplace MI rank agreement down to 0.72, but the Table 3 minimum is 0.85. It gives bootstrap-width rank agreement down to 0.89, against a table minimum of 0.94. It gives mean bootstrap width between 0.38 and 1.84, against 0.56 to 1.52 in the table. For tail reshaping, the text says Laplace EU changes by up to a factor of 2.3 and width by up to 1.39, against 1.8 and 1.30 in the table, and that mutual k-NN drops to 0.62, against 0.68. The AI-use statement says that "every number in the paper is generated by script from logged results", so these discrepancies suggest that the text and the table come from different runs. This section supports the abstract's claim that alignment-invariant transformations move EU. The direction of the effect survives either set of numbers, but the authors must reconcile them and confirm which run is reported before the paper can be trusted on its other figures.

### W7: Experiment labels are inconsistent between Table 1, Table 3 and §4.3
**Severity**: Minor
**Evidence Anchor**: text: Table 3 caption "Table 3: E2b (DINOv2-B, ﬁxed weight decay): align-"
**Confidence**: 5 (labelling read directly)

In Table 1, E2b is the falsifier about AU degrading as much as EU, and the scale-transformation prediction is P5. Yet Table 3 and the §4.3 heading label the scale and tail transformations "E2b", and §5 cites "(theory, E2, E2b)" as support for non-identification. A reader matching claims to pre-registered outcomes, which is the paper's main selling point, will be confused. Use one consistent labelling scheme.

### W8: Table 1 and the text record different outcomes for the d_eff prediction
**Severity**: Minor
**Evidence Anchor**: text: §4.4 "this pre-registered prediction"
**Confidence**: 4 (direct comparison)

Table 1 records the "Spec" row (d_eff predicts mean EU) as "partly", while §4.4 says that "this pre-registered prediction failed" for bootstrap width (-0.19). The row label "Spec" is also undefined. Because the paper's credibility depends on accurate reporting of outcomes, Table 1 should state the pre-registered criterion and give an outcome that agrees with the text.

### W9: A pre-submission instruction about unverified proofs was left in the AI-use statement
**Severity**: Minor
**Evidence Anchor**: text: AI use statement "must be checked line by line by the"
**Confidence**: 4 (the sentence is verbatim in the manuscript; I did not verify the proofs line by line)

The AI-use statement says the proofs "must be checked line by line by the authors before submission", while checklist item 2(b) claims complete proofs. Reviewers would read this as an admission that the theory has not been verified by a human. A quick read of the appendix found nothing obviously wrong, but the authors must do the check and then remove or restate the sentence.

### W10: Space is used poorly: key evidence sits in the supplement while main-text space goes unused
**Severity**: Minor
**Evidence Anchor**: text: §4.4 "with item-level EU agreement (Figure 3, Table 5 in the"
**Confidence**: 3 (based on the layout visible in the extracted text; page counts depend on the compiled PDF)

The full index comparison (Table 5), which bears directly on the headline claim (see W4), and the per-encoder results are in the supplement. Meanwhile the main text appears to end well before the page limit. The figures are small and their axis labels sparse in the extracted layout. Moving Table 5 into the main text and enlarging Figures 2 and 3 would let readers check the central comparison without the supplement.

### W11: Empirical scope is narrow for the practical recommendations: one dataset, upsampled 32-pixel images, frozen linear probes
**Severity**: Minor
**Evidence Anchor**: text: §5 Limitations "We study frozen encoders and linear probes on one"
**Confidence**: 3 (judgement about readership expectations at a statistics-and-ML venue)

The theory does not depend on dataset breadth, but Recommendation (i) is a practical prescription resting on 30 encoder pairs from one dataset (CIFAR-10H, upsampled from 32 to 224 pixels). The authors disclose this. A second dataset with soft labels, or a second modality, would make the recommendations credible beyond this setting. The unused main-text space (W10) leaves room for it.
