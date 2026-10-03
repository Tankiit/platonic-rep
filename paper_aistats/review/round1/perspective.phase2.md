contract_role: perspective
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: warn
trigger: "claimed implications for neighbouring fields are plausible yet only partially argued"

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I read this manuscript as a computational cognitive scientist who works on human label uncertainty and human-machine disagreement. My question is whether its framing, its use of the aleatoric/epistemic (AU/EU) split, and its practical recommendations make sense to readers outside the uncertainty-quantification community: people who use human soft labels, people who choose encoders for active learning, and people who use RSA-style alignment to compare models with brains. The paper is unusually candid. It reports a reliability ceiling for every human-entropy number and lists its falsified predictions next to the confirmed ones. Its central cross-field message, that high CKA does not mean shared blind spots, is well supported by the theory and by the identical-features experiment.

My concerns are about how far the empirical AU/EU story and the practitioner recommendations reach. First, the "partial" agreement metric controls for human entropy, a covariate with a reliability ceiling of 0.70. Psychometric readers will know that this control is incomplete. Second, the softmax EU and AU estimators look nearly collinear, so it is unclear how much of the empirical work is specifically about epistemic uncertainty. Third, the use cases the introduction gives as motivation (active learning, uncertainty transfer, shared blind spots) are never tested at the level of a decision. None of these overturns the non-identification result. Together they leave adjacent-field readers with gaps to fill on their own. That matches my Phase 1 warn condition and falls short of block, because the central cross-field claim is argued and supported, not just asserted.

### S1: Reliability ceilings make the human-label numbers interpretable
- **Evidence Anchor**: text: §4 Reliability and positive control (E0) "every AU number below should be read against it."
- Reporting a Spearman–Brown-corrected split-half ceiling for CIFAR-10H entropy (0.70, with a CI), and quoting model–human AU agreement as a fraction of that ceiling, is the standard of practice in the human-uncertainty literature. It is rarely seen in ML uncertainty papers. Readers from cognitive science can calibrate every AU number directly.

### S2: Falsified predictions are reported openly
- **Evidence Anchor**: table: Table 1 (pre-registered predictions and outcomes, including falsified E2b and E4)
- A pre-registration table that lists falsifiers, thresholds and outcomes, with "falsified" entries left visible, makes the claims auditable for readers who did not build the method. It also models good practice for neighbouring empirical communities.

### S3: The identical-features design is an accessible bridge for outside readers
- **Evidence Anchor**: text: §4.2 "Within each encoder the paired heads see literally the same features, so every alignment metric equals one."
- This design gives a non-specialist an argument they can check without the spectral theory: if alignment is at its maximum and EU still disagrees, alignment cannot be what determines EU. Corollary 8 and Table 2 make this concrete, and that will carry over well to the representational-convergence (Platonic hypothesis) audience.

### S4: Limitations are stated where an outside reader needs them
- **Evidence Anchor**: text: §5 Limitations "on real features only its ordinal predictions held."
- The paper says plainly that the exact identity holds only for Gaussian features. It also says the softmax estimators are approximations and that the evidence comes from one dataset with frozen encoders. This keeps adjacent readers from over-generalising the theory.

### S5: The partial-agreement recommendation is backed by evidence
- **Evidence Anchor**: text: §5 Recommendations "since raw EU agreement mostly reﬂects shared conﬁdence."
- Recommendation (iii) follows directly from Table 2 and Figure 2. Raw agreement is about 0.9 or higher for every pair, while partial agreement separates the conditions. Anyone who correlates model uncertainty with human ambiguity can act on this immediately.

### W1: Partialling out an unreliable human-entropy covariate does not "rule out" the ambiguity explanation
- **Severity**: Major
- **Evidence Anchor**: text: §2 EU agreement "Partial and stratiﬁed variants rule out the trivial explanation that both models are uncertain on intrinsically ambiguous images."
- **Confidence**: 4 (psychometrics of crowd-sourced soft labels; measurement error in covariates)
- The human-entropy covariate has a split-half reliability ceiling of 0.70 (§4 E0). The partial Spearman residualises on single-sample plug-in entropy (Appendix B, Statistics). When a covariate is measured with error, partialling it out removes only part of the construct it stands for, so residual "EU" agreement can still carry shared intrinsic ambiguity. Readers who know the attenuation and residual-confounding literature will see the word "rule out" as an overstatement. The comparisons within a single encoder (re-test 0.56 vs. 10%/100% 0.26) are probably robust, because every condition shares the same covariate. But the absolute partial levels, and the 0.10 figure across encoders, cannot be read as ambiguity-free. A remedy is to (a) soften the claim to "reduce", and (b) add a sensitivity analysis. Options include disattenuating the covariate, using the Dirichlet-posterior entropy already computed, or controlling for both split-half entropies, then reporting how the partial agreements move.

### W2: The softmax EU and AU estimators look nearly collinear, which blurs the paper's EU-specific claim
- **Severity**: Major
- **Evidence Anchor**: table: Table 2 (EU width and AU columns, every row)
- **Confidence**: 4 (AU/EU decomposition practice; empirical use of entropy-based decompositions)
- In Table 2, raw agreement for EU width and for AU match to within 0.01 in every row. In §4.4, Sρ predicts AU agreement (0.92) as well as EU agreement, and the authors concede that it indexes shared head behaviour rather than EU specifically. For readers outside UQ, the title promises a result about *epistemic* uncertainty. Yet the empirical estimators of EU and AU barely come apart on this data. The paper's EU-specific evidence then rests mainly on the theory and the closed-form u(x), whose rank agreement with the softmax estimators is never reported. The paper cites Wimmer et al. (2023), but it does not show the reader how far the EU summaries separate from AU here. I suggest reporting the item-level correlation between each EU summary and AU (and confidence) for each encoder, and saying explicitly which empirical findings remain EU-specific once that overlap is accounted for.

### W3: The motivating practical uses are never tested at the decision level
- **Severity**: Major
- **Evidence Anchor**: text: §1 Introduction "a cheap, label-free tool for transferring uncertainty estimates, selecting encoders for active learning"
- **Confidence**: 4 (active learning and uncertainty-transfer practice with soft-label targets)
- The introduction motivates the paper with active-learning encoder selection, uncertainty transfer and shared blind spots. Recommendation (i) tells practitioners not to use CKA or mutual k-NN for these purposes. But no experiment shows that the difference between CKA and Sρ changes a downstream decision. Examples would be overlap of top-k acquisition sets, active-learning curves, or selective-prediction risk when EU is transferred from one encoder to another. An item-level partial Spearman of 0.21 to 0.36 could still give large or small overlap in the top uncertain items that acquisition actually uses. As written, the recommendation is plausible but only partly argued for the practitioners it addresses. A minimal remedy is a top-k acquisition-overlap analysis on the existing heads. A stronger but costlier option is a small pool-based active-learning comparison using encoders ranked by CKA versus Sρ.

### W4: The E2b "AU not protected" conclusion compares residuals in which AU's construct-relevant variance has been removed
- **Severity**: Minor
- **Evidence Anchor**: text: §4.2 "the empirical conclusion is that identical features determine neither the residual EU ranking nor the residual AU ranking of a head."
- **Confidence**: 3 (interpretation of AU against human targets; the residualisation argument is conceptual, not re-computed)
- The partial AU agreement residualises model AU on both heads' confidence and on human entropy. Those are the two quantities that carry most of AU's meaning. What remains of AU is mostly variation specific to the head, so comparing it with residual EU does not cleanly test whether "AU agreement is protected". The authors report the falsification honestly. But adjacent readers may take it as evidence against the AU/EU distinction itself. The paper's own observation that AU's relation to human entropy is stable across head configurations points the other way. A sentence explaining the asymmetry in what the control removes, or an AU comparison that controls only for confidence, would prevent that misreading.

### W5: The stability of AU's relation to human labels is not pre-registered and not marked exploratory
- **Severity**: Minor
- **Evidence Anchor**: text: §4.2 "What is stable is AU’s relation to the human target"
- **Confidence**: 4 (checked against Table 1 and the paper's own exploratory-marking rule)
- §4 states that analyses not listed in Table 1 "are marked exploratory". This finding is not in Table 1 and carries no such label, yet it is the paper's main positive statement about human-label AU. It should be labelled exploratory, to stay consistent with the pre-registration framing that adjacent readers will rely on.

### W6: Recommendation (ii) is underspecified for the softmax heads practitioners actually use
- **Severity**: Minor
- **Evidence Anchor**: text: §5 Recommendations "Report EU together with the head’s prior strength relative to the feature scale."
- **Confidence**: 3 (practitioner reading; depends on Laplace/curvature details I did not verify)
- §4.3 says that for softmax estimators the effective prior strength is set by the curvature of the likelihood, not by weight decay alone. A practitioner following recommendation (ii) for a softmax probe therefore does not know which quantity to report. It could be weight decay times feature scale, the Laplace prior precision, or a curvature-adjusted effective ρ. One sentence of operational guidance would make this recommendation actionable.

### W7: In the tested regime, "Sρ at the head's own ρ" adds nothing over its ρ→0 canonical-correlation limit
- **Severity**: Minor
- **Evidence Anchor**: table: Table 5 (rows Sρ→0 and Sρ at the head's ρ)
- **Confidence**: 4 (read directly from Table 5 and §4.4)
- Table 5 shows Sρ→0 matching or slightly exceeding Sρ at the head's ρ in every column. §4.4 notes that all heads sit at deff(ρ)/d above 0.98. For a practitioner, the recommended index therefore collapses empirically to a mean-squared canonical correlation, which needs no knowledge of the head's prior. The theoretical case for using the head's own ρ is sound. But Recommendation (i) should say that its practical benefit over the ρ→0 index has not been shown on these data, and that it would matter only in regimes with stronger regularisation.

### W8: The evaluation never leaves the training distribution, though the "shared blind spots" framing points there
- **Severity**: Minor
- **Evidence Anchor**: absence: §4 Experiments and §5 Limitations — expected an evaluation of EU agreement on shifted or out-of-distribution inputs (or an explicit statement that the conclusions are restricted to in-distribution data); checked §1, §4.1–4.4, Table 1, §5 Limitations, Appendix B, Appendix C
- **Confidence**: 3 (safety and human-machine disagreement framing; in-distribution scope inferred from §4 data description)
- All EU agreement is measured on the CIFAR-10 test set, while the "blind spots" motivation most naturally concerns distribution shift, where EU matters most. The Limitations list other modalities and fine-tuning but not distribution shift. Adding that scope condition, or a small shifted-data check (for example, corrupted CIFAR-10), would tell adjacent readers how far the result transfers.

### W9: The implications for neighbouring human-uncertainty and RSA communities are not drawn out
- **Severity**: Minor
- **Evidence Anchor**: absence: §2 Setup and related work and §5 Discussion — expected discussion of what EU non-identification implies for RSA-style brain/model comparisons and for learning-from-human-disagreement (soft-label) work beyond Peterson et al. (2019); checked §1, §2, §5, References
- **Confidence**: 4 (core competence: human-label uncertainty and human-machine disagreement literature)
- The paper cites Kriegeskorte et al. (2008) and Stringer et al. (2019) and uses CIFAR-10H. But it never tells cognitive and neuroscience readers what follows for them. For example, high representational similarity between a model and a brain area would not imply shared uncertainty. Nor does it connect its human-entropy control to the literature on learning from annotator disagreement. One paragraph would substantiate the paper's implied cross-field relevance at low cost.
