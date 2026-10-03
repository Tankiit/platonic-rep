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
trigger: "Exposition is followable but has localised problems"

### D6: venue_fit_and_contribution
score: warn
trigger: "the readership payoff is asserted rather than demonstrated"

## Review Body

Journal-Fit Reviewer (senior area chair, statistical ML and uncertainty quantification). No venue criteria manifest was supplied, so I make no venue-criteria claims beyond general fit for a statistics/ML conference readership.

Overall view. The question (does representational alignment license transfer of epistemic uncertainty between heads?) sits well within a statistics-and-ML conference's scope. Bayesian linear heads, ridge resolvents, Laplace and bootstrap estimators are all core material for that readership. The manuscript is unusually candid: it pre-registers predictions, reports three failures, and its abstract states that the main cross-encoder difference is not statistically established. My concern is significance, not honesty. The theory is exact but shallow: by the authors' own account it is one application of Isserlis' theorem to an index family that already exists. The empirical half does not show the practical payoff its recommendations imply. The headline index comparison has an interval that touches zero. The proposed index cannot be told apart from linear predictivity. And because softmax EU summaries are nearly collinear with confidence, the real-data experiments cannot speak to EU specifically. What survives is a clean conceptual note: CKA weights the spectrum differently from posterior variance and ignores scale. In its current form that reads as a careful workshop-plus contribution more than a main-track one. Reframing the paper around what is actually established, and adding evidence where the payoff is now only asserted, could close the gap. I therefore score D6 at warn, not block. Writing quality is generally high, with localised presentation problems (D5 warn).

### S1: Pre-registration with all outcomes reported, including failures
The paper freezes predictions and falsifiers before computing any real-data result and reports every outcome, including three "not supported" verdicts (E2-b, E4, and the softmax arm of Spec). It also says plainly which operationalisations were fixed after the fact. This standard of reporting is rare in representation-similarity work and deserves credit from an editor.
**Evidence Anchor**: `table: Table 1, full verdict column across all eight pre-registered rows (E0 to Spec)`

### S2: Exact, interpretable account of what CKA discards
Theorem 2 and Proposition 3 show that CKA weights shared directions by squared variance, whereas EU agreement weights them by squared shrinkage factors. That explains the mismatch cleanly, and the explicit constructions in Corollary 4 make it concrete. The synthetic check, where the sample index predicts empirical EU correlation with error 0.010 while CKA's error is 0.37, confirms that the identity is implemented correctly.
**Evidence Anchor**: `figure: Figure 1a, filled (S-rho) versus open (CKA) markers against empirical EU correlation over 60 synthetic pairs`

### S3: Careful handling of estimator noise
The redraws with reliability correction separate genuine head disagreement from Monte Carlo noise, which gives the identical-features result real force: corrected EU agreement of 0.48 for 10% versus 100% of the data. Few papers on uncertainty agreement report re-test ceilings.
**Evidence Anchor**: `table: Table 3, "corrected" column for the 10% vs 100% and disjoint-halves rows`

### S4: In scope for a statistics/ML readership
The analytical core uses posterior variances of Bayesian linear heads, ridge smoothers and effective dimension. These are the venue's natural vocabulary, so the theory-plus-experiment balance suits the readership.
**Evidence Anchor**: `equation: Theorem 2, corr(uA, uB) = S-rho for jointly Gaussian features`

### S5: Proportionate limitations section
The limitations section names the dependence among the 30 encoder pairs, the single in-distribution dataset and the frozen-probe setting, and it does not claim analyses that were planned but never run.
**Evidence Anchor**: `text: Section 5, Limitations "Five encoders give 30 dependent pairs, which limits every cross-encoder conclusion."`

### W1: Theoretical contribution is thin relative to the main-track bar
**Problem**: The central identity follows from a single moment computation. The index it characterises is explicitly not new (ridge-CCA, GULP, Harvey et al.). The representation-only non-identification result is conceded to follow directly from Equation (1). The other results are either classical (Lemma 9, the sandwich covariance; Lemma 7, effective dimension) or direct specialisations (Corollaries 4 and 8, Proposition 5).
**Evidence Anchor**: `text: Section 3, after Proposition 3 "The proof is one application of Isserlis’ theorem; its value lies in the interpretation."`
**Why it matters**: When a paper's main claim to originality is the link between a known index and posterior-variance agreement, readers expect either depth (assumptions beyond Gaussianity, finite-sample rates, non-linear heads) or a demonstrated practical consequence. Without either, the contribution reads as an observation.
**Suggestion**: Pick at least one of the following. (a) Extend the identity beyond Gaussian features, for example with a fourth-cumulant correction that explains the 0.34 level offset reported on real features. (b) Give a finite-sample statement for the sample index. (c) Treat the GLM/Laplace case so that the theory covers the heads actually used. Alternatively, reframe the paper honestly as a short analytical note.
**Severity**: Major
**Confidence**: 4 — core expertise: Bayesian deep learning / UQ positioning

### W2: The practical recommendation is not empirically established
**Problem**: The recommendation to prefer whitened or S-rho indices over CKA rests on a cross-encoder comparison whose encoder-cluster interval reaches zero. The proposed index is also indistinguishable from linear predictivity, and the matched-rho advantage in the weight-decay sweep likewise touches zero.
**Evidence Anchor**: `table: Table 5, rows "Sρ−CKA" (+0.19 [+0.00, +0.50]) and "Sρ−linear pred." (-0.00 [-0.22, +0.11]) in the width (partial) column`
**Why it matters**: The readership payoff the paper offers, guidance on which index to use when transferring or validating EU, is asserted rather than demonstrated. The authors themselves say they have not shown a practical advantage over linear predictivity.
**Suggestion**: Add encoders (even 3 to 5 more would materially tighten encoder-cluster intervals), or pair the index comparison with a downstream task in which the choice of index changes a decision, such as acquisition-set overlap or uncertainty transfer error. Otherwise, demote the recommendation to a hypothesis.
**Severity**: Major
**Confidence**: 4 — core expertise: UQ evaluation practice

### W3: The real-data experiments cannot isolate epistemic uncertainty
**Problem**: For softmax probes, the EU summaries are nearly a function of confidence, and every index predicts AU agreement as well as it predicts EU agreement. The empirical half therefore measures shared head behaviour in general. The EU-specific mechanism rests on the Gaussian theory and the closed form alone.
**Evidence Anchor**: `text: Section 4.4 "All indices predict AU agreement about as well as EU agreement"`
**Why it matters**: For a UQ readership, the title's question is about epistemic uncertainty specifically. If the experiments cannot tell EU from AU or confidence, the empirical contribution to that question is limited. The paper concedes this in the Discussion, but the abstract and contributions still present the experiments as evidence about EU agreement.
**Suggestion**: Make the closed-form or Gaussian-process heads (where EU does not depend on labels) the primary empirical object. Alternatively, add an EU target that separates from confidence, such as out-of-distribution or subsampled-class inputs where epistemic and aleatoric uncertainty come apart. Then align the abstract with whichever choice is made.
**Severity**: Major
**Confidence**: 4 — core expertise: aleatoric/epistemic decomposition

### W4: One of the two "robust" findings has limited practical significance
**Problem**: The Discussion presents scale invariance hiding a prior–scale mismatch as robust. But the effect is the textbook coupling between ridge penalty and feature scale, and it disappears under ordinary validation tuning.
**Evidence Anchor**: `text: Section 4.3 "it vanishes when the weight decay is re-tuned."`
**Why it matters**: Practitioners who tune weight decay never meet this failure, so its weight as a headline result for the readership is modest.
**Suggestion**: Present it as a diagnostic caveat (report the scale together with the weight decay) rather than as one of two main robust findings, or show a realistic pipeline in which scale and prior end up mismatched despite tuning.
**Severity**: Minor
**Confidence**: 4 — core expertise: Bayesian linear models

### W5: Mutual k-NN nearly matches S-rho, which blurs the paper's account of why indices succeed or fail
**Problem**: Mutual k-NN shares CKA's scale and orthogonal invariance, yet it tracks closed-form and raw EU agreement almost as well as S-rho. This weakens the framing that CKA's specific invariances are what disqualify it. Its sensitivity to the spectral tail may be the operative property instead.
**Evidence Anchor**: `table: Table 7, BLR column, mutual k-NN 0.90 versus S-rho (head's rho) 0.92 and CKA 0.76`
**Why it matters**: Readers need to know which property of an index predicts EU agreement. The current narrative groups mutual k-NN with CKA in Section 2 but then reports it behaving more like S-rho.
**Suggestion**: Discuss mutual k-NN explicitly as a tail-sensitive but scale-invariant case, and state which of the two mechanisms (spectral weighting or scale) the cross-encoder data actually supports.
**Severity**: Minor
**Confidence**: 3 — adjacent: representation-similarity metrics

### W6: Positioning omits the literature on whether models share errors
**Problem**: The introduction motivates the question partly by asking whether aligned models share blind spots, but the related work does not engage with error-consistency or model-agreement studies of vision classifiers. Those studies address shared failures directly.
**Evidence Anchor**: `absence: Section 2 (Setup and Related Work) and the reference list — expected discussion of error-consistency or model-agreement work on vision classifiers; checked Introduction, Section 2, Discussion, References`
**Why it matters**: Without that literature, adjacent readers cannot judge what uncertainty agreement adds beyond error or prediction agreement, which is the paper's distinguishing claim.
**Suggestion**: Add a short paragraph contrasting EU agreement with error- and prediction-agreement measures, and say why uncertainty is the more informative target.
**Severity**: Minor
**Confidence**: 3 — adjacent field: applying general positioning standards

### W7: The pre-registration falsifier was triggered, which narrows the practical scope of the thesis
**Problem**: E4 predicted that head and data terms would dominate at high alignment. Instead the encoder accounted for 93% of raw EU disagreement, so the regime in which CKA's blindness to head and data would matter most was not reached by any real encoder pair tested.
**Evidence Anchor**: `text: Section 4.4, Swap design "alignment never reached a regime where head and data terms dominate."`
**Why it matters**: The thesis survives as a statement about index choice. But the scenario motivating the paper, aligned encoders whose heads nonetheless disagree because of prior and data, is shown only on identical features, not between distinct high-CKA encoders.
**Suggestion**: State this consequence in the introduction and abstract. Consider testing pairs with higher alignment, such as two seeds or two checkpoints of one architecture.
**Severity**: Minor
**Confidence**: 4 — core expertise: interpreting falsification outcomes

### W8: Table 1 uses an inconsistent verdict vocabulary and labels an uninformative test "supported"
**Problem**: Verdicts mix "supported", "not supported", "point est. only" and "closed form only". E2-a is marked "supported" even though its own footnote says the falsifier could not have fired.
**Evidence Anchor**: `table: Table 1, row E2-a (observed 0.26, verdict "supported", footnote †)`
**Why it matters**: Table 1 is the reader's map of what was confirmed. A "supported" verdict on an uninformative test overstates confirmation.
**Suggestion**: Use a fixed vocabulary (supported / not supported / inconclusive / uninformative) and recode E2-a as uninformative.
**Severity**: Minor
**Confidence**: 5 — core expertise: presentation of pre-registered results

### W9: The contributions list is padded with classical or tangential results
**Problem**: The contributions bullets list a classical bootstrap lemma, an in-sample resolvent bound that the paper itself shows does not extend to unobserved points, and a known effective-dimension identity alongside the main results.
**Evidence Anchor**: `text: Section 1, Contributions item 2 "also note an in-sample resolvent bound"`
**Why it matters**: The padding makes the theoretical contribution look broader than it is (see W1) and pulls attention away from the identity and the CKA limits.
**Suggestion**: Move Proposition 10 to the appendix and present Lemmas 7 and 9 as tools, not contributions.
**Severity**: Minor
**Confidence**: 4 — core expertise: editorial structure

### W10: "Post-review" labelling of main-text analyses confuses readers
**Problem**: Several main-text analyses and figures are tagged "post-review", and the appendix explains that these come from an internal review round. A conference reader has no context for this, and it mixes revision history into the scientific record.
**Evidence Anchor**: `text: Appendix C "All analyses in this section were added after an internal round of review and are exploratory."`
**Why it matters**: The "exploratory" status is the information that matters. The "post-review" provenance adds confusion, and it makes Section 4 read as a patched draft.
**Suggestion**: Label these analyses consistently as "exploratory (not pre-registered)" and drop the post-review terminology.
**Severity**: Minor
**Confidence**: 4 — core expertise: venue presentation conventions

Copyedit-level notes (below the finding threshold): Section 4.4 says "all reach 0.83–0.83", which is a degenerate range and should read "all reach 0.83". The upper bounds of 1.00 in Table 5 reflect the coarseness of a five-encoder resampling, and the caption should say so explicitly. Each line of the abstract is dense with numbers; cutting it to the three results the paper stands behind would help readability.
