# Peer Reviewer 3 (Perspective) — Phase 2 Paper-Visible Review

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
trigger: "implications for adjacent fields or practitioners are overstated, unbounded, or missing even though they are clearly relevant"

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

**Reviewer identity.** Computational cognitive scientist working on human label uncertainty and human-machine disagreement, with experience using crowd-sourced soft labels as uncertainty targets. I do not assess proof-level correctness (outside my remit and a declared blind spot).

**Confidence.** 4 of 5 for the human-uncertainty, reliability and practical-implication questions; 2 of 5 for anything that depends on the spectral theory itself. Confidence describes uncertainty and scope only. Calibration status: `NOT_CALIBRATED`.

**Summary from the perspective seat.** The paper asks whether CKA-aligned encoders give heads that are uncertain about the same inputs. Its answer is "not in general": CKA discards feature scale, which acts as a prior, and it weights spectral directions by squared variance, while EU counts every direction above the regularisation level. The mechanistic story is easy for an adjacent-field reader to follow, and the paper is unusually open about failed predictions, estimator noise and the reliability of its human covariate. My D4 concern is that the empirical bridge from this theory to the implications the introduction advertises (uncertainty transfer, encoder selection for active learning, shared blind spots) is thinner than the practitioner-facing recommendations and "robust findings" language suggest. Three things drive this. (a) The central empirical construct, confidence-partialled "residual EU", is defined only operationally and never checked against any external criterion. (b) Model "AU" is a hard-label ensemble entropy whose link to human ambiguity is overstated by the ceiling normalisation used. (c) The CCA-type recommendation rests on intervals that include zero and on indices that predict AU as well as they predict EU. None of these is asserted without evidence, so I score warn, not block. Each can be fixed by rewording and by one or two targeted analyses.

**Manuscript integrity note.** I found no instruction-like or reviewer-directed text in the manuscript. The AI-use statement is ordinary disclosure content.

### S1: Human-label reliability is treated as a first-class constraint
The paper reports a split-half, Spearman–Brown corrected reliability for CIFAR-10H entropy, tests sensitivity to the choice of entropy estimator, and says plainly that partial and stratified controls cannot fully remove ambiguity-driven agreement. Cognitive scientists using crowd labels rarely see this in ML papers, and it makes the AU numbers interpretable.
**Evidence Anchor**: `text: Section 2 EU agreement, "human entropy is itself a noisy covariate (reliability 0.70)"`

### S2: Failed pre-registered predictions are reported alongside the successful ones
Table 1 and the abstract report failed predictions (E2-b, E4, the Spec prediction for softmax heads) and mark post hoc operationalisations. This helps readers in adjacent empirical fields judge how much the confirmatory framing can carry.
**Evidence Anchor**: `text: Table 1 caption, "Pre-registered predictions and outcomes (frozen before any real-data result)"`

### S3: The mechanism is explained in terms an outsider can use
The two reasons CKA misses EU are given in one plain sentence each before any formalism, so a statistician or cognitive modeller can carry the intuition (scale is a prior; low-variance directions count for EU) into their own setting.
**Evidence Anchor**: `text: Section 1, "It normalises away the scale of the features, and a regularised head treats scale as part of its prior"`

### S4: A decision-level summary is offered next to correlations
Reporting overlap of the top-10% highest-uncertainty (acquisition) sets turns item-level rank agreement into something an active-learning practitioner can read directly (0.93 re-test vs 0.70 for 10% vs 100% data; 0.38 between encoders).
**Evidence Anchor**: `text: Table 3 caption, "top-10% = overlap of the 10% highest-uncertainty sets"`

### S5: The practical recommendation is partly self-bounded
The authors explicitly decline to claim a practical advantage of their own index over a simpler baseline.
**Evidence Anchor**: `text: Section 5 Recommendations, "we have not shown a practical advantage of Sρ over"`

### W1: "Residual EU" after partialling out confidence is an operational construct with no substantive validation, yet it carries a headline robust finding
For softmax probes the paper shows that EU summaries are almost a function of confidence (rank correlation about −0.999), so it reads all identical-feature results through partial agreement with confidence and human entropy regressed out. What remains is then described as "residual EU", and the Discussion lists its dependence on training data as one of two robust findings. For a reader outside the UQ community it is unclear what residual EU means. At M = 50 the re-test partial reliability is only 0.56, so a large share of the residual is Monte Carlo noise. After reliability correction the remainder is a confidence-orthogonal component of an ensemble-spread statistic, and nothing ties it to any reducible-uncertainty behaviour. The paper does not show that it predicts which items gain most from extra labels, error on held-out data, human disagreement beyond confidence, or anything else. Without such a link, "heads disagree about residual EU" could equally be read as "heads disagree about a noise-dominated residual of their confidence ranking". That reading undercuts the implications for transfer and acquisition.
Suggested remedy: (i) Give the construct a substantive gloss and rename it to match what is measured (for example, a confidence-residualised ensemble-spread ranking). (ii) Add at least one external-validity check, such as whether residual EU at 10% data predicts per-item loss reduction when moving to 100% data. That check is cheap, since both heads already exist. Otherwise, move the claim out of the "robust findings" sentence.
**Severity**: Major
**Evidence Anchor**: `text: Section 5 What the results say, "even identical features leave the residual EU ranking dependent on the"`
**Confidence**: 4 — routine concern in human-uncertainty work about residualised constructs and their reliability

### W2: The "fraction of ceiling" for AU versus human entropy uses the reliability itself rather than its square root, which conflicts with the paper's own correction elsewhere
Under classical test theory, the largest possible correlation between any predictor and an observed measure with reliability r_yy is sqrt(r_yy), here about 0.84, not 0.70. Dividing 0.34–0.43 by 0.70 gives the reported 0.48–0.61 "of the ceiling". Against sqrt(0.70) the figures are about 0.41–0.51. Table 3 applies the correct square-root (geometric-mean) disattenuation to head-head agreement, so the two corrections are inconsistent within one paper. The effect is to make model AU look more human-aligned than it is. Psychometrics and cognitive-science readers will notice this first.
Suggested remedy: Report the sqrt(r_yy) ceiling, or state explicitly which convention is used and why. Use the same convention in Section 4.2 and Table 6.
**Severity**: Minor
**Evidence Anchor**: `text: Section 4.2, "at 0.34–0.43 overall (0.48–0.61 of the ceiling)"`
**Confidence**: 4 — standard attenuation-correction convention in psychometrics

### W3: "AU" is a hard-label ensemble entropy, and the positive control validates it against a model-defined teacher rather than against human ambiguity
The heads are trained on CIFAR-10 hard labels, and AU is the mean member entropy. In the known-AU control, labels are drawn from a softmax teacher on each encoder's own features. The gate therefore shows that the pipeline recovers well-specified, model-generated noise. It does not show that model AU tracks the kind of perceptual ambiguity CIFAR-10H measures. Adjacent-field readers will take "aleatoric" to mean irreducible label ambiguity in the world. The text sometimes reads AU against human entropy as if it were an estimate of that quantity (for example, "AU's relation to the human target is stable"). This matters for the failed E2-b prediction too. If hard-label AU is mostly confidence, then the expectation that "AU agreement is protected" was never testable with these heads, and the failure says little about the AU/EU distinction in general.
Suggested remedy: State explicitly that model AU here is a hard-label predictive-entropy summary, not a human-ambiguity estimate. Optionally add a small soft-label-trained head (CIFAR-10H soft labels on a held-out split) as a contrast, which would test whether AU agreement behaves differently once the head is trained on the target it is compared to.
**Severity**: Minor
**Evidence Anchor**: `text: Appendix B Heads and Section 4 Data, "AU: mean member entropy" and "(50k, hard labels)"`
**Confidence**: 4 — direct experience training on soft versus hard labels

### W4: The practitioner recommendation to prefer whitened indices goes beyond the evidence and the paper's own reading of what the indices track
Recommendation (i) tells practitioners to prefer CCA-type indices or Sρ. The supporting comparison has an encoder-cluster interval of [0.00, 0.50]. Sρ is indistinguishable from linear predictivity and from its own ρ → 0 limit at the selected weight decays. And every index predicts AU agreement about as well as EU agreement (Sρ: 0.92), which the authors themselves read as indexing shared head behaviour rather than EU. A reader who deploys uncertainty estimates will take "prefer X" as established guidance. The negative half of the recommendation (do not use CKA to transfer EU) is well supported by the constructions and the scale experiment. The positive half is a hypothesis.
Suggested remedy: Split recommendation (i) into a supported negative claim and a labelled conjecture (for example, "the theory predicts, and our five-encoder data are consistent with, ..."). Also say that none of the indices is EU-specific in practice.
**Severity**: Minor
**Evidence Anchor**: `text: Section 5 Recommendations, "If an index is needed, prefer whitened (CCA-type) indices"`
**Confidence**: 4 — the evidence numbers are stated in the paper; judging how practitioners read guidance is within my remit

### W5: The motivating use cases are hypothetical and lie outside the tested regime
The introduction motivates the work with uncertainty transfer, encoder selection for active learning, and "shared blind spots". It also concedes that no one is known to use alignment this way. The experiments are in-distribution only, use upsampled 32-pixel images, and do not evaluate downstream active learning. The Limitations section discloses all of this, but readers from deployment-oriented fields have to reconstruct for themselves which implications survive. The between-encoder top-10% overlap of 0.38 is the most practically relevant number in the paper, yet it appears only in passing.
Suggested remedy: Bring the decision-level between-encoder result into the Discussion. State in the introduction that the motivating uses are hypothetical and that the OOD setting, where blind spots matter, is untested. A small OOD probe (for example, CIFAR-10-C or CIFAR-10.1 with the existing heads) would directly test the "shared blind spots" framing.
**Severity**: Minor
**Evidence Anchor**: `text: Section 5 Limitations, "out-of-distribution inputs, where shared blind spots matter most"`
**Confidence**: 4 — the scope gap is explicit in the text

### W6: No engagement with the human label-disagreement and human-uncertainty-alignment literature, although CIFAR-10H is central to the design
CIFAR-10H is cited only as a data source. The paper uses human entropy as a covariate and as an AU target, and it reports model-to-human AU correlations. It does not connect to work that treats annotator disagreement as signal: Uma et al. (2021, JAIR) survey learning from disagreement, Collins, Bhatt and Weller (2022, HCOMP) study soft labels from every annotator, and Peterson et al. (2019) themselves argue for training on human uncertainty. It also misses the representational-alignment framing of Sucholutsky et al. (2023), which treats humans as one more system to align with. That framing suggests a natural extension: treating the human label distribution as a sixth "encoder" and asking whether representational alignment predicts agreement with human uncertainty. This would widen the paper's reach to cognitive-science readers at little cost.
Suggested remedy: Add a short paragraph positioning the human-entropy analyses within this literature. Optionally report the "humans as a sixth system" agreement column using existing quantities.
**Severity**: Minor
**Evidence Anchor**: `absence: Section 2 Setup and Related Work — expected engagement with human label-disagreement and human-uncertainty-alignment literature beyond citing CIFAR-10H as a data source; checked Section 2, Section 4 E0 paragraph, Section 5 Discussion, References list`
**Confidence**: 4 — this is my home literature
