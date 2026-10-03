## Contract Paraphrase

D1 (methodology_rigor, mandatory) asks whether the empirical design can actually answer the stated research questions: whether model, dataset and layer choices are justified and not cherry-picked, whether similarity and uncertainty estimators are well specified, whether variance across seeds, samples or bootstrap resamples is reported with intervals, whether multiple comparisons and hyperparameter selection are controlled, and whether code, data, preprocessing and compute details are complete enough for an independent group to reproduce the headline numbers.

D2 (domain_accuracy, mandatory) concerns whether the paper's claims and its account of prior uncertainty-quantification and representation-similarity work match current domain evidence and terminology. From a methodology standpoint, this matters because a mischaracterised baseline or estimator distorts the comparison design, but scoring belongs to the domain seat.

D3 (argumentative_coherence, mandatory) asks whether the central thesis holds together internally: whether each conclusion follows from the evidence actually produced, whether correlational findings are kept separate from causal or mechanistic language, whether theoretical statements and empirical tests address the same quantity, and whether the scope of the claims stays within the tested conditions, without circular reasoning or post-hoc reframing.

D4 (cross_disciplinary_relevance, high) concerns whether framing, definitions and implications are accessible and substantiated for readers in adjacent fields. Methodologically, clear operational definitions help here, but this dimension is scored by the perspective seat.

D5 (writing_and_structure, normal) covers organisation, clarity, figure and table quality, and venue conventions. Clear reporting of experimental protocols overlaps with reproducibility, but the dimension is owned and scored by the editor seat.

D6 (venue_fit_and_contribution, mandatory) asks whether the work fits the target venue and makes an original, significant contribution. Methodological soundness is a precondition for a contribution to count, but judging fit and significance is the editor seat's job.

## Scoring Plan

### D1: methodology_rigor
dimension_id: D1
what_to_look_for: Precise operational definitions of the alignment and uncertainty-agreement measures; justified selection of models, datasets, layers and sample sizes; variability estimates (seeds, bootstrap, confidence intervals) on every headline quantity; null or permutation baselines and controls for confounds such as shared training data, model scale or accuracy; correction or acknowledgment for multiple comparisons and hyperparameter search; and complete reproducibility affordances (code, data splits, preprocessing, estimator settings, compute).
what_triggers_block: A headline empirical conclusion rests on point estimates with no variability, null baseline, or confound control, so the reported effect cannot be distinguished from estimator noise or a trivial explanation such as shared accuracy, but the gap is repairable with additional experiments.
what_triggers_warn: Supporting analyses lack intervals, seeds, or ablations, or key protocol details (estimator hyperparameters, sample counts, layer choices, preprocessing) are underspecified enough to impede exact replication, while the headline conclusion still appears supported.
what_triggers_fatal: The core measurement is invalid for the question posed, for example the agreement metric is mathematically guaranteed by construction, evaluation leaks information between the quantities being compared, or the estimator is biased in a way that produces the headline effect, so no added experiment on the existing design can rescue the main claim.

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: Whether each stated conclusion traces to a specific result; whether theorems or propositions and experiments test the same quantity under matching assumptions; whether correlational evidence is described without causal or mechanistic overreach; whether claimed generality matches the range of models and data tested; and whether negative or contradicting results are reported and reconciled rather than omitted.
what_triggers_block: A central claim is stated more broadly or more causally than the evidence licenses, or a theoretical result is used to support an empirical conclusion under assumptions the experiments do not satisfy, requiring substantial rewriting or new evidence to reconcile.
what_triggers_warn: Isolated overstatements, unaddressed counterexamples, or loosely connected secondary claims appear, but the core thesis remains consistent with the presented evidence after modest rewording.
what_triggers_fatal: The central thesis is internally contradicted by the paper's own reported results or relies on circular reasoning in which the conclusion is assumed in the definition or analysis, so the core argument cannot stand.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
