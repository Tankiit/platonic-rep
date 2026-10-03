## Contract Paraphrase

D1 (methodology_rigor, mandatory, owned by methodology): From a methodology-rigor standpoint, this dimension asks whether the experimental and analytical design can actually answer the questions the manuscript poses. That covers how data are chosen, processed and split; whether estimators, metrics and statistical procedures fit the data and the claims; whether variability (seeds, resamples, confidence intervals) and multiple comparisons are handled honestly; and whether enough procedural detail, code and data access exist for an independent group to reproduce the reported numbers. The bar is what a peer reviewer in statistical machine learning would expect.

D2 (domain_accuracy, mandatory, owned by domain): This dimension asks whether the manuscript's substantive claims agree with the current state of evidence in its field, whether prior methods and results are described faithfully, and whether technical terminology and reported domain facts are correct. Methodology rigor touches it only indirectly: a sound design built on a misrepresented premise still fails here. This seat does not score it.

D3 (argumentative_coherence, mandatory, owned by da, also eligible for methodology): Seen through methodology, this dimension asks whether the inferential chain from evidence to conclusion holds. Do the experiments and any formal results actually license the stated thesis, at the strength it is stated? Are the core claims mutually consistent? Is the argument free of fallacies such as circularity, overgeneralization from narrow conditions, or reading causal or identifiability conclusions into correlational or constructed evidence? The methodology seat scores it alongside the devil's advocate.

D4 (cross_disciplinary_relevance, high, owned by perspective): This dimension asks whether readers from neighboring fields can follow the framing, definitions and implications, and whether any claim that reaches across disciplines is backed by evidence rather than asserted. Methodology bears on it only when such claims depend on transferring a method or measure across settings without validation. This seat does not score it.

D5 (writing_and_structure, normal, owned by eic): This dimension covers organization, clarity of exposition, figure and table quality, and conformance to venue formatting conventions. From a methodology view, clear exposition of procedures supports reproducibility, but the score belongs to the editor seat. This seat does not score it.

D6 (venue_fit_and_contribution, mandatory, owned by eic): This dimension asks whether the work suits the configured venue and makes an original and significant contribution for that readership. Methodological soundness is necessary but not sufficient for contribution; the judgement of novelty and fit belongs to the editor seat. This seat does not score it.

## Scoring Plan

### D1: methodology_rigor
dimension_id: D1
what_to_look_for: A design whose experiments, estimators and metrics can answer the stated research questions; explicit data selection, preprocessing and splitting; appropriate baselines and controls; variability reported across seeds or resamples with intervals or tests; correction or acknowledgement of multiple comparisons; assumptions of formal results stated and checked against the experimental setup; enough hyperparameter, compute, code and data detail to reproduce headline numbers.
what_triggers_block: A repairable but material design or analysis defect that undermines a headline empirical result, such as missing key controls or baselines, results from a single run with no variability estimate on a claim that depends on small differences, data leakage or tuning on evaluation data, or a metric that does not measure the quantity the claim concerns.
what_triggers_warn: Gaps that weaken confidence without overturning headline results, such as incomplete reporting of hyperparameters or seeds, missing confidence intervals on secondary results, unstated preprocessing choices, limited ablations, or code and data availability not described.
what_triggers_fatal: The core empirical or formal evidence cannot support the central claim even in principle because of an unfixable design flaw, such as a construction that guarantees the reported outcome by definition, an evaluation that does not test the stated hypothesis, or reported numbers that are irreproducible or internally contradictory at a level that voids the main result.

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: A clear central thesis whose strength matches the evidence; formal statements whose hypotheses match the settings where they are applied; consistency between theory, experiments and conclusions; explicit scope conditions; absence of circular reasoning, overgeneralization from narrow conditions, or identifiability and causal conclusions drawn from evidence that cannot carry them; counterexamples and negative results addressed rather than omitted.
what_triggers_block: A material inferential gap that a substantial revision could close, such as conclusions stated more broadly than the experiments or theorems support, a formal result whose assumptions are not shown to hold in the empirical setting it is used to interpret, or an internal inconsistency between two core claims.
what_triggers_warn: Localized overstatement or loose reasoning that leaves the core thesis intact, such as hedging missing on secondary claims, scope conditions left implicit, alternative explanations acknowledged only in passing, or minor tension between a stated claim and a supporting figure or table.
what_triggers_fatal: The central thesis is logically unsupported or self-contradictory in a way no revision of the presented evidence can repair, such as an argument that assumes its own conclusion, a core theorem that is false or inapplicable to the claim it underwrites, or evidence that directly contradicts the headline conclusion.

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
