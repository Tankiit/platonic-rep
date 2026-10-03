# Editorial Decision Package

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from any candidate or prose-named profile. `PROFILE_MEASURED` is unavailable until a closed profile artifact and replay validator bind the target fields to the completed panel's `execution_topology_sha256`. All five seats are also `NOT_CALIBRATED` at emission (the DA card declares `NOT_CALIBRATED` explicitly).

## Manuscript Information
- **Title**: Alignment Does Not Identify Epistemic Uncertainty
- **Field (metadata)**: statistical machine learning: uncertainty quantification and representation similarity
- **Manuscript SHA-256 (review copy)**: `dc347f2ed1d8040ef36636756b056d1770ee6e1c91d7c84651dc285f0be94427`
- **Review round**: Round 1
- **Mode / contract**: `reviewer_full`, sprint contract `reviewer/reviewer_full/v2`
- **Criteria binding**: `criteria_binding_unavailable`. All five Phase 1 cards disclose it. This package makes **no venue-criteria or venue-alignment claim**. Binding status is not a score, failure condition or verdict.

## Review Panel Provenance (#540/#740)

- **Typed artifact**: `paper_aistats/review/round1/provenance.json` (replay-validated by the dispatching layer: PASS)
- **Artifact SHA-256 (raw bytes)**: `e7651ab35b9ac735b970c61c488c4f17a8d15492e724dc1fdd1b4e2e07df09cb`
- **Schema**: `review-panel-provenance/1.0`
- **Panel ID**: `uq-aistats2027-round1`
- **Contract SHA-256 (artifact field)**: `e9712090d2469fea15a37b8e22d4e137afbcb2bf38d5789939c5df56738ef7af`
- **Normalized manifest SHA-256**: `4448b21d027f9b468698394813e6cbe75944c54c01201f94d447e97590bd0ef3`
- **Execution topology SHA-256**: `c259b3f71e07a5edf63a2840650cfafb0d9380f68f5533b18fdd78f82f6e2603`
- **Fresh-context scope**: `within_panel_attempt_only`. This covers only invocation-context separation inside this panel attempt. It says nothing about retries or earlier rounds.

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | subagent-ab16a42225c287375 | false | claude-opus-5-5 | anthropic | null |
| R1 | methodology | model | subagent-ae32cb05205233eb7 | false | claude-opus-5-5 | anthropic | null |
| R2 | domain | model | subagent-a7d92b67e40d65718 | false | claude-opus-5-5 | anthropic | null |
| R3 | perspective | model | subagent-a97e719c8d3dd54ed | false | claude-opus-5-5 | anthropic | null |
| DA | da | model | subagent-a051618d4412a03ab | false | claude-opus-5-5 | anthropic | null |

| Provenance axis | Status (`true` / `false` / `unknown`) |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation (`fresh_context`) | true |
| Blind to peer outputs | true |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: Not computed (`independence_claim: not_computed_from_personas`). Role or persona diversity establishes only `role_separated`. The panel is **not** described as independent anywhere in this package, and no cross-family or same-model-majority aggregate is computed.
- **Correlated-error disclosure** (required; reason `same_model_family`): All model-executed review seats used one model family; role separation does not remove correlated-error risk.

Practical consequence for the reader: where several seats raise the same point, that agreement may partly reflect a shared model prior rather than separate lines of evidence. Consensus labels below count positions. They are not independent confirmations.

---

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting "Alignment Does Not Identify Epistemic Uncertainty". Five role-separated review seats assessed it: a Journal-Fit Reviewer (EIC), a methodology reviewer (R1), a domain reviewer (R2), a cross-disciplinary reviewer (R3) and a Devil's Advocate (DA). The provenance section above reports how these seats were executed. It is not reduced to an independence claim.

### Decision: Major Revision

### Sprint-Contract Audit (mechanical, v3.6.2)

**Step 1: Role-scoped scoring matrix.** Only eligible seats count. Ineligible `not_assessed` entries are excluded from numerator and denominator. No eligible seat abstained, and no seat declared a fatal block.

| Dim | Name | Priority | Eligible seats → assessed scores | Verdict (worst eligible) |
|---|---|---|---|---|
| D1 | methodology_rigor | mandatory | methodology: block (repairable) | block |
| D2 | domain_accuracy | mandatory | domain: warn | warn |
| D3 | argumentative_coherence | mandatory | da: block (repairable); methodology: warn | block |
| D4 | cross_disciplinary_relevance | high | perspective: warn | warn |
| D5 | writing_and_structure | normal | eic: warn | warn |
| D6 | venue_fit_and_contribution | mandatory | eic: block (repairable) | block |

**Step 2: Failure conditions.** The closed vocabulary recognised every expression.

| Condition | Severity | Quantifier | Expression | Evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 95 | any | any mandatory dimension has a fatal block | No seat declared fatal on D1, D2, D3 or D6 (all blocks are `repairable`) | false |
| F2 | 90 | any | any mandatory dimension scores 'block' | D1 (methodology), D3 (da) and D6 (eic) each have ≥1 assessed block | **true** |
| F3 | 70 | majority | two or more mandatory dimensions score 'warn' or worse | Per dimension: D1 n=1, owner block → T; D2 n=1, owner warn → T; D3 n=2, both seats ≥ warn → T; D6 n=1, owner block → T. 4 ≥ 2 | **true** |
| F4 | 60 | any | any high-priority dimension scores 'block' | D4 = warn | false |
| F5 | 40 | any | any dimension scores 'warn' or worse | D1 to D6 all ≥ warn | **true** |
| F0 | 10 | all | every dimension scores 'pass' | No dimension passes | false |

**Step 3: Precedence.** The fired conditions are F2 (90), F3 (70) and F5 (40). The highest severity is F2, whose action is `editorial_decision=major_revision`. No fired action was softened.

dimension_verdicts: [D1=block, D2=warn, D3=block, D4=warn, D5=warn, D6=block]
fired_conditions: [F2, F3, F5]
da_critical_adjudications: [C1=VALIDATED]
editorial_decision=major_revision

The decision is not `accept`, so no `[DA-CRITICAL-VS-ACCEPT]` marker applies. Cross-model blind decision check: `ARS_CROSS_MODEL` was not indicated for this run, so the check was not performed and nothing changes.

### DA CRITICAL Adjudication

**C1 (D3 argumentative_coherence): VALIDATED (repairable scope overreach).**

- **DA's argument** (da.phase2.md, CRITICAL table, C1, and "Strongest Counter-Argument"): the title thesis is caught in a dilemma. On a strict reading, representation-only non-identification follows from Eq. (1), so it is true by construction. On a substantive reading, classical alignment indices order EU agreement well according to the paper's own Table 5. The supported claim is narrower: scale-invariant, top-heavy indices such as CKA are poor EU proxies.
- **Check against the manuscript.** Table 5 (supplement) reports the following Spearman correlations for width-partial and BLR: Sρ→0 = 0.83 and 0.93; mutual k-NN = 0.72 and 0.90; CKA = 0.64 and 0.76. Linear predictivity also reaches 0.83 on width-partial. §4.4 confirms that "all heads sit in the small-ρ regime (deff(ρ)/d > 0.98), where Sρ is close to its canonical-correlation limit". The abstract's last sentence states that EU agreement is "a property of representation, prior and data jointly". The DA's numbers match the manuscript. Where the title and abstract say "alignment" without qualification, the paper's own across-encoder data support the claim for CKA, not for CCA-type or k-NN alignment indices.
- **Corroboration from other seats.** EIC W1 and W4 (framing overstates the contribution; S_rho not singled out in Table 5; these drive the D6 block). R1 W5 (CCA-type indices, not the head's ρ, carry the advantage; claim should narrow) and R1's D3 warn. R2 W1 ("the evidence supports 'CCA-type indices order EU agreement better than CKA'"). R3 W7.
- **Limits of the validation.** The "true by construction" horn is validated only as a framing point. The Corollary 4 construction and the spectral-weighting account are not definitional, and every seat that assessed the mathematics (R1 proof audit, R2 S2, DA "Genuine Strengths") found it correct. R1 scores D3 `warn` ("core thesis intact"), whereas the DA scores it `block`. That is a severity disagreement, not a disagreement about whether the problem exists. Under the contract, D3's verdict is the worst eligible score, so the disagreement does not change the arithmetic. It is recorded below as dissent.
- **Required author response.** Narrow the title, abstract and Contribution 2 to what the evidence supports (CKA and other scale-invariant, top-heavy indices, plus the representation-only non-identification that follows from Eq. (1), stated as such), or supply evidence that CCA-type and k-NN alignment also fail to identify EU agreement. Roadmap item R8.

**Methodology receipt attestation.** R1 used the `no_recomputable_statistics` path. The checker confirmed only that the declaration exists (`[RECEIPT-ATTESTATION: declaration-only]`). I did a bounded spot check of the manuscript and found no test statistic with df, no p-value and no discrete-scale mean with analytic N. The only interval is a Spearman–Brown CI, which none of the listed recompute procedures covers. The attestation is therefore consistent with the manuscript. This is a spot check, not a machine verification. The Fisher-z intervals in R1 W2 are R1's own illustrative computation, not a figure from the manuscript.

### Consensus Analysis

The consensus counts run over the four non-DA seats (EIC, R1, R2, R3). Silence is counted as silence, not agreement. The full sub-claim inventory appears in Part 3.

#### Points of Agreement

- **[CONSENSUS-3] SC-1. The EU-specific framing of the title and abstract is not supported by the empirical results.** EIC W1, R2 W3 and R3 W2 all cite E2b falsified (AU degrades as much as EU) and Sρ predicting AU agreement as well (0.92). R1 is silent on this sub-claim. DA M1 also raises it.
- **[CONSENSUS-3] SC-5. Linear predictivity ties Sρ (0.83) on the headline partial column, and the text does not say so.** Raised by EIC W4, R1 W5 and R2 W1. R3 is silent.
- **[CONSENSUS-3] SC-7. Numbers in the §4.3 prose contradict Table 3** (0.72 vs 0.85; 0.89 vs 0.94; 0.38–1.84 vs 0.56–1.52; 2.3 vs 1.8; 1.39 vs 1.30; 0.62 vs 0.68). Raised by EIC W6, R1 W1 and R2 W5. R3 is silent. The DA also notes it as a minor point. Three seats note that the qualitative direction survives under either set of numbers, but that the P5 and L9 outcomes in Table 1 are read off these values.
- **[CONSENSUS-3] SC-8. The "E2b" label is used for two different things** (the AU falsifier in Table 1 and the transformation experiment in Table 3 and §4.3). Raised by EIC W7, R1 W10 and R2 W5. R3 is silent. Transported severity differs (EIC and R1 Minor; R2 Major as part of its W5 bundle). R2 did not rate this sub-claim on its own, so the difference is recorded but not treated as a conflict.
- **[CONSENSUS-3] SC-9. Table 1 and the text give different outcomes for the d_eff prediction** ("partly" vs "failed"), and "partly"/"Spec" are undefined. Raised by EIC W8, R1 W4 and R2 W5. R3 is silent. Severity is recorded the same way as SC-8.
- **Corroborated (2/4).** SC-2: novelty of Sρ not positioned against ridge-regularised, prediction-based or decoding-based similarity families (EIC W2, R2 W8). SC-6: no interval or test for the headline 0.83 vs 0.64 comparison over 30 dependent pairs (EIC W5, R1 W2; DA M7). SC-10: the AI-use statement says the proofs still need line-by-line checking (EIC W9, R1 W12; DA minor). SC-11: the full Table 5 belongs in the main text (EIC W10, R1 W6). SC-13: the Prop. 3(ii) normalisation is not a mean when d_A ≠ d_B (R1 proof-audit summary, R2 W7).
- **Agreed strengths** (raised by all four non-DA seats and the DA). Pre-registration with falsified predictions reported in the main text (EIC S1, R1 S2, R2 S1, R3 S2, DA). Every seat that assessed the mathematics found the closed-form results correct (R1 S1 after a full proof audit, R2 S2, DA). The identical-features design is clean (EIC S3, R3 S3). Reliability ceilings make the AU numbers interpretable (R2 S4, R3 S1).

No sub-claim reached CONSENSUS-4.

#### Points of Disagreement

- **SC-4. The head's-ρ Sρ cannot be told apart from its ρ→0 (CCA) limit, so Recommendation (i) lacks support.** EIC W4, R1 W5 and R2 W1 rate this **Major**: they ask for a weight-decay sweep that moves d_eff/d well below 1, or for the claim to be narrowed to CCA-type indices. R3 W7 rates the same observation **Minor**: Recommendation (i) should simply say that its practical benefit over the ρ→0 index has not been shown. The DA (M4) corroborates.
  - *Type*: severity disagreement.
  - **Editor's resolution**: treat as **Major / must_fix**. All four seats agree that the problem exists, and the Table 5 evidence is in the manuscript. Methodological sufficiency is R1's domain, and R1 accepts either a sweep or narrowing. Narrowing, which is R3's lighter remedy, satisfies both positions. R3's view that a scoped sentence would be enough is preserved as dissent on severity. The author may choose either remedy (sweep or narrowing), as long as the claim no longer goes beyond Table 5.
- **D3 severity (R1 warn vs DA block).** R1: "the non-identification thesis ... survives", with localised overstatement. DA: "central claim unsupported as stated". The mechanical verdict uses the worst eligible score. C1 is validated as a scope overreach that can be repaired by narrowing, which both seats' remedies allow. The panel did not settle whether this is "localised" or "central". That question is recorded as unresolved dissent, and the author must address C1 either way.
- **Is the index itself claimed as new?** R2 S3 credits the authors for *not* claiming the ridge index as new ("restrict the novelty to the EU link"). EIC W2 and R2 W8 still require positioning against the ridge, decoding and shape-metric families. These views are compatible: the attribution is honest, but the positioning is incomplete. No arbitration is needed.

### Decision Rationale

The decision follows mechanically from contract condition F2: three mandatory dimensions carry a repairable block. D1 (R1) is blocked by the headline cross-encoder comparison and the identical-features contrast, which rest on point estimates with no interval or test, a partial-agreement metric with Monte Carlo re-test reliability of 0.56, and the §4.3/Table 3 conflicts. D3 (DA, C1 validated) is blocked because the title generalises from CKA to "alignment" while Table 5 shows CCA-type and k-NN indices ordering EU agreement well. D6 (EIC) is blocked by framing that is EU-specific although the results are not (SC-1), and by novelty not established against ridge, decoding and prediction-based similarity families (SC-2). F3 and F5 also fired with lower severity. No seat marked any block as fatal, and every seat names a concrete repair: narrowing the scope, adding inferential support, reconciling numbers, and positioning against related work. The decision is therefore not Reject.

A stricter outcome is not warranted because the mathematics was audited line by line by R1 and found correct, and every seat values the pre-registration and the identical-features design. A lighter outcome is not available under the contract once any mandatory dimension scores block. It is also not supported on substance: the abstract's two headline numbers (0.83 vs 0.64; 0.26 vs 0.56) are exactly the quantities several seats find under-supported (SC-6, SC-14). The §4.3 discrepancies also need a provenance audit, because the AI-use statement says every number is script-generated. A narrower paper is achievable within this round: alignment metrics such as CKA do not determine the uncertainty of regularised heads, together with the exact Gaussian identity that explains why. Re-review is required after revision.

### Blocking Issues (immutable source order)

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R1 | Title and abstract frame the result as EU-specific, but E2b was falsified and Sρ predicts AU agreement equally (D6 block driver) | EIC, R2, R3 (DA M1) | `text: §4.4 "First, Sρ predicts AU agreement as well"` | REV-SC-1 |
| R5 | Headline Sρ vs CKA comparison (0.83 vs 0.64, 30 dependent pairs) has no interval or test (D1 block trigger) | EIC, R1 (DA M7) | `text: §4.4 "EU agreement at Spearman 0.83, against 0.64 for CKA"` | REV-SC-6 |
| R8 | Title thesis generalises to "alignment", but Table 5 shows CCA-type and k-NN indices order EU agreement well (D3 block, C1) | DA (corroborated by EIC, R1, R2, R3) | `table: Table 5, rows Sρ→0 and mutual k-NN versus CKA` | REV-DA-C1 |

---

## Part 2: Revision Roadmap

Roadmap core: reviewer-owned, immutable and non-ranking. Items are listed in deterministic source order: seat order EIC, R1, R2, R3, DA, then each item's earliest finding ordinal. Severity and obligation never affect the order. `R<n>` and `S<n>` are transport references, not ranks. Severity and confidence are transported from the cards. Confidence is a self-reported scope disclosure only and carries no weight in any decision.

**Machine-core limitation.** No bound block manifest exists for this round. `block_manifest_sha256` and the exact `proposed_targets[]` block ids therefore cannot be filled without fabrication, so the closed `revision-roadmap/1.0` JSON core is not emitted here. Target sections are given as manuscript locators. The orchestrator should bind a block manifest before running `scripts/revision_roadmap.py`. Base draft SHA-256 for that binding: `dc347f2ed1d8040ef36636756b056d1770ee6e1c91d7c84651dc285f0be94427`. Editorial decision field value: `Major Revision`. Obligation counts: must_fix 8, should_fix 20, consider 11.

### Required Revisions (Must Fix)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Reframe the title, abstract and recommendations around head-level (prior- and data-dependent) uncertainty, or add an experiment in which EU and AU agreement diverge. Position against AU/EU disentanglement evidence (see S12). | SC-1 | major | `text: §4.4 "First, Sρ predicts AU agreement as well"` | EIC 4 (own reported outcomes); R2 3; R3 4 | EIC W1; R2 W3; R3 W2 (DA M1) | must_fix | section: title, abstract, §1 contributions, §5 | claim_scope_unsupported → claim: title/abstract EU-specificity |
| R2 | State exactly what Sρ adds over ridge-regularised prediction-based distances (e.g. GULP), the regularised shape-metric family (Williams et al. 2021) and decoding-based interpretations (Harvey et al. 2024) | SC-2 | major | `absence: §2 related work, §3 after Definition 1, References` | EIC 3; R2 3 | EIC W2; R2 W8 | must_fix | section: §2, §3 after Definition 1 | claim_scope_unsupported → claim: novelty of Sρ |
| R3 | Either run a weight-decay sweep that moves d_eff/d well below 1 and show that Sρ at the matched ρ beats Sρ→0 and the CCA limit, or narrow Recommendation (i) and the headline to CCA-type indices | SC-4 | major (R3 dissent: minor) | `table: Table 5, rows Sρ→0, Sρ (head's ρ), linear predictivity` | EIC 4; R1 4; R2 4; R3 4 | EIC W4; R1 W5; R2 W1; R3 W7 (DA M4) | must_fix | re_analysis: §4.4 cross-encoder (or section: §5 Rec. (i) if narrowing) | claim_scope_unsupported → claim: §5 Recommendation (i) |
| R4 | Report the linear-predictivity tie (0.83) and all Table 5 indices next to the headline comparison | SC-5 | major | `table: Table 5 rows "linear predictivity", "Sρ→0", "Sρ (head's ρ)"` | EIC 4; R1 4; R2 4 | EIC W4; R1 W5; R2 W1 | must_fix | section: §4.4, abstract | reporting_requirement_unmet → table: Table 5 / §4.4 |
| R5 | Give a CI or test for the difference in dependent Spearman correlations (cluster bootstrap over encoders or a permutation scheme that respects the crossed design), at the headline column and across Table 5. Drop the comparison from the abstract if it is not supported. | SC-6 | major | `text: §4.4 "EU agreement at Spearman 0.83, against 0.64 for CKA"` | EIC 3; R1 4 | EIC W5; R1 W2 (DA M7) | must_fix | re_analysis: §4.4, Table 5 | evidence_gap_remains → claim: Sρ orders EU agreement better than CKA |
| R6 | Regenerate the §4.3 prose and Table 3 from one logged run, state which run Table 1 (P5, L9) uses, and remove the stray "4.3" in the table row | SC-7 | major | `text: §4.3 "Laplace MI rank agreement down to 0.72"` | EIC 5; R1 5; R2 4 | EIC W6; R1 W1; R2 W5 | must_fix | re_analysis: §4.3, Table 3, Table 1 rows P5/L9 | method_reproducibility_unresolved → table: Table 3 |
| R7 | Characterise partial-agreement reliability as a function of M (or increase M), report reliability-corrected EU agreements, and redraw the 10% subsets and disjoint halves several times with intervals | SC-14 | major | `table: Table 2, row re-test floor, column EU partial (0.56)` | R1 4 | R1 W3 (DA M3) | must_fix | re_analysis: §4.2, Table 2, Appendix B | evidence_gap_remains → claim: abstract 0.26 vs 0.56 |
| R8 | Narrow the title, abstract and Contribution 2 to scale-invariant, top-heavy indices (CKA), and state that representation-only non-identification follows from Eq. (1); or show that CCA-type and k-NN alignment also fail to identify EU agreement | — (DA-CRITICAL) | critical | `table: Table 5, rows Sρ→0 and mutual k-NN versus CKA (width partial 0.83 and 0.72 vs 0.64; BLR 0.93 and 0.90 vs 0.76)` | DA 4 (kernel alignment and Bayesian linear regression) | DA C1 (corroborated: EIC W1/W4, R1 W5, R2 W1, R3 W7) | must_fix | section: title, abstract, §1 Contribution 2 | claim_scope_unsupported → claim: title thesis |

#### Required Item Details

- **R1**: Reframe EU-specificity (SC-1).
  - **Acceptance criteria**: The title and abstract claim only what the empirical results support for EU as distinct from AU, or a new experiment shows EU and AU agreement diverging.
- **R2**: Position Sρ against related similarity families (SC-2).
  - **Acceptance criteria**: §2 and §3 name the ridge, prediction, decoding and shape-metric families and state precisely what Sρ or the EU identity adds.
- **R3**: Test or narrow the head's-ρ claim (SC-4).
  - **Acceptance criteria**: Either the sweep result is reported with d_eff/d well below 1, or Recommendation (i) and the headline no longer claim an advantage of the head's ρ over the CCA limit.
- **R4**: Report all indices (SC-5).
  - **Acceptance criteria**: The main text reports the linear-predictivity tie and the full Table 5 pattern next to the headline.
- **R5**: Inference for the headline comparison (SC-6).
  - **Acceptance criteria**: A dependence-respecting interval or test for the Sρ minus CKA Spearman difference is reported, and the abstract wording matches its outcome.
- **R6**: Reconcile §4.3 and Table 3 (SC-7).
  - **Acceptance criteria**: Every §4.3 number matches Table 3, and the run that sources Table 1 is identified.
- **R7**: Partial-metric reliability (SC-14).
  - **Acceptance criteria**: Partial reliability against M, reliability-corrected agreements and subset-redraw intervals are reported, and the abstract magnitude claim is consistent with them.
- **R8**: Resolve C1 (DA-CRITICAL).
  - **Acceptance criteria**: The headline thesis is limited to indices for which the evidence shows non-identification, and the definitional component is labelled as such.

### Suggested Revisions (Should Fix / Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Present Theorem 2 as an interpretive lens built on an elementary Isserlis identity | SC-3 | minor | `text: Appendix A, Proof of Theorem 2 "Isserlis’ theorem (Isserlis, 1918) gives"` | EIC 4 | EIC W3 | consider | sentence: §1 Contribution 1 | interpretive_ambiguity_remains → claim: Contribution 1 |
| S2 | Use one consistent experiment-labelling scheme across Table 1, Table 3, §4.3 and §5 | SC-8 | minor (EIC, R1); major via R2 W5 bundle | `text: Table 3 caption "Table 3: E2b (DINOv2-B, ﬁxed weight decay)"` | EIC 5; R1 3; R2 4 | EIC W7; R1 W10; R2 W5 | should_fix | section: Table 1, Table 3, §4.3, §5 | reader_traceability_reduced → table: Table 1 |
| S3 | Make the Table 1 d_eff outcome agree with the text, and define "Spec" and "partly" or state the frozen criterion | SC-9 | minor (EIC); major via R1 W4 / R2 W5 bundles | `text: §4.4 "this pre-registered prediction"` | EIC 4; R1 4; R2 4 | EIC W8; R1 W4; R2 W5 | should_fix | section: Table 1, §4.4 | reporting_requirement_unmet → table: Table 1 row Spec |
| S4 | Complete the human line-by-line proof check and remove or restate the forward-looking sentence in the AI-use statement | SC-10 | minor | `text: AI use statement "must be checked line by line by the"` | EIC 4; R1 5 | EIC W9; R1 W12 (DA minor) | should_fix | sentence: AI-use statement | editorial_conformance_unmet → section: AI-use statement |
| S5 | Move the full index comparison (Table 5) into the main text and enlarge Figures 2 and 3 | SC-11 | minor | `text: §4.4 "with item-level EU agreement (Figure 3, Table 5 in the"` | EIC 3; R1 5 | EIC W10; R1 W6 | should_fix | section: §4.4, Figures 2–3 | reader_traceability_reduced → table: Table 5 |
| S6 | Add a second soft-label dataset or modality, or limit the practical recommendations to the tested setting | SC-12 | minor | `text: §5 Limitations "We study frozen encoders and linear probes on one"` | EIC 3 | EIC W11 | consider | new_data: §4 / section: §5 | claim_scope_unsupported → claim: §5 Recommendations |
| S7 | State the Prop. 3(ii) normalisation precisely (Σr²/√(d_A d_B)) and relate it to R²_CCA | SC-13 | minor `[SEVERITY-SOURCE: letter-fallback]` for R1 | `equation: Proposition 3(ii), Section 3` | R1 `[CONFIDENCE-SOURCE: report-level]` n/a; R2 4 | R1 proof-audit summary; R2 W7 | should_fix | sentence: §1 Contribution 1, §3, §4.4 | interpretive_ambiguity_remains → claim: Prop. 3(ii) label |
| S8 | Restate how informative E2a was (unreachable given the 0.56 floor), and give the frozen decision rules for E2b and S or relabel those verdicts as post hoc | SC-15 | major | `table: Table 1, row E2a` | R1 4 | R1 W4 | should_fix | section: Table 1, §4 | interpretive_ambiguity_remains → table: Table 1 |
| S9 | Correct "CKA is the weakest of the indices" (linear predictivity is lower in 4 of 5 columns) | SC-16 | minor | `text: §4.4 "the weakest of the indices we tested"` | R1 5 | R1 W6 | should_fix | sentence: §4.4 | claim_scope_unsupported → claim: §4.4 CKA weakest |
| S10 | Address Lemma 9 being applied outside its assumptions (case-resampling softmax heads), and define the six L9 units and how ties were scored | SC-17 | minor | `text: §4.3 "Laplace EU under tail reshaping; this held in 6 of 6"` | R1 4 | R1 W7 (DA minor) | should_fix | section: §4.3, Table 1 row L9 | interpretive_ambiguity_remains → table: Table 1 row L9 |
| S11 | Attribute the data-fraction E2 results to the data term rather than Corollary 8, and state whether weight decay was re-selected for subset heads | SC-18 | minor | `equation: Appendix B, Heads objective` | R1 3 | R1 W8 | should_fix | section: §4.2, Appendix B | interpretive_ambiguity_remains → claim: E2 mechanism |
| S12 | Report the promised stratified (human-entropy decile) analysis and the partial AU values | SC-19 | minor | `absence: §4.2, Table 2, Table 4` | R1 4 | R1 W9 | should_fix | re_analysis: §4.2, Table 2/4 | reporting_requirement_unmet → table: Table 2 |
| S13 | Give the full pre-registration hash and a dated external record | SC-20 | minor | `text: Appendix B, Pre-registration "its SHA-256 is 05aa7d680e7d7f11"` | R1 3 | R1 W10 | consider | sentence: Appendix B | method_reproducibility_unresolved → section: Appendix B |
| S14 | Justify evaluating Sρ on training ∪ test inputs on its own terms (e.g. a training-only ablation) rather than via Prop. 10 | SC-21 | minor | `text: §3, after Proposition 10 "points the ratio reached ≈5 (Appendix A). This is why"` | R1 4 | R1 W11 | consider | re_analysis: §3 / §4.4 | interpretive_ambiguity_remains → claim: Sρ input set |
| S15 | Correct the Figure 1(b) caption to the exact Corollary 4 expression with ρ | SC-22 | minor | `figure: Figure 1(b) caption and §4.1 construction with k = 9, m = 200` | R1 4 | R1 W13 | should_fix | sentence: Figure 1 caption | editorial_conformance_unmet → figure: Figure 1(b) |
| S16 | State that the finding reverses Kornblith et al.'s (2019) CKA-over-CCA preference for a different target, and explain why | SC-23 | major | `table: Table 5 (Supplement C), rows "linear predictivity", "S_rho to 0" and "S_rho (head's rho)", column "width partial"` | R2 4 | R2 W1 | should_fix | section: §2 / §5 | claim_scope_unsupported → claim: positioning vs CCA/CKA literature |
| S17 | Credit Ding et al. (2021) and Davari et al. (2023) as functional precedents for CKA's tail insensitivity | SC-24 | major | `text: Section 2 "Prior work examined the statistical reliability of these measures (Ding et al., 2021; Davari et al., 2023)"` | R2 4 | R2 W2 | should_fix | section: §2, §3 | claim_scope_unsupported → claim: novelty of tail argument |
| S18 | Engage with AU/EU disentanglement benchmarking (e.g. Mucsanyi, Kirchhof and Oh 2024) when interpreting the empirical half | SC-25 | major | `text: Section 4.4 "Sρ predicts AU agreement as well as EU agreement (0.92)"` | R2 3 | R2 W3 | should_fix | section: §2, §4.4, §5 | interpretive_ambiguity_remains → claim: empirical EU estimators |
| S19 | Credit the ridge sandwich covariance and randomized-prior work, and keep only the w⁴ corollary as the contribution | SC-26 | minor | `equation: Lemma 9, Section 3` | R2 4 | R2 W4 | should_fix | sentence: §1 Contribution 3, §3 | claim_scope_unsupported → claim: Contribution 3 |
| S20 | Present PRH as a hypothesis, justify the non-standard encoder taps, and scope the "coexist" sentence to the tested regime | SC-27 | minor | `text: Section 1 "The evidence for this convergence is kernel alignment"` | R2 4 | R2 W6 | consider | section: §1, §4, §5 | claim_scope_unsupported → claim: §5 PRH coexistence |
| S21 | Cite the Bayesian last-layer, neural-linear and distance-aware uncertainty lineage | SC-28 | minor | `absence: Section 2 Heads and epistemic uncertainty paragraph` | R2 4 | R2 W9 | consider | section: §2 | reader_traceability_reduced → section: §2 |
| S22 | Soften "rule out" to "reduce", and add a covariate-reliability sensitivity analysis (disattenuated, Dirichlet-posterior or both split-half entropies) | SC-29 | major | `text: §2 EU agreement "Partial and stratiﬁed variants rule out the trivial explanation that both models are uncertain on intrinsically ambiguous images."` | R3 4 | R3 W1 | should_fix | re_analysis: §2, §4.2 | claim_scope_unsupported → claim: ambiguity explanation ruled out |
| S23 | Report the item-level correlations between EU summaries, AU and confidence per encoder, and the closed-form u(x) vs softmax-estimator rank agreement. Say which findings remain EU-specific. | SC-30 | major | `table: Table 2 (EU width and AU columns, every row)` | R3 4 | R3 W2 | should_fix | re_analysis: §4.2, Table 2 | evidence_gap_remains → claim: EU-specific empirical findings |
| S24 | Add a decision-level analysis (e.g. top-k acquisition-set overlap on existing heads, or a small active-learning comparison) | SC-31 | major | `text: §1 Introduction "a cheap, label-free tool for transferring uncertainty estimates, selecting encoders for active learning"` | R3 4 | R3 W3 (DA M5, Unexamined Premise) | should_fix | re_analysis: §4 (new subsection) | evidence_gap_remains → claim: §5 Recommendation (i) practical value |
| S25 | Explain the asymmetry in what the partial control removes for AU vs EU, or add a confidence-only AU comparison | SC-32 | minor | `text: §4.2 "the empirical conclusion is that identical features determine neither the residual EU ranking nor the residual AU ranking of a head."` | R3 3 | R3 W4 | consider | sentence: §4.2 | interpretive_ambiguity_remains → claim: E2b AU conclusion |
| S26 | Label the AU–human stability finding exploratory | SC-33 | minor | `text: §4.2 "What is stable is AU’s relation to the human target"` | R3 4 | R3 W5 | should_fix | sentence: §4.2 | reporting_requirement_unmet → claim: AU stability |
| S27 | Make Recommendation (ii) operational for softmax heads (which prior-strength quantity to report) | SC-34 | minor | `text: §5 Recommendations "Report EU together with the head’s prior strength relative to the feature scale."` | R3 3 | R3 W6 | consider | sentence: §5 | interpretive_ambiguity_remains → claim: Recommendation (ii) |
| S28 | Add a distribution-shift scope condition or a small shifted-data check | SC-35 | minor | `absence: §4 Experiments and §5 Limitations` | R3 3 | R3 W8 | consider | sentence: §5 Limitations (or new_data: §4) | claim_scope_unsupported → claim: blind-spots framing |
| S29 | Add one paragraph on implications for RSA-style brain/model comparison and learning from annotator disagreement | SC-36 | minor | `absence: §2 Setup and related work and §5 Discussion` | R3 4 | R3 W9 | consider | section: §5 | reader_traceability_reduced → section: §5 |
| S30 | Qualify the scale-invariance pillar (Prop. 5): the discrepancy vanishes when weight decay is re-tuned by validation (max deviation 0.086) | — (DA MAJOR M2) | major | `text: §4.3 "largest deviation of any EU rank agreement or mean ratio from the untransformed head is 0.086"` | DA 4 | DA M2 | should_fix | sentence: abstract, §5 Recommendations | claim_scope_unsupported → claim: invariance pillar |
| S31 | Identify an actual proponent of using alignment as an EU proxy, or reframe the motivation | — (DA MAJOR M6) | major | `absence: §1 Introduction and §2 Setup and Related Work` | DA 3 | DA M6 | consider | section: §1 | interpretive_ambiguity_remains → section: §1 motivation |

Obligation tally: must_fix R1–R8 (8). should_fix S2, S3, S4, S5, S7, S8, S9, S10, S11, S12, S15, S16, S17, S18, S19, S22, S23, S24, S26, S30 (20). consider S1, S6, S13, S14, S20, S21, S25, S27, S28, S29, S31 (11). Total 39.

### Source-Traceability Checklist

Items follow source order. No work order is implied. Author triage is collected later as a separate, explicit sidecar.

- [ ] R1 — obligation `must_fix`: Reframe EU-specificity (SC-1)
- [ ] R2 — obligation `must_fix`: Position Sρ against ridge, decoding and shape-metric families (SC-2)
- [ ] S1 — obligation `consider`: Present Theorem 2 as an interpretive lens (SC-3)
- [ ] R3 — obligation `must_fix`: Test or narrow the head's-ρ claim (SC-4)
- [ ] R4 — obligation `must_fix`: Report the linear-predictivity tie and all indices (SC-5)
- [ ] R5 — obligation `must_fix`: Inference for 0.83 vs 0.64 (SC-6)
- [ ] R6 — obligation `must_fix`: Reconcile §4.3 and Table 3 (SC-7)
- [ ] S2 — obligation `should_fix`: Consistent experiment labels (SC-8)
- [ ] S3 — obligation `should_fix`: Table 1 d_eff outcome (SC-9)
- [ ] S4 — obligation `should_fix`: AI-use proof statement (SC-10)
- [ ] S5 — obligation `should_fix`: Table 5 into main text (SC-11)
- [ ] S6 — obligation `consider`: Second dataset or scoped recommendations (SC-12)
- [ ] S7 — obligation `should_fix`: Prop. 3(ii) normalisation (SC-13)
- [ ] R7 — obligation `must_fix`: Partial-metric reliability and redraws (SC-14)
- [ ] S8 — obligation `should_fix`: E2a informativeness and frozen rules (SC-15)
- [ ] S9 — obligation `should_fix`: "CKA weakest" correction (SC-16)
- [ ] S10 — obligation `should_fix`: Lemma 9 assumptions and L9 count (SC-17)
- [ ] S11 — obligation `should_fix`: Data-fraction pairs vs Corollary 8 (SC-18)
- [ ] S12 — obligation `should_fix`: Stratified analysis and partial AU (SC-19)
- [ ] S13 — obligation `consider`: Pre-registration verifiability (SC-20)
- [ ] S14 — obligation `consider`: Sρ input-set justification (SC-21)
- [ ] S15 — obligation `should_fix`: Figure 1(b) caption (SC-22)
- [ ] S16 — obligation `should_fix`: Kornblith reversal (SC-23)
- [ ] S17 — obligation `should_fix`: Ding/Davari credit (SC-24)
- [ ] S18 — obligation `should_fix`: Disentanglement literature (SC-25)
- [ ] S19 — obligation `should_fix`: Lemma 9 credit (SC-26)
- [ ] S20 — obligation `consider`: PRH framing and taps (SC-27)
- [ ] S21 — obligation `consider`: Bayesian last-layer lineage (SC-28)
- [ ] S22 — obligation `should_fix`: "Rule out" plus covariate sensitivity (SC-29)
- [ ] S23 — obligation `should_fix`: EU/AU/confidence overlap reporting (SC-30)
- [ ] S24 — obligation `should_fix`: Decision-level analysis (SC-31)
- [ ] S25 — obligation `consider`: AU residual asymmetry (SC-32)
- [ ] S26 — obligation `should_fix`: Label AU stability exploratory (SC-33)
- [ ] S27 — obligation `consider`: Operational Recommendation (ii) (SC-34)
- [ ] S28 — obligation `consider`: Distribution-shift scope (SC-35)
- [ ] S29 — obligation `consider`: Cross-field implications (SC-36)
- [ ] R8 — obligation `must_fix`: Resolve DA C1 (narrow the thesis)
- [ ] S30 — obligation `should_fix`: Qualify the Prop. 5 pillar under re-tuning (DA M2)
- [ ] S31 — obligation `consider`: Name a proponent or reframe the motivation (DA M6)

### Response Letter Template

Respond to every item above (R1–R8 and S1–S31) using `templates/revision_response_template.md`. For each item, give the action taken, the exact location of the change, and, for any item you decline, a reasoned justification. A declined item stays visible as unresolved at re-review. The DA's C1 must be answered explicitly whichever remedy you choose.

---

## Part 3: Appendix

### Step 1a — Reviewer Summary Matrix

The sprint cards carry dimension scores, not overall recommendations. Confidence is reported per finding. It is a self-report and is never aggregated.

| | EIC (Journal-Fit) | R1 (Methodology) | R2 (Domain) | R3 (Perspective) | DA |
|---|---|---|---|---|---|
| Assessed scores | D5 warn; D6 block (repairable) | D1 block (repairable); D3 warn | D2 warn | D4 warn | D3 block (repairable) |
| Overall recommendation | not stated (sprint card) | not stated | not stated | not stated | N/A (findings only) |
| Competence disclosure | Senior AC, Bayesian DL/UQ; no proof-level checking | Statistician; full proof audit; Table 2 recomputed from Table 4 | Representation similarity plus uncertainty | Computational cognitive science, human label uncertainty | Kernel alignment, Bayesian linear regression |
| Key strengths | Pre-registration; Corollary 4 mechanism; identical-features design; candid limitations | Proofs correct; falsifications reported; Table 2/4 consistent | Falsifications reported; correct CKA and Isserlis; honest ridge attribution; AU ceiling | Reliability ceilings; falsifications; identical-features design | Formal core sound; pre-registration candid |
| Weaknesses | 11 (W1–W11) | 13 (W1–W13) | 9 (W1–W9) | 9 (W1–W9) | 1 CRITICAL, 7 MAJOR, 4 minor points |

### Step 1b — Weakness Sub-Claim Inventory

Position codes: R = raised, C = corroborated, — = not-mentioned (silence, not opposition), D = disputed. Severity and confidence are transported from the cards.

| SC | Sub-claim (parent) | EIC | R1 | R2 | R3 | Disposition | Severity | DA |
|---|---|---|---|---|---|---|---|---|
| SC-1 | Empirical results not EU-specific; framing overstated | R (W1, major, 4) | — | R (W3, major, 3) | R (W2, major, 4) | CONSENSUS-3 (R1 silent) | major | M1 |
| SC-2 | Sρ novelty not positioned vs ridge, prediction and decoding families | R (W2, major, 3) | — | C (W8, major, 3) | — | corroborated 2/4 | major | — |
| SC-3 | Theorem 2 elementary; present as lens | R (W3, minor, 4) | — | — | — | single-reviewer | minor | — |
| SC-4 | Head's-ρ Sρ ≈ Sρ→0; Rec. (i) unsupported | R (W4, major, 4) | C (W5, major, 4) | C (W1, major, 4) | D-severity (W7, minor, 4) | SPLIT, arbitrated major | major | M4 |
| SC-5 | Linear-predictivity tie unreported | R (W4, major, 4) | C (W5, major, 4) | C (W1, major, 4) | — | CONSENSUS-3 (R3 silent) | major | — |
| SC-6 | Headline comparison lacks inference; dependent pairs | R (W5, major, 3) | C (W2, major, 4) | — | — | corroborated 2/4 | major | M7 |
| SC-7 | §4.3 prose vs Table 3 | R (W6, major, 5) | C (W1, major, 5) | C (W5, major, 4) | — | CONSENSUS-3 (R3 silent) | major | minor pt. |
| SC-8 | E2b label collision | R (W7, minor, 5) | C (W10, minor, 3) | C (W5 bundle, major, 4) | — | CONSENSUS-3 (R3 silent) | minor/major (bundle-inherited) | minor pt. |
| SC-9 | Table 1 d_eff "partly" vs "failed"; undefined Spec | R (W8, minor, 4) | C (W4 bundle, major, 4) | C (W5 bundle, major, 4) | — | CONSENSUS-3 (R3 silent) | minor/major (bundle-inherited) | — |
| SC-10 | AI-use statement: proofs unchecked | R (W9, minor, 4) | C (W12, minor, 5) | — | — | corroborated 2/4 | minor | minor pt. |
| SC-11 | Full Table 5 should be in main text | R (W10, minor, 3) | C (W6, minor, 5) | — | — | corroborated 2/4 | minor | — |
| SC-12 | Narrow empirical scope | R (W11, minor, 3) | — | — | — | single-reviewer | minor | — |
| SC-13 | Prop. 3(ii) "mean" mislabel when d_A ≠ d_B | — | R (proof audit; `[SEVERITY-SOURCE: letter-fallback]` minor; `[CONFIDENCE-SOURCE: report-level]` none given) | C (W7, minor, 4) | — | corroborated 2/4 | minor | — |
| SC-14 | Partial-metric reliability, M = 50, no disattenuation, single draw | — | R (W3, major, 4) | — | — | single-reviewer | major | M3 |
| SC-15 | E2a unreachable; S and E2b lack thresholds | — | R (W4, major, 4) | — | — | single-reviewer | major | — |
| SC-16 | "CKA weakest" contradicted by Table 5 | — | R (W6, minor, 5) | — | — | single-reviewer | minor | — |
| SC-17 | Lemma 9 outside assumptions; 6/6 units | — | R (W7, minor, 4) | — | — | single-reviewer | minor | minor pt. |
| SC-18 | Data-fraction pairs do not instantiate Cor. 8 | — | R (W8, minor, 3) | — | — | single-reviewer | minor | — |
| SC-19 | Stratified analysis and partial AU missing | — | R (W9, minor, 4) | — | — | single-reviewer | minor | — |
| SC-20 | Pre-registration not externally verifiable | — | R (W10, minor, 3) | — | — | single-reviewer | minor | — |
| SC-21 | Prop. 10 does not justify train ∪ test Sρ | — | R (W11, minor, 4) | — | — | single-reviewer | minor | — |
| SC-22 | Fig. 1(b) caption mismatch | — | R (W13, minor, 4) | — | — | single-reviewer | minor | — |
| SC-23 | Kornblith CKA-over-CCA reversal not discussed | — | — | R (W1, major, 4) | — | single-reviewer | major | — |
| SC-24 | Ding/Davari mischaracterised | — | — | R (W2, major, 4) | — | single-reviewer | major | — |
| SC-25 | Disentanglement literature missing | — | — | R (W3, major, 3) | — | single-reviewer | major | — |
| SC-26 | Lemma 9 is textbook; credit it | — | — | R (W4, minor, 4) | — | single-reviewer | minor | — |
| SC-27 | PRH framing and encoder taps | — | — | R (W6, minor, 4) | — | single-reviewer | minor | — |
| SC-28 | Bayesian last-layer lineage under-cited | — | — | R (W9, minor, 4) | — | single-reviewer | minor | — |
| SC-29 | Unreliable covariate does not "rule out" ambiguity | — | — | — | R (W1, major, 4) | single-reviewer | major | — |
| SC-30 | EU/AU/confidence overlap and closed-form vs softmax agreement unreported | — | — | — | R (W2, major, 4) | single-reviewer | major | — |
| SC-31 | No decision-level test of practical uses | — | — | — | R (W3, major, 4) | single-reviewer | major | M5 |
| SC-32 | AU residual asymmetry | — | — | — | R (W4, minor, 3) | single-reviewer | minor | — |
| SC-33 | AU stability not marked exploratory | — | — | — | R (W5, minor, 4) | single-reviewer | minor | — |
| SC-34 | Rec. (ii) underspecified | — | — | — | R (W6, minor, 3) | single-reviewer | minor | — |
| SC-35 | No out-of-distribution evaluation or scope | — | — | — | R (W8, minor, 3) | single-reviewer | minor | — |
| SC-36 | Cross-field implications not drawn | — | — | — | R (W9, minor, 4) | single-reviewer | minor | — |

**Surface-form parity (#216).** Every sub-claim was judged against manuscript locations named in the cards, not against how the cards were worded. No sub-claim was marked unevaluable. For C1 and SC-4, the key numbers (Table 5, §4.4 d_eff/d > 0.98) were checked directly in the manuscript.

**Process note.** The round directory has phase-2 lint records for the EIC, methodology, domain and DA cards, but none for the perspective card. The dispatching layer declared all five cards usable, and I synthesised them as given. I did not edit or supplement any card.

### Reviewer Report Summaries

- **Journal-Fit Reviewer (EIC)**: D5 warn, D6 block (repairable). Key point: the EU-specific framing and the unestablished novelty of Sρ overstate the contribution, but a narrower paper on non-identification plus the Gaussian identity can be salvaged.
- **Reviewer 1 (Methodology)**: D1 block (repairable), D3 warn. Key point: the proofs are correct, but the headline comparisons lack inference, the partial metric's reliability is uncharacterised, and §4.3 conflicts with Table 3.
- **Reviewer 2 (Domain)**: D2 warn. Key point: domain content is accurate, but positioning undersells prior work (CCA family, Ding/Davari, decoding and shape metrics, ridge sampling theory), and the empirical advantage is indistinguishable from CCA and linear predictivity.
- **Reviewer 3 (Perspective)**: D4 warn. Key point: the cross-field non-identification message holds, but partialling an unreliable covariate, near-collinear EU/AU estimators and untested decision-level uses leave adjacent-field readers with gaps.
- **Devil's Advocate**: D3 block (repairable). Recommendation N/A (findings only). Key challenge (C1, validated): the title thesis is either definitional or contradicted by Table 5, and the supported claim is that CKA-type indices are poor EU proxies.

## Attachment: Acronym Check (advisory, #849)

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 1218: ViT (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 9 | Defined after first use | CKA | 45 |
| Body | 31 | Not defined | CIFAR | 8 |
| Body | 227 | Not defined | MB | 6 |
| Body | 234 | Not defined | dAdB | 1 |
| Body | 381 | Not defined | DINOv2 | 6 |
| Body | 382 | Not defined | CLIP | 1 |
| Body | 385 | Not defined | MAE | 1 |
| Body | 396 | Not defined | NLL | 2 |
| Body | 581 | Not defined | MI | 3 |
| Body | 605 | Not defined | mKNN | 2 |
| Body | 662 | Not defined | NN | 6 |
| Body | 827 | Not defined | DCIC | 2 |
| Body | 834 | Not defined | AI | 6 |
| Body | 855 | Not defined | ReQ | 1 |
| Body | 871 | Not defined | CVF | 5 |
| Body | 871 | Not defined | IEEE | 5 |
| Body | 890 | Defined again | ICLR | 2 |
| Body | 896 | Defined again | CVPR | 4 |
| Body | 923 | Defined again | ICML | 3 |
| Body | 944 | Defined again | CVPR | 4 |
| Body | 955 | Defined again | CVPR | 4 |
| Body | 967 | Defined again | ICML | 3 |
| Body | 976 | Not defined | SVCCA | 1 |
| Body | 981 | Not defined | MIT | 1 |
| Body | 1062 | Not defined | GPU | 1 |
| Body | 1100 | Not defined | PROOFS | 1 |
| Body | 1105 | Not defined | AB | 11 |
| Body | 1107 | Not defined | ABQ | 2 |
| Body | 1110 | Not defined | BI | 1 |
| Body | 1179 | Not defined | CKA2 | 2 |
| Body | 1207 | Not defined | GT | 1 |
| Body | 1207 | Not defined | GZT | 1 |
| Body | 1213 | Not defined | PREREG | 1 |
| Body | 1214 | Not defined | SHA | 1 |
| Body | 1222 | Not defined | CEi | 1 |
| Body | 1224 | Not defined | BFGS | 1 |
| Body | 1228 | Not defined | GGN | 1 |
| Body | 1228 | Not defined | MAP | 1 |
| Body | 1240 | Not defined | GB | 1 |
| Body | 1240 | Not defined | MPS | 1 |
| Body | 1241 | Not defined | CPU | 1 |
| Body | 1243 | Not defined | BY | 1 |
| Body | 1243 | Not defined | CC | 1 |
| Body | 1243 | Not defined | NC | 1 |
| Body | 1243 | Not defined | SA | 1 |
