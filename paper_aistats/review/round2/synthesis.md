# Editorial Decision Package (Round 2)

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` stays unavailable until a closed profile artifact and replay validator bind the exact target fields to the completed panel's `execution_topology_sha256`. Every seat is also `NOT_CALIBRATED`.

## Manuscript Information
- **Title**: When Do Aligned Representations Agree on What They Do Not Know?
- **Manuscript ID**: not provided (panel `uq-aistats2027-round2`)
- **Submission date**: not provided
- **Decision date**: 2026-10-03
- **Review round**: Round 2 (a fresh full panel on the revised manuscript)
- **Contract**: `reviewer/reviewer_full/v2` (mode `reviewer_full`, panel_size 5)
- **Criteria binding**: `criteria_binding_unavailable`. All five Phase 1 cards (EIC, R1, R2, R3, DA) disclose `criteria_binding_unavailable`. This package therefore makes no venue-criteria or venue-alignment claim. The user-named target (AISTATS 2027) is not treated as a bound criteria manifest.
- **Base draft SHA-256** (`manuscript.md`): `df9b7b19be83a4e10397654fc70374d973f9232c51385f85162e5f604263f05a`

---

## Review Panel Provenance (#540/#740)

- **Typed artifact**: `paper_aistats/review/round2/provenance.json` (`review-panel-provenance/1.0`). The dispatching layer replay-validated it as PASS.
- **Artifact SHA-256** (raw bytes, computed by the synthesizer): `d5c6a99f08fc2896b1c9a670b1835c39b81c078019911823640113c284a1981a`
- **Panel ID**: `uq-aistats2027-round2`
- **Contract SHA-256** (as recorded in artifact): `e9712090d2469fea15a37b8e22d4e137afbcb2bf38d5789939c5df56738ef7af`
- **Normalized manifest SHA-256**: `a8cd236303a1f7828e7a7aa8e9b7b3f7add5988cad924c1fc7c3ede2432ab811`
- **Execution topology SHA-256**: `c259b3f71e07a5edf63a2840650cfafb0d9380f68f5533b18fdd78f82f6e2603`
- **Fresh-context scope**: `within_panel_attempt_only`. This scope does not compare retries or earlier rounds.

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | subagent-ad3c74aefa4785313 | false | claude-opus-5-5 | anthropic | null |
| R1 | methodology | model | subagent-a2ad56387f7c71705 | false | claude-opus-5-5 | anthropic | null |
| R2 | domain | model | subagent-a7d338913dcdb8d20 | false | claude-opus-5-5 | anthropic | null |
| R3 | perspective | model | subagent-aa8b0b9be17c7e0e7 | false | claude-opus-5-5 | anthropic | null |
| DA | da | model | subagent-a9cae7bdf1053354e | false | claude-opus-5-5 | anthropic | null |

| Provenance axis | Status (`true` / `false` / `unknown`) |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation (`fresh_context`) | true |
| Blind to peer outputs | true |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: not computed (`independence_claim: not_computed_from_personas`). Persona or role diversity establishes only `role_separated`. The five seats are **not** described as independent reviewers.
- **Correlated-error disclosure** (required; reason `same_model_family`): "All model-executed review seats used one model family; role separation does not remove correlated-error risk."
- **DA seat execution note**: the first DA Phase 2 card failed conformance: `[PROTOCOL-VIOLATION: reviewer=da, contract=reviewer/reviewer_full/v2, phase2_lint_failed=DA-TABLE-PARSE] attempt 1 unusable; seat re-run from Phase 1 in fresh contexts` (source: `da.attempt_log.txt`). The failed attempt is excluded and was not read for this synthesis. The DA card used here is the re-run (`da.phase1.md` → `da.phase2.md`, both lint PASS). The `fresh_context` axis has the fixed scope `within_panel_attempt_only`. It does not show that the re-run's context was new relative to the history of the failed attempt.
- **Panel completeness**: all five Phase 2 cards passed conformance lint. There is no `[PANEL-SHRUNK]` condition.
- **Methodology receipt attestation**: the R1 card takes the `no_recomputable_statistics` path. The checker confirmed only that the declaration exists: `[RECEIPT-ATTESTATION: declaration-only — applicability not machine-verified; adjudication judges the attestation]`. Editor's judgement: the declaration is plausible. A pattern scan of the manuscript for t, F, or χ² statistics with df, for p-values, and for mean±SD reporting returned no matches. This is consistent with the card's statement that only rank and partial correlations, percentile intervals, ratios and Shapley shares are reported. The scan is a spot check, not full verification, so the attestation is accepted as declaration-level only.
- **Cross-model decision check (#518)**: `ARS_CROSS_MODEL` is not set, so the check did not run. Behaviour is unchanged.

---

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting the revised manuscript "When Do Aligned Representations Agree on What They Do Not Know?". It was reviewed by five role-separated seats: the Journal-Fit Reviewer (EIC), R1 Methodology, R2 Domain, R3 Perspective, and the Devil's Advocate (DA). Their execution provenance is reported above and is not reduced to a binary independence claim. All five seats ran on one model family.

### Decision: Major Revision

### Sprint-contract audit (v3.6.2 mechanical protocol)

**Step 0: binding check.** This is an explicitly unbound run. All five cards disclose `criteria_binding_unavailable`. No venue-alignment claim is made.

**Step 1: role-scoped scoring matrix.** Only eligible seats are counted. Ineligible `not_assessed` values are excluded from numerator and denominator.

| Dim | Name | Priority | Eligible roles | Assessed eligible scores | Fatal declared | Verdict (worst) |
|---|---|---|---|---|---|---|
| D1 | methodology_rigor | mandatory | methodology | R1 = warn | no | warn |
| D2 | domain_accuracy | mandatory | domain | R2 = warn | no | warn |
| D3 | argumentative_coherence | mandatory | da, methodology | R1 = warn; DA = warn | no | warn |
| D4 | cross_disciplinary_relevance | high | perspective | R3 = warn | no | warn |
| D5 | writing_and_structure | normal | eic | EIC = warn | no | warn |
| D6 | venue_fit_and_contribution | mandatory | eic | EIC = warn | no | warn |

Every dimension has at least one assessed eligible seat, so there is no `[DIMENSION-UNASSESSED]`.

**Step 2: failure conditions.** All expressions parse within the closed vocabulary. There is no `[EXPRESSION-UNRECOGNISED]`.

| ID | Severity | Quantifier | Expression | Evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 95 | any | any mandatory dimension has a fatal block | No seat declared a fatal block on D1, D2, D3 or D6 | false |
| F2 | 90 | any | any mandatory dimension scores 'block' | Mandatory verdicts are all warn | false |
| F3 | 70 | majority | two or more mandatory dimensions score 'warn' or worse | D1 (n=1, owner R1 warn → true); D2 (n=1, owner R2 warn → true); D3 (n=2, both R1 and DA warn → true); D6 (n=1, owner EIC warn → true). 4 ≥ 2 | **true** |
| F4 | 60 | any | any high-priority dimension scores 'block' | D4 = warn | false |
| F5 | 40 | any | any dimension scores 'warn' or worse | D1–D6 all warn | **true** |
| F0 | 10 | all | every dimension scores 'pass' | No dimension passes | false |

**Step 3: precedence.** The fired conditions are F3 (70) and F5 (40). F3 has the highest severity, so its action applies: `editorial_decision=major_revision`. F3's action is not softened.

```
dimension_verdicts: [D1=warn, D2=warn, D3=warn, D4=warn, D5=warn, D6=warn]
fired_conditions: [F3, F5]
da_critical_adjudications: []
editorial_decision=major_revision
```

The DA card's `#### CRITICAL` table has no rows, so there are no DA CRITICAL IDs. No `C<n> rejection rationale:` line is required, and no `[DA-CRITICAL-VS-ACCEPT]` marker applies because the decision is not accept.

### DA CRITICAL adjudication

No CRITICAL items were raised (empty CRITICAL table in `da.phase2.md`). For visibility, the DA's five **MAJOR** items (M1–M5, all on D3) are listed below with their corroboration status. They do not take part in consensus counting and are not CRITICAL adjudications.

| DA item | Gist (from DA card) | Corroborated by non-DA seats? | Roadmap disposition |
|---|---|---|---|
| M1 | CKA is said to fail by discarding scale, yet the cross-encoder winner Sρ→0 (CCA) is invariant to scale and every invertible map, and mutual k-NN shares CKA's cQ invariance yet reaches 0.90 | Partly. EIC W5 and R2 W1 raise the mutual k-NN part (→ REV-05). The CCA-invariance tension is DA-only among seats | REV-05 (corroborating); REV-40 (DA-only part) |
| M2 | Theorem 2 concerns label-free u(x); the cross-encoder test uses partial bootstrap width (−0.999 with confidence, −0.08 to 0.31 with u(x)); indices predict AU equally (0.92) | Yes. EIC W3 (→ REV-03); R1 W3 (→ REV-15) | Corroborating source on REV-03 and REV-15 |
| M3 | The scale finding is headlined as "robust" but comes from rescaling at frozen weight decay and vanishes on re-tuning | Yes. EIC W4 and R2 W10 (→ REV-04) | Corroborating source on REV-04 |
| M4 | The second mechanism is not operative at the selected weight decays (d_eff/d > 0.98); "explains" overreaches, "is compatible with" is supported | Partly. R1 W3 raises the ρ-mismatch (→ REV-15). The d_eff/d regime and the "explains" wording are DA-only | REV-41 (DA-only part) |
| M5 | The "beyond estimator noise" claim on identical features rests on M = 50 reliability corrections. Corrected AU degrades equally, so the result is not EU-specific | Yes. R1 W2 and R3 W1 (→ REV-13, REV-14) | Corroborating source on REV-13 and REV-14 |

### Consensus Analysis

Consensus is computed per sub-claim over the four non-DA seats (EIC, R1, R2, R3). `not-mentioned` counts as silence, not agreement. The full inventory is in Part 4.

#### Points of Agreement (Consensus)
- **[CONSENSUS-4]**: none.
- **[CONSENSUS-3]**: none.

Because each seat scores only its owned dimensions, the four seats covered largely different ground. No sub-claim reached three or more seats.

#### Corroborated findings (2/4, no conflict; below the consensus bar)
- **SC-04: The scale-as-prior result is over-presented as a "robust" headline.** Raised by EIC W4 and R2 W10, both Minor. R1 and R3 are silent. DA M3 points the same way (not counted). Both seats' remedies are compatible: present it as a diagnostic caveat, namely that CKA cannot tell whether priors match feature scales.
- **SC-06: Missing literature on error consistency and functional or prediction-level similarity.** Raised by EIC W6 and R2 W4, both Minor. R1 and R3 are silent.
- **SC-08: Table 1 credits E2-a as "supported" although its falsifier could not fire, and the verdict vocabulary is inconsistent.** Raised by EIC W8 and R1 W8 (sub 1), both Minor. R2 and R3 are silent.
- **SC-12: Residual ("partial") EU after confidence partialling has unclear construct validity.** The residual holds about 0.2–0.6% of rank variance and has no external validation. Raised by R1 W2 (sub 1) and R3 W1 (sub 1), both Major. EIC and R2 are silent.
- **SC-13: The Discussion's "robust" claim that identical features leave residual EU dependent on training data beyond noise is overclaimed.** Raised by R1 W2 (sub 2) and R3 W1 (sub 2), both Major. EIC and R2 are silent. DA M5 points the same way (not counted).
- **SC-17: The human-reliability ceiling fraction divides by r rather than √r.** Raised by R1 W4 and R3 W2, both Minor. EIC and R2 are silent. The corrected range is about 0.41–0.51, not 0.48–0.61.

All other sub-claims are single-reviewer findings (1/4). Each is retained and assessed against its anchored evidence (Part 4).

#### Points of Disagreement (SPLIT)

Both splits are **severity disagreements** (Major against Minor) on a shared sub-claim. No seat disputes that either problem exists. The Journal-Fit Reviewer is a party to both splits, so the resolution below is the synthesizer's arbitration under the evidence-first and expertise-first principles, with the record stated openly.

**Disagreement 1 (SC-02): Is the positive half of Recommendation (i) ("prefer whitened / Sρ indices") empirically established?**
- **EIC view (W2, Major, Confidence 4)**: The readership payoff is "asserted rather than demonstrated". Table 5 gives Sρ−CKA +0.19 [+0.00, +0.50] and Sρ−linear pred. −0.00 [−0.22, +0.11]. The remedy is to add encoders or a downstream decision task, or else demote the recommendation to a hypothesis.
- **R3 view (W4, Minor, Confidence 4)**: The negative half (do not use CKA to transfer EU) is well supported and the positive half is a hypothesis. The remedy is to split Recommendation (i) into a supported negative claim and a labelled conjecture, and to say that no index is EU-specific in practice.
- **Disagreement type**: severity disagreement. The remedies are compatible because EIC's fallback is R3's remedy.
- **Editor's resolution**: obligation `must_fix`. The **minimum required action** is the remedy both seats share: restate the positive recommendation as a conjecture, separated from the supported negative claim. EIC's evidence-adding route (more encoders or a downstream decision task) is offered as an author-chosen alternative that would let the positive claim stand. It is not required.
- **Rationale**: On evidence, both seats cite the same Table 5 intervals, and the manuscript itself concedes "we have not shown a practical advantage of Sρ over" linear predictivity. The problem therefore exists without dispute, and only its weight differs. On expertise, contribution significance is the EIC's owned dimension (D6) and practitioner uptake is R3's (D4). Both are within competence, and the difference reflects different stakes, not different evidence. The remedy is the same under either severity, so the severity difference does not change the required action. Both severities are carried on the roadmap row.

**Disagreement 2 (SC-05): Mutual k-NN tracks EU agreement nearly as well as Sρ but is not discussed. What follows for the paper's account of why indices succeed or fail?**
- **EIC view (W5, Minor, Confidence 3)**: Mutual k-NN shares CKA's invariances but behaves like Sρ (Table 7 BLR: 0.90 against 0.92 against CKA 0.76). This blurs the framing that CKA's invariances disqualify it. The remedy is to discuss mutual k-NN explicitly and say which mechanism the data support.
- **R1 view (W6 sub 2, Minor, Confidence 4)**: Mutual k-NN reaches 0.85–0.90, close to Sρ, and this bears on the post hoc choice of the headline column.
- **R2 view (W1 sub 2, Major, Confidence 4)**: Mutual k-NN is the main metric behind the platonic-representation convergence claim. The paper's critique applies to CKA much more than to that metric, and the Discussion and Recommendations never address mutual k-NN. The remedy is to discuss it as a separate case, theoretically, given its invariances and tail sensitivity.
- **Disagreement type**: severity disagreement (Major from R2 against Minor from EIC and R1). The remedies are compatible.
- **Editor's resolution**: obligation `must_fix`. Required action: an explicit treatment of mutual k-NN as a scale-invariant but tail-sensitive index, stating which mechanism (spectral weighting or scale) the cross-encoder data actually support.
- **Rationale**: Under expertise-first, the significance of mutual k-NN for the convergence literature is a domain matter, so R2's Major weighting governs the obligation. On evidence, all three seats read the same Table 7 values, and DA M1 independently points the same way (not counted). Because the remedies coincide, adopting R2's weighting adds no action the other seats would reject. The disagreement concerns weight only and is recorded, not erased.

No sub-claim is unresolved dissent. No existence or direction disagreement was found.

### Decision Rationale

The decision follows mechanically from the contract. All four mandatory dimensions (D1, D2, D3, D6) scored `warn` from their eligible seats. This triggers F3 (severity 70, majority quantifier, four mandatory dimensions at warn or worse) → `major_revision`. No block and no fatal block was declared, so F1 and F2 did not fire, and no DA CRITICAL issue exists.

The substance supports this outcome. Every seat that checked the theory found it correct: R1 re-derived each result in Appendix A, and R2 and the DA confirmed the identity and the limits. Every seat also credits the paper's candid pre-registration and failure reporting (EIC S1, R1 S2, R2 S4, R3 S2). The problems lie in the evidential bridge from theory to the real-data and practitioner-facing claims, not in correctness:
- the positive index recommendation rests on a five-encoder interval that touches zero (EIC W2, R3 W4), and R1 W1 shows that this interval is not a valid inferential statement at G = 5;
- the confidence-residualised "EU" construct carries a "robust" finding without validation or intervals (R1 W2, R3 W1, DA M5);
- the softmax experiments cannot isolate EU from general head agreement (EIC W3, DA M2), and the matched-ρ tests use a ρ the paper itself says is not the softmax estimators' effective prior (R1 W3);
- the platonic-representation positioning misstates which metric carries the convergence claim (R2 W1);
- the theoretical increment is judged thin for a main-track bar (EIC W1).

A stricter decision is not warranted. Every scoring seat explicitly chose warn over block, and each says the issues are repairable by rewording plus targeted analyses without a new design (R1 D1 note, R3 Review Body, EIC D6 note). A lighter decision is ruled out by F3: a minor revision would soften a fired condition's action, which the contract forbids. The revised manuscript will need re-review.

### Blocking Issues (0–3, immutable source order)

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R2 | Positive half of Recommendation (i) not empirically established | EIC (W2), R3 (W4) | `table: Table 5, rows "Sρ−CKA" (+0.19 [+0.00, +0.50]) and "Sρ−linear pred." (-0.00 [-0.22, +0.11]) in the width (partial) column` | REV-02 |
| R5 | Five-cluster percentile bootstrap is degenerate; intervals and "0.09 of resamples" are not inferential | R1 (W1) | `text: Appendix C, "keep every pair of distinct sampled encoders (with multiplicity," and "pairs; intervals are 2.5–97.5% percentiles."` | REV-11 |
| R7 | "Robust" claim that identical features leave residual EU data-dependent beyond noise is overclaimed | R1 (W2), R3 (W1); DA M5 corroborating | `text: Section 5 What the results say, "even identical features leave the residual EU ranking dependent on the"` | REV-13 |

### Round-1 item status (external verification, non-voting)

The following comes from `verification.md`, a separate manuscript-only check of the round-1 roadmap. It is **not a panel seat**. It is not counted in consensus, the scoring matrix, or the decision arithmetic, and none of its items are added to the roadmap below.
- Round-1 required items: 7 of 8 are fully addressed (R1–R6, R8). R7 (the reliability-versus-M and redraw evidence) is partially addressed: reliability versus M is shown for one encoder only, and the three redraws carry no intervals. None were rated not addressed or made worse.
- Round-1 suggested items: 22 of 31 fully addressed, 8 partially addressed, 1 not addressed (S29: RSA and annotator-disagreement discussion).
- The verification's top residual issues overlap with this panel's findings. Its R7 residual matches REV-13 and REV-14. Its scale-invariance tension matches DA M1 and REV-40. Its point on EU-specificity matches REV-03. It also lists new copy-level inconsistencies that no panel seat raised: the closed-form prose values 0.93 and 0.74 against Table 5's 0.92 and 0.76; "All indices predict AU..." against linear predictivity's AU of 0.53; Fig. 3a's 0.60 against Table 6's 0.61. Because no seat raised these, they are **not** roadmap items. The authors may wish to check them.

---

## Part 2: Revision Roadmap

**Editorial obligation rule used.**
- Major-severity findings, and splits with a Major side, are `must_fix`.
- Minor findings that correct a stated claim, number, attribution or verdict, and all corroborated Minor findings, are `should_fix`.
- Remaining Minor, optional-scope items are `consider`.

Obligation is an editorial gate and does not rank the work. Rows follow immutable source order (seat order EIC, R1, R2, R3, DA, then finding ordinal, then sub-claim ordinal). Severity and confidence are transported from the cards. Confidence is self-reported scope metadata and carries no weight.

`[BLOCK-MANIFEST-UNAVAILABLE]`: no bound block manifest was supplied for this round. The `proposed_targets` below are therefore section-level locators, not manifest block IDs. A closed `revision-roadmap/1.0` machine artifact can be built only after a block manifest is bound (`block_manifest_sha256`). This package does not fabricate block IDs.

### Required Revisions (Must Fix)

| Transport ref | Item | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Consensus | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | REV-01 | Deepen the theory (non-Gaussian or fourth-cumulant correction, a finite-sample statement, or a GLM/Laplace treatment), or reframe the paper as a short analytical note | SC-01 | major | `text: Section 3, after Proposition 3 "The proof is one application of Isserlis’ theorem; its value lies in the interpretation."` | 4 (Bayesian DL / UQ positioning) | EIC W1 | single-reviewer | must_fix | section: §1 Contributions, §3 | claim_scope_unsupported → claim: contribution significance (§1) |
| R2 | REV-02 | Split Recommendation (i) into a supported negative claim and a labelled conjecture; optionally add encoders or a downstream decision task | SC-02 | major (EIC W2) / minor (R3 W4); SPLIT | `table: Table 5, rows "Sρ−CKA" (+0.19 [+0.00, +0.50]) and "Sρ−linear pred." (-0.00 [-0.22, +0.11]) in the width (partial) column` | EIC 4 (UQ evaluation practice); R3 4 (practitioner reading) | EIC W2; R3 W4 | SPLIT (arbitrated) | must_fix | section: §5 Recommendations; abstract | claim_scope_unsupported → claim: Recommendation (i) |
| R3 | REV-03 | Make closed-form or GP heads the primary empirical object, or add an EU target that separates from confidence (OOD or subsampled classes); align the abstract with that choice | SC-03 | major | `text: Section 4.4 "All indices predict AU agreement about as well as EU agreement"` | 4 (aleatoric/epistemic decomposition) | EIC W3; DA M2 corroborating | single-reviewer (+DA) | must_fix | re_analysis: §4.2–4.4; abstract | evidence_gap_remains → claim: EU-specific empirical support |
| R4 | REV-05 | Discuss mutual k-NN as a scale-invariant, tail-sensitive case and state which mechanism the cross-encoder data support | SC-05 | major (R2 W1) / minor (EIC W5, R1 W6); SPLIT | `table: Table 7, BLR column, mutual k-NN 0.90 versus S-rho (head's rho) 0.92 and CKA 0.76` | EIC 3; R1 4; R2 4 | EIC W5; R1 W6 (sub 2); R2 W1 (sub 2); DA M1 corroborating | SPLIT (arbitrated) | must_fix | section: §3 discussion, §5 | interpretive_ambiguity_remains → claim: why CKA fails (§5) |
| R5 | REV-11 | Replace the G = 5 percentile cluster bootstrap with a leave-one-encoder-out jackknife or exact enumeration over the 126 multisets (or an explicit exclusion rule), or with an exact permutation test over the 10 pairs; label intervals as descriptive; fix the abstract's "(0.00 to 0.50)" | SC-11 | major | `text: Appendix C, "keep every pair of distinct sampled encoders (with multiplicity," and "pairs; intervals are 2.5–97.5% percentiles."` | 5 (direct enumeration) | R1 W1 | single-reviewer | must_fix | re_analysis: Table 5, wd sweep, Appendix C, abstract | reporting_requirement_unmet → table: Table 5 |
| R6 | REV-12 | Report the residual variance fraction with each partial agreement; give "residual EU" a substantive gloss or rename it; add an external-validity check (for example, whether residual EU at 10% predicts per-item loss reduction at 100%) | SC-12 | major | `text: Section 4.2 and Appendix B, "probes the bootstrap width has rank correlation -0.999" and "M = 50: 0.998, M = 100: 0.999. We use M = 50."` | R1 4; R3 4 | R1 W2 (sub 1); R3 W1 (sub 1) | corroborated (2/4) | must_fix | re_analysis: §4.2, Tables 2–3 | evidence_gap_remains → claim: residual-EU construct |
| R7 | REV-13 | Remove the identical-features "beyond estimator noise" claim from the "robust" findings, or support it with the intervals in REV-14 | SC-13 | major | `text: Section 5 What the results say, "even identical features leave the residual EU ranking dependent on the"` | R1 4; R3 4 | R1 W2 (sub 2); R3 W1 (sub 2); DA M5 corroborating | corroborated (2/4) | must_fix | sentence: §5 "Two further findings are robust" | claim_scope_unsupported → claim: §5 robust finding 2 |
| R8 | REV-14 | Run the identical-features analyses at M ≥ 200 for all encoders; bootstrap over items and subset draws for intervals on corrected agreements; apply the correction to the wd/10 and wd×10 rows | SC-14 | major | `text: Section 4.2 and Appendix B, "probes the bootstrap width has rank correlation -0.999" and "M = 50: 0.998, M = 100: 0.999. We use M = 50."` | 4 | R1 W2 (sub 3); DA M5 corroborating | single-reviewer (+DA) | must_fix | re_analysis: Tables 2–3, Fig. 3a | evidence_gap_remains → table: Table 3 |
| R9 | REV-15 | Evaluate Sρ at a curvature-calibrated ρ (for example, the Laplace GGN prior-to-curvature ratio), or reframe the bootstrap analyses as testing CCA-type against CKA-type weighting only | SC-15 | major | `text: Section 4.3 and Section 4.2, "prior strength is set by the curvature of the likelihood," and "u(x), which ignores labels, correlates with it at only"` | 4 | R1 W3 (sub 1); DA M2/M4 corroborating | single-reviewer (+DA) | must_fix | re_analysis: Table 5, Fig. 3b | claim_scope_unsupported → claim: matched-ρ predictions |
| R10 | REV-16 | Test the fourth-cumulant explanation of the 0.34 level offset directly from cached features, or restrict the claim to ordinal agreement | SC-16 | major | `text: Section 4.3 and Section 4.2, "prior strength is set by the curvature of the likelihood," and "u(x), which ignores labels, correlates with it at only"` | 4 | R1 W3 (sub 2) | single-reviewer | must_fix | re_analysis: §4.4 closed-form level | evidence_gap_remains → claim: fourth-cumulant attribution |
| R11 | REV-27 | Correct the attribution: the main metric in Huh et al. (2024) is mutual k-NN, not kernel alignment | SC-27 | major | `text: §1 Introduction, "a claim made mainly through kernel alignment (Huh et al., 2024)"` | 4 (platonic-representation metrics) | R2 W1 (sub 1) | single-reviewer | must_fix | sentence: §1 Introduction; abstract opening | claim_scope_unsupported → claim: convergence-literature attribution |
| R12 | REV-28 | Narrow "aligned representations" in the title and abstract to the CKA-specific conclusions where appropriate | SC-28 | major | `text: §1 Introduction, "a claim made mainly through kernel alignment (Huh et al., 2024)"` | 4 | R2 W1 (sub 3) | single-reviewer | must_fix | sentence: title, abstract | claim_scope_unsupported → manuscript: title/abstract scope |
| R13 | REV-29 | State that the encoder set has no model-scale axis and that the paper cannot address the scale-dependent convergence trend | SC-29 | major | `text: §1 Introduction, "a claim made mainly through kernel alignment (Huh et al., 2024)"` | 4 | R2 W1 (sub 4) | single-reviewer | must_fix | sentence: §5 Limitations | claim_scope_unsupported → section: §5 Limitations |
| R14 | REV-40 | Reconcile the scale-blindness argument with the fact that the winning Sρ→0 (CCA) is equally scale-invariant; narrow the claim to the Gaussian model, or show a regime where scale-sensitive indices win | SC-40 | major | `table: Table 5, rows Sρ→0, Sρ (head's ρ) and mutual k-NN, closed-form and width (partial) columns` | 4 | DA M1 (CCA-invariance part) | DA-only (not counted) | must_fix | section: abstract, §1, §5 Recommendations | interpretive_ambiguity_remains → claim: "discards exactly the two pieces of information" |
| R15 | REV-41 | Replace "Theorem 2 explains" with "is compatible with" for the second mechanism, given d_eff/d > 0.98 at the selected weight decays | SC-41 | major | `text: §4.3 "The closed-form EU barely moves (mean ratio within"` | 4 | DA M4 (regime part) | DA-only (not counted) | must_fix | sentence: §4.3, §5 | claim_scope_unsupported → claim: tail-weighting mechanism on real data |

#### Required Item Details

**R1: Theoretical depth or honest reframing (REV-01)**
- **Problem**: The central identity follows from one moment computation. The index is not new, and the remaining results are classical or direct specialisations.
- **Source**: EIC W1 ("The proof is one application of Isserlis’ theorem; its value lies in the interpretation.")
- **Requirement**: Pick one of (a) a non-Gaussian or fourth-cumulant correction, (b) a finite-sample statement for the sample index, or (c) a GLM/Laplace treatment. Alternatively, reframe the paper as a short analytical note.
- **Acceptance criteria**: The revised manuscript either contains one of the three extensions with a proof, or its abstract and contributions explicitly present the work as an analytical note without implying broader theory.

**R2: Positive index recommendation (REV-02)**
- **Problem**: The preference for whitened indices or Sρ rests on intervals that touch zero, and Sρ cannot be distinguished from linear predictivity.
- **Source**: EIC W2 (Major); R3 W4 (Minor). This is a SPLIT, arbitrated above.
- **Requirement**: Separate the supported negative claim from a labelled conjecture, and state that no index is EU-specific in practice. Optionally, add encoders or a downstream decision task.
- **Acceptance criteria**: Recommendation (i) and the abstract present the "prefer whitened/Sρ" half as a conjecture or as consistent-with-evidence wording, unless new encoders or a downstream task yield an interval for the difference that excludes zero.

**R3: EU isolation in the real-data experiments (REV-03)**
- **Problem**: Softmax EU summaries are nearly a function of confidence, and indices predict AU as well as EU.
- **Source**: EIC W3; DA M2 corroborating.
- **Requirement**: Either make closed-form or GP heads the primary empirical object, or add an EU target that separates from confidence. Then align the abstract with that choice.
- **Acceptance criteria**: The abstract and contributions no longer present softmax-probe results as evidence about EU specifically, or a new EU target with reported EU and AU separation is included.

**R4: Mutual k-NN (REV-05)**
- **Problem**: Mutual k-NN shares CKA's invariances but tracks EU agreement close to Sρ, and the paper never discusses it.
- **Source**: EIC W5; R1 W6; R2 W1. This is a SPLIT, arbitrated above. DA M1 corroborates.
- **Requirement**: Add an explicit discussion of mutual k-NN as a tail-sensitive, scale-invariant index, and state which mechanism the data support.
- **Acceptance criteria**: §3 or §5 contains a paragraph on mutual k-NN that cites its Table 7 values and names the supported mechanism (spectral weighting or scale).

**R5: Cluster bootstrap with five encoders (REV-11)**
- **Problem**: With five clusters the percentile bootstrap gives lattice-valued bounds (upper limits of 1.00, lower limits of +0.00). The "0.09 of resamples" figure is not a valid one-sided p-value.
- **Source**: R1 W1.
- **Requirement**: Use a jackknife, an exact enumeration with a stated exclusion rule, or an exact permutation test. Present G = 5 intervals as descriptive, and state how resamples without a valid pair are handled.
- **Acceptance criteria**: Table 5, the weight-decay sweep and the abstract report intervals from a stated non-degenerate procedure, or are labelled as descriptive, and no sentence reads "0.09" as a test outcome.

**R6: Validity of the residual-EU construct (REV-12)**
- **Problem**: After residualising on confidence, well under 1% of rank variance remains. That residual plausibly reflects non-max class structure or noise and has no external validation.
- **Source**: R1 W2 (sub 1); R3 W1 (sub 1).
- **Requirement**: Report the residual variance fraction, give the construct a gloss or rename it, and add at least one external-validity check.
- **Acceptance criteria**: Each partial agreement is accompanied by its residual variance fraction, and the construct is either renamed or tied to an external criterion with a reported result.

**R7: "Robust" identical-features claim (REV-13)**
- **Problem**: An exploratory, interval-free result is called robust in the Discussion.
- **Source**: R1 W2 (sub 2); R3 W1 (sub 2); DA M5.
- **Requirement**: Remove the claim from the robust-findings sentence, or support it with REV-14's intervals.
- **Acceptance criteria**: §5 either no longer calls this finding robust, or cites intervals over items and subset draws at M ≥ 200 that exclude the re-test ceiling.

**R8: Intervals for reliability-corrected agreements (REV-14)**
- **Problem**: The corrected agreements (0.48, 0.68) are ratio estimators with no intervals. The weight-decay rows are uncorrected, and M = 50 is justified by raw stability rather than partial reliability.
- **Source**: R1 W2 (sub 3).
- **Requirement**: Use M ≥ 200 for all encoders, bootstrap over items and subset draws, and correct the weight-decay rows.
- **Acceptance criteria**: Table 3 (or its successor) reports intervals for every corrected agreement at M ≥ 200, and the wd/10 and wd×10 rows carry the reliability correction.

**R9: Matched-ρ tested at the wrong ρ (REV-15)**
- **Problem**: Sρ is evaluated at ρ = wd, but the paper states that the effective prior of the softmax estimators is set by likelihood curvature.
- **Source**: R1 W3 (sub 1).
- **Requirement**: Evaluate Sρ at a curvature-calibrated ρ, or reframe the bootstrap analyses as weighting-type tests only.
- **Acceptance criteria**: Either Table 5 adds Sρ at a curvature-calibrated ρ, or the matched-ρ claims are explicitly restricted to the closed-form estimator.

**R10: Fourth-cumulant attribution (REV-16)**
- **Problem**: The 0.34 level offset is attributed to fourth cumulants without a test.
- **Source**: R1 W3 (sub 2).
- **Requirement**: Compute the empirical fourth-moment form from cached features, or restrict the claim to ordinal agreement.
- **Acceptance criteria**: The manuscript reports the computed fourth-moment correction and its effect on the level offset, or removes the attribution and claims only ordinal agreement.

**R11: Attribution of the convergence metric (REV-27)**
- **Problem**: The introduction attributes the convergence claim to kernel alignment, but the main metric in Huh et al. (2024) is mutual k-NN.
- **Source**: R2 W1 (sub 1).
- **Requirement**: Correct the attribution.
- **Acceptance criteria**: The introduction and abstract attribute the convergence claim to the metrics Huh et al. (2024) actually use, with mutual k-NN identified as the primary one.

**R12: Title and abstract scope (REV-28)**
- **Problem**: "Aligned representations" claims more than the CKA-specific conclusions support.
- **Source**: R2 W1 (sub 3).
- **Requirement**: Narrow the framing where appropriate.
- **Acceptance criteria**: The title or abstract either scopes the negative conclusions to CKA, or justifies the broader phrasing with reference to the other indices tested.

**R13: Model-scale axis (REV-29)**
- **Problem**: The five encoders have no scale axis, so the paper cannot speak to the scale-dependent convergence trend.
- **Source**: R2 W1 (sub 4).
- **Requirement**: State this limitation.
- **Acceptance criteria**: §5 Limitations explicitly says that the encoder set has no model-scale axis and that conclusions about scale-dependent convergence are out of scope.

**R14: Scale argument against the CCA winner (REV-40)**
- **Problem**: CKA's scale-blindness is given as a reason it fails, but the best index (Sρ→0/CCA) is invariant to every invertible linear map.
- **Source**: DA M1. This part is DA-only among seats, and the DA is not counted in consensus.
- **Requirement**: Narrow the claim to the Gaussian model, or show a regime where scale-sensitive indices win. Separate what whitening fixes (tail weighting) from what only Sρ at the head's ρ detects (prior–scale mismatch).
- **Acceptance criteria**: The abstract, §1 and Recommendation (i) no longer present scale-blindness as a reason CKA underperforms the recommended scale-invariant indices on the real data, or a result is added in which a scale-sensitive index outperforms them.

**R15: "Explains" against "is compatible with" (REV-41)**
- **Problem**: At the selected weight decays d_eff/d > 0.98, so the tail-weighting mechanism is not operative for the closed-form estimator.
- **Source**: DA M4. The regime part is DA-only.
- **Requirement**: Soften the causal wording.
- **Acceptance criteria**: §4.3 and §5 describe Theorem 2 as compatible with, not explaining, the real-data tail-reshaping observations.

### Suggested Revisions (Should Fix / Consider)

| Transport ref | Item | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Consensus | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | REV-04 | Present scale-as-prior as a diagnostic caveat (CKA cannot detect whether priors match feature scales), not as a "robust" headline | SC-04 | minor | `text: Section 4.3 "it vanishes when the weight decay is re-tuned."` | EIC 4; R2 4 | EIC W4; R2 W10; DA M3 corroborating | corroborated (2/4) | should_fix | sentence: §5 What the results say; abstract | claim_scope_unsupported → claim: §5 robust finding 1 |
| S2 | REV-06 | Add a paragraph on error consistency and functional or prediction-level similarity (for example, Geirhos et al. 2020, Mania et al. 2019, Jiang et al. 2022, Baek et al. 2022, Bansal et al. 2021, Klabunde et al.) and say what uncertainty agreement adds | SC-06 | minor | `absence: Section 2 (Setup and Related Work) and the reference list — expected discussion of error-consistency or model-agreement work on vision classifiers; checked Introduction, Section 2, Discussion, References` | EIC 3; R2 4 | EIC W6; R2 W4 | corroborated (2/4) | should_fix | section: §2 | reader_traceability_reduced → section: §2 |
| S3 | REV-07 | State in the introduction and abstract that E4's falsifier fired (encoder 93%) and narrow the scope accordingly; consider higher-alignment pairs (seeds or checkpoints) | SC-07 | minor | `text: Section 4.4, Swap design "alignment never reached a regime where head and data terms dominate."` | 4 | EIC W7 | single-reviewer | should_fix | sentence: §1, abstract | claim_scope_unsupported → claim: motivating scenario |
| S4 | REV-08 | Use a fixed verdict vocabulary (supported / not supported / inconclusive / uninformative) and recode E2-a as uninformative or "untested (exploratory support)" | SC-08 | minor | `table: Table 1, row E2-a (observed 0.26, verdict "supported", footnote †)` | EIC 5; R1 5 | EIC W8; R1 W8 (sub 1) | corroborated (2/4) | should_fix | section: Table 1 | acceptance_criterion_unmet → table: Table 1 |
| S5 | REV-09 | Move Proposition 10 to the appendix and present Lemmas 7 and 9 as tools, not contributions | SC-09 | minor | `text: Section 1, Contributions item 2 "also note an in-sample resolvent bound"` | 4 | EIC W9 | single-reviewer | should_fix | section: §1 Contributions | editorial_conformance_unmet → section: §1 |
| S6 | REV-10 | Replace "post-review" labels with "exploratory (not pre-registered)" | SC-10 | minor | `text: Appendix C "All analyses in this section were added after an internal round of review and are exploratory."` | 4 | EIC W10 | single-reviewer | should_fix | sentence: §4, Appendix C | editorial_conformance_unmet → manuscript: labels |
| S7 | REV-17 | Use √r_xx (≈0.837) as the human-reliability ceiling; correct 0.48–0.61 to about 0.41–0.51 and harmonise with Table 3's correction | SC-17 | minor | `text: Appendix D, Table 6, "0.36 (0.52 of ceiling); test acc. 0.98"` | R1 5; R3 4 | R1 W4; R3 W2 | corroborated (2/4) | should_fix | sentence: §4.2, Table 6 | reporting_requirement_unmet → table: Table 6 |
| S8 | REV-18 | Relabel the split-half "95% CI" as split variability, or bootstrap over images and annotators | SC-18 | minor | `text: Section 4, Reliability and positive control, "(Spearman–Brown corrected, 95% CI [0.69, 0.71])"` | 4 | R1 W5 | single-reviewer | should_fix | sentence: §4 Reliability | reporting_requirement_unmet → claim: reliability CI |
| S9 | REV-19 | Report all agreement columns in the main text or name the pre-registered column; handle multiplicity across indices and columns | SC-19 | minor | `table: Table 7 (Appendix D), rows linear predictivity and mutual k-NN across width, width partial, MI, BLR, AU` | 4 | R1 W6 (sub 1) | single-reviewer | should_fix | section: §4.4, Table 5 | claim_scope_unsupported → claim: headline index ordering |
| S10 | REV-20 | State that the Lemma 7 closed-form "prediction" is near-identity; explain 0.90 rather than about 1 | SC-20 | minor | `text: Section 4.4, Label-free prediction, "rank-correlates with mean closed-form EU at 0.90, as"` | 4 | R1 W7 | single-reviewer | should_fix | sentence: §4.4, Table 1 Spec row | interpretive_ambiguity_remains → table: Table 1 |
| S11 | REV-21 | Qualify E2b-tail 6/6 as six dependent settings without a pre-specified magnitude threshold | SC-21 | minor | `text: Table 1 caption, "Uninformative in hindsight: the re-test partial agreement at M = 50 is itself ≈0.56,"` | 5 | R1 W8 (sub 2) | single-reviewer | should_fix | sentence: Table 1 / §4.3 | interpretive_ambiguity_remains → table: Table 1 |
| S12 | REV-22 | Extend the weight-decay grid by at least one decade on each side | SC-22 | minor | `text: Section 4.3, "as c2 (from 10−5 at c = 0.1 to 10−1 at c = 10), which"` | 4 | R1 W9 | single-reviewer | consider | re_analysis: §4.3, Appendix B grid | evidence_gap_remains → claim: c² scaling |
| S13 | REV-23 | Define symmetric linear predictivity and the input set and sample size for mutual k-NN and Sρ; give the converse Corollary 4 parameters | SC-23 | minor | `absence: Section 4.4 and Appendix B — expected a definition of symmetric linear predictivity (regulariser, held-out protocol, symmetrisation) and of the input set and sample size used for mutual k-NN and S_rho; checked Section 2, Section 4.4, Table 5 caption, Appendix B, Appendix C` | 3 | R1 W10 | single-reviewer | should_fix | section: Appendix B | method_reproducibility_unresolved → section: Appendix B |
| S14 | REV-24 | Drop or condition the Fig. 1b "≈ k/(k+m)" approximation (it needs w_p close to 1) | SC-24 | minor | `figure: Figure 1b caption (k = 9, lambda_s = 10, lambda_p = 0.1, rho = 0.01)` | 5 | R1 W11 | single-reviewer | should_fix | sentence: Fig. 1 caption | reporting_requirement_unmet → figure: Figure 1b |
| S15 | REV-25 | Give replicate-based intervals for the E4 Shapley shares and equal prominence to the partial-disagreement variant (61/30/10) | SC-25 | minor | `text: Section 4.4, Swap design, "the encoder accounts for 93%"` | 3 | R1 W12 | single-reviewer | should_fix | re_analysis: §4.4 Swap design | evidence_gap_remains → claim: E4 encoder share |
| S16 | REV-26 | Deposit the pre-registration with a remote timestamp or registry for future work | SC-26 | minor | `text: Appendix B, Pre-registration, "so the hash certiﬁes content, not time"` | 3 | R1 W13 | single-reviewer | consider | other (surface_id: preregistration_record): Appendix B | method_reproducibility_unresolved → section: Appendix B |
| S17 | REV-30 | Attribute the λ² against uniform spectral-reweighting view of CKA/CCA to Kornblith et al. (2019) | SC-30 | minor | `text: §3 after Proposition 3, "its value lies in the interpretation. In the eigenbasis, CKA"` | 4 | R2 W2 | single-reviewer | should_fix | sentence: §3 | reader_traceability_reduced → claim: spectral reweighting attribution |
| S18 | REV-31 | Fix the Stringer et al. (2019) citation (biological cortex) for "learned representations" | SC-31 | minor | `text: §3 after Proposition 3, "Since learned representations have long" and "(Stringer et al., 2019; Agrawal et al., 2022)"` | 5 | R2 W3 | single-reviewer | should_fix | sentence: §3 | reader_traceability_reduced → claim: spectral-tail citation |
| S19 | REV-32 | Connect u(x) and d_eff(ρ) to the ridge leverage-score literature (Bach 2013; Alaoui and Mahoney 2015) | SC-32 | minor | `text: §2 Heads and epistemic uncertainty, "the EU score; it interpolates between a" and "Mahalanobis leverage (ρ →0)"` | 4 | R2 W5 | single-reviewer | consider | sentence: §2 | reader_traceability_reduced → section: §2 |
| S20 | REV-33 | Discuss stochastic or noise-aware shape metrics (Duong et al. 2023) | SC-33 | minor | `absence: Section 2 Representations and alignment — expected discussion of stochastic or noise-covariance-aware shape metrics as the closest second-order similarity measures; checked Section 2, Section 3 positioning paragraph after Definition 1, References list` | 3 | R2 W6 | single-reviewer | consider | sentence: §2 | reader_traceability_reduced → section: §2 |
| S21 | REV-34 | Justify the feature tap and pooling, or add a robustness check (CLS or post-norm features; ViT high-norm artifacts) | SC-34 | minor | `text: §4 Data, encoders and heads, "pooled patch tokens (spatial mean for ConvNeXt) of the raw block output at relative depths 0.5, 0.75, 1"` | 3 | R2 W7 | single-reviewer | consider | re_analysis: §4 features | claim_scope_unsupported → claim: generality of Table 5 ordering |
| S22 | REV-35 | Add a d-versus-n caveat to the whitened-index recommendation, or prefer Sρ at non-vanishing ρ | SC-35 | minor | `text: §5 Recommendations, "If an index is needed, prefer"` | 4 | R2 W8 | single-reviewer | should_fix | sentence: §5 Recommendations | claim_scope_unsupported → claim: Recommendation (i) |
| S23 | REV-36 | Reserve "EU" for closed-form and Laplace/MI quantities, or qualify bootstrap width as an EU proxy throughout | SC-36 | minor | `text: §4.2, "probes the bootstrap width has rank correlation -0.999"` | 4 | R2 W9 | single-reviewer | should_fix | section: abstract, tables, §5 | interpretive_ambiguity_remains → manuscript: EU terminology |
| S24 | REV-37 | State that model AU is a hard-label predictive-entropy summary, not a human-ambiguity estimate; optionally add a soft-label-trained head | SC-37 | minor | `text: Appendix B Heads and Section 4 Data, "AU: mean member entropy" and "(50k, hard labels)"` | 4 | R3 W3 | single-reviewer | should_fix | sentence: §4.2, Appendix B | interpretive_ambiguity_remains → claim: AU interpretation |
| S25 | REV-38 | State that the motivating uses are hypothetical and OOD is untested; bring the between-encoder top-10% overlap (0.38) into the Discussion; optionally run a small OOD probe | SC-38 | minor | `text: Section 5 Limitations, "out-of-distribution inputs, where shared blind spots matter most"` | 4 | R3 W5 | single-reviewer | should_fix | section: §1, §5 | claim_scope_unsupported → claim: practical implications |
| S26 | REV-39 | Position the human-entropy analyses within the literature on learning from disagreement and human-uncertainty alignment (Uma et al. 2021; Collins et al. 2022; Peterson et al. 2019; Sucholutsky et al. 2023); optionally report a "humans as a sixth system" column | SC-39 | minor | `absence: Section 2 Setup and Related Work — expected engagement with human label-disagreement and human-uncertainty-alignment literature beyond citing CIFAR-10H as a data source; checked Section 2, Section 4 E0 paragraph, Section 5 Discussion, References list` | 4 | R3 W6 | single-reviewer | consider | section: §2 | reader_traceability_reduced → section: §2 |

Copy-level notes that the EIC placed below the finding threshold are not roadmap items: "0.83–0.83" should read "0.83", Table 5's 1.00 upper bounds need a caption note, and the abstract is dense with numbers. They are listed here for the author's convenience only.

### Source-Traceability Checklist

> Immutable source order. This is not a work order. Author triage (`will_address` / `wont_address` / `not_on_point`) is collected later in the separate author-adjudication sidecar.

- [ ] REV-01 / R1: obligation `must_fix`. Deepen the theory or reframe as an analytical note.
- [ ] REV-02 / R2: obligation `must_fix`. Split Recommendation (i) into a negative claim and a conjecture.
- [ ] REV-03 / R3: obligation `must_fix`. Isolate EU empirically or align the abstract.
- [ ] REV-04 / S1: obligation `should_fix`. Recast scale-as-prior as a diagnostic caveat.
- [ ] REV-05 / R4: obligation `must_fix`. Discuss mutual k-NN and the operative mechanism.
- [ ] REV-06 / S2: obligation `should_fix`. Add the error-consistency and functional-similarity literature.
- [ ] REV-07 / S3: obligation `should_fix`. Disclose the consequence of the E4 falsifier in the intro and abstract.
- [ ] REV-08 / S4: obligation `should_fix`. Fix the Table 1 verdict vocabulary and the E2-a verdict.
- [ ] REV-09 / S5: obligation `should_fix`. Trim the contributions list.
- [ ] REV-10 / S6: obligation `should_fix`. Relabel "post-review" as "exploratory".
- [ ] REV-11 / R5: obligation `must_fix`. Replace the degenerate G = 5 bootstrap.
- [ ] REV-12 / R6: obligation `must_fix`. Validate or rename residual EU; report residual variance.
- [ ] REV-13 / R7: obligation `must_fix`. Remove or support the "robust" identical-features claim.
- [ ] REV-14 / R8: obligation `must_fix`. Add intervals at M ≥ 200; correct the wd rows.
- [ ] REV-15 / R9: obligation `must_fix`. Use a curvature-calibrated ρ or reframe.
- [ ] REV-16 / R10: obligation `must_fix`. Test the fourth-cumulant attribution or claim ordinal agreement only.
- [ ] REV-17 / S7: obligation `should_fix`. Use the √r ceiling.
- [ ] REV-18 / S8: obligation `should_fix`. Relabel the split-half CI.
- [ ] REV-19 / S9: obligation `should_fix`. Report all columns and handle multiplicity.
- [ ] REV-20 / S10: obligation `should_fix`. Qualify the Lemma 7 prediction.
- [ ] REV-21 / S11: obligation `should_fix`. Qualify the E2b-tail 6/6 count.
- [ ] REV-22 / S12: obligation `consider`. Extend the weight-decay grid.
- [ ] REV-23 / S13: obligation `should_fix`. Define the baselines and input sets.
- [ ] REV-24 / S14: obligation `should_fix`. Fix the Fig. 1b approximation.
- [ ] REV-25 / S15: obligation `should_fix`. Add E4 Shapley intervals.
- [ ] REV-26 / S16: obligation `consider`. Add an external pre-registration timestamp.
- [ ] REV-27 / R11: obligation `must_fix`. Correct the Huh et al. metric attribution.
- [ ] REV-28 / R12: obligation `must_fix`. Narrow the title and abstract scope.
- [ ] REV-29 / R13: obligation `must_fix`. Disclose the missing model-scale axis.
- [ ] REV-30 / S17: obligation `should_fix`. Attribute spectral reweighting to Kornblith et al.
- [ ] REV-31 / S18: obligation `should_fix`. Fix the Stringer citation.
- [ ] REV-32 / S19: obligation `consider`. Add the leverage-score link.
- [ ] REV-33 / S20: obligation `consider`. Add stochastic shape metrics.
- [ ] REV-34 / S21: obligation `consider`. Justify or check the feature tap and pooling.
- [ ] REV-35 / S22: obligation `should_fix`. Add the d-versus-n caveat.
- [ ] REV-36 / S23: obligation `should_fix`. Qualify the "EU" terminology.
- [ ] REV-37 / S24: obligation `should_fix`. Clarify that AU is a hard-label summary.
- [ ] REV-38 / S25: obligation `should_fix`. Bound the motivating uses; surface the 0.38 overlap.
- [ ] REV-39 / S26: obligation `consider`. Add the human-disagreement literature.
- [ ] REV-40 / R14: obligation `must_fix`. Reconcile the scale argument with the CCA winner (DA M1).
- [ ] REV-41 / R15: obligation `must_fix`. Change "explains" to "is compatible with" (DA M4).

Obligation counts: must_fix 15, should_fix 20, consider 6 (41 items in total).

### Journal-Supplied Deadline

- **Exact deadline from source letter**: NOT PROVIDED. No deadline or work estimate is inferred.

### Response Letter Template

Please respond to every item (R1–R15 and S1–S26) using `templates/revision_response_template.md`. For each, give a response and a description of the revision, or a reason for not adopting it. Mark changes in the manuscript and include a cross-reference table of new locations. Declining a `must_fix` item is permitted, but it stays visible as unresolved at re-review.

---

## Part 3: Reviewer Report Summary (Appendix)

Sprint-contract cards report dimension scores, not an overall recommendation. No card stated an Accept/Minor/Major/Reject recommendation, and none is invented here. Confidence is per finding unless a card states a report-level value.

| Seat | Role | Owned-dimension scores | Overall recommendation | Confidence |
|---|---|---|---|---|
| EIC | Journal-Fit Reviewer (senior AC, statistical ML / UQ) | D5 warn; D6 warn | not stated in card | per finding (3–5) |
| R1 | Methodology (statistician; Bayesian linear models, resampling) | D1 warn; D3 warn | not stated in card | per finding (3–5) |
| R2 | Domain (representation similarity) | D2 warn | not stated in card | per finding (3–5) |
| R3 | Perspective (computational cognitive science; human label uncertainty) | D4 warn | not stated in card | report-level 4/5 (human uncertainty, practice), 2/5 (spectral theory); per finding 4 |
| DA | Devil's Advocate | D3 warn | N/A (findings only) | per finding (3–4) |

### Journal-Fit Review Report Summary
- Key point: The work is a candid, correct but thin analytical contribution whose practical payoff (index recommendation, EU-specific evidence) is asserted rather than demonstrated. D6 warn, not block.

### Reviewer 1 (Methodology) Summary
- Key point: All proofs re-derived and correct. The uncertainty statements are invalid or missing in places: the degenerate five-cluster bootstrap, interval-free corrected agreements, matched ρ evaluated at the wrong estimator prior, and the √r ceiling. All are repairable, so warn.

### Reviewer 2 (Domain) Summary
- Key point: Domain content is accurate. The positioning misstates the platonic-representation metric (mutual k-NN), omits several neighbouring literatures, and misattributes a few sources. D2 warn.

### Reviewer 3 (Perspective) Summary
- Key point: The mechanism is accessible to adjacent fields, but "residual EU" lacks substantive validation, model AU is over-read against human ambiguity, and the practitioner recommendation outruns the evidence. D4 warn.

### Devil's Advocate Summary
- Recommendation: N/A (findings only).
- Key challenge: There is no CRITICAL challenge. The strongest counter-argument (M1–M5) is that on real data neither "scale" nor "tail weighting" does measurable work, and a parsimonious account (general linear-readout similarity) fits the results, so the identity is "a correct theorem about a quantity the experiments do not isolate."

---

## Part 4: Step 1b Weakness Sub-Claim Inventory (working record)

Positions are given for the four non-DA seats. Seats not listed for a sub-claim are `not-mentioned` (silence). DA corroboration is noted but not counted. Severity and confidence are transported from each card's per-finding tags. No sub-claim was introduced that a reviewer did not raise.

| sub_claim_id | parent_weakness | reviewer_id | position | evidence_pointer | severity | confidence | Disposition |
|---|---|---|---|---|---|---|---|
| SC-01 | EIC W1 thin theory | EIC | raised | EIC W1 anchor (§3 after Prop. 3) | major | 4 | single-reviewer |
| SC-02 | EIC W2 / R3 W4 positive recommendation | EIC | raised | Table 5 Sρ−CKA, Sρ−lin. pred. | major | 4 | SPLIT (severity) |
| SC-02 | | R3 | disputed (severity: minor; compatible remedy) | §5 Recommendations "If an index is needed, prefer whitened" | minor | 4 | |
| SC-03 | EIC W3 EU not isolable | EIC | raised | §4.4 "All indices predict AU..." | major | 4 | single-reviewer (+DA M2) |
| SC-04 | EIC W4 / R2 W10 scale finding over-presented | EIC | raised | §4.3 "vanishes when ... re-tuned" | minor | 4 | corroborated 2/4 (+DA M3) |
| SC-04 | | R2 | corroborated | §5 "scale hides the prior–scale mismatch" | minor | 4 | |
| SC-05 | EIC W5 / R1 W6 / R2 W1 mutual k-NN | EIC | raised | Table 7 BLR column | minor | 3 | SPLIT (severity) |
| SC-05 | | R1 | corroborated | Table 7 rows across columns | minor | 4 | |
| SC-05 | | R2 | disputed (severity: major; compatible remedy) | §1 "mainly through kernel alignment" + Table 7 | major | 4 | |
| SC-06 | EIC W6 / R2 W4 functional-similarity literature | EIC | raised | absence: §2, References | minor | 3 | corroborated 2/4 |
| SC-06 | | R2 | corroborated | absence: Intro, §2, §5, References | minor | 4 | |
| SC-07 | EIC W7 E4 falsifier scope | EIC | raised | §4.4 Swap design | minor | 4 | single-reviewer |
| SC-08 | EIC W8 / R1 W8 E2-a verdict | EIC | raised | Table 1 row E2-a | minor | 5 | corroborated 2/4 (+DA obs.) |
| SC-08 | | R1 | corroborated | Table 1 caption † | minor | 5 | |
| SC-09 | EIC W9 padded contributions | EIC | raised | §1 Contributions item 2 | minor | 4 | single-reviewer |
| SC-10 | EIC W10 post-review labels | EIC | raised | Appendix C | minor | 4 | single-reviewer |
| SC-11 | R1 W1 degenerate cluster bootstrap | R1 | raised | Appendix C bootstrap description | major | 5 | single-reviewer |
| SC-12 | R1 W2 / R3 W1 residual-EU construct validity | R1 | raised | §4.2 / App. B −0.999; M = 50 | major | 4 | corroborated 2/4 |
| SC-12 | | R3 | corroborated | §5 "residual EU ranking" | major | 4 | |
| SC-13 | R1 W2 / R3 W1 "robust" identical-features claim | R1 | raised | §4.2 / App. B (Discussion "robust") | major | 4 | corroborated 2/4 (+DA M5) |
| SC-13 | | R3 | corroborated | §5 "even identical features leave..." | major | 4 | |
| SC-14 | R1 W2 no intervals on corrected agreements; wd rows | R1 | raised | §4.2 / App. B | major | 4 | single-reviewer (+DA M5) |
| SC-15 | R1 W3 matched ρ at wd, not effective prior | R1 | raised | §4.3 / §4.2 curvature; u(x) correlation | major | 4 | single-reviewer (+DA M2, M4) |
| SC-16 | R1 W3 fourth-cumulant attribution untested | R1 | raised | §4.3 / §4.2 (same anchor) | major | 4 | single-reviewer |
| SC-17 | R1 W4 / R3 W2 √r ceiling | R1 | raised | App. D Table 6 | minor | 5 | corroborated 2/4 |
| SC-17 | | R3 | corroborated | §4.2 "0.48–0.61 of the ceiling" | minor | 4 | |
| SC-18 | R1 W5 split-half CI | R1 | raised | §4 Reliability | minor | 4 | single-reviewer |
| SC-19 | R1 W6 post hoc column; multiplicity | R1 | raised | Table 7 | minor | 4 | single-reviewer |
| SC-20 | R1 W7 Lemma 7 near-identity | R1 | raised | §4.4 Label-free prediction | minor | 4 | single-reviewer |
| SC-21 | R1 W8 E2b-tail 6/6 dependent settings | R1 | raised | Table 1 caption | minor | 5 | single-reviewer |
| SC-22 | R1 W9 grid boundaries | R1 | raised | §4.3 c² | minor | 4 | single-reviewer |
| SC-23 | R1 W10 underspecified baselines | R1 | raised | absence: §4.4, App. B/C | minor | 3 | single-reviewer |
| SC-24 | R1 W11 Fig. 1b approximation | R1 | raised | Fig. 1b caption | minor | 5 | single-reviewer |
| SC-25 | R1 W12 E4 Shapley uncertainty | R1 | raised | §4.4 Swap design | minor | 3 | single-reviewer |
| SC-26 | R1 W13 pre-registration timing | R1 | raised | App. B Pre-registration | minor | 3 | single-reviewer |
| SC-27 | R2 W1 Huh et al. metric misattributed | R2 | raised | §1 "mainly through kernel alignment" | major | 4 | single-reviewer |
| SC-28 | R2 W1 title/abstract scope | R2 | raised | §1 (same anchor) | major | 4 | single-reviewer |
| SC-29 | R2 W1 no model-scale axis | R2 | raised | §1 (same anchor) | major | 4 | single-reviewer |
| SC-30 | R2 W2 Kornblith reweighting attribution | R2 | raised | §3 after Prop. 3 | minor | 4 | single-reviewer |
| SC-31 | R2 W3 Stringer misattribution | R2 | raised | §3 after Prop. 3 | minor | 5 | single-reviewer |
| SC-32 | R2 W5 leverage-score link | R2 | raised | §2 Heads and EU | minor | 4 | single-reviewer |
| SC-33 | R2 W6 stochastic shape metrics | R2 | raised | absence: §2, §3, References | minor | 3 | single-reviewer |
| SC-34 | R2 W7 feature tap / pooling | R2 | raised | §4 Data, encoders and heads | minor | 3 | single-reviewer |
| SC-35 | R2 W8 whitened-index d-versus-n caveat | R2 | raised | §5 Recommendations | minor | 4 | single-reviewer |
| SC-36 | R2 W9 "EU" label for softmax summaries | R2 | raised | §4.2 −0.999 | minor | 4 | single-reviewer |
| SC-37 | R3 W3 AU is hard-label entropy | R3 | raised | App. B Heads, §4 Data | minor | 4 | single-reviewer |
| SC-38 | R3 W5 hypothetical uses; OOD untested | R3 | raised | §5 Limitations | minor | 4 | single-reviewer |
| SC-39 | R3 W6 human-disagreement literature | R3 | raised | absence: §2, §4 E0, §5, References | minor | 4 | single-reviewer |
| SC-40 | DA M1 scale argument against scale-invariant CCA winner | DA | raised (not counted) | Table 5 rows Sρ→0, Sρ, mutual k-NN | major | 4 | DA-only |
| SC-41 | DA M4 second mechanism inoperative at d_eff/d > 0.98 | DA | raised (not counted) | §4.3 "closed-form EU barely moves" | major | 4 | DA-only |

**Surface-form parity check (#216).** Each sub-claim was assessed on its paper evidence. No sub-claim was down-rated for informal wording or credited for technical specificity. The two SPLIT resolutions rest on shared table evidence and role competence, not on how precisely a card was phrased. No sub-claim was found unevaluable.

**Manuscript-as-data note.** R3 reports no instruction-like or reviewer-directed text in the manuscript. The synthesizer opened the manuscript only to spot-check the methodology attestation, because no DA CRITICAL items required adjudication. It found nothing aimed at the reviewers or the editor.

## Attachment: Acronym Check (advisory, #849)

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 141: GULP (the initials before its parentheses do not spell it)
- Body, line 1598: ViT (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 35 | Not defined | CIFAR | 9 |
| Body | 37 | Not defined | CCA | 6 |
| Body | 60 | Defined again | CKA | 54 |
| Body | 114 | Not defined | AND | 1 |
| Body | 114 | Not defined | SETUP | 1 |
| Body | 114 | Not defined | WORK | 1 |
| Body | 121 | Not defined | RdA | 1 |
| Body | 124 | Not defined | AB | 13 |
| Body | 131 | Not defined | NN | 2 |
| Body | 144 | Not defined | RSA | 1 |
| Body | 165 | Not defined | AI | 8 |
| Body | 180 | Not defined | AU | 18 |
| Body | 198 | Not defined | THEORY | 1 |
| Body | 202 | Not defined | MB | 9 |
| Body | 204 | Not defined | ABMB | 1 |
| Body | 214 | Not defined | BI | 2 |
| Body | 258 | Not defined | dAdB | 3 |
| Body | 449 | Not defined | DINOv2 | 13 |
| Body | 450 | Not defined | CLIP | 8 |
| Body | 453 | Not defined | MAE | 8 |
| Body | 464 | Not defined | NLL | 2 |
| Body | 1083 | Not defined | MI | 2 |
| Body | 1119 | Not defined | mKNN | 1 |
| Body | 1155 | Not defined | DCIC | 2 |
| Body | 1186 | Not defined | ReQ | 1 |
| Body | 1205 | Not defined | CVF | 5 |
| Body | 1205 | Not defined | IEEE | 5 |
| Body | 1225 | Defined again | ICLR | 3 |
| Body | 1233 | Not defined | PMLR | 1 |
| Body | 1237 | Defined again | CVPR | 4 |
| Body | 1264 | Defined again | ICML | 5 |
| Body | 1290 | Defined again | CVPR | 4 |
| Body | 1307 | Defined again | CVPR | 4 |
| Body | 1323 | Defined again | ICML | 5 |
| Body | 1333 | Not defined | SVCCA | 1 |
| Body | 1338 | Not defined | MIT | 1 |
| Body | 1343 | Defined again | ICLR | 3 |
| Body | 1348 | Defined again | ICML | 5 |
| Body | 1363 | Defined again | ICML | 5 |
| Body | 1433 | Not defined | GPU | 1 |
| Body | 1472 | Not defined | PROOFS | 1 |
| Body | 1479 | Not defined | ABQ | 2 |
| Body | 1554 | Not defined | CKA2 | 2 |
| Body | 1582 | Not defined | GT | 1 |
| Body | 1582 | Not defined | GZT | 1 |
| Body | 1588 | Not defined | PREREG | 2 |
| Body | 1589 | Not defined | SHA | 1 |
| Body | 1590 | Not defined | FROZEN | 1 |
| Body | 1602 | Not defined | CEi | 1 |
| Body | 1604 | Not defined | BFGS | 1 |
| Body | 1608 | Not defined | GGN | 1 |
| Body | 1608 | Not defined | MAP | 1 |
| Body | 1620 | Not defined | GB | 1 |
| Body | 1620 | Not defined | MPS | 1 |
| Body | 1621 | Not defined | CPU | 1 |
| Body | 1623 | Not defined | BY | 1 |
| Body | 1623 | Not defined | CC | 1 |
| Body | 1623 | Not defined | NC | 1 |
| Body | 1623 | Not defined | SA | 1 |
| Body | 1626 | Not defined | POST | 1 |
| Body | 1626 | Not defined | REVIEW | 1 |
