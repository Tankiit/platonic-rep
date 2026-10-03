# Response to Round-2 Panel (Major Revision, no blocks)

Title: "When Do Aligned Representations Agree on What They Do Not Know?" kept (author's choice), with the scoping subtitle "CKA, Prior Strength and the Spectral Tail" (REV-28). New code: `uq_lab/uq_revision2.py`; numbers via `uq_lab/make_paper_assets.py`. All new analyses are labelled exploratory.

## Required revisions

| Item | Status | What changed |
|---|---|---|
| REV-01 thin theory | Partly | Contributions condensed: supporting results declared elementary/classical. The theorem is framed as an interpretive identity whose level offset is now explained exactly (REV-16). No new theorem added. |
| REV-02 Rec. (i) positive half | Done | Split into "Supported: do not use CKA" and "Conjecture, not established here". The abstract states that no index beats linear predictivity or mutual k-NN. |
| REV-03 EU not isolated | Scoped | Stated in Discussion and Limitations; closed-form/GP heads and an EU target separating from confidence (OOD, held-out classes) named as next steps. Not run (time). |
| REV-05 mutual k-NN | Done | §3 explains mutual k-NN as scale-invariant but tail-sensitive (drops to 0.62 under tail reshaping). §4.4 and §5 state that the well-performing indices are all tail-sensitive and mostly scale-blind, so the cross-encoder data support the spectral-weighting mechanism, not the scale mechanism. |
| REV-11 degenerate cluster bootstrap | Done | Withdrawn. Replaced by leave-one-encoder-out ranges (Table 5, scorecard, abstract, sweep). S_rho − CKA > 0 in 5/5 subsets (partial), 4/5 (closed form); differences from linear predictivity and mutual k-NN change sign. The appendix explains the withdrawal. |
| REV-12 residual-EU construct | Done | The residual fraction is reported: about 0.3% of the rank variance of EU width (MI about 4%). The text says the finding concerns fine structure and has no external validation. |
| REV-13 "robust" claim | Done | "Robust" removed; the identical-features observation is presented as diagnostic, with caveats. |
| REV-14 M≥200, intervals, wd rows | Done | Identical-features analysis at M=200 for all encoders, 3 draws, wd pairs, item-bootstrap intervals (Table 3). |
| REV-15 rho is not the effective prior | Done | Curvature-calibrated ρ_eff = wd / mean p(1−p) (median about 100× wd). S_rho at ρ_eff does not improve prediction (0.78 vs 0.83), reported as a negative result. The sweep is described as a trend. |
| REV-16 fourth-cumulant claim | Done | Gaussianization test: with Gaussian surrogates of the same joint covariance, MAE between S_rho and closed-form Pearson agreement falls from 0.34 to 0.019. |
| REV-27 Huh et al. attribution | Done | Abstract and §1 now say the PRH rests mainly on mutual k-NN and its growth with scale. |
| REV-28 title scope | Done | Subtitle added (title kept at the author's request). |
| REV-29 no model-scale axis | Done | Limitations. |
| REV-40 scale-blindness vs CCA | Done | Abstract, §3, §5: whitened indices are equally scale-blind; only S_rho at the head's ρ on the head's feature scaling is not (S_rho(cQφ) = S_{ρ/c²}(φ)). |
| REV-41 "explains" | Done | "is compatible with". |

## Should-fix / consider (selected)

- AU fraction of ceiling now divides by √reliability (attainable maximum √0.70 = 0.84).
- Split-half "95% CI" relabelled as the range over random splits.
- Scorecard vocabulary: supported / not supported / partly supported / inconclusive. F-data is "inconclusive" (the test could not fire); S is "inconclusive".
- "Post-review" labels replaced by "exploratory".
- Kornblith et al. credited for the eigen-weighting view; the EU score is connected to ridge leverage scores (Alaoui & Mahoney 2015); Stringer et al. now cited as cortex, Agrawal et al. for learned representations; error consistency (Geirhos et al. 2020) and learning-from-disagreement (Uma et al. 2021) added.
- Re-tuned weight decays at grid boundaries noted as bounds (Appendix).
- Figure 1(b) caption approximation removed.
- Figures 2 and 4 moved to the supplement for space.

## Not done

REV-01 (new theory), REV-03 (new EU target / GP heads), noise-aware RSA discussion, Lemma 7 recast. New citations (Alaoui & Mahoney 2015, Geirhos et al. 2020, Uma et al. 2021) should be included in the author's citation check.
