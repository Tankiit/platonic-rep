# Domain Review (Peer Reviewer 2)

Reviewer identity (Card #3): researcher in representation similarity and representation learning (CKA, CCA variants, shape metrics, platonic-representation claims), familiar with self-supervised vision encoders. Possible blind spot: Bayesian-statistics details. Calibration status: `NOT_CALIBRATED`. criteria_binding_unavailable, so no venue-alignment claim is made.

contract_role: domain
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: warn
trigger: "secondhand or incomplete attribution, missing recent relevant references"

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

The paper asks whether representational alignment between two frozen encoders implies that linear heads on them agree about epistemic uncertainty (EU). Its central result is a Gaussian identity: the correlation of two quadratic-form EU scores equals a ridge-regularised spectral agreement index, with CKA as the large-ρ limit and the R²_CCA (mean squared canonical correlation) as the small-ρ limit. I checked the domain content of the theory against the representation-similarity literature I know. The identity follows correctly from Isserlis' theorem. The CKA formula, its invariance to isotropic scaling and orthogonal maps, the statement of Proposition 3(ii) (including the √(d_A d_B) normalisation when dimensions differ), and the descriptions of Ding et al. (2021), Davari et al. (2023), Kornblith et al. (2019) and the ridge-CCA lineage (Vinod; Bach and Jordan) are all accurate. The uncertainty side is also represented fairly: the paper acknowledges the limits of entropy-based decompositions and the entanglement of AU and EU estimators, and it does not oversell the empirical cross-encoder result. On domain accuracy, nothing I found invalidates a core conclusion. The negative claim (CKA is neither necessary nor sufficient for EU agreement) rests on a correct constructive argument.

My concerns are about positioning and attribution, which together justify a warn on D2. (a) The paper treats the convergence literature as resting on CKA, while the main metric of the platonic-representation paper it cites is mutual k-NN. Its own results show mutual k-NN performing close to S_ρ on several columns, and the paper does not discuss what this means for the convergence claims it opens with. (b) The spectral-reweighting reading of CKA versus CCA, which drives the "why CKA fails" argument, is already in Kornblith et al. (2019), but the paper does not attribute it. (c) Several neighbouring lines of work are missing: prediction-level and functional similarity (shared errors, disagreement), ridge leverage scores, and stochastic or noise-aware shape metrics. (d) There are small misattributions and terminology stretches. I list each separately below. On the card's generalisation focus: the claims are scoped to frozen linear and softmax probes on CIFAR-10H, in-distribution, with five ImageNet-scale encoders. The Limitations section states this honestly. The encoder set has no scale axis, though, so the paper cannot speak to the scale-dependent convergence trend that motivates it. I note this inside W1 and do not count it twice.

Questions for the authors (domain): (1) GULP (Boix-Adserà et al., 2022) studies ridge-regularised prediction distances across regularisation levels. Do its limiting regimes coincide exactly with Proposition 3, or only qualitatively? Please state the precise relation, because the novelty of the S_ρ interpolation itself hinges on it. (2) Why does mutual k-NN, which is invariant to scale and orthogonal maps but sensitive to the tail, track EU agreement nearly as well as S_ρ on raw width, closed form and AU (Table 7)? Does the theory predict this? (3) Would CLS-token or final-norm features change the CKA versus CCA-type ordering in Table 5?

### S1: Honest and accurate placement of S_ρ within the ridge-CCA family
The paper explicitly disclaims novelty of the index, and correctly locates its contribution in the link to posterior-variance agreement. That claim is faithful to the prior work it cites.
**Evidence Anchor**: text: §3 after Definition 1, "What we add is the exact link between this family and agreement of posterior variances"

### S2: Correct limiting relations to CKA and R²_CCA
Proposition 3 correctly recovers population linear CKA as ρ→∞. It also correctly states that the ρ→0 limit equals the mean squared canonical correlation only when d_A = d_B. Prior informal statements often blur this normalisation detail.
**Evidence Anchor**: equation: Proposition 3 (Limits), §3 and its proof in Appendix A

### S3: Faithful representation of the CKA-reliability literature
Ding et al. (2021) and Davari et al. (2023) are described accurately: insensitivity to low-variance but functionally relevant directions, and manipulability without functional change. The paper also frames its own tail argument as a special case rather than a discovery.
**Evidence Anchor**: text: §2 Representations and alignment, "Prior work has already shown that CKA can miss functionally relevant"

### S4: Transparent treatment of the UQ-decomposition caveats and failed predictions
The paper acknowledges known limits of entropy-based EU/AU decompositions and estimator entanglement. It reports that the AU-protection prediction and the label-free prediction for softmax heads failed. This keeps its domain claims calibrated.
**Evidence Anchor**: table: Table 1 (pre-registered predictions and outcomes, including E2-b and E4 marked not supported)

### W1: Positioning against the platonic-representation literature misstates which metric carries the convergence claim
The introduction says the convergence claim is made "mainly through kernel alignment" and cites Huh et al. (2024). The abstract opens with CKA as the standard evidence. But the main alignment measure in Huh et al. is mutual k-NN overlap, with CKA and other indices secondary. This matters for the paper's thesis. The paper's own Table 7 shows mutual k-NN tracking EU agreement nearly as well as S_ρ for raw width (0.85 vs 0.87), closed form (0.90 vs 0.92) and AU (0.86 vs 0.92), and well above CKA. So the critique applies to CKA much more than to the metric behind the motivating claim. The Discussion and Recommendations never address mutual k-NN. The title's "aligned representations" therefore claims more than the CKA-specific conclusions support. The five encoders also have no model-scale axis, so the paper cannot engage with the scale-dependent convergence trend that platonic-representation work emphasises. Suggested remedy: correct the attribution, discuss mutual k-NN as a separate case (theoretically, given its invariances and its sensitivity to the tail), and narrow the framing of the title and abstract to CKA where appropriate.
**Severity**: Major
**Evidence Anchor**: text: §1 Introduction, "a claim made mainly through kernel alignment (Huh et al., 2024)"
**Confidence**: 4 — direct familiarity with the platonic-representation paper's metric choices; Table 7 numbers read from the manuscript

### W2: Spectral-reweighting view of CKA versus CCA is not attributed to its source
The "why CKA fails" argument says that CKA weights shared eigendirections by squared variance while whitened indices weight them uniformly. Kornblith et al. (2019) already wrote linear regression, CCA and linear CKA as differently weighted sums of eigenvector correlations, with CKA weighting by eigenvalue products. The paper cites Kornblith et al. only for the CKA definition and its preference over CCA for layer identification, not for this decomposition. The new part is the w_i(ρ)² weighting tied to EU, and that remains new. But the reader should be told that the λ² versus uniform contrast is established.
**Severity**: Minor
**Evidence Anchor**: text: §3 after Proposition 3, "its value lies in the interpretation. In the eigenbasis, CKA"
**Confidence**: 4 — familiar with the CKA/CCA/regression reweighting section of Kornblith et al. (2019)

### W3: Biological-cortex result cited for spectral tails of learned representations
Stringer et al. (2019) report power-law eigenspectra of mouse visual-cortex population responses, not of learned network representations. Citing it as evidence that "learned representations" have long tails is a misattribution. Agrawal et al. (2022) is the appropriate source for learned encoders. Either rephrase to say "neural and learned representations" or cite Stringer et al. only for the biological analogy.
**Severity**: Minor
**Evidence Anchor**: text: §3 after Proposition 3, "Since learned representations have long" and "(Stringer et al., 2019; Agrawal et al., 2022)"
**Confidence**: 5 — the Stringer et al. paper is a neuroscience recording study

### W4: Functional and prediction-level similarity literature is absent
The paper motivates the question with "shared blind spots" and active-learning transfer. A direct literature already asks whether models share errors or uncertain inputs, using functional rather than representational measures. Examples are error consistency (Geirhos et al., NeurIPS 2020), model similarity and test-set reuse (Mania et al., NeurIPS 2019), disagreement-based generalisation estimates (Jiang et al., ICLR 2022), agreement-on-the-line (Baek et al., NeurIPS 2022), model stitching (Bansal, Nakkiran and Barak, NeurIPS 2021), and the representational-versus-functional survey by Klabunde et al. These are the natural alternative to representation-only indices for predicting EU agreement, and the paper should position S_ρ against them. Linear predictivity is the only functional-flavoured index included.
**Severity**: Minor
**Evidence Anchor**: absence: Introduction, Section 2 related work and Section 5 Discussion — expected discussion of functional or prediction-level model similarity and error-consistency work as the competing route to detecting shared blind spots; checked Introduction, Section 2, Section 5 Recommendations, References list
**Confidence**: 4 — these works are standard in the model-similarity literature; the absence was checked against the full reference list

### W5: EU score is a ridge leverage score, but that literature is not connected
At training points, u(x) is (up to scaling) the ridge leverage score, and Lemma 7's d_eff(ρ) is the sum of ridge leverage scores, i.e. the statistical degrees of freedom. The paper cites Zhang (2005) and Caponnetto and De Vito (2007) for d_eff but never uses the term "leverage score". It also omits the ridge-leverage literature in kernel methods (e.g. Bach, COLT 2013; Alaoui and Mahoney, NeurIPS 2015). Making this link would place Propositions 5 and 6 within a known theory and help readers from kernel methods.
**Severity**: Minor
**Evidence Anchor**: text: §2 Heads and epistemic uncertainty, "the EU score; it interpolates between a" and "Mahalanobis leverage (ρ →0)"
**Confidence**: 4 — standard identity in ridge regression and kernel approximation theory

### W6: Noise-aware representational similarity is not discussed
EU agreement compares second-order (covariance) structure induced by heads. The closest representation-similarity line is stochastic shape metrics, which compare representations through their noise covariances (Duong et al., ICLR 2023, building on Williams et al., 2021, which the paper cites). The related-work paragraph concludes that prior indices "concern predictions or decodable information". That description overlooks this line, which concerns stochastic and second-order structure directly.
**Severity**: Minor
**Evidence Anchor**: absence: Section 2 Representations and alignment — expected discussion of stochastic or noise-covariance-aware shape metrics as the closest second-order similarity measures; checked Section 2, Section 3 positioning paragraph after Definition 1, References list
**Confidence**: 3 — confident the line of work exists; less certain how closely its formal objects map onto S_ρ

### W7: Feature tap and pooling choices bear on the CKA-versus-CCA comparison and are non-canonical
Features are mean-pooled patch tokens of the raw block output (at the last depth, apparently before the final norm). This is not the representation usually compared or probed for CLIP and supervised ViTs (pooled or CLS output after the final norm). In ViT residual streams, high-norm artifact tokens and a few large-magnitude feature dimensions are known to exist (e.g. Darcet et al., "Vision Transformers Need Registers", ICLR 2024). These load the top eigendirections, which is exactly where CKA concentrates its weight. The empirical CKA deficit in Table 5 may therefore be partly specific to this tap. This is consistent with the paper's mechanism, but it limits how far the empirical ordering generalises. Robustness to pooling is listed as not run.
**Severity**: Minor
**Evidence Anchor**: text: §4 Data, encoders and heads, "pooled patch tokens (spatial mean for ConvNeXt) of the raw block output at relative depths 0.5, 0.75, 1"
**Confidence**: 3 — familiar with ViT activation artifacts; magnitude of the effect on these specific features is unknown

### W8: Recommendation for whitened indices omits their known high-dimensional failure mode
Recommendation (i) advises preferring whitened (CCA-type) indices. Kornblith et al. (2019) showed that indices invariant to invertible linear transformations become uninformative when feature dimension approaches the number of examples. Sample estimates of the ρ→0 limit are also biased upward when d/n is not small. Here n is large relative to d is a few hundred, but practitioners using the recommendation in low-data regimes (active learning, which the paper names as a motivation) would hit exactly this failure. The recommendation should carry the d versus n caveat, or should prefer S_ρ at a non-vanishing ρ for that reason.
**Severity**: Minor
**Evidence Anchor**: text: §5 Recommendations, "If an index is needed, prefer"
**Confidence**: 4 — the invariance argument against CCA-type indices is a central point of Kornblith et al. (2019)

### W9: "EU" label for softmax-probe summaries stretches field terminology
The paper calls the summed bootstrap quantile width an EU summary. Yet it reports that width is almost a deterministic function of max-probability on every encoder. In the UQ literature the term "epistemic" implies reducibility with data. A quantity with rank correlation of magnitude about 0.999 with confidence is better described as predictive or total uncertainty. The paper concedes the point in its Discussion. Still, the "EU agreement" label in the abstract, tables and recommendations for softmax probes invites the conflation that the cited work of Wimmer et al. and Mucsányi et al. warns against. Suggest reserving "EU" for the closed form and Laplace/MI quantities, or qualifying the bootstrap width as an EU proxy throughout.
**Severity**: Minor
**Evidence Anchor**: text: §4.2, "probes the bootstrap width has rank correlation -0.999"
**Confidence**: 4 — standard EU/AU terminology; empirical numbers taken from the manuscript

### W10: Scale-as-prior equivalence is textbook, but the Discussion presents it as a robust finding
That rescaling features at fixed ridge penalty is equivalent to rescaling the prior is a standard property of ridge regression. Proposition 5 states it correctly. The Discussion, however, lists it among the "robust" findings, which overstates its novelty relative to prior knowledge. The domain-relevant point is the narrower one the paper makes elsewhere: CKA cannot detect whether priors are matched to feature scales. The Discussion should be framed that way.
**Severity**: Minor
**Evidence Anchor**: text: §5 What the results say, "scale hides the prior–scale mismatch that moves EU"
**Confidence**: 4 — elementary ridge-regression property
