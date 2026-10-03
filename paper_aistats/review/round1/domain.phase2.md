contract_role: domain
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: warn
trigger: "mild overstatement of how prior work relates to the contribution"

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

I review as a researcher in representation similarity (CKA, CCA variants, shape metrics, platonic-representation claims) who also works on uncertainty. The paper asks whether representational alignment tells us whether two linear heads will agree on epistemic uncertainty (EU). It derives an exact Gaussian identity linking the correlation of Bayesian-linear posterior variances to a ridge-regularised alignment index S_rho. CKA and a squared-canonical-correlation quantity are the two limits of that index. The paper then tests the predictions on five frozen encoders with CIFAR-10H.

The domain content is mostly accurate. The linear CKA definition and its invariance group are stated correctly. The Isserlis-based covariance of Gaussian quadratic forms is applied correctly. The ridge interpolation between correlation and covariance analyses is credited to classical sources, and the d_eff identity is credited to Zhang (2005) and Caponnetto and De Vito (2007). The authors report falsified pre-registered predictions openly, which is rare in this literature and strengthens how much a reader can trust the domain claims.

My D2 concern is positioning. In several places the manuscript undersells how much of its spectral argument the representation-similarity literature already contains. In particular, the low-variance-tail insensitivity of CKA is already documented. The bootstrap-versus-posterior relationship is textbook ridge theory. And the empirical advantage of S_rho is, on the authors' own Table 5, indistinguishable from the classical canonical-correlation limit and from linear predictivity on the headline partial metric. The EU-specific framing also needs to engage with recent evidence that practical AU and EU estimators are not disentangled. That evidence bears directly on the falsified E2b prediction and on the caveat that S_rho predicts AU as well as EU.

None of these issues breaks the core claim. Corollary 8 alone shows that identical features do not imply EU agreement, and that holds by construction. But the novelty and the recommendation (i) are stated more strongly than prior work allows. Hence warn rather than pass.

### S1: Transparent reporting of falsified pre-registered predictions
The pre-registration table lists failed falsifiers (E2b, E4) and a partly failed d_eff prediction next to the confirmed ones. The Discussion then explicitly narrows the claims to non-identification rather than irrelevance of the representation. This is accurate scholarly positioning that respects the evidence.
**Evidence Anchor**: text: Abstract "predictions that failed are reported alongside those that held"

### S2: Correct statement of CKA, its invariances, and the Gaussian identity
Population linear CKA is defined in Kornblith et al.'s form, with the correct invariance group (isotropic scale and orthogonal maps, not general invertible maps). The covariance of Gaussian quadratic forms via Isserlis' theorem is derived correctly. The rho-to-infinity limit recovering CKA follows cleanly.
**Evidence Anchor**: equation: Theorem 2 and Proposition 3(i), Section 3, with proofs in Appendix A

### S3: Honest attribution of the ridge-interpolation idea
The authors do not claim the regularised index itself as new. They credit canonical ridge and kernel CCA, and restrict the novelty to the EU link. That is the correct framing of the theory contribution.
**Evidence Anchor**: text: Section 3 after Definition 1 "Ridge interpolation between correlation and covariance analyses is classical (Vinod, 1976; Bach and Jordan, 2002)"

### S4: Domain-appropriate AU reference with a reliability ceiling
Using CIFAR-10H soft labels as an aleatoric target with a Spearman-Brown corrected split-half ceiling is the right practice in the human-uncertainty literature. Reporting every AU number against that ceiling prevents over-reading the AU results.
**Evidence Anchor**: text: Section 4 E0 "reliability ceiling of CIFAR-10H entropy is 0.70"

### W1: The empirical advantage attributed to S_rho is indistinguishable from classical canonical-correlation similarity and from linear predictivity
The abstract headline and recommendation (i) present S_rho at the head's own rho as "the better-founded choice". But Table 5 does not separate S_rho at the head's rho from the exploratory rho-to-0 limit, which is a squared-canonical-correlation index from the SVCCA/PWCCA/R2_CCA family (Raghu et al., 2017; Morcos et al., 2018; Kornblith et al., 2019):

- On the headline partial bootstrap metric, both score 0.83. Symmetric linear predictivity also scores 0.83.
- On raw width and MI, the rho-to-0 limit slightly beats S_rho at the head's rho.

Section 4.4 itself notes that all heads sit in the small-rho regime (d_eff/d above 0.98). So the evidence supports "CCA-type indices order EU agreement better than CKA", not the superiority of a new head-calibrated index. This matters for domain positioning because Kornblith et al. (2019) reached the opposite preference (CKA over CCA) for identifying corresponding layers. The paper should state that its finding reverses that preference for a different target, and explain why. The headline comparison should name CCA and linear predictivity rather than only CKA.
**Severity**: Major
**Evidence Anchor**: table: Table 5 (Supplement C), rows "linear predictivity", "S_rho to 0" and "S_rho (head's rho)", column "width partial"
**Confidence**: 4 (read directly off the authors' table; expert familiarity with the CCA-similarity family)

### W2: Prior work on CKA's functional blind spots is described as only "statistical reliability", which overstates the novelty of the tail argument
Section 2 puts Ding et al. (2021) and Davari et al. (2023) under "statistical reliability" and then claims "we ask instead" what alignment implies downstream. But both papers already ask functional questions:

- Ding et al. ground similarity measures in functional behaviour (probe accuracy and out-of-distribution performance). They show specifically that CKA fails to detect the removal of low-variance principal components that change function.
- Davari et al. show CKA can be manipulated substantially without changing model behaviour.

These results are the direct precedent for the paper's "CKA is dominated by the top eigendirections" mechanism and for the CKA half of Proposition 6 (tail reshaping keeps CKA high). The EU-specific consequences are new. The spectral-insensitivity observation and the dissociation of CKA from function are not, and Section 3's in-eigenbasis discussion should credit them.
**Severity**: Major
**Evidence Anchor**: text: Section 2 "Prior work examined the statistical reliability of these measures (Ding et al., 2021; Davari et al., 2023)" and "we ask instead what they imply for downstream uncertainty"
**Confidence**: 4 (close familiarity with both cited works)

### W3: The EU-specific framing is not positioned against evidence that practical AU and EU estimators are entangled
The title and abstract frame the result around epistemic uncertainty specifically. Yet several of the authors' own results show that their softmax-head EU estimators do not isolate EU:

- E2b is falsified: AU degrades as much as EU.
- S_rho predicts AU agreement as well as EU agreement.
- Raw EU agreement is mostly shared confidence.

This pattern matches the recent disentanglement benchmarking literature. Mucsanyi, Kirchhof and Oh (NeurIPS 2024, "Benchmarking Uncertainty Disentanglement") report that common AU and EU estimators are highly correlated in practice. The paper cites only Wimmer et al. (2023) and does not use this literature to interpret its empirical half. The theory (label-free Bayesian-linear posterior variance) is cleanly epistemic. The empirical claims are therefore claims about "head uncertainty" more broadly. The framing should separate the two, or justify why the residualised bootstrap width counts as EU.
**Severity**: Major
**Evidence Anchor**: text: Section 4.4 "Sρ predicts AU agreement as well as EU agreement (0.92)"
**Confidence**: 3 (familiar with the disentanglement literature; Bayesian-statistics details are partly outside my core expertise)

### W4: Lemma 9 is presented as a contribution but restates textbook ridge sampling theory and known bootstrap-versus-posterior results
Contribution 3 lists the "characterisation of bootstrap ensembles as a shrunk posterior". But two parts of it are already established. The sandwich covariance sigma^2 A^-1 G A^-1 is the standard sampling covariance of the ridge estimator (Hoerl and Kennard, 1970). In the deep-ensemble literature, the observation that a bootstrap without a prior term under-represents posterior variance in low-data directions motivated randomized prior functions (Osband, Aslanides and Cassirer, NeurIPS 2018). The derived direction weights w_i^4 and the resulting tail prediction are a useful corollary. The lemma itself should be credited rather than claimed.
**Severity**: Minor
**Evidence Anchor**: equation: Lemma 9, Section 3, proof in Appendix A
**Confidence**: 4 (standard ridge-regression result)

### W5: Reported numbers in Section 4.3 do not match Table 3
The Section 4.3 text, which explicitly cites Table 3, gives values that the table does not contain:

| Quantity | Text | Table 3 |
|---|---|---|
| Laplace MI rank agreement, minimum | 0.72 | 0.85 |
| Bootstrap-width rank agreement, minimum | 0.89 | 0.94 |
| Mean bootstrap-width ratio, range | 0.38 to 1.84 | 0.56 to 1.52 |
| Tail Laplace EU change, maximum factor | 2.3 | 1.8 (or about 1.12 inverse) |
| Tail bootstrap-width change, maximum factor | 1.39 | 1.30 |
| Mutual k-NN at tail s = 4 | 0.62 | 0.68 |

In addition, Table 1 grades the d_eff prediction "partly", while Section 4.4 says it "failed". Appendix B lists d_eff analyses as exploratory, yet Table 1 lists them as pre-registered. The label "E2b" is also used for both the AU falsifier (Table 1) and the transformation experiment (Table 3, Section 4.3). These may come from different encoders, depths or estimators, but the text does not say so. Because P5 and L9 are reported as confirmed predictions, the factual record of these results has to be reconciled.
**Severity**: Major
**Evidence Anchor**: text: Section 4.3 "Laplace MI rank agreement down to 0.72" and "mutual k-NN, which is sensitive to the tail, drops to 0.62"
**Confidence**: 4 (direct comparison of text and table)

### W6: The platonic-representation framing and encoder taps do not match the regime in which convergence was claimed
The Introduction states convergence as established fact and calls kernel alignment "the evidence" for it. But Huh et al. (2024) put it forward as a hypothesis, mainly supported by mutual-kNN alignment that grows with model scale and capability, often across modalities. The earlier CKA/CCA works cited (Raghu et al., 2017; Morcos et al., 2018; Kornblith et al., 2019) are not evidence of cross-objective convergence.

The experiments differ from that regime in three ways:

- They use mean-pooled patch tokens of raw block outputs, not each model's standard embedding (CLS token or CLIP projected embedding).
- They use ViT-B-scale models.
- They use 32-pixel images upsampled to 224 pixels.

The highest CKA reached is 0.837, and E4 concedes that alignment never reached the regime where head and data terms dominate. The theory-level conclusion (Corollary 8) is unaffected. But the Discussion sentence that PRH convergence "can therefore coexist" with EU disagreement is supported empirically only well below the PRH regime. It should be stated that way, and the non-standard taps should be justified against the PRH protocol.
**Severity**: Minor
**Evidence Anchor**: text: Section 1 "The evidence for this convergence is kernel alignment" and Section 4 "of the raw block output at relative depths 0.5, 0.75, 1"
**Confidence**: 4 (core expertise in representational-convergence claims)

### W7: "Mean squared canonical correlation" mislabels the rho-to-0 limit when dimensions differ
Proposition 3(ii) gives the limit as the sum of r_j^2 divided by sqrt(d_A d_B). That equals the mean squared canonical correlation only when d_A = d_B. With unequal dimensions there are at most min(d_A, d_B) canonical correlations, so this normalisation is not the mean. It also differs from the R2_CCA normalisation used by Kornblith et al. (2019). The encoders compared have different widths (for example ConvNeXt-S against the ViT-B models). Contribution 1 and Section 4.4 should use the precise normalisation and relate it to the established CCA similarity indices.
**Severity**: Minor
**Evidence Anchor**: equation: Proposition 3(ii), Section 3
**Confidence**: 4 (direct reading of the stated limit)

### W8: Missing the regularised-shape-metric and decoding-based lineages of similarity measures
The closest conceptual precedent for "what alignment implies about downstream linear heads under ridge regularisation" is the line of work that interprets similarity measures through optimal linear decoding with a regularisation parameter. Williams et al. (2021) already define a regularised shape-metric family that interpolates between CCA-like and Procrustes-like distances. The paper cites them only for metric properties. Harvey, Lipshutz and Williams (2024, "What representational similarity measures imply about decodable information") relate CKA, CCA and Procrustes to the alignment of ridge-regularised linear decoders. Neither connection is discussed. S_rho should be positioned as a member, or a variant, of these families, so that readers can see what the EU identity adds beyond the existing decoding interpretation.
**Severity**: Major
**Evidence Anchor**: absence: Section 2 related work and the References list — expected discussion of the regularised shape-metric family and of decoding-based interpretations of CKA and CCA under ridge regularisation; checked Section 2, Section 3 after Definition 1, Section 5 Discussion, References
**Confidence**: 3 (confident in the Williams et al. regularised family; confident the Harvey et al. line exists, less certain of every detail of its results)

### W9: Bayesian last-layer and distance-aware uncertainty lineage is under-cited
The EU score u(x) is the posterior variance of a Bayesian last layer on frozen features. It is linked only to Mahalanobis OOD scoring (Lee et al., 2018) and DDU (Mukhoti et al., 2023). The neural-linear / Bayesian-last-layer literature (for example, Snoek et al. 2015 for scalable Bayesian optimisation with neural features, and Riquelme et al. 2018's bandit showdown) is not cited. Neither is distance-aware single-model uncertainty (Liu et al., 2020, SNGP). That literature has already documented that last-layer EU depends on feature scale and prior strength. The omission is not load-bearing for the identity, but it understates how established Proposition 5's "scale is a prior change" observation is.
**Severity**: Minor
**Evidence Anchor**: absence: Section 2 Heads and epistemic uncertainty paragraph — expected citations to Bayesian last-layer, neural-linear and distance-aware uncertainty methods; checked Section 2, Section 3 Proposition 5 discussion, Section 5, References
**Confidence**: 4 (standard literature in deep uncertainty quantification)
