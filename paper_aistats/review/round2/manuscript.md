# When Do Aligned Representations Agree on What They Do Not Know?

When Do Aligned Representations Agree on What They Do Not
Know?
Anonymous Author
Anonymous Institution
Abstract
Centred kernel alignment (CKA) is the stan-
dard evidence that diﬀerent vision models
learn similar representations. We ask when
linear heads on two encoders agree about
what they do not know, and show that high
CKA does not imply it, for two reasons we
make exact.
For Bayesian linear heads on
Gaussian features, the correlation across in-
puts of two heads’ epistemic variances equals
a ridge-regularised alignment index Sρ at
the heads’ own prior strength. CKA is the
ρ →∞limit of this index and weights spec-
tral directions by squared variance, whereas
epistemic uncertainty (EU) counts every di-
rection above ρ; we construct pairs with
CKA > 0.99 and EU correlation 0.06, and
the converse.
CKA is also blind to fea-
ture scale, which acts on EU as a change of
prior. Representation-only non-identiﬁcation
follows already from the dependence of EU
on prior and data; our results say which spec-
tral information an index needs and predict
that whitened, canonical-correlation-type in-
dices should outperform CKA for weakly reg-
ularised heads. On ﬁve frozen encoders with
CIFAR-10H labels, the evidence is consis-
tent with this but limited: across 30 encoder
pairs, CCA-type indices and linear predictiv-
ity order conﬁdence-controlled EU agreement
at Spearman ≈0.83 against 0.64 for CKA, but
with ﬁve encoders the interval for the diﬀer-
ence includes zero (0.00 to 0.50). Heads on
identical features (CKA = 1) trained on dif-
ferent data disagree beyond estimator noise,
and rescaling features at ﬁxed weight decay
shifts mean Laplace EU over three orders
Preliminary work. Under review by AISTATS 2027. Do
not distribute.
of magnitude while CKA stays at one. For
softmax probes, EU summaries are almost
a function of conﬁdence (rank correlation -
0.999 to -0.997), so EU and aleatoric agree-
ment behave alike; this and other failed pre-
registered predictions are reported in full.
1
INTRODUCTION
Neural networks trained with diﬀerent objectives, ar-
chitectures and data are converging toward similar
representations, a claim made mainly through kernel
alignment (Huh et al., 2024): centred kernel alignment
(CKA; Kornblith et al., 2019), canonical correlations
(Raghu et al., 2017; Morcos et al., 2018), and mutual
nearest-neighbour overlap. A natural but untested in-
ference is that aligned encoders also induce heads that
are unsure about the same inputs, which would make
alignment a label-free tool for transferring uncertainty
estimates, selecting encoders for active learning, or ar-
guing that aligned models share blind spots. We are
not aware of work that uses alignment this way ex-
plicitly; our question is whether doing so would be
justiﬁed.
We study this for epistemic uncertainty (EU), the re-
ducible part that reﬂects limited data (Kendall and
Gal, 2017; Hüllermeier and Waegeman, 2021). At one
level the answer is immediate: the EU of a head de-
pends on its prior and training data, which no function
of two representations can see. The useful question is
sharper: which information about a pair of represen-
tations controls EU agreement once the head is ﬁxed,
and does the most widely used index, CKA, carry
it? Our answer is that CKA discards exactly the two
pieces of information that matter. It normalises away
the scale of the features, and a regularised head treats
scale as part of its prior. And it weights spectral di-
rections by their squared variance, whereas EU counts
every direction above the regularisation level, however
small its variance.
Contributions.

1. An exact identity (Theorem 2): for Gaussian fea-
tures and Bayesian linear heads, the correlation of
two heads’ epistemic variances equals Sρ, the co-
sine between ridge smoothers at the heads’ prior
strengths, an index from the ridge-CCA family.
CKA is its ρ →∞limit and a normalised sum
of squared canonical correlations its ρ →0 limit
(Proposition 3).
2. Why CKA fails: CKA is neither necessary nor
suﬃcient for EU agreement (Corollary 4); its in-
variances are not invariances of EU (Propositions 5
and 6); and even identical features give imperfect
EU agreement once priors diﬀer (Corollary 8). We
also note an in-sample resolvent bound (Propo-
sition 10), the identity between mean EU and
deﬀ(ρ) (Lemma 7), and the consequence of the
classical bootstrap–posterior relation for agreement
(Lemma 9).
3. Pre-registered and post-review experiments
on ﬁve frozen encoders with CIFAR-10H human la-
bels, using bootstrap, Laplace and closed-form es-
timators, re-test reliabilities, encoder-cluster inter-
vals and reliability ceilings. Conﬁrmed and failed
predictions are reported alike (Table 1).
2
SETUP AND RELATED WORK
Representations
and
alignment.
An
encoder
ϕA : X
→RdA is frozen; inputs x ∼P.
We
centre features and write ΣA = E[ϕAϕ⊤
A], ΣAB =
E[ϕAϕ⊤
B].
Population linear CKA is CKA(A, B) =
∥ΣAB∥2
F /(∥ΣA∥F ∥ΣB∥F ) (Kornblith et al., 2019); it
is invariant to ϕ 7→cQϕ for c > 0 and orthogonal Q.
Mutual k-NN with cosine similarity (Huh et al., 2024)
shares this invariance. Prior work has already shown
that CKA can miss functionally relevant diﬀerences: it
is insensitive to low-variance directions that matter for
downstream behaviour (Ding et al., 2021) and can be
manipulated without changing function (Davari et al.,
2023). Our tail argument is a special case of this in-
sensitivity, made exact for one downstream quantity.
Other lines of work regularise shape metrics (Williams
et al., 2021), deﬁne distances through ridge-regularised
linear prediction (GULP; Boix-Adserà et al., 2022), or
interpret CKA and CCA as average alignment of op-
timal linear decoders (Harvey et al., 2024); see also
RSA (Kriegeskorte et al., 2008). These works concern
predictions or decodable information.
We ask what
alignment implies for the uncertainty of a head.
Heads and epistemic uncertainty.
A head is ﬁt-
ted on n labelled examples with feature matrix ΦA ∈
Rn×dA. For a Bayesian linear head with prior w ∼
N(0, τ 2I) and noise variance σ2, the posterior vari-
ance of w⊤ϕA(x) is
vA(x) = σ2ϕA(x)⊤(Φ⊤
AΦA+λI)−1ϕA(x),
λ = σ2/τ 2,
(1)
which does not depend on the labels (Rasmussen and
Williams, 2006).
With ˆΣA = Φ⊤
AΦA/n and ρ =
λ/n, nvA(x)/σ2 = ϕA(x)⊤(ˆΣA + ρI)−1ϕA(x).
We
call the population version uA(x) = ϕA(x)⊤(ΣA +
ρAI)−1ϕA(x) the EU score; it interpolates between a
Mahalanobis leverage (ρ →0) and a squared norm
(ρ
→
∞), connecting to Bayesian last-layer and
neural-linear models (Snoek et al., 2015; Riquelme
et al., 2018) and to distance-aware feature-space un-
certainty (Lee et al., 2018; van Amersfoort et al., 2020;
Liu et al., 2020; Mukhoti et al., 2023). In experiments
we also use softmax heads with bootstrap ensem-
bles (Efron and Tibshirani, 1993; Lakshminarayanan
et al., 2017), summarised by the mutual informa-
tion (Houlsby et al., 2011) and by quantile-interval
widths, and last-layer Laplace (Daxberger et al., 2021).
Entropy-based decompositions have known limitations
(Wimmer et al., 2023), and in practice AU and EU
estimators are strongly entangled (Mucsányi et al.,
2024); we therefore report several EU summaries, in-
cluding the closed form, which does not depend on
labels.
EU agreement.
For two models we measure agree-
ment as the correlation of their EU scores across in-
puts; in experiments, as Spearman correlation, partial
Spearman controlling for both models’ conﬁdence and
human-label entropy, and Spearman within human-
entropy deciles. Partial and stratiﬁed variants reduce,
but cannot remove, the trivial explanation that both
models are uncertain on intrinsically ambiguous im-
ages, because human entropy is itself a noisy covariate
(reliability 0.70); Section 4.2 reports a sensitivity anal-
ysis over covariate estimators.
3
THEORY
All proofs are in Appendix A. Let wi(ρ) = λi/(λi + ρ)
for eigenvalues λi of a covariance.
Deﬁnition 1 (Spectral agreement index). For sym-
metric MA, MB let
S(MA, MB) =
tr(MAΣABMBΣBA)
√
tr
(
(MAΣA)2)
tr
(
(MBΣB)2),
and Sρ := S
(
(ΣA + ρAI)−1, (ΣB + ρBI)−1)
.
On a sample, Sρ is the Frobenius cosine between the
ridge smoothers H = Φ(Φ⊤Φ+nρI)−1Φ⊤, computable
with d × d matrices only.
Sρ is not a new simi-
larity measure: ridge interpolation between correla-
tion and covariance analyses is classical (Vinod, 1976;

Bach and Jordan, 2002), and GULP (Boix-Adserà
et al., 2022) compares representations through ridge-
regularised linear predictors, which is closely related.
Comparisons of optimal linear readouts also underlie
CKA and CCA (Harvey et al., 2024). What we add
is the exact link between this family and agreement
of posterior variances, which identiﬁes the head’s own
prior strength as the right regularisation level.
Theorem
2
(EU
agreement
identity).
If
(ϕA(x), ϕB(x))
is
zero-mean
jointly
Gaussian
under x ∼P, then for any symmetric MA, MB,
corr
(
ϕ⊤
AMAϕA,
ϕ⊤
BMBϕB
)
=
S(MA, MB).
In
particular corr(uA, uB) = Sρ.
Proposition 3 (Limits). With ρA = ρB = ρ: (i)
limρ→∞Sρ = CKA(A, B); (ii) if ΣA, ΣB are invert-
ible, limρ→0 Sρ = ∑
j r2
j/√dAdB, where rj are the
canonical correlations of A and B; for dA = dB = d
this is the mean squared canonical correlation (the
R2
CCA similarity), and otherwise the sum of squared
canonical correlations normalised by √dAdB.
The proof is one application of Isserlis’ theorem; its
value lies in the interpretation. In the eigenbasis, CKA
weighs a shared direction by λ2
i , whereas Sρ weighs
it by wi(ρ)2: every direction with λi ≫ρ counts as
one, however small its variance.
This gives a con-
crete prediction: when heads are weakly regularised,
whitened (CCA-type) indices should track EU agree-
ment better than CKA, reversing the preference for
CKA over CCA that Kornblith et al. (2019) estab-
lished for a diﬀerent target (identifying corresponding
layers). Since learned representations have long spec-
tral tails (Stringer et al., 2019; Agrawal et al., 2022),
the two indices read diﬀerent parts of the spectrum.
Corollary 4 (CKA is neither necessary nor suﬃcient).
Let ϕA = (s, a), ϕB = (s, b) with s ∈Rk, a, b ∈Rm
independent Gaussians with covariances λsI, λpI, λpI.
Then CKA =
kλ2
s
kλ2s+mλ2p and corr(uA, uB) =
kw2
s
kw2s+mw2p .
Hence for every ϵ > 0 there are pairs with CKA ≥1−ϵ
and corr(uA, uB) ≤ϵ, and pairs with CKA ≤ϵ and
corr(uA, uB) ≥1 −ϵ.
Proposition
5
(Scale
is
a
prior
change). For
c
>
0 and orthogonal Q,
CKA(ϕ, cQϕ)
=
1,
cosine
k-NN
graphs
are
unchanged,
and
v(cQΦ, cQx; λ)
=
v(Φ, x; λ/c2).
Under
Gaus-
sian features, the EU agreement between the head
on cQϕ and the head on ϕ at the same λ is
∑
i wi(ρ)wi(ρ/c2)
/√∑
i wi(ρ)2 ∑
i wi(ρ/c2)2,
which
is < 1 unless the spectrum is ﬂat on its support.
Proposition 6 (Tail reshaping). Let Σ = UΛU ⊤, let
T be a set of tail directions, and let Ts scale them
by s. With η = ∑
i∈T λ2
i / ∑
i/∈T λ2
i , CKA(ϕ, Tsϕ) =
1+s2η
√
(1+η)(1+s4η) ≥1 −(s2 −1)2η, whereas the eﬀective
dimension changes by ∑
i∈T
[
wi(ρ/s2) −wi(ρ)
]
.
Lemma 7 (Mean EU is eﬀective dimension). On the
training points,
1
n
∑
i v(xi) = σ2
n deﬀ(ρ) with deﬀ(ρ) =
∑
i
ˆλi
ˆλi+ρ
(Zhang, 2005; Caponnetto and De Vito,
2007).
Together, Proposition 6 and lemma 7 give a transfor-
mation under which CKA stays above 1 −(s2 −1)2η
while mean EU changes by a factor of order |T |/deﬀ.
The participation ratio (∑λi)2/ ∑λ2
i , a common “ef-
fective rank”, is top-heavy like CKA and does not track
deﬀ(ρ).
Corollary
8
(Identical
features
do
not
imply
EU
agreement).
Two
heads
on
the
same
fea-
tures
with
prior
strengths
ρ1
̸=
ρ2
(for
a
summed likelihood with ﬁxed λ,
ρ
=
λ/n,
so
diﬀerent
sample
sizes
suﬃce)
have
EU
agree-
ment ∑
i wi(ρ1)wi(ρ2)/
√∑
i wi(ρ1)2 ∑
i wi(ρ2)2 un-
der Gaussian features, while every alignment metric
equals its maximum.
Lemma 9 (Bootstrap is a shrunk posterior; classi-
cal). For ﬁxed-design ridge with residual resampling,
Cov( ˆw) = σ2A−1GA−1 with G = Φ⊤Φ, A = G + λI,
versus the posterior σ2A−1; along eigendirection i the
ratio is gi/(gi + λ).
This is the textbook sandwich covariance of ridge re-
gression; the gap between bootstrap and posterior
variance also motivates randomised-prior ensembles
(Osband et al., 2018). Our only addition is the con-
sequence for agreement: the bootstrap EU score is
ϕ⊤(Σ + ρ)−1Σ(Σ + ρ)−1ϕ, so Theorem 2 applies with
direction weights w4
i rather than w2
i . Bootstrap EU
is therefore more top-heavy than posterior or Laplace
EU, and we predict it moves less under tail reshaping.
Proposition 10 (In-sample resolvent bound). For
Gram matrices K, L on n observed points and noise
s2 > 0, the posterior covariances P(G) = G −G(G +
s2I)−1G satisfy ∥P(K) −P(L)∥op ≤∥K −L∥op.
The bound uses the unnormalised Gram in operator
norm, exactly the information that CKA discards. It
does not extend with constant one to unobserved eval-
uation points: in random trials with partially observed
points the ratio reached ≈5 (Appendix A). Separately,
because EU combines the training design (through the
posterior) with the evaluation inputs (where it is read
out), we compute Sρ on training ∪evaluation inputs;
a training-only variant is reported in Section 4.4.
4
EXPERIMENTS
Pre-registration.
Predictions, falsiﬁers and gate
thresholds were frozen in code before any real-data re-
sult was computed (hash in the supplement); Table 1

lists every one with its outcome. Analyses not listed
there are marked exploratory.
Data, encoders and heads.
We use CIFAR-10
(Krizhevsky, 2009) training images (50k, hard labels)
to ﬁt heads and the 10k test images with CIFAR-
10H soft labels (about 50 annotators per image; Pe-
terson et al., 2019) for evaluation only.
Five frozen
encoders diﬀer in objective and architecture: DINOv2
ViT-B/14 (Oquab et al., 2024), CLIP ViT-B/16 im-
age tower (Radford et al., 2021; Cherti et al., 2023),
supervised ViT-B/16 (Dosovitskiy et al., 2021; Steiner
et al., 2022), ConvNeXt-S (Liu et al., 2022) and MAE
ViT-B/16 (He et al., 2022), all via timm/open_clip
(Wightman, 2019). Images are upsampled from 32 to
224 pixels (a shared limitation). Features are mean-
pooled patch tokens (spatial mean for ConvNeXt) of
the raw block output at relative depths 0.5, 0.75, 1,
centred and divided by one scalar computed on the
training set; no per-dimension standardisation is ap-
plied, since it would undo the transformations of
Propositions 5 and 6. Heads are softmax regressions
with weight decay chosen per encoder and depth by
validation NLL on a held-out 10k training split. EU
is estimated with M = 50 Poisson-bootstrap mem-
bers (quantile width q.95 −q.05 summed over classes,
and mutual information), with class-block last-layer
Laplace, and with the closed form u(x) at ρ equal to
the weight decay.
Reliability and positive control (E0).
The split-
half reliability ceiling of CIFAR-10H entropy is 0.70
(Spearman–Brown corrected, 95% CI [0.69, 0.71]); ev-
ery AU number below should be read against it. Plug-
in and Dirichlet-posterior entropy estimators agree at
Spearman 0.95. In a known-AU control, where labels
are drawn from a well-speciﬁed softmax teacher on
each encoder’s features, model AU recovers the true
AU with Spearman between 0.89 and 0.96 across en-
coders and temperatures, so the pre-registered gate
passes.
4.1
Synthetic veriﬁcation of the theory
Over 60 random representation pairs (shared and pri-
vate blocks with power-law spectra, random mixings,
one third with Laplace instead of Gaussian latents)
and four values of ρ, the sample Sρ predicts the em-
pirical EU correlation with mean absolute error 0.010
(correlation 0.999), while CKA has mean absolute er-
ror 0.37 and correlation 0.32 with EU agreement (Fig-
ure 1a). The error of CKA shrinks as ρ grows (0.57
at ρ = 0.01 to 0.07 at ρ = 10), as Proposition 3 pre-
dicts.
The construction of Corollary 4 with k = 9
shared and m = 200 private directions gives CKA
= 0.998 with EU correlation 0.06; the converse con-
0.0
0.5
1.0
alignment index
0.00
0.25
0.50
0.75
1.00
EU correlation
(a)
Sρ
CKA
0
200
400
private dims m (k = 9)
0.00
0.25
0.50
0.75
1.00
(b)
CKA
EU corr.
Figure 1:
(a) Empirical EU correlation versus Sρ
(ﬁlled) and CKA (open) over 60 synthetic pairs and
four ρ. (b) The construction of Corollary 4 (k = 9,
λs = 10, λp = 0.1, ρ = 0.01): CKA stays near one
as private dimensions are added while EU correlation
falls as kw2
s/(kw2
s + mw2
p) ≈k/(k + m); lines are the
exact closed forms of Corollary 4.
struction gives CKA = 0.003 with EU correlation 0.96
(Figure 1b). With identical features and a ﬁxed ridge,
a head trained on 1% of the data has EU correlation
0.76 with the full-data head, below the population pre-
diction of Corollary 8 (0.86): ﬁnite-sample covariance
error adds a data term on top of the change in ρ.
4.2
Identical features, diﬀerent heads (E2)
Within each encoder the paired heads see literally the
same features, so CKA and every other alignment in-
dex equals one. Raw EU agreement is high for all pairs
(Table 2), but this is shared conﬁdence: for softmax
probes the bootstrap width has rank correlation -0.999
to -0.997 with the max-probability and at least 0.994
with AU on every encoder, whereas the closed-form
u(x), which ignores labels, correlates with it at only
-0.08 to 0.31. We therefore read E2 through the par-
tial agreement, controlling for both heads’ conﬁdence
and human entropy.
It falls from a re-test value of
0.56 (two ensembles with identical conﬁguration) to
0.36 for disjoint halves of the training set and 0.26 for
10% versus 100% of the data (Figure 2). Because the
head objective averages the loss, the weight decay sets
the same ρ at every sample size, so this drop reﬂects
the ﬁnite-sample data term (Section 4.1) rather than
the change of ρ in Corollary 8; weight decay was not
re-selected for subset heads. Falsiﬁer E2-a (> 0.9) is
not triggered, but in hindsight it could not have ﬁred,
since the re-test value itself is far below 0.9 (Table 1).
Estimator noise and redraws (post-review, ex-
ploratory).
The re-test partial agreement is limited
by Monte Carlo noise: for DINOv2 it rises from 0.45
at M = 10 to 0.60 at M = 50 and 0.82 at M = 200
(mutual information: 0.90 at M = 200; Figure 3a).

Table 1: Pre-registered predictions and outcomes (frozen before any real-data result). The last column is the
verdict on the prediction. †Uninformative in hindsight: the re-test partial agreement at M = 50 is itself ≈0.56,
so the threshold could not be reached. ‡The frozen falsiﬁer had no numeric threshold; our operationalisation is
post hoc (Appendix B). The CI for S is an encoder-cluster bootstrap (Section 4.4).
id
prediction (falsiﬁer)
observed
verdict
E0
pipeline recovers known AU (gate: Spearman ≥0.5)
0.89–0.96
passed
E2-a
data term matters on identical features (falsiﬁer: partial
EU agr. 10% vs 100% > 0.9)†
0.26
supported
E2-b
AU agreement protected relative to EU (falsiﬁer: AU
degrades as much)‡
drop 0.10 vs 0.10
not supported
E2b-scale rescaling moves EU at ﬁxed wd, not when re-tuned
(Prop. 5)
Laplace ×0.02–62
supported
E2b-tail
bootstrap EU moves less than Laplace EU (Lemma 9)
6/6
supported
E4
head and data dominate at high alignment (falsiﬁer: en-
coder share > 80%)
93%
not supported
S
Sρ predicts EU agreement better than CKA (falsiﬁer: no
better)‡
0.83 vs 0.64; diﬀ. CI [0.00, 0.50]
point est. only
Spec
deﬀ(ρ) predicts mean EU
closed form 0.90; bootstrap -0.19 closed form only
Table 2: E2: heads on identical frozen features (all
alignment metrics = 1). Item-level Spearman agree-
ment, mean (min–max) over ﬁve encoders; partial =
controlling for both models’ conﬁdence and human en-
tropy.
pair
EU width
EU partial
AU
re-test ﬂoor
1.00 (0.99–1.00)
0.56
1.00 (1.00–1.00)
25% vs 100% data 0.94 (0.90–0.96)
0.36
0.94 (0.91–0.97)
10% vs 100% data 0.90 (0.84–0.94)
0.26
0.90 (0.85–0.94)
disjoint halves
0.91 (0.88–0.94)
0.36
0.92 (0.89–0.95)
wd ÷10
0.98 (0.98–0.99)
0.24
0.98 (0.97–0.99)
wd ×10
0.97 (0.97–0.98)
0.49
0.97 (0.96–0.98)
We therefore redrew the subsets 3 times per encoder
and divided each partial agreement by the geometric
mean of the two heads’ own re-test reliabilities (Ta-
ble 3). The reliability-corrected EU agreement is 0.48
(range over encoders 0.26–0.66) for 10% versus 100%
of the data and 0.68 for disjoint halves, clearly below
one: heads on identical features disagree about resid-
ual EU beyond estimator noise. At the decision level,
the overlap of the 10% highest-uncertainty (acquisi-
tion) sets is 0.93 between re-test ensembles but 0.70
between the 10% and 100% heads. Replacing the plug-
in human entropy by the Dirichlet-posterior estimator
or by two independent split-half entropies changes the
partial agreements by at most 0.001, because conﬁ-
dence absorbs almost all shared variance.
Falsiﬁer E2-b is triggered: AU agreement on iden-
tical features degrades about as much as EU agree-
ment (raw drop 0.10 vs. 0.10; corrected AU agreement
0.52 vs. 0.48 for EU). The pre-registered contrast that
AU agreement is protected is not supported for soft-
max probes, consistent with the near-collinearity of
their AU and EU summaries. In an exploratory ob-
re-test
25%
10%
halves
wd/10
wd×10
0.00
0.25
0.50
0.75
1.00
agreement
raw Spearman
EU width
EU MI
AU
re-test
25%
10%
halves
wd/10
wd×10
partial Spearman
Figure 2: E2: agreement between heads on identical
features, mean over ﬁve encoders (bars: min–max).
Right: partial Spearman controlling for both heads’
conﬁdence and human entropy; the dashed line is the
mean partial EU agreement between diﬀerent encoders
at the ﬁnal depth.
servation, AU’s relation to the human target is stable:
per encoder, Spearman between model AU and human
entropy varies by at most 0.05 across the eight head
conﬁgurations, at 0.34–0.43 overall (0.48–0.61 of the
ceiling).
4.3
Transformations that preserve alignment
(E2b)
Applying ϕ 7→cQϕ leaves CKA and mutual k-NN
at one for both encoders tested (DINOv2 and CLIP;
Table 4).
At ﬁxed weight decay, EU moves with c:
across c ∈[0.1, 10] the mean Laplace logit variance
changes by a factor between 0.017 and 61.7, the mean
bootstrap width between 0.38 and 1.84, and EU rank-
ings change (Laplace MI rank agreement down to 0.72,
bootstrap width down to 0.89). When the weight de-
cay is re-tuned by validation, the selected value scales

Table 3: Post-review redraws (exploratory): heads on
identical features, mean over 5 encoders × 3 indepen-
dent subset draws (M = 50). Partial = controlling
for both heads’ conﬁdence and human entropy; cor-
rected = partial divided by the geometric mean of the
two heads’ own re-test partial reliabilities; top-10% =
overlap of the 10% highest-uncertainty sets.
pair
raw partial corrected top-10%
re-test
EU width 1.00
0.56
1
0.93
EU MI
0.99
0.76
1
0.90
AU
1.00
0.87
1
0.97
disjoint halves EU width 0.91
0.36
0.68
0.72
EU MI
0.91
0.52
0.70
0.73
AU
0.92
0.53
0.61
0.74
10% vs 100%
EU width 0.90
0.27
0.48
0.70
EU MI
0.89
0.39
0.50
0.70
AU
0.91
0.46
0.52
0.72
as c2 (from 10−5 at c = 0.1 to 10−1 at c = 10), which
is exactly the prior change of Proposition 5, and the
largest deviation of any EU rank agreement or mean
ratio from the untransformed head is 0.086 (the re-
tuned grid is coarse, one decade per step). The eﬀect
therefore operates through the prior, as Proposition 5
says; it vanishes when the weight decay is re-tuned.
The point is not that practitioners will meet this fail-
ure after tuning, but that CKA cannot tell whether
two heads’ priors are matched to their feature scales.
Tail reshaping (scaling the directions outside the top
95% of variance by s ∈{0.5, 2, 4}) keeps CKA at
or above 0.956 while the mean Laplace logit variance
changes by up to a factor 2.3 and the mean bootstrap
width by up to 1.39 (ranges over both encoders); mu-
tual k-NN, which is sensitive to the tail, drops to 0.62.
The closed-form EU barely moves (mean ratio within
0.04 of one): at the selected weight decay the heads are
in the small-ρ regime, where u(x) approaches the lever-
age, which is invariant to every invertible linear map.
The softmax estimators move because their eﬀective
prior strength is set by the curvature of the likelihood,
not by the weight decay alone. Lemma 9 predicts that
bootstrap EU moves less than Laplace EU under tail
reshaping; this held in 6 of 6 settings (3 values of s ×
2 encoders, comparing rank changes). The lemma is
derived for residual resampling, so this is a heuristic
extrapolation to Poisson-resampled softmax heads.
4.4
Across encoders: which index tracks EU
agreement?
For all 10 encoder pairs at three matched relative
depths (30 pairs) we compute CKA, mutual k-NN
(k = 10), symmetric linear predictivity, Sρ at the two
heads’ own ρ and its ρ →0 limit, on training ∪test in-
puts, together with item-level EU agreement (Table 5
and ﬁg. 4). For partial bootstrap EU agreement, Sρ,
Sρ→0 and linear predictivity all reach 0.83–0.83, mu-
tual k-NN 0.72 and CKA 0.64. This ordering is what
Proposition 3 predicts for weakly regularised heads,
but the evidence is weak: pairs share encoders, and
the encoder-cluster bootstrap interval for Sρ −CKA is
[0.00, 0.50] (0.09 of resamples ≤0). Sρ is indistinguish-
able from linear predictivity (diﬀerence ≈0) and from
its own ρ →0 limit, so at the selected weight decays
we cannot attribute any advantage to the head’s prior
strength. Falsiﬁer S (“no better than CKA”) is not
triggered on the point estimate but would not survive
a test at the 5% level. CKA is not uninformative, and
for raw agreement linear predictivity (0.41) is weaker
still; the highest-CKA pair at the ﬁnal depth (CKA
0.837) reaches a partial EU agreement of only 0.21.
All indices predict AU agreement about as well as EU
agreement (Sρ: 0.92), so they index shared head be-
haviour rather than EU speciﬁcally. On real features
Theorem 2 holds only ordinally: Sρ ranks the closed-
form Pearson agreement at 0.93 (CKA: 0.74) but its
level is oﬀby 0.34, as expected when fourth cumu-
lants of non-Gaussian features enter the covariance of
quadratic forms.
Does the head’s ρ matter?
(post-review, ex-
ploratory).
At the selected weight decays all heads
have deﬀ(ρ)/d > 0.98, so Sρ ≈Sρ→0. We therefore re-
ﬁt all 15 encoder–depth heads (M = 20) at a common
weight decay ρ ∈{10−3, 10−2, 10−1, 1}, which lowers
deﬀ(ρ)/d to 0.06–0.29 at ρ = 1 (Figure 3b). As pre-
dicted, the advantage of evaluating the index at the
matched ρ grows with ρ: for closed-form EU agree-
ment Sρ −Sρ→0 moves from -0.02 at ρ = 10−3 to
+0.05 at ρ = 1 (0.94 vs. 0.89; CKA 0.69), and for
partial bootstrap agreement to +0.08 (0.96 vs. 0.88;
CKA 0.71).
The encoder-cluster intervals at ρ = 1
([-0.01, +0.13] and [+0.00, +0.13]) touch zero, and
strongly regularised heads underﬁt for some encoders
(accuracy 0.50–0.97), so this is a trend, not an estab-
lished eﬀect. Computing Sρ on training inputs only
instead of training ∪test changes its correlation with
EU agreement by at most 0.03. Between diﬀerent en-
coders, the top-10% acquisition sets overlap by only
0.38 on average.
Label-free
prediction.
Across
encoders
and
depths,
deﬀ(ρ)
from
unlabelled
training
features
rank-correlates with mean closed-form EU at 0.90, as
Lemma 7 predicts, but not with the mean bootstrap
width of softmax heads (-0.19), whose magnitude
follows accuracy; this pre-registered prediction failed
for softmax heads.

Table 4: E2b (ﬁxed weight decay): alignment of the transformed to the original features, Spearman rank
agreement of EU with the untransformed head, and mean EU relative to it, for both encoders (width = bootstrap
quantile width; Laplace = MI for the rank, logit variance for the ratio). The text summarises ranges over both
encoders.
DINOv2-B
CLIP-B/16
CKA mKNN
rank agr.
mean ratio
CKA mKNN
rank agr.
mean ratio
transform
width Lapl. width Lapl.
width Lapl. width
Lapl.
c =0.1
1.000
1.00
0.94
0.89
0.56
0.017 1.000
1.00
0.89
0.85
0.38
0.0301
c =0.3
1.000
1.00
0.98
0.94
0.74
0.125 1.000
1.00
0.96
0.95
0.62
0.182
c =1.0
1.000
1.00
1.00
1.00
1.00
1
1.000
1.00
1.00
1.00
1.00
1
c =3.0
1.000
1.00
0.99
0.91
1.29
6.78
1.000
1.00
0.98
0.92
1.45
4.24
c =10.0
1.000
1.00
0.97
0.85
1.52
61.7
1.000
1.00
0.95
0.72
1.84
39.3
tail s =0.5 1.000
0.96
1.00
1.00
1.02
0.894 1.000
0.95
1.00
0.99
0.98
0.788
tail s =2.0 0.998
0.89
1.00
0.99
1.07
1.27
0.999
0.86
0.99
0.98
1.13
1.47
tail s =4.0 0.956
0.68
0.98
0.95
1.30
1.8
0.982
0.62
0.97
0.92
1.39
2.28
Table 5: Across 30 encoder pairs (5 encoders × 3 depths): Spearman correlation between each alignment index
and item-level agreement, with 95% encoder-cluster bootstrap intervals (2000 resamples of encoders; pairs share
encoders, so the eﬀective sample is small). Last row: the diﬀerence Sρ −CKA.
index
width (partial)
closed form
AU
CKA
0.64 [0.48, 1.00]
0.76 [0.14, 1.00]
0.68 [0.30, 0.87]
mutual k-NN
0.72 [0.62, 1.00]
0.90 [0.62, 1.00]
0.86 [0.50, 1.00]
linear predictivity
0.83 [0.76, 1.00]
0.56 [0.32, 1.00]
0.53 [0.30, 1.00]
Sρ→0
0.83 [0.72, 1.00]
0.93 [0.63, 1.00]
0.93 [0.50, 1.00]
Sρ (head’s ρ)
0.83 [0.72, 1.00]
0.92 [0.60, 1.00]
0.92 [0.50, 1.00]
Sρ−CKA
+0.19 [+0.00, +0.50]
+0.16 [-0.15, +0.50]
+0.24 [+0.00, +0.50]
Sρ−linear pred.
-0.00 [-0.22, +0.11]
+0.37 [-0.10, +0.56]
+0.40 [+0.00, +0.59]
Swap design (E4).
In a 2 × 2 × 2 design on the
highest-CKA pair (DINOv2-B vs. supervised ViT-
B/16, CKA 0.837; encoder × weight decay {w, 10w}
× disjoint training half), the encoder accounts for 93%
of the Shapley decomposition of raw EU disagreement,
the training half for 9% and the weight decay for essen-
tially none (-2%). This triggers the pre-registered E4
falsiﬁer (> 80%): among the encoders we could test,
alignment never reached a regime where head and data
terms dominate. In an exploratory variant using par-
tial disagreement, the shares are 61%, 30% and 10%
for encoder, data and weight decay (re-test disagree-
ment 0.33).
5
DISCUSSION
What the results say.
That a representation-only
index cannot determine EU follows from Equation (1);
our contribution is to say which information an index
needs and why CKA lacks it. Theorem 2 identiﬁes the
relevant summary for linear-Gaussian heads, spectral
overlap above the head’s prior strength weighted by
wi(ρ)2, and explains two observations: CCA-type in-
dices tracked EU agreement at least as well as CKA
for weakly regularised heads, and the matched-ρ index
gained ground as regularisation increased. Both are
directionally consistent with the theory, but with ﬁve
encoders neither diﬀerence is statistically established.
Two further ﬁndings are robust: CKA’s invariance to
scale hides the prior–scale mismatch that moves EU
by orders of magnitude, and even identical features
leave the residual EU ranking dependent on the train-
ing data beyond estimator noise.
Between diﬀerent
encoders, however, the representation is the dominant
source of EU disagreement (E4), so our results do not
say that the encoder is irrelevant; they say that CKA
does not tell us how EU will agree. For softmax probes,
EU and AU summaries are nearly collinear with conﬁ-
dence, so the empirical part of this paper cannot sep-
arate EU-speciﬁc from general head agreement; the
EU-speciﬁc mechanisms are established by the theory
and the closed form.
Recommendations.
(i) Do not use CKA to trans-
fer or validate EU. If an index is needed, prefer
whitened (CCA-type) indices or Sρ evaluated at the
head’s prior strength on the inputs where EU is used;
we have not shown a practical advantage of Sρ over
linear predictivity.
(ii) Report the weight decay to-
gether with the feature normalisation used (here a
single scalar on centred features), since the pair de-
termines the eﬀective prior; for Laplace heads, re-

101
102
ensemble size M
0.5
0.6
0.7
0.8
0.9
re-test partial agr.
(a)
EU width
EU MI
AU
10−3
10−2
10−1
100
weight decay = ρ
0.7
0.8
0.9
Spearman with EU agr.
(b)
Sρ
Sρ →0
CKA
Figure 3: Post-review analyses.
(a) Re-test partial
agreement of two independent ensembles versus en-
semble size M (DINOv2, ﬁnal layer): the noise ﬂoor
in Table 2 is a Monte Carlo limit. (b) Weight-decay
sweep over 30 encoder pairs: Spearman correlation of
Sρ at the matched ρ, its ρ →0 limit and CKA with
closed-form (solid) and partial bootstrap (dashed) EU
agreement.
0.25
0.50
0.75
alignment index
0.5
0.6
0.7
0.8
EU agreement (width)
(a) raw
Sρ
CKA
mKNN
0.25
0.50
0.75
alignment index
0.0
0.1
0.2
0.3
0.4
(b) partial
Figure 4:
Across 30 encoder pairs (5 encoders, 3
depths): item-level bootstrap EU agreement, raw (a)
and partial (b), against Sρ at the heads’ ρ (ﬁlled),
CKA (open) and mutual k-NN (crosses).
port the prior precision relative to the trace of the
feature covariance. (iii) Report conﬁdence-controlled
(partial) agreement with re-test reliabilities, since raw
EU agreement of softmax probes mostly reﬂects shared
conﬁdence and the partial measure is noisy at common
ensemble sizes.
Limitations.
The theory is exact for Gaussian fea-
tures and linear-Gaussian heads; on real features only
its ordinal predictions held.
Softmax heads enter
through bootstrap and Laplace approximations; the
bootstrap lemma is for residual resampling with ﬁxed
design, and our Laplace drops cross-class curvature.
Five encoders give 30 dependent pairs, which limits
every cross-encoder conclusion. We study frozen en-
coders and linear probes on one dataset with human la-
bels (CIFAR-10H) at low native resolution and only in-
distribution; out-of-distribution inputs, where shared
blind spots matter most, ﬁne-tuned models, other
modalities and the DCIC datasets are not covered, and
we do not evaluate downstream active-learning per-
formance. The positive control replaces the planned
DCIC synthetic dataset with a known softmax teacher
on each encoder’s features. Post-review analyses (Fig-
ure 3, Table 3, the cluster intervals) are exploratory.
Planned analyses that were not run (class/residual de-
composition, AU-versus-function agreement, robust-
ness to k, pooling and probe class) are not claimed.
AI use statement
We used generative AI tools (a large language model
coding and writing assistant) for implementation of
the experiment code from the authors’ design scaf-
fold, numerical veriﬁcation of theoretical claims, draft-
ing of proofs, drafting and editing of the manuscript
text, generation of ﬁgures and tables, assembly of the
bibliography, and a simulated multi-reviewer critique
whose ﬁndings informed this revision.
The research
question, pre-registered predictions and falsiﬁers, and
experimental design were speciﬁed by the authors be-
fore AI assistance. All AI-generated code was checked
against independent reference implementations by unit
tests (17 tests, including 9 theory checks), every proof
was checked numerically on random instances, and ev-
ery number in the paper is generated by script from
logged results. We take responsibility for the ﬁnal con-
tent of this work, including text, claims, and artifacts
produced with the aid of generative AI.
References
Agrawal, K. K., Mondal, A. K., Ghosh, A., and
Richards, B. (2022). α-ReQ: Assessing representa-
tion quality in self-supervised learning by measuring
eigenspectrum decay. In Advances in Neural Infor-
mation Processing Systems (NeurIPS).
Bach, F. R. and Jordan, M. I. (2002). Kernel inde-
pendent component analysis.
Journal of Machine
Learning Research, 3:1–48.
Boix-Adserà, E., Lawrence, H., Stepaniants, G., and
Rigollet, P. (2022). GULP: A prediction-based met-
ric between representations. In Advances in Neural
Information Processing Systems (NeurIPS).
Caponnetto, A. and De Vito, E. (2007). Optimal rates
for the regularized least-squares algorithm. Founda-
tions of Computational Mathematics, 7(3):331–368.
Cherti, M., Beaumont, R., Wightman, R., et al.
(2023).
Reproducible scaling laws for contrastive
language-image learning.
In IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition
(CVPR).
Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G.,
and Belilovsky, E. (2023). Reliability of CKA as a

similarity measure in deep learning. In International
Conference on Learning Representations (ICLR).
Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen,
R., Bauer, M., and Hennig, P. (2021). Laplace redux
– eﬀortless Bayesian deep learning. In Advances in
Neural Information Processing Systems (NeurIPS).
Ding, F., Denain, J.-S., and Steinhardt, J. (2021).
Grounding representation similarity through statis-
tical testing.
In Advances in Neural Information
Processing Systems (NeurIPS).
Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.
(2021). An image is worth 16x16 words: Transform-
ers for image recognition at scale. In International
Conference on Learning Representations (ICLR).
Efron, B. and Tibshirani, R. J. (1993). An Introduction
to the Bootstrap. Chapman & Hall.
Harvey, S. E., Lipshutz, D., and Williams, A. H.
(2024). What representational similarity measures
imply about decodable information. In Proceedings
of UniReps: the Second Edition of the Workshop on
Unifying Representations in Neural Models, volume
285 of PMLR, pages 140–151.
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Gir-
shick, R. (2022). Masked autoencoders are scalable
vision learners. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR).
Houlsby, N., Huszár, F., Ghahramani, Z., and Lengyel,
M. (2011).
Bayesian active learning for classi-
ﬁcation and preference learning.
arXiv preprint
arXiv:1112.5745.
Huh, M., Cheung, B., Wang, T., and Isola, P. (2024).
The platonic representation hypothesis. In Interna-
tional Conference on Machine Learning (ICML).
Hüllermeier, E. and Waegeman, W. (2021). Aleatoric
and epistemic uncertainty in machine learning: An
introduction to concepts and methods.
Machine
Learning, 110(3):457–506.
Isserlis, L. (1918).
On a formula for the product-
moment coeﬃcient of any order of a normal fre-
quency distribution in any number of variables.
Biometrika, 12(1/2):134–139.
Kendall, A. and Gal, Y. (2017). What uncertainties do
we need in Bayesian deep learning for computer vi-
sion? In Advances in Neural Information Processing
Systems (NeurIPS).
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G.
(2019). Similarity of neural network representations
revisited. In International Conference on Machine
Learning (ICML).
Kriegeskorte, N., Mur, M., and Bandettini, P. (2008).
Representational similarity analysis – connecting
the branches of systems neuroscience. Frontiers in
Systems Neuroscience, 2:4.
Krizhevsky, A. (2009). Learning multiple layers of fea-
tures from tiny images. Technical report, University
of Toronto.
Lakshminarayanan, B., Pritzel, A., and Blundell, C.
(2017). Simple and scalable predictive uncertainty
estimation using deep ensembles.
In Advances in
Neural Information Processing Systems (NeurIPS).
Lee, K., Lee, K., Lee, H., and Shin, J. (2018). A simple
uniﬁed framework for detecting out-of-distribution
samples and adversarial attacks.
In Advances in
Neural Information Processing Systems (NeurIPS).
Liu, J. Z., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss,
T., and Lakshminarayanan, B. (2020). Simple and
principled uncertainty estimation with deterministic
deep learning via distance awareness. In Advances in
Neural Information Processing Systems (NeurIPS).
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell,
T., and Xie, S. (2022). A ConvNet for the 2020s.
In IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
Morcos, A. S., Raghu, M., and Bengio, S. (2018). In-
sights on representational similarity in neural net-
works with canonical correlation.
In Advances in
Neural Information Processing Systems (NeurIPS).
Mucsányi, B., Kirchhof, M., and Oh, S. J. (2024).
Benchmarking uncertainty disentanglement:
Spe-
cialized uncertainties for specialized tasks. In Ad-
vances in Neural Information Processing Systems
(NeurIPS), Datasets and Benchmarks Track.
Mukhoti, J., Kirsch, A., van Amersfoort, J., Torr, P.
H. S., and Gal, Y. (2023). Deep deterministic un-
certainty: A new simple baseline.
In IEEE/CVF
Conference on Computer Vision and Pattern Recog-
nition (CVPR).
Oquab, M., Darcet, T., Moutakanni, T., et al. (2024).
DINOv2: Learning robust visual features without
supervision. Transactions on Machine Learning Re-
search.
Osband, I., Aslanides, J., and Cassirer, A. (2018).
Randomized prior functions for deep reinforcement
learning. In Advances in Neural Information Pro-
cessing Systems (NeurIPS).
Peterson, J. C., Battleday, R. M., Griﬃths, T. L., and
Russakovsky, O. (2019). Human uncertainty makes
classiﬁcation more robust. In IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV).
Radford, A., Kim, J. W., Hallacy, C., et al. (2021).
Learning transferable visual models from natural
language supervision. In International Conference
on Machine Learning (ICML).

Raghu,
M.,
Gilmer,
J.,
Yosinski,
J.,
and Sohl-
Dickstein, J. (2017).
SVCCA: Singular vector
canonical correlation analysis for deep learning dy-
namics and interpretability. In Advances in Neural
Information Processing Systems (NeurIPS).
Rasmussen, C. E. and Williams, C. K. I. (2006). Gaus-
sian Processes for Machine Learning. MIT Press.
Riquelme, C., Tucker, G., and Snoek, J. (2018). Deep
Bayesian bandits showdown: An empirical compar-
ison of Bayesian deep networks for Thompson sam-
pling. In International Conference on Learning Rep-
resentations (ICLR).
Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish,
N., Sundaram, N., Patwary, M. M. A., Prabhat, and
Adams, R. P. (2015). Scalable Bayesian optimiza-
tion using deep neural networks. In International
Conference on Machine Learning (ICML).
Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R.,
Uszkoreit, J., and Beyer, L. (2022). How to train
your ViT? data, augmentation, and regularization
in vision transformers.
Transactions on Machine
Learning Research.
Stringer, C., Pachitariu, M., Steinmetz, N., Carandini,
M., and Harris, K. D. (2019). High-dimensional ge-
ometry of population responses in visual cortex. Na-
ture, 571:361–365.
van Amersfoort, J., Smith, L., Teh, Y. W., and Gal,
Y. (2020).
Uncertainty estimation using a single
deep deterministic neural network. In International
Conference on Machine Learning (ICML).
Vinod, H. D. (1976). Canonical ridge and economet-
rics of joint production. Journal of Econometrics,
4(2):147–166.
Wightman,
R.
(2019).
Pytorch
image
mod-
els.
https://github.com/huggingface/
pytorch-image-models.
Williams, A. H., Kunz, E., Kornblith, S., and Linder-
man, S. W. (2021). Generalized shape metrics on
neural representations. In Advances in Neural In-
formation Processing Systems (NeurIPS).
Wimmer, L., Sale, Y., Hofman, P., Bischl, B., and
Hüllermeier, E. (2023). Quantifying aleatoric and
epistemic uncertainty in machine learning: Are con-
ditional entropy and mutual information appropri-
ate measures? In Conference on Uncertainty in Ar-
tiﬁcial Intelligence (UAI).
Zhang, T. (2005).
Learning bounds for kernel re-
gression using eﬀective data dimensionality. Neural
Computation, 17(9):2077–2098.

CHECKLIST
1. For all models and algorithms presented, check if
you include:
(a) A clear description of the mathematical set-
ting, assumptions, algorithm, and/or model.
[Yes] Sections 2 and 3.
(b) An analysis of the properties and complexity
(time, space, sample size) of any algorithm.
[Yes] Sρ needs only d×d matrices, O(nd2+d3)
(Section 3); compute in Appendix B.
(c) (Optional) Anonymized source code, with
speciﬁcation of all dependencies, including
external libraries. [Yes] Supplementary code
with requirements.txt.
2. For any theoretical claim, check if you include:
(a) Statements of the full set of assumptions
of all theoretical results.
[Yes] Each state-
ment lists its assumptions (Gaussian fea-
tures, linear-Gaussian heads, ﬁxed design
where relevant).
(b) Complete proofs of all theoretical results.
[Yes] Appendix A.
(c) Clear explanations of any assumptions. [Yes]
Sections 3 and 5.
3. For all ﬁgures and tables that present empirical
results, check if you include:
(a) The code, data, and instructions needed
to reproduce the main experimental results
(either in the supplemental material or as
a URL). [Yes] Supplementary code; public
datasets and weights.
(b) All the training details (e.g., data splits, hy-
perparameters, how they were chosen). [Yes]
Section 4 and appendix B.
(c) A clear deﬁnition of the speciﬁc measure or
statistics and error bars (e.g., with respect to
the random seed after running experiments
multiple times).
[Yes] Appendix B; re-test
ﬂoors quantify bootstrap seed variability.
(d) A description of the computing infrastructure
used. (e.g., type of GPUs, internal cluster, or
cloud provider). [Yes] Appendix B.
4. If you are using existing assets (e.g., code, data,
models) or curating/releasing new assets, check if
you include:
(a) Citations of the creator If your work uses ex-
isting assets. [Yes]
(b) The license information of the assets, if ap-
plicable. [Yes] Appendix B.
(c) New assets either in the supplemental mate-
rial or as a URL, if applicable. [Yes] Code
only.
(d) Information
about
consent
from
data
providers/curators. [Not Applicable]
(e) Discussion of sensible content if applicable,
e.g., personally identiﬁable information or of-
fensive content. [Not Applicable]
5. If you used crowdsourcing or conducted research
with human subjects, check if you include:
(a) The full text of instructions given to partic-
ipants and screenshots. [Not Applicable] We
use the public CIFAR-10H labels.
(b) Descriptions of potential participant risks,
with links to Institutional Review Board
(IRB) approvals if applicable.
[Not Appli-
cable]
(c) The estimated hourly wage paid to partici-
pants and the total amount spent on partic-
ipant compensation. [Not Applicable]

When Do Aligned Representations Agree on What They Do Not
Know?
Supplementary Materials
A
PROOFS
Throughout, wi(ρ) = λi/(λi + ρ), and M matrices are symmetric.
Proof of Theorem 2.
Write a = ϕA(x), b = ϕB(x), P = MA, Q = MB.
For a zero-mean Gaussian
vector, Isserlis’ theorem (Isserlis, 1918) gives E[aiajbkbl] = (ΣA)ij(ΣB)kl + (ΣAB)ik(ΣAB)jl + (ΣAB)il(ΣAB)jk.
Contracting with PijQkl and using the symmetry of P, Q,
E[a⊤Pa b⊤Qb] = tr(PΣA) tr(QΣB) + 2 tr(PΣABQΣBA).
Since E[a⊤Pa] = tr(PΣA), Cov(a⊤Pa, b⊤Qb) = 2 tr(PΣABQΣBA). Setting b = a, Q = P gives Var(a⊤Pa) =
2 tr((PΣA)2), and likewise for b. Dividing gives S(MA, MB). For uA, uB take MA = (ΣA + ρAI)−1, MB =
(ΣB + ρBI)−1.
□
Proof of Proposition 3.
(i) As ρ →∞, ρMA →I and ρMB →I.
Multiplying numerator and both
factors of the denominator by ρ2, the numerator tends to tr(ΣABΣBA) = ∥ΣAB∥2
F and the denominator to
√
tr(Σ2
A) tr(Σ2
B) = ∥ΣA∥F ∥ΣB∥F . (ii) As ρ →0, MA →Σ−1
A ; the numerator tends to tr(Σ−1
A ΣABΣ−1
B ΣBA) =
∥Σ−1/2
A
ΣABΣ−1/2
B
∥2
F = ∑
j r2
j, and tr((Σ−1
A ΣA)2) = dA. The limit is therefore ∑
j r2
j/√dAdB, which equals the
mean squared canonical correlation only when dA = dB.
□
Proof of Corollary 4.
ΣAB = diag(λsIk, 0) and MA = diag((λs + ρ)−1Ik, (λp + ρ)−1Im), so the numerator
of Sρ is kw2
s and tr((MAΣA)2) = kw2
s + mw2
p; CKA follows the same way with λ in place of w. High CKA,
low agreement: take ρ ≤λp (so wp ≥1
2, ws ≤1) and m ≥4k/ϵ; then corr ≤k/(k + m/4) ≤ϵ. Then take
λp/λs ≤
√
ϵk/m, so CKA = 1/(1 + (m/k)(λp/λs)2) ≥1/(1 + ϵ) ≥1 −ϵ. Low CKA, high agreement: take ρ ≤λs
(so ws ≥1
2) and k ≥4m/ϵ; then corr = 1/(1 + mw2
p/(kw2
s)) ≥1 −4m/k ≥1 −ϵ. Then take λp/λs ≥
√
k/(mϵ),
so CKA ≤kλ2
s/(mλ2
p) ≤ϵ.
□
Proof of Proposition 5.
CKA and cosine similarities are invariant to ϕ 7→cQϕ.
For the variance,
v(cQΦ, cQx; λ) = σ2c2x⊤Q⊤(c2QΦ⊤ΦQ⊤+ λI)−1Qx = σ2x⊤(Φ⊤Φ + (λ/c2)I)−1x. So the transformed head
is the original head with prior strength ρ/c2. Apply Theorem 2 to the same features with MA = (Σ + ρc−2I)−1,
MB = (Σ + ρI)−1, ΣAB = Σ: the numerator is ∑
i wi(ρ/c2)wi(ρ) and the denominators are ∑
i wi(·)2. By
Cauchy–Schwarz the ratio equals one iﬀwi(ρ)/wi(ρ/c2) = (λi + ρ/c2)/(λi + ρ) is constant over the support, i.e.
iﬀthe nonzero λi are equal (for c ̸= 1).
□
Proof of Proposition 6.
With Ts = U diag(si)U ⊤(si = s on T , 1 otherwise), E[ϕ(Tsϕ)⊤] = UΛ diag(si)U ⊤
and Cov(Tsϕ) = UΛ diag(s2
i )U ⊤. Hence CKA = ∑λ2
i s2
i /
√∑λ2
i
∑λ2
i s4
i = (1 + s2η)/
√
(1 + η)(1 + s4η). More-
over 1−CKA2 = η(s2 −1)2/((1+η)(1+s4η)) ≤η(s2 −1)2, and CKA ≥CKA2 since CKA ≤1. The transformed
eigenvalues are s2λi on T , and s2λi/(s2λi + ρ) = wi(ρ/s2).
□
Proof of Lemma 7.
∑
i v(xi) = σ2 tr(Φ(Φ⊤Φ + λI)−1Φ⊤) = σ2 ∑
i gi/(gi + λ) with gi = nˆλi the eigenvalues
of Φ⊤Φ; divide by n and use λ/n = ρ.
□
Proof of Corollary 8.
Theorem 2 with ΣA = ΣB = ΣAB = Σ, Mj = (Σ + ρjI)−1.
□

Proof of Lemma 9.
With y = Φw + ε, Cov(ε) = σ2I and ˆw = A−1Φ⊤y, Covε( ˆw) = σ2A−1Φ⊤ΦA−1 =
σ2A−1GA−1. In the eigenbasis of G this is σ2gi/(gi +λ)2, against the posterior σ2/(gi +λ). Residual resampling
estimates this noise covariance as n →∞. The EU score is ϕ⊤Cov( ˆw)ϕ = σ2
n ϕ⊤(ˆΣ + ρ)−1 ˆΣ(ˆΣ + ρ)−1ϕ, so in
Theorem 2 MΣ has eigenvalues w2
i and the direction weights are w4
i .
□
Proof of Proposition 10.
G and (G + s2I)−1 commute, so P(G) = G(G + s2I)−1[(G + s2I) −G] = s2(I −
s2(G + s2I)−1). Hence P(K) −P(L) = s4[(L + s2I)−1 −(K + s2I)−1] = s4(K + s2I)−1(K −L)(L + s2I)−1, and
∥(G + s2I)−1∥op ≤s−2 for G ⪰0.
□
The bound fails for unobserved points.
For Z = T ∪E with only T observed, P(G) = G −GZT (GT T +
s2I)−1GT Z. In 3000 random trials (feature dimension 1–11, |T | ∈[3, 30), |E| ∈[1, 20), s2 ∈[10−2, 10]) the ratio
∥P(K) −P(L)∥op/∥K −L∥op reached 4.97, so no constant-one bound holds there.
B
EXPERIMENTAL DETAILS
Pre-registration.
The PREREG dictionary (claims, falsiﬁers, gate) was frozen before any real-data com-
putation; its SHA-256 is 05aa7d680e7d7f11ec6ccca283c8c56ddf021a971ae97de23d94528935d3cc70 (ﬁle
PREREG_FROZEN.txt in the supplementary code, frozen 2026-10-02 before any real-data computation; no exter-
nal registry was used, so the hash certiﬁes content, not time). The frozen falsiﬁers for E2-b (“AU degrades as
much as EU”) and S (“no better than CKA”) carried no numeric threshold; the operational rules in Table 1
were ﬁxed by us when computing them and are therefore post hoc. Analyses not in the pre-registration (the Sρ
evaluation at ρ →0, the deﬀ-versus-participation-ratio contrast, the exploratory E4 variant, and all post-review
analyses in Appendix C) are exploratory.
Features.
Taps at relative depths 0.5, 0.75, 1 of the ﬂattened block list; mean over patch tokens excluding preﬁx
tokens (ViTs) or spatial mean (ConvNeXt) of the raw block output; fp16 cache; bicubic upsampling 32 →224
with each encoder’s own normalisation. DINOv2 uses img_size=224 with interpolated position embeddings.
Heads.
Softmax regression, objective ∑
i wi CEi/ ∑
i wi + wd
2 ∥W∥2, full-batch L-BFGS (strong Wolfe, ≤200
iterations), all M members trained in one tensor; agreement with scikit-learn’s optimum to 5×10−4 in probability.
Weight decay grid {10−5, . . . , 10−1}, chosen by validation NLL on a ﬁxed 10k split of the training set with hard
labels. Poisson(1) bootstrap weights. EU summaries: summed 5–95% quantile width over classes and mutual
information; AU: mean member entropy. Laplace: class-block-diagonal GGN at the MAP of member 0 with
prior precision n · wd, 100 Monte Carlo logit samples. Closed-form EU: u(x) = ϕ⊤(ˆΣ + ρI)−1ϕ with ρ = wd.
Ensemble size.
M-stability (DINOv2-B, ﬁnal layer): Spearman between independent ensembles of size M and
2M: M = 10: 0.992, M = 25: 0.997, M = 50: 0.998, M = 100: 0.999. We use M = 50.
Statistics.
Item-level Spearman; partial Spearman on ranks residualised on both models’ conﬁdence (max
mean probability) and human plug-in entropy; stratiﬁed Spearman within human-entropy deciles.
CIFAR-
10H ceiling: split votes into halves without replacement (multivariate hypergeometric), Spearman between half
entropies, Spearman–Brown corrected, 50 repetitions.
Compute.
One Apple M2 Pro laptop (16 GB), PyTorch MPS for feature extraction (68–113 images/s) and
CPU ﬂoat32/ﬂoat64 for heads and linear algebra.
Licences.
CIFAR-10H: CC BY-NC-SA 4.0. Model weights via timm and open_clip under their respective
licences.
C
POST-REVIEW ANALYSES
All analyses in this section were added after an internal round of review and are exploratory.

Encoder-cluster bootstrap.
The 30 encoder pairs share encoders, so pairs are not independent. We resample
the ﬁve encoders with replacement (2000 times), keep every pair of distinct sampled encoders (with multiplicity,
all three depths), and recompute each Spearman correlation and each diﬀerence of correlations on the resampled
pairs; intervals are 2.5–97.5% percentiles. With ﬁve encoders the resampling distribution is coarse, and intervals
are wide.
Reliability and redraws.
For each encoder (ﬁnal depth, selected weight decay, M = 50) we draw three
independent 10% subsets and three independent splits into halves; every head is ﬁtted twice with diﬀerent
Poisson weights. A pair’s corrected agreement is its partial agreement divided by √rxxryy, where rxx is the
partial agreement between the two replicates of head x. The re-test ensembles use seeds 0 and 1 of the full-data
head.
Weight-decay sweep.
All 15 encoder–depth heads are reﬁtted with M = 20 at a common weight decay
ρ ∈{10−3, 10−2, 10−1, 1} (no validation selection), and agreement and indices are recomputed for the 30 pairs.
Training-only Sρ uses the same 10k training subset without test inputs.
D
ADDITIONAL RESULTS
Table 6: E2 per encoder: Spearman (partial) agreement for EU width, mutual information and AU, and model
AU versus human entropy (fraction of ceiling).
encoder
pair
EU width
EU MI
AU
DINOv2-B
re-test ﬂoor
1.00 (0.61)
0.99 (0.78)
1.00
DINOv2-B
25% vs 100% data
0.96 (0.48)
0.95 (0.54)
0.97
DINOv2-B
10% vs 100% data
0.94 (0.40)
0.93 (0.42)
0.94
DINOv2-B
disjoint halves
0.94 (0.45)
0.93 (0.55)
0.95
DINOv2-B
wd ÷10
0.99 (0.28)
0.96 (0.19)
0.99
DINOv2-B
wd ×10
0.98 (0.55)
0.98 (0.53)
0.98
DINOv2-B
AU vs human
0.36 (0.52 of ceiling); test acc. 0.98
CLIP-B/16
re-test ﬂoor
1.00 (0.52)
0.99 (0.73)
1.00
CLIP-B/16
25% vs 100% data
0.95 (0.33)
0.94 (0.50)
0.95
CLIP-B/16
10% vs 100% data
0.90 (0.21)
0.89 (0.39)
0.90
CLIP-B/16
disjoint halves
0.91 (0.30)
0.91 (0.52)
0.92
CLIP-B/16
wd ÷10
0.98 (0.11)
0.97 (0.35)
0.97
CLIP-B/16
wd ×10
0.97 (0.45)
0.96 (0.58)
0.96
CLIP-B/16
AU vs human
0.43 (0.61 of ceiling); test acc. 0.95
ViT-B/16 sup.
re-test ﬂoor
1.00 (0.55)
0.99 (0.72)
1.00
ViT-B/16 sup.
25% vs 100% data
0.95 (0.43)
0.95 (0.49)
0.96
ViT-B/16 sup.
10% vs 100% data
0.93 (0.25)
0.92 (0.30)
0.93
ViT-B/16 sup.
disjoint halves
0.94 (0.39)
0.93 (0.48)
0.95
ViT-B/16 sup.
wd ÷10
0.99 (0.28)
0.97 (0.21)
0.99
ViT-B/16 sup.
wd ×10
0.98 (0.43)
0.97 (0.48)
0.98
ViT-B/16 sup.
AU vs human
0.35 (0.50 of ceiling); test acc. 0.98
ConvNeXt-S
re-test ﬂoor
1.00 (0.55)
0.99 (0.78)
1.00
ConvNeXt-S
25% vs 100% data
0.93 (0.39)
0.93 (0.54)
0.94
ConvNeXt-S
10% vs 100% data
0.88 (0.28)
0.88 (0.41)
0.89
ConvNeXt-S
disjoint halves
0.88 (0.40)
0.88 (0.59)
0.89
ConvNeXt-S
wd ÷10
0.98 (0.12)
0.97 (0.35)
0.98
ConvNeXt-S
wd ×10
0.97 (0.50)
0.97 (0.63)
0.97
ConvNeXt-S
AU vs human
0.34 (0.48 of ceiling); test acc. 0.97
MAE-B/16
re-test ﬂoor
0.99 (0.58)
0.99 (0.76)
1.00
MAE-B/16
25% vs 100% data
0.90 (0.19)
0.90 (0.41)
0.91
MAE-B/16
10% vs 100% data
0.84 (0.17)
0.84 (0.35)
0.85
MAE-B/16
disjoint halves
0.88 (0.24)
0.87 (0.48)
0.89
MAE-B/16
wd ÷10
0.99 (0.44)
0.98 (0.68)
0.99
MAE-B/16
wd ×10
0.97 (0.52)
0.97 (0.66)
0.98
MAE-B/16
AU vs human
0.38 (0.55 of ceiling); test acc. 0.90

Table 7: Across 30 encoder pairs: Spearman correlation between alignment indices and item-level agreement of
EU (and AU), all columns.
index
width
width partial
MI
BLR
AU
CKA
0.66
0.64
0.67
0.76
0.68
mutual k-NN
0.85
0.72
0.86
0.90
0.86
linear predictivity
0.41
0.83
0.44
0.56
0.53
Sρ→0
0.88
0.83
0.90
0.93
0.93
Sρ (head’s ρ)
0.87
0.83
0.89
0.92
0.92
