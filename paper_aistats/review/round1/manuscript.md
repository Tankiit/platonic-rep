# Alignment Does Not Identify Epistemic Uncertainty

Alignment Does Not Identify Epistemic Uncertainty
Anonymous Author
Anonymous Institution
Abstract
Representations learned by diﬀerent vision
models are increasingly similar under kernel-
alignment metrics such as CKA and mutual
nearest neighbours.
It is tempting to read
high alignment as evidence that two mod-
els will also agree about what they do not
know.
We show that alignment does not
identify epistemic uncertainty (EU), and we
characterise why. For Bayesian linear heads
on Gaussian features, the correlation across
inputs of two models’ epistemic variances
equals a ridge-regularised alignment index
Sρ at the heads’ own prior strength; linear
CKA is its ρ →∞limit and weights spec-
tral directions by squared variance, whereas
EU weights every direction above ρ almost
equally. CKA is therefore neither necessary
nor suﬃcient for EU agreement (we construct
pairs with CKA > 0.99 and EU correla-
tion 0.06, and the converse), and transforma-
tions that leave alignment invariant move EU
through an implicit change of prior. On ﬁve
frozen vision encoders with CIFAR-10H la-
bels, heads on identical features disagree sub-
stantially once their training data diﬀer: af-
ter controlling for conﬁdence and human am-
biguity, EU rank agreement between heads
trained on 10% and 100% of the data is 0.26,
against a re-test ﬂoor of 0.56. Across 30 en-
coder pairs, Sρ orders EU agreement better
than CKA (Spearman 0.83 vs. 0.64).
Pre-
registered predictions that failed are reported
alongside those that held:
aleatoric agree-
ment degraded about as much as EU agree-
ment, the representation still dominated EU
disagreement between the most aligned en-
coders we tested, and the identity holds only
Preliminary work. Under review by AISTATS 2027. Do
not distribute.
ordinally on real, non-Gaussian features. EU
agreement is a property of representation,
prior and data jointly, not of the represen-
tation alone.
1
INTRODUCTION
Neural networks trained with diﬀerent objectives, ar-
chitectures and data are converging toward similar rep-
resentations (Huh et al., 2024). The evidence for this
convergence is kernel alignment: centred kernel align-
ment (CKA; Kornblith et al., 2019), canonical corre-
lations (Raghu et al., 2017; Morcos et al., 2018), and
mutual nearest-neighbour overlap (Huh et al., 2024).
A natural next step is to treat alignment as a proxy
for agreement in downstream uncertainty: if two en-
coders induce nearly the same kernel, linear probes
on top of them should be unsure about the same in-
puts. This would make alignment a cheap, label-free
tool for transferring uncertainty estimates, selecting
encoders for active learning, or arguing that aligned
models share their blind spots.
We argue that this step is unjustiﬁed for epistemic un-
certainty (EU), the reducible part that reﬂects limited
data (Kendall and Gal, 2017; Hüllermeier and Waege-
man, 2021), as opposed to aleatoric uncertainty (AU),
which reﬂects intrinsic ambiguity of the input. The
reason is structural. Alignment metrics are designed to
be invariant to a group of transformations of the rep-
resentation (orthogonal maps and isotropic scale for
CKA; similarity transforms for cosine nearest neigh-
bours), and they normalise away the scale of the ker-
nel. Epistemic uncertainty of a regularised head is not
invariant to these transformations: rescaling features
at a ﬁxed weight decay is a change of prior, and the
posterior variance depends on how the spectrum com-
pares to the prior strength. Alignment and EU also
read diﬀerent parts of the spectrum: CKA is domi-
nated by the few top eigendirections, while EU is dom-
inated by the many directions just above the regular-
isation level.
Contributions.

1. An exact identity (Theorem 2): for Gaussian fea-
tures and Bayesian linear heads, the Pearson corre-
lation of two models’ epistemic variances equals Sρ,
the cosine between the two ridge smoothers at the
heads’ prior strengths. CKA is the ρ →∞limit
and mean squared canonical correlation the ρ →0
limit (Proposition 3).
2. Non-identiﬁcation results: CKA is neither nec-
essary nor suﬃcient for EU agreement (Corol-
lary 4); invariances of alignment metrics are not
invariances of EU (Propositions 5 and 6); identical
features do not imply EU agreement (Corollary 8).
3. What does control EU: an in-sample resolvent
bound in operator norm (Proposition 10), a label-
free identity between mean EU and the eﬀective
dimension deﬀ(ρ) (Lemma 7), and a characterisa-
tion of bootstrap ensembles as a shrunk posterior
(Lemma 9).
4. Pre-registered experiments on ﬁve frozen en-
coders (DINOv2,
CLIP, supervised ViT, Con-
vNeXt, MAE) with CIFAR-10H human labels, us-
ing bootstrap, Laplace and closed-form estimators
and reliability ceilings for every AU number. We
report conﬁrmed and falsiﬁed predictions alike.
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
shares this invariance. Prior work examined the sta-
tistical reliability of these measures (Ding et al., 2021;
Davari et al., 2023), their metric properties (Williams
et al., 2021), and their relation to RSA (Kriegesko-
rte et al., 2008); we ask instead what they imply for
downstream uncertainty.
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
(ρ →∞), connecting to feature-space uncertainty
scores (Lee et al., 2018; Mukhoti et al., 2023).
In
experiments we also use softmax heads with boot-
strap ensembles (Efron and Tibshirani, 1993; Laksh-
minarayanan et al., 2017), summarised by the mutual
information (Houlsby et al., 2011) and by quantile-
interval widths, and last-layer Laplace (Daxberger
et al., 2021).
We are aware that entropy-based de-
compositions have known limitations (Wimmer et al.,
2023); we therefore report several EU summaries.
EU agreement.
For two models we measure agree-
ment as the correlation of their EU scores across in-
puts; in experiments, as Spearman correlation, partial
Spearman controlling for both models’ conﬁdence and
human-label entropy, and Spearman within human-
entropy deciles. Partial and stratiﬁed variants rule out
the trivial explanation that both models are uncertain
on intrinsically ambiguous images.
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
with d × d matrices only. Ridge interpolation between
correlation and covariance analyses is classical (Vinod,
1976; Bach and Jordan, 2002); what is new here is its
exact link to EU.
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
canonical correlations of A and B.
In the eigenbasis, CKA weighs a shared direction by
λ2
i , whereas Sρ weighs it by wi(ρ)2: every direction
with λi ≫ρ counts as one, however small its variance.
Since learned representations have long spectral tails
(Stringer et al., 2019; Agrawal et al., 2022), the two
indices read diﬀerent parts of the spectrum.

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
Corollary 8 (Identical features do not imply EU
agreement). Two heads on the same features with
prior strengths ρ1
̸=
ρ2,
e.g. the same λ but
n1
̸=
n2 examples (ρ
=
λ/n), have EU agree-
ment ∑
i wi(ρ1)wi(ρ2)/
√∑
i wi(ρ1)2 ∑
i wi(ρ2)2 un-
der Gaussian features, while every alignment metric
equals its maximum.
Lemma 9 (Bootstrap is a shrunk posterior). For
ﬁxed-design ridge with residual resampling, Cov( ˆw) =
σ2A−1GA−1 with G = Φ⊤Φ, A = G + λI, versus
the posterior σ2A−1; along eigendirection i the ratio
is gi/(gi + λ).
Thus the bootstrap EU score is ϕ⊤(Σ + ρ)−1Σ(Σ +
ρ)−1ϕ, and Theorem 2 applies with direction weights
w4
i rather than w2
i . Bootstrap EU is therefore more
top-heavy than posterior or Laplace EU, and we pre-
dict it moves less under tail reshaping.
Proposition 10 (In-sample resolvent bound). For
Gram matrices K, L on n observed points and noise
s2 > 0, the posterior covariances P(G) = G −G(G +
s2I)−1G satisfy ∥P(K) −P(L)∥op ≤∥K −L∥op.
The bound uses the unnormalised Gram in operator
norm, exactly the information that CKA discards. It
does not extend with constant one to unobserved eval-
uation points: in random trials with partially observed
points the ratio reached ≈5 (Appendix A). This is why
we evaluate Sρ on training ∪evaluation inputs.
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

Table 1: Pre-registered predictions and outcomes (thresholds frozen before any real-data result). E2a, E4 and S
are falsiﬁers: “falsiﬁed” means the falsifying condition occurred.
id
criterion
observed
outcome
E0
known-AU recovery ≥0.5 (gate)
0.89–0.96
passed
E2a
partial EU agr. 10% vs 100% > 0.9 falsiﬁes
0.26
not falsiﬁed
E2b
AU degrades as much as EU falsiﬁes
drop 0.10 vs 0.10
falsiﬁed
P5
scale moves EU at ﬁxed wd, not when re-tuned Laplace ×0.02–62
conﬁrmed
L9
bootstrap moves less than Laplace (tail)
6/6
conﬁrmed
E4
representation share > 80% falsiﬁes
93%
falsiﬁed
S
Sρ no better than CKA falsiﬁes
0.83 vs 0.64
not falsiﬁed
Spec deﬀpredicts mean EU
BLR 0.90, boot. -0.19 partly
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
four ρ.
(b) The construction of Corollary 4: CKA
stays near one as private dimensions are added while
EU correlation falls as k/(k + m); lines are the closed
forms.
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
struction gives CKA = 0.003 with EU correlation 0.96
(Figure 1b). With identical features and a ﬁxed ridge,
a head trained on 1% of the data has EU correlation
0.76 with the full-data head, below the population pre-
diction of Corollary 8 (0.86): ﬁnite-sample covariance
error adds a data term on top of the change in ρ.
4.2
Identical features, diﬀerent heads (E2)
Within each encoder the paired heads see literally the
same features, so every alignment metric equals one.
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
Raw EU agreement is high for all pairs (Table 2), but
this is largely shared conﬁdence: both EU and AU of
softmax heads are dominated by the predictive margin.
After controlling for both heads’ conﬁdence and hu-
man entropy, quantile-width EU agreement falls from
a re-test ﬂoor of 0.56 (two ensembles with identical
conﬁguration) to 0.36 for disjoint halves of the training
set and 0.26 for 10% versus 100% of the data (mutual
information: 0.76 to 0.37; Figure 2). For comparison,
heads on diﬀerent encoders at the ﬁnal depth reach a
mean partial agreement of 0.10. Falsiﬁer E2a (partial
agreement > 0.9 for 10% vs. 100%) is not triggered:
the data term is not negligible.
Falsiﬁer E2b is triggered. AU agreement on identical
features degrades about as much as EU agreement:
the raw drop from the re-test ﬂoor is 0.10 for AU and
0.10 for EU, and the partial agreement retains 53%
of its re-test value for AU against 46% for EU. We
therefore cannot claim the pre-registered contrast that
AU agreement is protected; the empirical conclusion is
that identical features determine neither the residual
EU ranking nor the residual AU ranking of a head.
What is stable is AU’s relation to the human target:
per encoder, Spearman between model AU and human
entropy varies by at most 0.05 across the eight head
conﬁgurations, at 0.34–0.43 overall (0.48–0.61 of the
ceiling).

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
Table 3: E2b (DINOv2-B, ﬁxed weight decay): align-
ment of the transformed to the original features,
Spearman rank agreement of EU with the untrans-
formed head, and mean EU relative to it (width =
bootstrap quantile width; Laplace = logit variance for
the ratio, MI for the rank).
rank agreement
mean ratio
transform
CKA mKNN width Laplace width Laplace
c =0.1
1.000
1.00
0.94
0.89
0.56
0.017
c =0.3
1.000
1.00
0.98
0.94
0.74
0.125
c =1.0
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
c =10.0
1.000
1.00
0.97
0.85
1.52
61.7
tail s =0.5 1.000
0.96
1.00
1.00
1.02
0.894
tail s =2.0 0.998
0.89
1.00
0.99
1.07
1.27
tail s =4.0 0.956
0.68
0.98
0.95
1.30
1.8
4.3
Transformations that preserve alignment
(E2b)
Applying ϕ 7→cQϕ leaves CKA and mutual k-NN at
one (Table 3). At ﬁxed weight decay, EU moves with
c: across c ∈[0.1, 10] the mean Laplace logit variance
changes by a factor between 0.017 and 61.7, the mean
bootstrap width between 0.38 and 1.84, and EU rank-
ings change (Laplace MI rank agreement down to 0.72,
bootstrap width down to 0.89). When the weight de-
cay is re-tuned by validation, the selected value scales
as c2 (from 10−5 at c = 0.1 to 10−1 at c = 10), which
is exactly the prior change of Proposition 5, and the
largest deviation of any EU rank agreement or mean
ratio from the untransformed head is 0.086 (the re-
tuned grid is coarse, one decade per step).
Tail re-
shaping (scaling the directions outside the top 95% of
variance by s ∈{0.5, 2, 4}) keeps CKA at or above
0.956 while the mean Laplace EU changes by up to a
factor 2.3 and the mean bootstrap width by up to 1.39;
mutual k-NN, which is sensitive to the tail, drops to
0.62. The closed-form EU barely moves (mean ratio
within 0.04 of one): at the selected weight decay the
heads are in the small-ρ regime, where u(x) approaches
the leverage, which is invariant to every invertible lin-
ear map, and deﬀchanges only through tail eigenvalues
near ρ (Proposition 6). The softmax estimators move
because their eﬀective prior strength is set by the cur-
vature of the likelihood, not by the weight decay alone.
Lemma 9 predicts that bootstrap EU moves less than
Laplace EU under tail reshaping; this held in 6 of 6
tail settings.
4.4
Across encoders: Sρ versus CKA
For all 10 encoder pairs at three matched relative
depths (30 pairs), we compute CKA, mutual k-NN
(k = 10), symmetric linear predictivity, and Sρ at the
two heads’ own ρ on training ∪test inputs, together
with item-level EU agreement (Figure 3, Table 5 in the
supplement). Across pairs, Sρ orders partial bootstrap
EU agreement at Spearman 0.83, against 0.64 for CKA
and 0.72 for mutual k-NN; for closed-form EU agree-
ment the values are 0.92, 0.76 and 0.90. Falsiﬁer S is
not triggered. CKA is thus not uninformative, but it is
the weakest of the indices we tested, and the highest-
CKA pair at the ﬁnal depth (CKA 0.837) reaches a
partial EU agreement of only 0.21. Two caveats limit
the claim.
First, Sρ predicts AU agreement as well
as EU agreement (0.92), so it indexes shared head be-
haviour rather than EU speciﬁcally.
Second, Theo-
rem 2 holds only ordinally on real features: Sρ ranks
the closed-form Pearson agreement across pairs at 0.93
(CKA: 0.74), but its level is oﬀby 0.34 on average,
as expected when fourth cumulants of non-Gaussian
features enter the covariance of quadratic forms. At
the selected weight decays all heads sit in the small-
ρ regime (deﬀ(ρ)/d > 0.98), where Sρ is close to its
canonical-correlation limit.
Label-free prediction and swap design.
Across
encoders and depths, deﬀ(ρ) from unlabelled training
features rank-correlates with mean closed-form EU at
0.90, as Lemma 7 predicts, but not with the mean
bootstrap width of softmax heads (-0.19), whose mag-
nitude follows accuracy; this pre-registered prediction
failed.
Swap design (E4).
In a 2 × 2 × 2 design on the
highest-CKA pair (DINOv2-B vs. supervised ViT-
B/16, CKA 0.837; encoder × weight decay {w, 10w}
× disjoint training half), the encoder accounts for 93%
of the Shapley decomposition of raw EU disagreement,
the training half for 9% and the weight decay for es-
sentially none (-2%). This triggers the pre-registered

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
Figure 3:
Across 30 encoder pairs (5 encoders, 3
depths): item-level bootstrap EU agreement, raw (a)
and partial (b), against Sρ at the heads’ ρ (ﬁlled),
CKA (open) and mutual k-NN (crosses).
E4 falsiﬁer (> 80%): among the encoders we could
test, alignment never reached the regime where head
and data terms dominate, so our claim is restricted to
non-identiﬁcation (theory, E2, E2b), not to the irrele-
vance of the representation. In an exploratory variant
using partial disagreement (controlling for conﬁdence
and human entropy), the shares are 61%, 30% and
10% for encoder, data and weight decay, with a re-test
disagreement of 0.33.
5
DISCUSSION
What
the
results
say.
For EU, alignment is
the wrong summary of a pair of representations.
Theorem 2 identiﬁes the relevant one for linear-
Gaussian heads:
spectral overlap above the head’s
prior strength, weighted by wi(ρ)2.
CKA reads the
top of the spectrum, and mutual nearest neighbours
read local geometry that is invariant to the scale on
which the prior acts. Empirically, Sρ orders EU agree-
ment across encoders better than either. Yet even per-
fect alignment leaves EU underdetermined: on identi-
cal features the residual EU ranking depends on the
training data and weight decay. The convergence doc-
umented by Huh et al. (2024) can therefore coexist
with disagreement about what models do not know.
Our data also qualify our own hypotheses. Between
diﬀerent encoders the representation term dominates
(E4), so non-identiﬁcation does not mean that the en-
coder is irrelevant; it means that alignment scores do
not tell us how EU will agree. And the residual AU
ranking was not more stable than the residual EU
ranking, so the asymmetry we expected between AU
and EU is not supported for softmax probes, although
the theory’s EU-speciﬁc mechanisms (scale and tail in-
variances; labels not entering the posterior variance)
stand.
Recommendations.
(i) Do not use CKA or mutual
k-NN to transfer or validate EU; if a single index is
needed, Sρ at the head’s own ρ on training ∪evalu-
ation inputs is the better-founded choice. (ii) Report
EU together with the head’s prior strength relative
to the feature scale. (iii) Report conﬁdence-controlled
(partial) agreement, since raw EU agreement mostly
reﬂects shared conﬁdence.
Limitations.
The theory is exact for Gaussian fea-
tures and linear-Gaussian heads; on real features only
its ordinal predictions held.
Softmax heads enter
through bootstrap and Laplace approximations; the
bootstrap lemma is for residual resampling with ﬁxed
design, and our Laplace drops cross-class curvature.
We study frozen encoders and linear probes on one
dataset with human labels (CIFAR-10H) at low native
resolution; ﬁne-tuned models, other modalities and the
DCIC datasets are not covered.
The positive con-
trol replaces the planned DCIC synthetic dataset with
a known softmax teacher on each encoder’s features.
Planned analyses that were not run (class/residual de-
composition, AU-versus-function agreement, robust-
ness to k, pooling and probe class) are not claimed.
AI use statement
We used generative AI tools (a large language model
coding and writing assistant) for implementation of
the experiment code from the authors’ design scaf-
fold, numerical veriﬁcation of theoretical claims, draft-
ing of proofs, drafting and editing of the manuscript
text, generation of ﬁgures and tables, and assem-
bly of the bibliography. The research question, pre-
registered predictions and falsiﬁers, and experimental
design were speciﬁed by the authors before AI assis-
tance. All AI-generated code was checked against in-
dependent reference implementations by unit tests (17
tests, including 9 theory checks); proofs were checked
numerically and must be checked line by line by the
authors before submission; every number in the paper
is generated by script from logged results. We take
responsibility for the ﬁnal content of this work, in-
cluding text, claims, and artifacts produced with the
aid of generative AI.
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
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell,
T., and Xie, S. (2022). A ConvNet for the 2020s.
In IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
Morcos, A. S., Raghu, M., and Bengio, S. (2018). In-
sights on representational similarity in neural net-
works with canonical correlation.
In Advances in
Neural Information Processing Systems (NeurIPS).
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

Alignment Does Not Identify Epistemic Uncertainty:
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
A ΣA)2) = dA.
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
The PREREG dictionary (claims, falsiﬁers, gate) was frozen before any real-data computation;
its SHA-256 is 05aa7d680e7d7f11.... Analyses not in the pre-registration (the Sρ evaluation at ρ →0 and the
deﬀ-versus-participation-ratio contrast) are exploratory.
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
ADDITIONAL RESULTS

Table 4: E2 per encoder: Spearman (partial) agreement for EU width, mutual information and AU, and model
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
Table 5: Across 30 encoder pairs: Spearman correlation between alignment indices and item-level agreement of
EU (and AU).
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
