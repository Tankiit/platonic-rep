# EU-specific follow-up (exploratory)

The frozen original PREREG is unchanged. This follow-up tests whether representation
similarity predicts *latent posterior variance agreement* and acquisition-set
agreement, rather than softmax width residualized against confidence.

Use exact Bayesian linear regression and an exact RBF Gaussian process. Linear heads are equivalent to a linear-kernel Gaussian
process: C = (tau^-2 I + X.T diag(1/noise_variance) X)^-1;
EU(x) = x.T C x; AU(x) = prescribed observation variance.
This is a Gaussian regression likelihood on frozen image features, not a validated
classification uncertainty estimator. No labels or human soft targets enter the
posterior variance. A held-out image class is a coverage intervention, not proof
of semantic OOD uncertainty or accuracy. Alignment measures are not new.

## Design

- Metrics and heads share the same training-centering/scalar-normalization.
- Alignment inputs capped at 1024; posterior pool capped at 1024 by default
  (`--max-train`), keeping exact RBF GP memory bounded.
- Five controlled-data seeds, fixed priors 0.1/1/10, nested training fractions
  0.1/0.25/1. Each fraction uses the SAME prior; actual posterior variance must
  decrease pointwise as observations are added.
- Homoscedastic AU control and known heteroscedastic AU. Controlled noise metadata
  is independent of coverage. In the image arm noise is class-parity metadata,
  intentionally artificial; stratified outcomes accompany pooled outcomes.
- CKA, cosine mutual k-NN (k=10), bidirectional held-out linear predictivity,
  CCA-type smoother limit (regularized at 1e-6), smoother similarities at 0.01/1.
  These are prespecified comparators, not selected after results.
- Alignment inputs, posterior training inputs and posterior query inputs are
  disjoint. Training centering and scalar scaling are frozen across data fractions.
- Primary outputs: EU Spearman agreement and top-10% acquisition overlap.
  Report both within-ID and within-held-out agreement and noise strata.
- Output metric-prediction correlations are descriptive, not independent-pair
  inference. Pairs share encoders; do not bootstrap pairs or treat their count as
  sample size. Separate seed/encoder/prior/fraction results; no pooled headline.
- Synthetic views share a latent source and are controls, not independent encoders.
  For image experiments, assess leave-one-encoder-out stability before any claim
  that an index is better. No metric superiority claim is established by this runner.
- Constant AU has undefined rank correlation, intentionally stored as NaN.
  Acquisition ties use stable index ordering; random-overlap expectation is 0.1.

## Run

From `uq_lab`, with NumPy/SciPy/pandas/scikit-learn installed:

```bash
python uq_eu_specific.py --output uq_runs/eu_specific_synthetic
python uq_eu_specific.py --head rbf --output uq_runs/eu_specific_synthetic_rbf
```

For real images first extract caches (requires torch/timm/open_clip and pretrained
weight downloads; the repository only includes JSON sidecars):

```bash
python prepare_eu_features.py --data-root /path/to/cifar-10-batches-py --root uq_runs
python uq_eu_specific.py --mode features --root uq_runs --labels uq_runs/cifar10_train_labels.npy --heldout 0 --output uq_runs/eu_specific_class0
```

Run the final-layer class sensitivity suite (five seeds, ten withheld classes):

```bash
python run_eu_image_suite.py --head linear --root uq_runs --output-root uq_runs
python run_eu_image_suite.py --head rbf --root uq_runs --output-root uq_runs
```

Each class writes into a distinct output directory. The feature
arm reserves every fifth training image as query, withholds the selected class
from all posterior training, and uses the remaining images for alignment/posterior
partitions. Original test caches are checked for encoder completeness but the
query arm uses disjoint TRAIN images. No test-label or human-label files needed.
Depths are separate views, not independent encoders.

A pilot extraction can use `--limit-train 3000 --limit-test 1000` with a
separate root. Do not reuse a root with caches of a different size or ordering.
RBF lengthscale is the median pairwise distance of up to 256 posterior-pool inputs,
frozen across training sizes. For the image arm run both `--head linear` and
`--head rbf` into separate directories.

Files: `manifest.json`, `pairs.csv`, `diagnostics.csv`, `metric_prediction.csv`, `leave_one_encoder_out.csv`.
The manifest records script/alignment-code SHA-256, input-array and label SHA-256, cache metadata, and CLI settings. Preserve the matching source
commit and input-cache metadata for reproducibility.

## Remaining evidence before a paper claim

Run the complete frozen-image arm, inspect ID/held-out variance and monotonicity,
compare indices across encoder omissions and held-out classes, then test whether
acquisition rankings improve an actual downstream learning curve. Add an independently justified classifier EU estimator to assess head-family
sensitivity. Known-noise regression does not by itself establish classification
EU/AU separation, and set overlap does not establish acquisition utility.

For a final-layer pilot, use `--layers 2`. Sample and feature dimensions are
recorded in pair rows. A CCA-type index can saturate when feature dimension
exceeds alignment sample size; inspect `metric_span` before interpreting rank
correlations. Saturated metrics are not evidence of meaningful predictivity.

The checked image pilot uses a reproducible 3000-image training sample and
1000-image test sample (sampling seed 2026). Queries are 600 reserved training
images. Prescribed AU is either constant 0.1 or class-parity variance 0.1/1.
Training contains no examples from the class being withheld. Label values are
used only for withholding/noise metadata; no predictive means are evaluated.
