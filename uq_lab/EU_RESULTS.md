# EU-specific experiment results

Exploratory Gaussian-regression EU on frozen features. These runs do not establish classification EU or acquisition utility.

The tables use constant AU, prior variance 1, and the largest training fraction. Entries are mean correlations across seeds with the seed range. Pair observations share encoders; ranges are sensitivity summaries, not confidence intervals.

## Across withheld classes

Each cell averages seeds within a withheld class, then reports the mean and min–max across classes. These are descriptive sensitivity summaries, not confidence intervals.

### linear

Withheld classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.83 [0.77, 0.86] | 0.72 [0.56, 0.93] | 0.80 [0.72, 0.87] | 0.53 [-0.03, 0.92] |
| cka | 0.95 [0.92, 0.96] | 0.81 [0.65, 0.94] | 0.93 [0.87, 0.95] | 0.59 [0.14, 0.91] |
| mknn | 0.94 [0.92, 0.95] | 0.80 [0.61, 0.94] | 0.93 [0.86, 0.95] | 0.58 [0.10, 0.89] |
| predictivity | 0.83 [0.76, 0.87] | 0.72 [0.52, 0.82] | 0.83 [0.79, 0.87] | 0.61 [0.15, 0.89] |
| smoother_001 | 0.83 [0.77, 0.86] | 0.72 [0.56, 0.93] | 0.80 [0.72, 0.87] | 0.53 [-0.03, 0.92] |
| smoother_1 | 0.90 [0.86, 0.92] | 0.79 [0.70, 0.92] | 0.88 [0.84, 0.93] | 0.56 [0.18, 0.84] |

### rbf

Withheld classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.82 [0.70, 0.90] | 0.79 [0.64, 0.92] | 0.72 [0.61, 0.88] | 0.51 [-0.10, 0.91] |
| cka | 0.90 [0.85, 0.94] | 0.82 [0.71, 0.87] | 0.81 [0.73, 0.89] | 0.54 [0.13, 0.95] |
| mknn | 0.90 [0.85, 0.94] | 0.83 [0.71, 0.89] | 0.80 [0.71, 0.89] | 0.53 [0.09, 0.94] |
| predictivity | 0.81 [0.76, 0.87] | 0.75 [0.56, 0.94] | 0.76 [0.71, 0.80] | 0.58 [0.11, 0.87] |
| smoother_001 | 0.82 [0.70, 0.90] | 0.79 [0.64, 0.92] | 0.72 [0.61, 0.88] | 0.51 [-0.10, 0.91] |
| smoother_1 | 0.88 [0.80, 0.94] | 0.84 [0.73, 0.93] | 0.80 [0.71, 0.93] | 0.53 [0.13, 0.85] |

## eu_specific_image_linear_class0

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 0.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.47; observed mean top-10% set overlap: 0.42 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.83 [0.81, 0.83] | 0.67 [0.61, 0.73] | 0.81 [0.76, 0.85] | 0.92 [0.89, 0.93] |
| cka | 0.96 [0.95, 0.96] | 0.75 [0.68, 0.79] | 0.95 [0.93, 0.98] | 0.81 [0.77, 0.83] |
| mknn | 0.95 [0.95, 0.96] | 0.76 [0.70, 0.79] | 0.94 [0.93, 0.98] | 0.81 [0.77, 0.83] |
| predictivity | 0.82 [0.72, 0.87] | 0.52 [0.34, 0.60] | 0.85 [0.73, 0.90] | 0.75 [0.68, 0.78] |
| smoother_001 | 0.83 [0.81, 0.83] | 0.67 [0.61, 0.73] | 0.81 [0.76, 0.85] | 0.92 [0.89, 0.93] |
| smoother_1 | 0.91 [0.90, 0.92] | 0.71 [0.64, 0.80] | 0.90 [0.88, 0.93] | 0.84 [0.82, 0.87] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.054 and a maximum span of 0.145.

Held-out/ID mean EU ratio across views/seeds: 1.21–3.61.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class1

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 1.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.40; observed mean top-10% set overlap: 0.29 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.77 [0.77, 0.79] | 0.56 [0.38, 0.65] | 0.78 [0.77, 0.79] | 0.70 [0.66, 0.76] |
| cka | 0.94 [0.94, 0.95] | 0.82 [0.72, 0.88] | 0.94 [0.93, 0.95] | 0.89 [0.87, 0.92] |
| mknn | 0.94 [0.94, 0.94] | 0.82 [0.72, 0.88] | 0.93 [0.93, 0.94] | 0.89 [0.85, 0.92] |
| predictivity | 0.84 [0.79, 0.87] | 0.76 [0.68, 0.84] | 0.84 [0.79, 0.88] | 0.89 [0.84, 0.99] |
| smoother_001 | 0.77 [0.77, 0.79] | 0.56 [0.38, 0.65] | 0.78 [0.77, 0.79] | 0.70 [0.66, 0.76] |
| smoother_1 | 0.88 [0.87, 0.89] | 0.74 [0.61, 0.81] | 0.88 [0.87, 0.89] | 0.84 [0.79, 0.88] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.057 and a maximum span of 0.140.

Held-out/ID mean EU ratio across views/seeds: 1.22–2.00.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class2

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 2.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.43; observed mean top-10% set overlap: 0.42 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.86 [0.85, 0.87] | 0.93 [0.88, 0.98] | 0.81 [0.79, 0.85] | 0.82 [0.76, 0.88] |
| cka | 0.95 [0.94, 0.96] | 0.92 [0.86, 0.96] | 0.94 [0.94, 0.96] | 0.74 [0.68, 0.81] |
| mknn | 0.93 [0.92, 0.94] | 0.90 [0.81, 0.98] | 0.93 [0.90, 0.94] | 0.78 [0.71, 0.85] |
| predictivity | 0.85 [0.83, 0.87] | 0.82 [0.74, 0.89] | 0.87 [0.87, 0.88] | 0.53 [0.41, 0.64] |
| smoother_001 | 0.86 [0.85, 0.87] | 0.93 [0.88, 0.98] | 0.81 [0.79, 0.85] | 0.82 [0.76, 0.88] |
| smoother_1 | 0.91 [0.89, 0.93] | 0.92 [0.86, 0.98] | 0.88 [0.84, 0.89] | 0.76 [0.68, 0.83] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.087 and a maximum span of 0.181.

Held-out/ID mean EU ratio across views/seeds: 1.08–3.10.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class3

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 3.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.45; observed mean top-10% set overlap: 0.38 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.84 [0.83, 0.87] | 0.72 [0.65, 0.79] | 0.80 [0.75, 0.85] | -0.03 [-0.08, 0.04] |
| cka | 0.95 [0.94, 0.98] | 0.81 [0.74, 0.90] | 0.94 [0.94, 0.95] | 0.22 [0.15, 0.27] |
| mknn | 0.94 [0.90, 0.96] | 0.79 [0.69, 0.88] | 0.94 [0.93, 0.95] | 0.19 [0.12, 0.27] |
| predictivity | 0.84 [0.75, 0.92] | 0.81 [0.72, 0.87] | 0.83 [0.73, 0.87] | 0.43 [0.35, 0.55] |
| smoother_001 | 0.84 [0.83, 0.87] | 0.72 [0.65, 0.79] | 0.80 [0.75, 0.85] | -0.03 [-0.08, 0.04] |
| smoother_1 | 0.92 [0.92, 0.95] | 0.84 [0.80, 0.87] | 0.90 [0.87, 0.93] | 0.18 [0.14, 0.24] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.043 and a maximum span of 0.147.

Held-out/ID mean EU ratio across views/seeds: 1.19–2.23.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class4

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 4.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.36; observed mean top-10% set overlap: 0.27 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.81 [0.79, 0.85] | 0.67 [0.57, 0.74] | 0.80 [0.77, 0.85] | 0.72 [0.59, 0.83] |
| cka | 0.94 [0.94, 0.95] | 0.77 [0.60, 0.87] | 0.94 [0.92, 0.95] | 0.85 [0.75, 0.96] |
| mknn | 0.93 [0.93, 0.95] | 0.79 [0.61, 0.88] | 0.92 [0.89, 0.95] | 0.84 [0.73, 0.95] |
| predictivity | 0.87 [0.87, 0.88] | 0.74 [0.58, 0.84] | 0.87 [0.87, 0.88] | 0.78 [0.64, 0.89] |
| smoother_001 | 0.81 [0.79, 0.85] | 0.67 [0.57, 0.74] | 0.80 [0.77, 0.85] | 0.72 [0.59, 0.83] |
| smoother_1 | 0.86 [0.84, 0.89] | 0.76 [0.68, 0.80] | 0.86 [0.83, 0.89] | 0.74 [0.61, 0.85] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.062 and a maximum span of 0.166.

Held-out/ID mean EU ratio across views/seeds: 1.01–1.81.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class5

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 5.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.45; observed mean top-10% set overlap: 0.47 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.84 [0.83, 0.85] | 0.76 [0.63, 0.86] | 0.81 [0.75, 0.85] | 0.38 [0.28, 0.59] |
| cka | 0.95 [0.94, 0.96] | 0.78 [0.52, 0.87] | 0.95 [0.94, 0.95] | 0.54 [0.37, 0.67] |
| mknn | 0.94 [0.94, 0.95] | 0.76 [0.49, 0.87] | 0.95 [0.94, 0.96] | 0.52 [0.37, 0.67] |
| predictivity | 0.82 [0.77, 0.87] | 0.79 [0.54, 0.92] | 0.81 [0.75, 0.87] | 0.78 [0.72, 0.88] |
| smoother_001 | 0.84 [0.83, 0.85] | 0.76 [0.63, 0.86] | 0.81 [0.75, 0.85] | 0.38 [0.28, 0.59] |
| smoother_1 | 0.90 [0.88, 0.93] | 0.79 [0.62, 0.86] | 0.88 [0.82, 0.92] | 0.42 [0.33, 0.59] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.057 and a maximum span of 0.156.

Held-out/ID mean EU ratio across views/seeds: 1.55–2.77.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class6

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 6.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.44; observed mean top-10% set overlap: 0.40 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.85 [0.83, 0.87] | 0.82 [0.81, 0.84] | 0.87 [0.83, 0.92] | 0.81 [0.68, 0.87] |
| cka | 0.94 [0.92, 0.96] | 0.94 [0.92, 0.96] | 0.93 [0.90, 0.96] | 0.91 [0.89, 0.94] |
| mknn | 0.92 [0.88, 0.95] | 0.94 [0.91, 0.96] | 0.90 [0.85, 0.95] | 0.89 [0.85, 0.92] |
| predictivity | 0.82 [0.77, 0.85] | 0.81 [0.67, 0.86] | 0.79 [0.76, 0.85] | 0.77 [0.72, 0.84] |
| smoother_001 | 0.85 [0.83, 0.87] | 0.82 [0.81, 0.84] | 0.87 [0.83, 0.92] | 0.81 [0.68, 0.87] |
| smoother_1 | 0.90 [0.88, 0.92] | 0.89 [0.87, 0.92] | 0.93 [0.92, 0.95] | 0.84 [0.78, 0.89] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.054 and a maximum span of 0.144.

Held-out/ID mean EU ratio across views/seeds: 0.99–3.47.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class7

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 7.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.41; observed mean top-10% set overlap: 0.35 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.84 [0.81, 0.89] | 0.75 [0.66, 0.85] | 0.72 [0.67, 0.81] | 0.35 [0.27, 0.49] |
| cka | 0.92 [0.89, 0.95] | 0.65 [0.58, 0.69] | 0.87 [0.83, 0.93] | 0.14 [0.02, 0.41] |
| mknn | 0.92 [0.89, 0.94] | 0.61 [0.54, 0.69] | 0.86 [0.83, 0.93] | 0.10 [-0.07, 0.41] |
| predictivity | 0.82 [0.76, 0.87] | 0.61 [0.52, 0.80] | 0.82 [0.79, 0.85] | 0.15 [-0.02, 0.25] |
| smoother_001 | 0.84 [0.81, 0.89] | 0.76 [0.66, 0.85] | 0.72 [0.67, 0.79] | 0.36 [0.27, 0.50] |
| smoother_1 | 0.90 [0.88, 0.93] | 0.70 [0.68, 0.73] | 0.84 [0.81, 0.89] | 0.18 [0.03, 0.42] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.055 and a maximum span of 0.150.

Held-out/ID mean EU ratio across views/seeds: 1.31–2.53.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class8

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 8.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.44; observed mean top-10% set overlap: 0.35 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.83 [0.81, 0.83] | 0.74 [0.70, 0.79] | 0.79 [0.75, 0.83] | 0.34 [0.16, 0.53] |
| cka | 0.96 [0.95, 0.96] | 0.86 [0.82, 0.91] | 0.93 [0.92, 0.96] | 0.38 [0.13, 0.61] |
| mknn | 0.95 [0.94, 0.96] | 0.85 [0.81, 0.91] | 0.94 [0.92, 0.95] | 0.36 [0.10, 0.55] |
| predictivity | 0.76 [0.72, 0.87] | 0.69 [0.59, 0.81] | 0.79 [0.77, 0.83] | 0.41 [0.16, 0.59] |
| smoother_001 | 0.83 [0.81, 0.83] | 0.74 [0.70, 0.79] | 0.79 [0.75, 0.83] | 0.34 [0.16, 0.53] |
| smoother_1 | 0.91 [0.90, 0.92] | 0.84 [0.81, 0.87] | 0.87 [0.85, 0.92] | 0.37 [0.12, 0.52] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.059 and a maximum span of 0.135.

Held-out/ID mean EU ratio across views/seeds: 0.96–2.79.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_linear_class9

Head: linear; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 9.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.43; observed mean top-10% set overlap: 0.37 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.82 [0.79, 0.85] | 0.56 [0.34, 0.72] | 0.81 [0.78, 0.85] | 0.29 [0.04, 0.73] |
| cka | 0.94 [0.93, 0.95] | 0.74 [0.65, 0.84] | 0.93 [0.92, 0.94] | 0.40 [0.13, 0.76] |
| mknn | 0.94 [0.93, 0.96] | 0.75 [0.65, 0.84] | 0.93 [0.93, 0.95] | 0.38 [0.13, 0.73] |
| predictivity | 0.81 [0.73, 0.87] | 0.72 [0.65, 0.79] | 0.81 [0.75, 0.85] | 0.57 [0.38, 0.75] |
| smoother_001 | 0.82 [0.79, 0.85] | 0.56 [0.34, 0.72] | 0.81 [0.78, 0.85] | 0.29 [0.04, 0.73] |
| smoother_1 | 0.91 [0.88, 0.93] | 0.74 [0.55, 0.86] | 0.90 [0.87, 0.93] | 0.42 [0.15, 0.81] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.049 and a maximum span of 0.138.

Held-out/ID mean EU ratio across views/seeds: 1.30–2.43.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class0

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 0.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.43; observed mean top-10% set overlap: 0.37 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.87 [0.87, 0.87] | 0.66 [0.62, 0.67] | 0.69 [0.66, 0.72] | 0.84 [0.79, 0.92] |
| cka | 0.93 [0.93, 0.94] | 0.82 [0.75, 0.84] | 0.79 [0.77, 0.79] | 0.74 [0.72, 0.77] |
| mknn | 0.93 [0.93, 0.93] | 0.82 [0.78, 0.84] | 0.78 [0.77, 0.79] | 0.74 [0.72, 0.77] |
| predictivity | 0.83 [0.72, 0.87] | 0.62 [0.52, 0.67] | 0.77 [0.72, 0.82] | 0.73 [0.65, 0.79] |
| smoother_001 | 0.87 [0.87, 0.87] | 0.66 [0.62, 0.67] | 0.69 [0.66, 0.72] | 0.84 [0.79, 0.92] |
| smoother_1 | 0.91 [0.90, 0.92] | 0.73 [0.67, 0.77] | 0.79 [0.77, 0.81] | 0.77 [0.73, 0.83] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.096 and a maximum span of 0.190.

Held-out/ID mean EU ratio across views/seeds: 1.31–2.47.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class1

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 1.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.33; observed mean top-10% set overlap: 0.25 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.70 [0.66, 0.73] | 0.64 [0.51, 0.77] | 0.61 [0.54, 0.68] | 0.77 [0.71, 0.81] |
| cka | 0.85 [0.82, 0.88] | 0.81 [0.71, 0.85] | 0.73 [0.70, 0.83] | 0.95 [0.89, 0.99] |
| mknn | 0.85 [0.82, 0.88] | 0.81 [0.75, 0.84] | 0.73 [0.68, 0.82] | 0.94 [0.89, 0.98] |
| predictivity | 0.79 [0.77, 0.82] | 0.83 [0.71, 0.92] | 0.71 [0.64, 0.78] | 0.87 [0.76, 0.94] |
| smoother_001 | 0.70 [0.66, 0.73] | 0.64 [0.51, 0.77] | 0.61 [0.54, 0.68] | 0.77 [0.71, 0.81] |
| smoother_1 | 0.82 [0.78, 0.84] | 0.77 [0.66, 0.84] | 0.71 [0.66, 0.78] | 0.85 [0.79, 0.92] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.102 and a maximum span of 0.179.

Held-out/ID mean EU ratio across views/seeds: 1.15–1.75.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class2

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 2.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.39; observed mean top-10% set overlap: 0.32 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.86 [0.84, 0.87] | 0.92 [0.85, 0.95] | 0.71 [0.71, 0.71] | 0.81 [0.77, 0.84] |
| cka | 0.94 [0.93, 0.95] | 0.87 [0.84, 0.88] | 0.78 [0.77, 0.81] | 0.77 [0.72, 0.82] |
| mknn | 0.93 [0.90, 0.95] | 0.89 [0.87, 0.90] | 0.73 [0.67, 0.77] | 0.82 [0.75, 0.87] |
| predictivity | 0.84 [0.81, 0.85] | 0.77 [0.71, 0.79] | 0.76 [0.73, 0.78] | 0.60 [0.58, 0.65] |
| smoother_001 | 0.86 [0.84, 0.87] | 0.92 [0.85, 0.95] | 0.71 [0.71, 0.71] | 0.81 [0.77, 0.84] |
| smoother_1 | 0.90 [0.88, 0.92] | 0.93 [0.88, 0.95] | 0.77 [0.75, 0.79] | 0.79 [0.71, 0.84] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.124 and a maximum span of 0.252.

Held-out/ID mean EU ratio across views/seeds: 1.06–2.22.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class3

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 3.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.40; observed mean top-10% set overlap: 0.31 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.82 [0.75, 0.88] | 0.68 [0.64, 0.69] | 0.76 [0.71, 0.88] | -0.10 [-0.26, -0.02] |
| cka | 0.91 [0.84, 0.94] | 0.84 [0.79, 0.89] | 0.87 [0.84, 0.92] | 0.14 [-0.03, 0.22] |
| mknn | 0.91 [0.87, 0.95] | 0.84 [0.79, 0.90] | 0.87 [0.84, 0.94] | 0.11 [-0.07, 0.20] |
| predictivity | 0.83 [0.73, 0.88] | 0.94 [0.92, 0.96] | 0.80 [0.71, 0.84] | 0.41 [0.21, 0.53] |
| smoother_001 | 0.82 [0.75, 0.88] | 0.68 [0.64, 0.69] | 0.76 [0.71, 0.88] | -0.10 [-0.26, -0.02] |
| smoother_1 | 0.89 [0.85, 0.93] | 0.83 [0.79, 0.84] | 0.86 [0.83, 0.93] | 0.13 [-0.04, 0.22] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.094 and a maximum span of 0.186.

Held-out/ID mean EU ratio across views/seeds: 1.16–1.68.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class4

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 4.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.33; observed mean top-10% set overlap: 0.24 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.77 [0.71, 0.81] | 0.83 [0.80, 0.88] | 0.69 [0.66, 0.73] | 0.78 [0.78, 0.78] |
| cka | 0.88 [0.84, 0.92] | 0.83 [0.77, 0.87] | 0.77 [0.70, 0.87] | 0.88 [0.87, 0.92] |
| mknn | 0.86 [0.83, 0.90] | 0.86 [0.77, 0.89] | 0.76 [0.68, 0.85] | 0.89 [0.87, 0.93] |
| predictivity | 0.87 [0.84, 0.89] | 0.85 [0.77, 0.89] | 0.76 [0.72, 0.83] | 0.83 [0.79, 0.84] |
| smoother_001 | 0.77 [0.71, 0.81] | 0.83 [0.80, 0.88] | 0.69 [0.66, 0.73] | 0.78 [0.78, 0.78] |
| smoother_1 | 0.80 [0.75, 0.88] | 0.88 [0.84, 0.95] | 0.74 [0.68, 0.84] | 0.79 [0.78, 0.82] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.094 and a maximum span of 0.194.

Held-out/ID mean EU ratio across views/seeds: 1.08–1.59.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class5

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 5.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.40; observed mean top-10% set overlap: 0.36 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.78 [0.72, 0.85] | 0.88 [0.79, 0.94] | 0.70 [0.62, 0.75] | 0.22 [0.18, 0.26] |
| cka | 0.88 [0.83, 0.94] | 0.86 [0.72, 0.93] | 0.79 [0.68, 0.85] | 0.37 [0.27, 0.42] |
| mknn | 0.90 [0.85, 0.94] | 0.87 [0.72, 0.93] | 0.81 [0.71, 0.89] | 0.35 [0.27, 0.42] |
| predictivity | 0.78 [0.71, 0.83] | 0.83 [0.76, 0.96] | 0.75 [0.66, 0.83] | 0.65 [0.56, 0.70] |
| smoother_001 | 0.78 [0.72, 0.85] | 0.88 [0.79, 0.94] | 0.70 [0.62, 0.75] | 0.22 [0.18, 0.26] |
| smoother_1 | 0.85 [0.81, 0.92] | 0.91 [0.84, 0.97] | 0.79 [0.68, 0.85] | 0.26 [0.20, 0.32] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.113 and a maximum span of 0.214.

Held-out/ID mean EU ratio across views/seeds: 1.33–2.11.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class6

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 6.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.41; observed mean top-10% set overlap: 0.33 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.90 [0.89, 0.92] | 0.89 [0.84, 0.92] | 0.88 [0.84, 0.92] | 0.91 [0.89, 0.94] |
| cka | 0.90 [0.88, 0.92] | 0.86 [0.83, 0.91] | 0.88 [0.84, 0.92] | 0.81 [0.75, 0.87] |
| mknn | 0.86 [0.81, 0.90] | 0.86 [0.81, 0.91] | 0.85 [0.79, 0.90] | 0.76 [0.67, 0.83] |
| predictivity | 0.76 [0.70, 0.78] | 0.71 [0.63, 0.78] | 0.76 [0.68, 0.79] | 0.64 [0.56, 0.71] |
| smoother_001 | 0.90 [0.89, 0.92] | 0.89 [0.84, 0.92] | 0.88 [0.84, 0.92] | 0.91 [0.89, 0.94] |
| smoother_1 | 0.94 [0.93, 0.95] | 0.89 [0.86, 0.91] | 0.93 [0.92, 0.94] | 0.84 [0.82, 0.87] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.088 and a maximum span of 0.156.

Held-out/ID mean EU ratio across views/seeds: 1.13–2.18.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class7

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 7.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.35; observed mean top-10% set overlap: 0.28 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.85 [0.79, 0.89] | 0.82 [0.76, 0.86] | 0.69 [0.66, 0.73] | 0.38 [0.27, 0.41] |
| cka | 0.89 [0.77, 0.96] | 0.71 [0.62, 0.80] | 0.73 [0.70, 0.78] | 0.13 [0.03, 0.20] |
| mknn | 0.88 [0.77, 0.94] | 0.71 [0.66, 0.80] | 0.71 [0.70, 0.72] | 0.09 [-0.02, 0.20] |
| predictivity | 0.83 [0.78, 0.89] | 0.56 [0.47, 0.63] | 0.74 [0.71, 0.78] | 0.11 [0.01, 0.16] |
| smoother_001 | 0.84 [0.78, 0.89] | 0.82 [0.76, 0.86] | 0.68 [0.66, 0.70] | 0.40 [0.27, 0.47] |
| smoother_1 | 0.89 [0.83, 0.93] | 0.73 [0.67, 0.82] | 0.76 [0.75, 0.77] | 0.16 [0.03, 0.24] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.101 and a maximum span of 0.202.

Held-out/ID mean EU ratio across views/seeds: 1.12–1.86.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class8

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 8.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.39; observed mean top-10% set overlap: 0.32 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.85 [0.84, 0.87] | 0.80 [0.76, 0.84] | 0.71 [0.66, 0.73] | 0.17 [0.13, 0.21] |
| cka | 0.94 [0.93, 0.94] | 0.87 [0.84, 0.89] | 0.87 [0.83, 0.90] | 0.28 [0.18, 0.37] |
| mknn | 0.94 [0.93, 0.95] | 0.87 [0.85, 0.89] | 0.88 [0.83, 0.92] | 0.28 [0.21, 0.37] |
| predictivity | 0.76 [0.71, 0.85] | 0.75 [0.66, 0.87] | 0.77 [0.75, 0.81] | 0.47 [0.38, 0.56] |
| smoother_001 | 0.85 [0.84, 0.87] | 0.80 [0.76, 0.84] | 0.71 [0.66, 0.73] | 0.17 [0.13, 0.21] |
| smoother_1 | 0.91 [0.90, 0.92] | 0.87 [0.83, 0.90] | 0.83 [0.79, 0.87] | 0.29 [0.20, 0.38] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.108 and a maximum span of 0.178.

Held-out/ID mean EU ratio across views/seeds: 0.92–2.11.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_image_rbf_class9

Head: rbf; mode: features; seeds: [0, 1, 2, 3, 4]; held-out class: 9.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.38; observed mean top-10% set overlap: 0.33 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.85 [0.84, 0.87] | 0.79 [0.71, 0.84] | 0.75 [0.73, 0.78] | 0.27 [0.22, 0.37] |
| cka | 0.93 [0.93, 0.94] | 0.75 [0.72, 0.79] | 0.89 [0.85, 0.93] | 0.34 [0.27, 0.47] |
| mknn | 0.93 [0.93, 0.94] | 0.76 [0.72, 0.80] | 0.89 [0.85, 0.93] | 0.31 [0.27, 0.39] |
| predictivity | 0.79 [0.73, 0.84] | 0.68 [0.63, 0.77] | 0.79 [0.76, 0.83] | 0.44 [0.38, 0.59] |
| smoother_001 | 0.85 [0.84, 0.87] | 0.79 [0.71, 0.84] | 0.75 [0.73, 0.78] | 0.27 [0.22, 0.37] |
| smoother_1 | 0.91 [0.90, 0.92] | 0.82 [0.77, 0.85] | 0.85 [0.83, 0.87] | 0.39 [0.28, 0.50] |

Near-saturated metrics (span < 1e-4): cca_type. Their rank correlations can depend on tiny numerical differences and should not support a metric-superiority claim.

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.097 and a maximum span of 0.172.

Held-out/ID mean EU ratio across views/seeds: 1.13–2.01.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_synthetic

Head: linear; mode: synthetic; seeds: [0, 1, 2, 3, 4]; held-out class: controlled coverage shift.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.97; observed mean top-10% set overlap: 0.78 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.88 [0.77, 0.95] | 0.88 [0.77, 0.96] | 0.88 [0.73, 0.95] | 0.85 [0.77, 0.92] |
| cka | 0.84 [0.64, 0.99] | 0.83 [0.74, 0.98] | 0.84 [0.66, 0.99] | 0.87 [0.74, 0.99] |
| mknn | 0.83 [0.64, 1.00] | 0.87 [0.79, 0.99] | 0.83 [0.67, 1.00] | 0.87 [0.74, 1.00] |
| predictivity | 0.88 [0.68, 1.00] | 0.88 [0.83, 0.97] | 0.89 [0.78, 1.00] | 0.89 [0.78, 1.00] |
| smoother_001 | 0.88 [0.76, 0.99] | 0.85 [0.74, 0.96] | 0.87 [0.76, 0.99] | 0.89 [0.76, 0.99] |
| smoother_1 | 0.89 [0.77, 0.96] | 0.88 [0.81, 0.95] | 0.88 [0.77, 0.96] | 0.91 [0.87, 0.96] |

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.044 and a maximum span of 0.148.

Held-out/ID mean EU ratio across views/seeds: 12.65–16.97.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## eu_specific_synthetic_rbf

Head: rbf; mode: synthetic; seeds: [0, 1, 2, 3, 4]; held-out class: controlled coverage shift.

900 pair/configuration records; 450 diagnostic records; all contraction checks passed: True.

Observed mean cross-encoder EU agreement: 0.93; observed mean top-10% set overlap: 0.68 (chance expectation 0.10). The following table reports how well each metric predicts those outcomes, not the outcomes themselves.

| Metric | EU ranking | Acquisition overlap | ID-only EU | Held-out-only EU |
|---|---|---|---|---|
| cca_type | 0.78 [0.67, 0.86] | 0.61 [0.45, 0.74] | 0.80 [0.69, 0.95] | 0.67 [0.54, 0.72] |
| cka | 0.89 [0.66, 0.98] | 0.81 [0.56, 0.94] | 0.92 [0.76, 0.99] | 0.91 [0.85, 0.96] |
| mknn | 0.92 [0.81, 0.99] | 0.84 [0.56, 0.97] | 0.91 [0.72, 1.00] | 0.94 [0.85, 0.99] |
| predictivity | 0.87 [0.82, 0.90] | 0.75 [0.56, 0.87] | 0.90 [0.80, 0.99] | 0.83 [0.71, 0.90] |
| smoother_001 | 0.80 [0.77, 0.86] | 0.58 [0.41, 0.81] | 0.83 [0.67, 0.96] | 0.68 [0.41, 0.86] |
| smoother_1 | 0.91 [0.79, 0.99] | 0.80 [0.55, 0.97] | 0.94 [0.89, 0.98] | 0.90 [0.80, 0.99] |

With the same six alignment scores held fixed, changing prior/training fraction changes EU agreement by a median span of 0.039 and a maximum span of 0.121.

Held-out/ID mean EU ratio across views/seeds: 4.68–15.32.

Source SHA-256: `c1403c0af96ab3003f62dfe175c2ed562d798bb9bfe18b7fa959a7f67fbf017b`.

## What remains

The image results are a small pilot. Extend to the full image caches, inspect leave-one-encoder-out sensitivity, validate acquisition utility on learning curves, and add a classification EU estimator with an independently supported likelihood. Known AU is prescribed, not estimated from human disagreement.
