# Auto-generated numbers for the paper (verify before pasting)

## Table 1 -> Section 6.1

- **HO**: AUC = 0.71 [0.67, 0.75]; F1 = 0.59; r = 0.46 [0.40, 0.52]; time = 54.4 min; U = -102
- **AO**: AUC = 0.75 [0.71, 0.79]; F1 = 0.63; r = 0.54 [0.48, 0.59]; time = 0.9 min; U = -66
- **HAC**: AUC = 0.79 [0.76, 0.83]; F1 = 0.60; r = 0.64 [0.59, 0.68]; time = 4.6 min; U = -50

## Pairwise tests -> Section 6.2 (Holm-corrected)

- HO vs AO: n = 600; DeLong Z = -1.55, p = .122; validity diff = -0.08, Williams t = -2.07, p = .039, q = -0.10; McNemar p = .224
- HO vs HAC: n = 600; DeLong Z = -3.52, p = .001; validity diff = -0.17, Williams t = -5.25, p < .001, q = -0.25; McNemar p = .224
- AO vs HAC: n = 600; DeLong Z = -3.41, p = .001; validity diff = -0.10, Williams t = -5.39, p < .001, q = -0.15; McNemar p = .951

## Fairness -> Section 6.4

- HO / gender / DPD: 0.08 [0.01, 0.16] (BCa)
- HO / gender / EOD: 0.04 [0.00, 0.14] (BCa)
- HO / region / DPD: 0.05 [0.00, 0.12] (BCa)
- HO / region / EOD: 0.07 [0.00, 0.18] (BCa)
- AO / gender / DPD: 0.12 [0.04, 0.20] (BCa)
- AO / gender / EOD: 0.09 [0.01, 0.20] (BCa)
- AO / region / DPD: 0.05 [0.00, 0.14] (BCa)
- AO / region / EOD: 0.06 [0.00, 0.15] (BCa)
- HAC / gender / DPD: 0.01 [0.00, 0.02] (BCa)
- HAC / gender / EOD: 0.09 [0.01, 0.21] (BCa)
- HAC / region / DPD: 0.03 [0.00, 0.10] (BCa)
- HAC / region / EOD: 0.02 [0.00, 0.06] (BCa)

### Paired gap differences

- HO - AO / gender / DPD: -0.03 [-0.13, +0.07] significant=False
- HO - AO / gender / EOD: -0.05 [-0.18, +0.08] significant=False
- HO - AO / region / DPD: -0.00 [-0.10, +0.08] significant=False
- HO - AO / region / EOD: +0.01 [-0.10, +0.14] significant=False
- HO - HAC / gender / DPD: +0.07 [+0.01, +0.19] significant=True
- HO - HAC / gender / EOD: -0.05 [-0.18, +0.09] significant=False
- HO - HAC / region / DPD: +0.02 [-0.07, +0.12] significant=False
- HO - HAC / region / EOD: +0.06 [-0.03, +0.19] significant=False
- AO - HAC / gender / DPD: +0.11 [+0.04, +0.21] significant=True
- AO - HAC / gender / EOD: +0.00 [-0.15, +0.14] significant=False
- AO - HAC / region / DPD: +0.02 [-0.07, +0.12] significant=False
- AO - HAC / region / EOD: +0.04 [-0.04, +0.18] significant=False

## HAC confidence bands -> Section 6.4 / 6.5

| band   |   n |   override_rate |   final_accuracy |   ai_accuracy |   overrides_that_helped |   overrides_that_harmed |   net_accuracy_gain_human_vs_ai |
|:-------|----:|----------------:|-----------------:|--------------:|------------------------:|------------------------:|--------------------------------:|
| low    | 148 |           0.716 |            0.588 |         0.466 |                      62 |                      44 |                           0.122 |
| mid    | 230 |           0.587 |            0.574 |         0.613 |                      63 |                      72 |                          -0.039 |
| high   | 222 |           0.131 |            0.730 |         0.761 |                      11 |                      18 |                          -0.032 |

## Inter-rater reliability -> Section 5.3

- {'pipeline': 'HO', 'n_candidates_multi_rated': 600, 'mean_raters_per_candidate': 2.8933333333333335, 'ICC1_single_rater': 0.17337927334835684, 'ICC1k_average_rater': 0.37763798911731444}

> Replace the synthetic-benchmark language in Sections 4.1, 7.5 and 8 with a description of your REAL data source, sampling, consent and ethics approval.