# Hiring-pipeline analysis toolkit
Reproduces every statistic in *"Human–AI Collaborative Hiring: An Empirical Comparison of
Human-Only, AI-Only, and Human–AI Employee Selection…"* from **your real data**.

## 1. Install & self-test
    pip install -r requirements.txt
    python test_metrics.py            # verifies DeLong/F1/Holm/Williams/DPD/EOD vs independent code
    python generate_testing_data.py --out testing_data.csv      # SYNTHETIC, for a dry run only
    python analyze_hiring.py --data testing_data.csv --outdir result_output --group-cols gender region

## 2. Prepare your real data (one CSV, long format)
One row per **candidate × pipeline × reviewer-or-model**. See `data_template.csv`.

| column | required | meaning |
|---|---|---|
| candidate_id | yes | same ID across all pipelines |
| pipeline | yes | `HO`, `AO`, `HAC` (+ any baselines/ablations, e.g. `HAC_noLabel`, `RAND`) |
| decision | yes | 0/1 final hire decision of that row |
| quality_label | yes | 0/1 ground truth (good hire), **identical for a candidate in every row** |
| gender / region / … | yes (≥1) | protected-attribute columns; pass with `--group-cols` |
| score | recommended | continuous suitability score 0–1 → AUC, validity r, threshold sensitivity |
| quality_score | recommended | continuous criterion (e.g. 6-month performance composite) for validity r |
| time_minutes | recommended | time per decision → efficiency & cost |
| reviewer_id | optional | enables ICC and clustered GEE |
| ai_decision, ai_confidence | HAC rows | AI recommendation and confidence (|score−τ| normalised 0–1) → override/band analysis |

Multiple reviewers per candidate (HO): scores are averaged, decisions majority-voted (ties → hire).

## 3. Run
    python analyze_hiring.py --data my_real_data.csv --outdir results \
        --pipelines HO AO HAC --group-cols gender region \
        --gamma 1.5 --hourly-rate 40 --ai-cost 0.05 --alpha 0.15 --beta 0.40 --n-boot 10000

Add ablations/baselines by listing them in `--pipelines HO AO HAC HAC_noLabel RAND`.
Use `--human-pipelines` / `--ai-pipelines` (name prefixes) to control which time columns are costed.

## 4. Outputs → where they go in the paper
| file | paper location |
|---|---|
| table1_main.csv | Table 1 (§6.1): AUC±CI, F1, validity r±CI, time, cost, utility |
| table2_pairwise_tests.csv | §6.2: DeLong, Williams (Cohen's q), McNemar, Holm-corrected p |
| table3_fairness.csv, table3b_… | Table 3 (§6.4): DPD/EOD with BCa CIs; paired gap differences |
| table4_hac_bands.csv, table4b_hac_gee.csv | §6.4–6.5: override rate/benefit by confidence band; OR from GEE |
| table5_threshold_sensitivity.csv | §6.4 sensitivity analysis |
| fig_accuracy / fig_fairness / fig_efficiency_quality / fig_roc (.png, 300 dpi) | Figures 3–5 (+ROC) |
| paper_snippets.md | ready-to-paste numbers |
| data_quality_report.txt | check class balance and subgroup sizes first |

## 5. After you have real results, edit the manuscript
1. §4.1/§4.2: replace "synthetic benchmark" with the real source, sampling, N, inclusion criteria, anonymisation.
2. Tables 1–3, Figures 3–5, §6, abstract, highlights, conclusion: replace every number; re-check that each
   hypothesis (H1–H4) is still supported — **report null or reversed findings honestly**.
3. §4.3, Declarations: real ethics-approval ID, consent statement, participant counts.
4. §7.5 and §8: remove the synthetic-data caveats that no longer apply; add the real limitations.
5. Delete the "(synthetic)" line in Appendix A.4 and the Section 9 repository text; upload code + de-identified data.

## Notes & caveats
* DPD/EOD are **absolute max–min gaps**, so their bootstrap CIs are bounded at 0 (a lower bound near 0 is expected).
* Groups with n < 20 give unstable gaps — the data-quality report warns you.
* `ai_confidence` must be on the same normalised scale as `--alpha/--beta`.
* Cost model: human time × hourly rate for `--human-pipelines`, plus per-case AI cost for `--ai-pipelines`.
* GEE (clustered on reviewer) is used instead of a full mixed model; for a random-slope GLMM use R `lme4`.
* Never publish results generated from `testing_data.csv`.
