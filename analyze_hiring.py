#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_hiring.py
=================
Statistical analysis pipeline for the paper

  "Human-AI Collaborative Hiring: An Empirical Comparison of Human-Only,
   AI-Only, and Human-AI Employee Selection for Accuracy, Fairness,
   Efficiency, and Decision Quality"

Feed it YOUR real data (one CSV, long format: one row per candidate x pipeline
x reviewer-or-model) and it reproduces every quantity the paper reports:

  Table 1  accuracy / efficiency / decision quality  (AUC, F1, validity r, time, cost, utility)
  Table 2  pairwise tests (DeLong AUC, Williams validity, McNemar) + Holm correction
  Table 3  fairness gaps (DPD, EOD) with BCa bootstrap CIs + paired gap differences
  Table 4  HAC confidence-band / override analysis (+ GEE logistic regression)
  Table 5  threshold sensitivity
  Figures  accuracy, fairness, efficiency-vs-quality, ROC
  Extras   inter-rater reliability ICC, data-quality report, paper_snippets.md

Usage
-----
  python analyze_hiring.py --data my_data.csv --outdir results
  python analyze_hiring.py --help

See README.md for the exact column schema.
"""
import argparse
import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
REQUIRED_COLS = ["candidate_id", "pipeline", "decision", "quality_label"]
OPTIONAL_COLS = ["score", "quality_score", "time_minutes", "reviewer_id",
                 "ai_decision", "ai_confidence", "role_family", "age",
                 "education", "years_experience"]


# --------------------------------------------------------------------------
# 1. Loading, validation, aggregation
# --------------------------------------------------------------------------
def load_data(path, group_cols):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS + list(group_cols) if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing required column(s): {missing}\n"
                 f"        Required: {REQUIRED_COLS + list(group_cols)}")
    df["pipeline"] = df["pipeline"].astype(str).str.strip()
    for c in ["decision", "quality_label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["score", "quality_score", "time_minutes", "ai_decision", "ai_confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def data_quality_report(df, group_cols, pipelines, min_group=20):
    lines = ["DATA QUALITY REPORT", "=" * 60]
    lines.append(f"Rows: {len(df)} | Unique candidates: {df.candidate_id.nunique()}")
    lines.append(f"Pipelines found: {sorted(df.pipeline.unique())}")
    missing_pipes = [p for p in pipelines if p not in set(df.pipeline)]
    if missing_pipes:
        lines.append(f"[WARN] Requested pipelines not in data: {missing_pipes}")
    for c in ["decision", "quality_label"]:
        n_na = df[c].isna().sum()
        if n_na:
            lines.append(f"[WARN] {n_na} missing values in '{c}' (rows will be dropped)")
        bad = df[c].dropna()[~df[c].dropna().isin([0, 1])]
        if len(bad):
            lines.append(f"[ERROR] '{c}' must be 0/1; found values like {bad.unique()[:5]}")
    # label consistency: one quality_label per candidate
    inc = df.groupby("candidate_id")["quality_label"].nunique()
    if (inc > 1).any():
        lines.append(f"[ERROR] {int((inc > 1).sum())} candidates have >1 distinct quality_label")
    base = df.drop_duplicates("candidate_id")["quality_label"].mean()
    lines.append(f"Base rate of good hires (quality_label=1): {base:.3f}")
    for g in group_cols:
        cnt = df.drop_duplicates("candidate_id")[g].value_counts()
        lines.append(f"Group sizes for '{g}': {cnt.to_dict()}")
        small = cnt[cnt < min_group]
        if len(small):
            lines.append(f"[WARN] '{g}' groups with n<{min_group}: {small.to_dict()} "
                         f"-> fairness gaps will be unstable")
    cand_per_pipe = df.groupby("pipeline").candidate_id.nunique().to_dict()
    lines.append(f"Candidates per pipeline: {cand_per_pipe}")
    if len(set(cand_per_pipe.values())) > 1:
        lines.append("[WARN] Pipelines evaluated different numbers of candidates; "
                     "paired tests use only the shared candidates.")
    for opt in OPTIONAL_COLS:
        if opt not in df.columns:
            lines.append(f"[INFO] Optional column '{opt}' absent -> related analyses skipped")
    return "\n".join(lines)


def aggregate_to_candidate(df, group_cols):
    """One row per candidate x pipeline: mean score, majority decision, summed time."""
    agg = {"decision": "mean", "quality_label": "first"}
    if "score" in df.columns:
        agg["score"] = "mean"
    if "quality_score" in df.columns:
        agg["quality_score"] = "first"
    if "time_minutes" in df.columns:
        agg["time_minutes"] = "sum"   # total human time across all reviewers of a candidate
    for g in group_cols:
        agg[g] = "first"
    if "ai_decision" in df.columns:
        agg["ai_decision"] = "mean"
    if "ai_confidence" in df.columns:
        agg["ai_confidence"] = "mean"
    for c in ["role_family", "age", "education", "years_experience"]:
        if c in df.columns:
            agg[c] = "first"
    out = df.groupby(["candidate_id", "pipeline"], as_index=False).agg(agg)
    out["decision"] = (out["decision"] >= 0.5).astype(int)  # majority vote, ties -> hire
    if "ai_decision" in out:
        out["ai_decision"] = (out["ai_decision"] >= 0.5).astype(int)
    return out


# --------------------------------------------------------------------------
# 2. Accuracy metrics
# --------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_auc_cov(preds, y):
    """Fast DeLong (Sun & Xu 2014). preds: k x n; y: 0/1. Returns aucs (k,), cov (k,k)."""
    preds = np.atleast_2d(preds)
    order = (-y).argsort(kind="stable")
    preds = preds[:, order]
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present.")
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r], ty[r], tz[r] = _midrank(pos[r]), _midrank(neg[r]), _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, s


def delong_ci(auc, var, level=0.95):
    se = np.sqrt(max(var, 1e-12))
    z = stats.norm.ppf(0.5 + level / 2)
    return max(0, auc - z * se), min(1, auc + z * se)


def delong_pair_test(s_a, s_b, y):
    aucs, cov = delong_auc_cov(np.vstack([s_a, s_b]), y)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return aucs[0] - aucs[1], z, 2 * stats.norm.sf(abs(z))


def f1_prec_rec(y, d):
    tp = int(((d == 1) & (y == 1)).sum())
    fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def validity(score, criterion):
    r, p = stats.pearsonr(score, criterion)
    n = len(score)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return r, lo, hi, p


def williams_test(r12, r13, r23, n):
    """Williams (1959) test: dependent, overlapping correlations (criterion=1)."""
    det = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2
    denom = 2 * ((n - 1) / (n - 3)) * det + (rbar**2) * (1 - r23) ** 3
    t = (r12 - r13) * np.sqrt(((n - 1) * (1 + r23)) / denom)
    p = 2 * stats.t.sf(abs(t), n - 3)
    q = np.arctanh(np.clip(r12, -0.9999, 0.9999)) - np.arctanh(np.clip(r13, -0.9999, 0.9999))
    return t, p, q


def mcnemar_exact(correct_a, correct_b):
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        return b, c, 1.0
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
# 3. Fairness metrics + bootstrap
# --------------------------------------------------------------------------
def group_rates(d, y, g):
    sel, tpr = {}, {}
    for lvl in pd.unique(g):
        m = g == lvl
        sel[lvl] = d[m].mean() if m.sum() else np.nan
        mp = m & (y == 1)
        tpr[lvl] = d[mp].mean() if mp.sum() else np.nan
    return sel, tpr


def gap(vals):
    v = [x for x in vals.values() if not np.isnan(x)]
    return (max(v) - min(v)) if len(v) >= 2 else np.nan


def dpd(d, y, g):
    return gap(group_rates(d, y, g)[0])


def eod(d, y, g):
    return gap(group_rates(d, y, g)[1])


def boot_ci(stat_fn, n, n_boot=2000, seed=42, level=0.95):
    """BCa bootstrap CI over candidate indices; falls back to percentile."""
    idx = np.arange(n)
    try:
        res = stats.bootstrap((idx,), stat_fn, vectorized=False, n_resamples=n_boot,
                              confidence_level=level, method="BCa",
                              random_state=np.random.default_rng(seed))
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError
        return float(lo), float(hi), "BCa"
    except Exception:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            vals.append(stat_fn(rng.integers(0, n, n)))
        vals = np.array([v for v in vals if not np.isnan(v)])
        a = (1 - level) / 2
        return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)), "percentile"


# --------------------------------------------------------------------------
# 4. Per-pipeline summary (Table 1)
# --------------------------------------------------------------------------
def pipeline_table(cand, raw, pipelines, ref, gamma, hourly_rate, ai_cost,
                   human_prefix=("HO", "HAC"), ai_prefix=("AO", "HAC")):
    rows = []
    time_ref = None
    if "time_minutes" in cand.columns:
        ref_sub = cand[cand.pipeline == ref]
        if not ref_sub.empty and "time_minutes" in ref_sub.columns:
            time_ref = ref_sub["time_minutes"].mean()
    for p in pipelines:
        sub = cand[cand.pipeline == p]
        if sub.empty:
            continue
        y, d = sub.quality_label.values.astype(int), sub.decision.values.astype(int)
        row = {"pipeline": p, "n_candidates": len(sub)}
        f1, prec, rec = f1_prec_rec(y, d)
        row.update(F1=f1, precision=prec, recall=rec, accuracy=(y == d).mean(),
                   selection_rate=d.mean())
        if "score" in sub.columns and sub.score.notna().all() and len(np.unique(y)) == 2:
            aucs, cov = delong_auc_cov(sub.score.values, y)
            lo, hi = delong_ci(aucs[0], cov[0, 0])
            row.update(AUC=aucs[0], AUC_lo=lo, AUC_hi=hi)
            crit = sub["quality_score"].values if "quality_score" in sub.columns else y
            mask = sub.score.notna().values
            if "quality_score" in sub.columns:
                mask = mask & ~pd.isna(crit)
            if mask.sum() >= 3 and len(np.unique(y[mask])) == 2:
                r, rlo, rhi, rp = validity(sub.score.values[mask], np.asarray(crit)[mask])
                row.update(validity_r=r, validity_lo=rlo, validity_hi=rhi, validity_p=rp)
        tp = int(((d == 1) & (y == 1)).sum())
        fp = int(((d == 1) & (y == 0)).sum())
        row["utility_U"] = tp - gamma * fp
        row["utility_per_100"] = 100 * (tp - gamma * fp) / len(sub)
        if "time_minutes" in sub.columns and sub["time_minutes"].notna().any():
            t = sub["time_minutes"]
            row["time_mean_min"] = t.mean()
            row["time_sd_min"] = t.std()
            cost = t.mean() / 60 * hourly_rate if p.startswith(human_prefix) else 0.0
            cost += ai_cost if p.startswith(ai_prefix) else 0.0
            row["cost_per_decision"] = cost
        rows.append(row)
    out = pd.DataFrame(rows)
    if "cost_per_decision" in out and ref in set(out.pipeline):
        ref_cost = out.loc[out.pipeline == ref, "cost_per_decision"].iloc[0]
        if ref_cost > 0:
            out["cost_relative_to_ref"] = out["cost_per_decision"] / ref_cost
    if time_ref:
        out["time_saving_vs_ref_pct"] = 100 * (1 - out["time_mean_min"] / time_ref)
    return out


# --------------------------------------------------------------------------
# 5. Pairwise tests (Table 2)
# --------------------------------------------------------------------------
def wide(cand, pipelines, col):
    w = cand.pivot(index="candidate_id", columns="pipeline", values=col)
    return w[[p for p in pipelines if p in w.columns]]


def pairwise_tests(cand, pipelines):
    rows = []
    ws = wide(cand, pipelines, "score") if "score" in cand.columns else None
    wd = wide(cand, pipelines, "decision")
    wy = cand.drop_duplicates("candidate_id").set_index("candidate_id")["quality_label"]
    wq = (cand.drop_duplicates("candidate_id").set_index("candidate_id")["quality_score"]
          if "quality_score" in cand.columns else wy)
    avail = [p for p in pipelines if p in wd.columns]
    for a, b in combinations(avail, 2):
        common = wd[[a, b]].dropna().index
        if ws is not None:
            common = common.intersection(ws[[a, b]].dropna().index)
        if wq is not None:
            common = common.intersection(wq.dropna().index)
        y = wy.loc[common].values.astype(int)
        row = {"comparison": f"{a} vs {b}", "n_shared": len(common)}
        if ws is not None and len(np.unique(y)) == 2:
            dauc, z, p = delong_pair_test(ws.loc[common, a].values, ws.loc[common, b].values, y)
            row.update(dAUC=dauc, DeLong_Z=z, DeLong_p=p)
            r12 = stats.pearsonr(ws.loc[common, a], wq.loc[common])[0]
            r13 = stats.pearsonr(ws.loc[common, b], wq.loc[common])[0]
            r23 = stats.pearsonr(ws.loc[common, a], ws.loc[common, b])[0]
            t, pv, q = williams_test(r12, r13, r23, len(common))
            row.update(d_validity=r12 - r13, Williams_t=t, Williams_p=pv, Cohens_q=q)
        ca = (wd.loc[common, a].values == y).astype(int)
        cb = (wd.loc[common, b].values == y).astype(int)
        b01, c01, pm = mcnemar_exact(ca, cb)
        row.update(McNemar_a_only_correct=b01, McNemar_b_only_correct=c01, McNemar_p=pm)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["DeLong_p", "Williams_p", "McNemar_p"]:
        if col in out:
            out[col + "_holm"] = holm(out[col].values)
    return out


# --------------------------------------------------------------------------
# 6. Fairness table (Table 3)
# --------------------------------------------------------------------------
def fairness_table(cand, pipelines, group_cols, n_boot, seed):
    rows = []
    counter = 0
    for p in pipelines:
        sub = cand[cand.pipeline == p].reset_index(drop=True)
        if sub.empty:
            continue
        y, d = sub.quality_label.values.astype(int), sub.decision.values.astype(int)
        for g in group_cols:
            gv = sub[g].values
            for name, fn in (("DPD", dpd), ("EOD", eod)):
                est = fn(d, y, gv)
                lo, hi, how = boot_ci(lambda i: fn(d[i], y[i], gv[i]), len(sub), n_boot, seed + counter)
                counter += 1
                lo = max(0.0, lo)   # gaps are non-negative (max-min)
                hi = min(1.0, hi)
                rows.append({"pipeline": p, "attribute": g, "metric": name,
                             "gap": est, "ci_lo": lo, "ci_hi": hi, "ci_method": how})
    return pd.DataFrame(rows)


def fairness_diffs(cand, pipelines, group_cols, n_boot, seed):
    rows = []
    wd = wide(cand, pipelines, "decision")
    first = cand.drop_duplicates("candidate_id").set_index("candidate_id")
    avail = [p for p in pipelines if p in wd.columns]
    counter = 0
    for a, b in combinations(avail, 2):
        common = wd[[a, b]].dropna().index
        y = first.loc[common, "quality_label"].values.astype(int)
        da, db = wd.loc[common, a].values.astype(int), wd.loc[common, b].values.astype(int)
        for g in group_cols:
            gv = first.loc[common, g].values
            for name, fn in (("DPD", dpd), ("EOD", eod)):
                est = fn(da, y, gv) - fn(db, y, gv)
                lo, hi, how = boot_ci(lambda i: fn(da[i], y[i], gv[i]) - fn(db[i], y[i], gv[i]),
                                      len(common), n_boot, seed + counter)
                counter += 1
                rows.append({"comparison": f"{a} - {b}", "attribute": g, "metric": name,
                             "gap_difference": est, "ci_lo": lo, "ci_hi": hi,
                             "significant_95": bool(lo > 0 or hi < 0), "ci_method": how})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 7. HAC confidence-band / override analysis (Table 4)
# --------------------------------------------------------------------------
def override_analysis(raw, hac_name, alpha, beta, ao_name):
    need = {"ai_confidence", "ai_decision"}
    if not need.issubset(raw.columns):
        return None, None
    h = raw[(raw.pipeline == hac_name)].dropna(subset=["ai_confidence", "ai_decision"]).copy()
    if h.empty:
        return None, None
    h["band"] = pd.cut(h.ai_confidence, [-np.inf, alpha, beta, np.inf], right=False,
                        labels=["low", "mid", "high"])
    h["override"] = (h.decision != h.ai_decision).astype(int)
    h["final_correct"] = (h.decision == h.quality_label).astype(int)
    h["ai_correct"] = (h.ai_decision == h.quality_label).astype(int)
    h["override_helped"] = ((h.override == 1) & (h.final_correct == 1)).astype(int)
    h["override_harmed"] = ((h.override == 1) & (h.ai_correct == 1) & (h.final_correct == 0)).astype(int)
    tab = h.groupby("band", observed=True).agg(
        n=("decision", "size"), override_rate=("override", "mean"),
        final_accuracy=("final_correct", "mean"), ai_accuracy=("ai_correct", "mean"),
        overrides_that_helped=("override_helped", "sum"),
        overrides_that_harmed=("override_harmed", "sum")).reset_index()
    tab["net_accuracy_gain_human_vs_ai"] = tab.final_accuracy - tab.ai_accuracy
    gee_txt = None
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        cluster = h["reviewer_id"] if "reviewer_id" in h.columns else h["candidate_id"]
        m = smf.gee("final_correct ~ C(band, Treatment('high'))", groups=cluster, data=h,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        coef = pd.DataFrame({"coef": m.params, "OR": np.exp(m.params),
                             "OR_lo": np.exp(m.conf_int()[0]), "OR_hi": np.exp(m.conf_int()[1]),
                             "p": m.pvalues})
        gee_txt = coef.reset_index().rename(columns={"index": "term"})
    except Exception as e:  # statsmodels missing or non-convergence
        gee_txt = pd.DataFrame({"note": [f"GEE skipped: {e}"]})
    return tab, gee_txt


# --------------------------------------------------------------------------
# 8. Threshold sensitivity (Table 5)
# --------------------------------------------------------------------------
def threshold_sensitivity(cand, pipelines, group_cols, gamma, thresholds):
    rows = []
    if "score" not in cand.columns:
        return pd.DataFrame()
    for p in pipelines:
        sub = cand[cand.pipeline == p].dropna(subset=["score"])
        if sub.empty:
            continue
        y = sub.quality_label.values.astype(int)
        for t in thresholds:
            d = (sub.score.values >= t).astype(int)
            f1, _, rec = f1_prec_rec(y, d)
            row = {"pipeline": p, "threshold": t, "F1": f1, "recall": rec,
                   "selection_rate": d.mean(),
                   "utility_U": int(((d == 1) & (y == 1)).sum()) - gamma * int(((d == 1) & (y == 0)).sum())}
            for g in group_cols:
                row[f"DPD_{g}"] = dpd(d, y, sub[g].values)
                row[f"EOD_{g}"] = eod(d, y, sub[g].values)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9. Inter-rater reliability (one-way ICC, handles unbalanced raters)
# --------------------------------------------------------------------------
def icc_oneway(raw, pipeline, col="score"):
    if "reviewer_id" not in raw.columns or col not in raw.columns:
        return None
    s = raw[raw.pipeline == pipeline].dropna(subset=[col, "reviewer_id"])
    grp = s.groupby("candidate_id")[col]
    ks = grp.size()
    keep = ks[ks >= 2].index
    if len(keep) < 5:
        return None
    s = s[s.candidate_id.isin(keep)]
    grp = s.groupby("candidate_id")[col]
    n, ks = grp.ngroups, grp.size().values
    N = ks.sum()
    gm = s[col].mean()
    ssb = (grp.size() * (grp.mean() - gm) ** 2).sum()
    ssw = ((s[col] - grp.transform("mean")) ** 2).sum()
    msb, msw = ssb / (n - 1), ssw / (N - n)
    k0 = (N - (ks**2).sum() / N) / (n - 1)
    icc1 = (msb - msw) / (msb + (k0 - 1) * msw)
    icck = (msb - msw) / msb
    return {"pipeline": pipeline, "n_candidates_multi_rated": int(n),
            "mean_raters_per_candidate": float(ks.mean()),
            "ICC1_single_rater": float(icc1), "ICC1k_average_rater": float(icck)}


# --------------------------------------------------------------------------
# 10. Figures
# --------------------------------------------------------------------------
def make_figures(t1, fair, cand, pipelines, outdir, group_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams["font.family"] = "serif"
    pal = ["#2e75b6", "#2a9d8f", "#7b4397", "#c9a227", "#7f7f7f", "#b3261e", "#1f3864"]
    names = list(t1.pipeline)
    x = np.arange(len(names))

    # Fig: accuracy
    if {"AUC", "F1", "validity_r"}.issubset(t1.columns):
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        w = 0.26
        for k, (col, lab, c) in enumerate([("AUC", "AUC", "#1f3864"), ("F1", "F1-score", "#2e75b6"),
                                            ("validity_r", "Validity r", "#c9a227")]):
            bars = ax.bar(x + (k - 1) * w, t1[col], w, label=lab, color=c)
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                        f"{r.get_height():.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("Score"); ax.legend(frameon=False)
        ax.set_title("Predictive accuracy and criterion-related validity", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{outdir}/fig_accuracy.png", dpi=300); plt.close(fig)

    # Fig: fairness
    if fair is not None and not fair.empty:
        fig, axes = plt.subplots(1, len(group_cols), figsize=(5 * len(group_cols), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, g in zip(axes, group_cols):
            sub = fair[fair.attribute == g]
            for k, m in enumerate(["DPD", "EOD"]):
                s = sub[sub.metric == m].set_index("pipeline").reindex(names)
                yerr = np.vstack([s.gap - s.ci_lo, s.ci_hi - s.gap]).clip(min=0)
                ax.bar(x + (k - 0.5) * 0.35, s.gap, 0.35, yerr=yerr, capsize=2,
                       label={"DPD": "Demographic parity diff.", "EOD": "Equal opportunity diff."}[m],
                       color=["#2e75b6", "#b3261e"][k], alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title(g, weight="bold")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Fairness gap (95% CI)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Subgroup fairness gaps by pipeline", weight="bold")
        fig.tight_layout(); fig.savefig(f"{outdir}/fig_fairness.png", dpi=300); plt.close(fig)

    # Fig: efficiency vs quality
    if {"time_mean_min", "utility_U"}.issubset(t1.columns):
        fig, ax1 = plt.subplots(figsize=(8.6, 4.6))
        ax1.bar(x, t1.time_mean_min, 0.5, color=pal[: len(x)], alpha=0.85)
        ax1.set_ylabel("Time per decision (min)"); ax1.set_xticks(x); ax1.set_xticklabels(names)
        ax2 = ax1.twinx()
        ax2.plot(x, t1.utility_U, "o-", color="#b3261e", lw=2)
        ax2.set_ylabel("Utility index U", color="#b3261e")
        ax1.set_title("Efficiency vs decision quality", weight="bold")
        fig.tight_layout(); fig.savefig(f"{outdir}/fig_efficiency_quality.png", dpi=300); plt.close(fig)

    # Fig: ROC
    if "score" in cand.columns:
        fig, ax = plt.subplots(figsize=(5.4, 5))
        for k, p in enumerate(names):
            s = cand[cand.pipeline == p].dropna(subset=["score"])
            y = s.quality_label.values.astype(int)
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y, s.score.values)
                auc = t1.loc[t1.pipeline == p, "AUC"].iloc[0] if "AUC" in t1 else np.nan
                ax.plot(fpr, tpr, color=pal[k % len(pal)], lw=2, label=f"{p} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(frameon=False); ax.set_title("ROC curves", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{outdir}/fig_roc.png", dpi=300); plt.close(fig)


# --------------------------------------------------------------------------
# 11. Paper snippets (numbers ready to paste into Section 6)
# --------------------------------------------------------------------------
def fmt_p(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def write_snippets(path, t1, pw, fair, fdiff, bands, icc, args):
    L = ["# Auto-generated numbers for the paper (verify before pasting)\n"]
    L.append("## Table 1 -> Section 6.1\n")
    for _, r in t1.iterrows():
        s = f"- **{r.pipeline}**: "
        parts = []
        if "AUC" in r and not pd.isna(r.AUC):
            parts.append(f"AUC = {r.AUC:.2f} [{r.AUC_lo:.2f}, {r.AUC_hi:.2f}]")
        parts.append(f"F1 = {r.F1:.2f}")
        if "validity_r" in r and not pd.isna(r.validity_r):
            parts.append(f"r = {r.validity_r:.2f} [{r.validity_lo:.2f}, {r.validity_hi:.2f}]")
        if "time_mean_min" in r and not pd.isna(r.time_mean_min):
            parts.append(f"time = {r.time_mean_min:.1f} min")
        parts.append(f"U = {r.utility_U:.0f}")
        L.append(s + "; ".join(parts))
    L.append("\n## Pairwise tests -> Section 6.2 (Holm-corrected)\n")
    for _, r in pw.iterrows():
        bits = [f"n = {int(r.n_shared)}"]
        if "DeLong_Z" in r and not pd.isna(r.DeLong_Z):
            bits.append(f"DeLong Z = {r.DeLong_Z:.2f}, {fmt_p(r.DeLong_p_holm)}")
            bits.append(f"validity diff = {r.d_validity:.2f}, Williams t = {r.Williams_t:.2f}, "
                        f"{fmt_p(r.Williams_p_holm)}, q = {r.Cohens_q:.2f}")
        bits.append(f"McNemar {fmt_p(r.McNemar_p_holm)}")
        L.append(f"- {r.comparison}: " + "; ".join(bits))
    L.append("\n## Fairness -> Section 6.4\n")
    for _, r in fair.iterrows():
        L.append(f"- {r.pipeline} / {r.attribute} / {r.metric}: {r.gap:.2f} "
                 f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] ({r.ci_method})")
    if fdiff is not None and not fdiff.empty:
        L.append("\n### Paired gap differences\n")
        for _, r in fdiff.iterrows():
            L.append(f"- {r.comparison} / {r.attribute} / {r.metric}: {r.gap_difference:+.2f} "
                     f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] significant={r.significant_95}")
    if bands is not None:
        L.append("\n## HAC confidence bands -> Section 6.4 / 6.5\n")
        L.append(bands.to_markdown(index=False, floatfmt=".3f"))
    if icc:
        L.append("\n## Inter-rater reliability -> Section 5.3\n")
        L.append(f"- {icc}")
    L.append("\n> Replace the synthetic-benchmark language in Sections 4.1, 7.5 and 8 "
             "with a description of your REAL data source, sampling, consent and ethics approval.")
    open(path, "w", encoding="utf-8").write("\n".join(L))


# --------------------------------------------------------------------------
# 12. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV file (see README for schema)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--pipelines", nargs="+", default=["HO", "AO", "HAC"],
                    help="Pipeline names in display order (add ablations/baselines here)")
    ap.add_argument("--ref", default="HO", help="Reference pipeline for relative cost / time saving")
    ap.add_argument("--hac", default="HAC", help="Name of the human-AI pipeline (for override analysis)")
    ap.add_argument("--ao", default="AO")
    ap.add_argument("--group-cols", nargs="+", default=["gender"], help="Protected-attribute columns")
    ap.add_argument("--gamma", type=float, default=1.5, help="False-hire cost weight in utility U")
    ap.add_argument("--hourly-rate", type=float, default=40.0, help="Reviewer fully-loaded cost / hour")
    ap.add_argument("--ai-cost", type=float, default=0.05, help="AI inference cost per candidate")
    ap.add_argument("--human-pipelines", nargs="+", default=["HO", "HAC"],
                    help="Pipeline-name prefixes whose time_minutes are human reviewer time (costed hourly)")
    ap.add_argument("--ai-pipelines", nargs="+", default=["AO", "HAC"],
                    help="Pipeline-name prefixes that incur the per-case AI cost")
    ap.add_argument("--alpha", type=float, default=0.15, help="Low/mid confidence cut-point")
    ap.add_argument("--beta", type=float, default=0.40, help="Mid/high confidence cut-point")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples (paper used 10000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw = load_data(args.data, args.group_cols)
    report = data_quality_report(raw, args.group_cols, args.pipelines)
    print(report)
    open(f"{args.outdir}/data_quality_report.txt", "w", encoding="utf-8").write(report)
    if "[ERROR]" in report:
        sys.exit("\nFix the [ERROR] items above and re-run.")

    raw = raw.dropna(subset=["decision", "quality_label"])
    pipes = [p for p in args.pipelines if p in set(raw.pipeline)]
    cand = aggregate_to_candidate(raw, args.group_cols)

    t1 = pipeline_table(cand, raw, pipes, args.ref, args.gamma, args.hourly_rate, args.ai_cost,
                        tuple(args.human_pipelines), tuple(args.ai_pipelines))
    pw = pairwise_tests(cand, pipes)
    fair = fairness_table(cand, pipes, args.group_cols, args.n_boot, args.seed)
    fdiff = fairness_diffs(cand, pipes, args.group_cols, args.n_boot, args.seed)
    bands, gee = override_analysis(raw, args.hac, args.alpha, args.beta, args.ao)
    sens = threshold_sensitivity(cand, pipes, args.group_cols, args.gamma, np.arange(0.3, 0.71, 0.05))
    icc = icc_oneway(raw, args.ref)

    for name, tab in [("table1_main", t1), ("table2_pairwise_tests", pw), ("table3_fairness", fair),
                      ("table3b_fairness_pairwise_diff", fdiff), ("table4_hac_bands", bands),
                      ("table4b_hac_gee", gee), ("table5_threshold_sensitivity", sens)]:
        if tab is not None and len(tab):
            tab.to_csv(f"{args.outdir}/{name}.csv", index=False)
    json.dump({"icc": icc, "args": vars(args)}, open(f"{args.outdir}/results.json", "w"), indent=2)

    make_figures(t1, fair, cand, pipes, args.outdir, args.group_cols)
    write_snippets(f"{args.outdir}/paper_snippets.md", t1, pw, fair, fdiff, bands, icc, args)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== TABLE 1 ===\n", t1.round(3).to_string(index=False))
    print("\n=== PAIRWISE TESTS ===\n", pw.round(4).to_string(index=False))
    print("\n=== FAIRNESS ===\n", fair.round(3).to_string(index=False))
    if bands is not None:
        print("\n=== HAC BANDS ===\n", bands.round(3).to_string(index=False))
    print(f"\nAll outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
