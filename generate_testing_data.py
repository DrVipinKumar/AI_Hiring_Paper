#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_testing_data.py
------------------------
Creates a SYNTHETIC but realistic CSV in the exact schema expected by
analyze_hiring.py so you can test the full pipeline end-to-end BEFORE
loading real data.

The generator mimics a real hiring funnel: job-relevant ability, protected
attributes, multiple human reviewers with individual bias/noise, an AI
model with measurable (and slightly biased) accuracy, and a human-AI
collaborative pipeline with confidence-dependent overrides.

!!  The numbers it produces are synthetic. NEVER report them as real findings.  !!

  python generate_testing_data.py --n 600 --out testing_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


EDU_MAP = {"HighSchool": 0.0, "Bachelor": 1.0, "Master": 1.7, "PhD": 2.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of candidates")
    ap.add_argument("--reviewers", type=int, default=14, help="number of distinct reviewers")
    ap.add_argument("--out", default="testing_data.csv")
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n = a.n

    # ---- Candidate attributes (realistic base rates) ----
    gender = rng.choice(["F", "M"], n, p=[0.44, 0.56])
    region = rng.choice(["Metro", "Non-metro"], n, p=[0.58, 0.42])
    role = rng.choice(["Software", "Analytics", "Operations"], n, p=[0.46, 0.32, 0.22])
    education = rng.choice(["HighSchool", "Bachelor", "Master", "PhD"], n, p=[0.12, 0.50, 0.28, 0.10])
    experience = rng.integers(0, 16, n)
    age = rng.integers(22, 58, n)

    edu_val = np.array([EDU_MAP[e] for e in education])
    role_effect = np.where(role == "Software", 0.25,
                  np.where(role == "Analytics", 0.10, -0.05))

    # latent job-relevant ability (continuous)
    ability = (0.35 * (experience / 10.0)
               + 0.30 * (edu_val / 2.2)
               + 0.25 * rng.normal(size=n)
               + role_effect
               + 0.05 * (age - 35) / 10.0)
    ability = ability - ability.mean()   # center for realistic score balance

    # 6-month performance composite (ground-truth criterion)
    quality_score = 0.80 * ability + 0.20 * rng.normal(size=n)
    # ~40% good hires, realistic class imbalance
    threshold = np.quantile(quality_score, 0.60)
    y = (quality_score >= threshold).astype(int)

    # reviewer pool with individual skill & bias
    reviewer_ids = [f"R{i:02d}" for i in range(a.reviewers)]
    rev_skill = rng.normal(0.90, 0.20, a.reviewers)          # signal weight
    rev_bias_region = rng.normal(0, 0.12, a.reviewers)      # Non-metro bias
    rev_bias_gender = rng.normal(0, 0.10, a.reviewers)      # gender bias
    rev_noise = rng.uniform(0.6, 0.9, a.reviewers)

    rows = []
    for i in range(n):
        cid = f"C{i:04d}"
        base = dict(candidate_id=cid, quality_label=int(y[i]),
                    quality_score=float(round(quality_score[i], 4)),
                    gender=gender[i], region=region[i], role_family=role[i],
                    age=int(age[i]), education=education[i],
                    years_experience=int(experience[i]))

        # ---------- AI-only ----------
        ai_logit = (1.95 * ability[i]
                    + 0.20 * (1 if gender[i] == "M" else 0)
                    + 0.10 * (1 if region[i] == "Metro" else 0)
                    + rng.normal(0, 0.66))
        s_ao = float(sigmoid(ai_logit))
        conf = float(min(1.0, abs(s_ao - 0.5) * 2.0))
        d_ao = int(s_ao >= 0.5)
        rows.append({**base, "pipeline": "AO", "score": round(s_ao, 4),
                     "decision": d_ao,
                     "time_minutes": round(float(abs(rng.normal(0.9, 0.1))), 3),
                     "reviewer_id": "",
                     "ai_decision": d_ao, "ai_confidence": round(conf, 4)})

        # ---------- AI-only unstructured (text-only) ----------
        s_au = float(sigmoid(1.28 * ability[i] + rng.normal(0, 0.97)))
        rows.append({**base, "pipeline": "AO_unstructured", "score": round(s_au, 4),
                     "decision": int(s_au >= 0.5),
                     "time_minutes": round(float(abs(rng.normal(0.7, 0.1))), 3),
                     "reviewer_id": "",
                     "ai_decision": int(s_au >= 0.5),
                     "ai_confidence": round(float(min(1.0, abs(s_au - 0.5) * 2.0)), 4)})

        # ---------- Random baseline ----------
        s_rnd = float(rng.random())
        rows.append({**base, "pipeline": "RAND", "score": round(s_rnd, 4),
                     "decision": int(rng.random() < 0.40),
                     "time_minutes": round(float(abs(rng.normal(0.1, 0.02))), 3),
                     "reviewer_id": ""})

        # ---------- Human-only: 2-4 reviewers ----------
        k = int(rng.integers(2, 5))
        chosen = rng.choice(a.reviewers, size=k, replace=False)
        for ridx in chosen:
            s_ho = sigmoid(rev_skill[ridx] * ability[i]
                           + rev_bias_region[ridx] * (0.3 if region[i] == "Non-metro" else 0)
                           + rev_bias_gender[ridx] * (0.25 if gender[i] == "F" else 0)
                           + rng.normal(0, rev_noise[ridx]))
            # realistic review time: ~12-25 min, log-normal-ish
            t = float(np.clip(rng.lognormal(mean=2.9, sigma=0.25), 8, 40))
            rows.append({**base, "pipeline": "HO", "score": round(float(s_ho), 4),
                         "decision": int(s_ho >= 0.5),
                         "time_minutes": round(t, 3),
                         "reviewer_id": reviewer_ids[ridx]})

        # ---------- Human-AI collaborative ----------
        ridx = int(rng.integers(0, a.reviewers))
        human_view = sigmoid(1.45 * ability[i] + rng.normal(0, 0.60)
                            + rev_bias_region[ridx] * (0.2 if region[i] == "Non-metro" else 0))
        w_human = 0.45 if conf >= 0.40 else 0.62
        s_hac = (1 - w_human) * s_ao + w_human * human_view
        d_hac = int(s_hac >= 0.5)
        # override behaviour: humans override more on low confidence
        override_prob = 0.46 if conf < 0.40 else (0.12 if conf < 0.70 else 0.03)
        overridden = bool(rng.random() < override_prob)
        if overridden:
            d_hac = 1 - d_ao
        # time: quick when high confidence, longer when reviewing low-conf AI
        if conf >= 0.40:
            t = float(np.clip(rng.normal(1.2, 0.35), 0.4, 3.0))
        else:
            t = float(np.clip(rng.normal(6.5, 1.4), 3.5, 12.0))
        rows.append({**base, "pipeline": "HAC", "score": round(float(s_hac), 4),
                     "decision": d_hac,
                     "time_minutes": round(t, 3),
                     "reviewer_id": reviewer_ids[ridx],
                     "ai_decision": d_ao, "ai_confidence": round(conf, 4)})

    df = pd.DataFrame(rows)
    # stable column order
    cols = ["candidate_id", "pipeline", "decision", "quality_label", "quality_score",
            "gender", "region", "role_family", "age", "education", "years_experience",
            "score", "time_minutes", "reviewer_id", "ai_decision", "ai_confidence"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows ({n} candidates, {a.reviewers} reviewers) to {a.out}")


if __name__ == "__main__":
    main()
