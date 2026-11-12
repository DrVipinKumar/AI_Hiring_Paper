"""Self-test: checks the statistics in analyze_hiring.py against independent implementations.
Run:  python test_metrics.py"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
import analyze_hiring as A

rng = np.random.default_rng(0)
n = 300
y = rng.integers(0, 2, n)
s1 = rng.normal(size=n) + 0.9 * y
s2 = rng.normal(size=n) + 0.5 * y

# 1. DeLong AUC equals sklearn AUC
aucs, cov = A.delong_auc_cov(np.vstack([s1, s2]), y)
assert abs(aucs[0] - roc_auc_score(y, s1)) < 1e-10 and abs(aucs[1] - roc_auc_score(y, s2)) < 1e-10
print("OK  DeLong AUC == sklearn roc_auc_score")

# 2. DeLong variance ~ bootstrap variance of AUC
boots = [roc_auc_score(y[i], s1[i]) for i in (rng.integers(0, n, n) for _ in range(1000))]
ratio = np.sqrt(cov[0, 0]) / np.std(boots)
assert 0.85 < ratio < 1.15, ratio
print(f"OK  DeLong SE within 15% of bootstrap SE (ratio={ratio:.2f})")

# 3. F1 matches sklearn
d = (s1 > 0.4).astype(int)
assert abs(A.f1_prec_rec(y, d)[0] - f1_score(y, d)) < 1e-12
print("OK  F1 == sklearn f1_score")

# 4. Holm matches the textbook example
adj = A.holm([0.01, 0.04, 0.03])
assert np.allclose(adj, [0.03, 0.06, 0.06]), adj
print("OK  Holm correction")

# 5. Williams test: identical correlations -> t = 0
t, p, q = A.williams_test(0.5, 0.5, 0.6, 200)
assert abs(t) < 1e-12 and p > 0.99
print("OK  Williams test null case")

# 6. DPD / EOD hand-computed
dd = np.array([1, 1, 0, 0, 1, 0, 0, 0]); yy = np.array([1, 0, 1, 0, 1, 1, 0, 1])
g = np.array(list("AAAABBBB"))
assert abs(A.dpd(dd, yy, g) - abs(0.5 - 0.25)) < 1e-12
assert abs(A.eod(dd, yy, g) - abs(0.5 - 1/3)) < 1e-12
print("OK  DPD / EOD hand-computed example")
print("\nAll self-tests passed.")
