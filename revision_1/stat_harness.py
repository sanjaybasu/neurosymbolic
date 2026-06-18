"""Statistical harness for the v1 -> v2 (improved) precision change on the adjudicated flag set.

All inputs are the EXISTING dual-clinician adjudication labels (no new adjudication): the improved
tool flags a strict subset of v1, so its precision on this corpus is computable from labels already in
hand. This quantifies the IN-SAMPLE precision gain with significance and confidence intervals, and the
per-rule attribution. The unbiased (out-of-sample) confirmation comes from the random-sample gold
standard (separate packet + estimator).

Tests:
  1. Wilson 95% CIs for v1 PPV and v2 PPV.
  2. Exact sign test on discordant removals (each removed flag is a correct or incorrect removal).
  3. Hypergeometric enrichment: probability the suppression removed >= the observed number of false
     positives if it had removed the same number of flags at random.
  4. Seeded bootstrap 95% CI for v1 PPV, v2 PPV, and their difference (paired over the 107 cases).
  5. Per-suppression-rule x adjudication-category attribution (107) + corpus-wide counts (1519).

Output: audit/derived/stat_harness.json , audit/derived/suppression_rule_attribution.csv
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

SEED = 20260617
N_BOOT = 10000
R1 = Path(os.environ.get("NEUROSYMBOLIC_REVISION1","."))
OUT = R1 / "audit/derived"


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
    return ((c - h)/d, (c + h)/d)


def main():
    adj = pd.read_csv(OUT / "improved_suppression_audit.csv")  # 107 rows: case_id, category, kept, suppressed_rules
    note = pd.read_csv(OUT / "improved_pharmacy_per_note.csv")  # 1519 rows: v1, improved, suppressed_rules
    assert len(adj) == 107

    is_true = (adj["category"].str.upper() == "A").values  # category A = true contraindication
    kept = (adj["kept"] == 1).values
    n = len(adj)
    n_true = int(is_true.sum())
    n_false = n - n_true

    v1_tp, v1_n = n_true, n
    v2_tp, v2_n = int((kept & is_true).sum()), int(kept.sum())
    removed = ~kept
    n_removed = int(removed.sum())
    removed_false = int((removed & ~is_true).sum())
    removed_true = int((removed & is_true).sum())

    v1_ppv, v2_ppv = v1_tp / v1_n, v2_tp / v2_n
    res = {
        "n_adjudicated": n, "n_true_A": n_true, "n_false_BCD": n_false,
        "v1": {"flagged": v1_n, "true": v1_tp, "PPV": round(v1_ppv, 4),
               "PPV_wilson95": [round(x, 4) for x in wilson(v1_tp, v1_n)]},
        "v2_improved": {"flagged": v2_n, "true": v2_tp, "PPV": round(v2_ppv, 4),
                        "PPV_wilson95": [round(x, 4) for x in wilson(v2_tp, v2_n)]},
        "ppv_gain_abs": round(v2_ppv - v1_ppv, 4),
        "removed_total": n_removed, "removed_false_positives": removed_false,
        "removed_true_positives": removed_true,
        "true_positives_retained": f"{v2_tp}/{n_true}",
    }

    # 2. exact sign test: each removal is "correct" (removed a false positive) or "incorrect"
    # one-sided P that all/most removals are correct under a fair coin (H0: removal direction random)
    from scipy.stats import binomtest
    sign = binomtest(removed_false, n_removed, 0.5, alternative="greater")
    res["sign_test_removals"] = {
        "correct_removals": removed_false, "incorrect_removals": removed_true,
        "p_one_sided": sign.pvalue}

    # 3. hypergeometric enrichment: among 107 with n_false false, draw n_removed; P(X >= removed_false)
    # sf(k-1) = P(X >= k)
    res["hypergeometric_enrichment"] = {
        "population": n, "false_in_population": n_false, "draws": n_removed,
        "observed_false_removed": removed_false,
        "p_at_least_observed": float(hypergeom.sf(removed_false - 1, n, n_false, n_removed))}

    # 4. seeded bootstrap CI for PPVs and difference (paired resample of the 107)
    rng = np.random.default_rng(SEED)
    d_v1, d_v2, d_diff = [], [], []
    idx = np.arange(n)
    for _ in range(N_BOOT):
        b = rng.choice(idx, size=n, replace=True)
        t = is_true[b]
        k = kept[b]
        p1 = t.mean()  # all are v1-flagged -> PPV = fraction true
        kn = k.sum()
        if kn == 0:
            continue
        p2 = (t & k).sum() / kn
        d_v1.append(p1); d_v2.append(p2); d_diff.append(p2 - p1)
    res["bootstrap"] = {
        "seed": SEED, "iterations": N_BOOT,
        # 6-decimal precision avoids double-rounding when re-expressed as a 1-decimal percentage
        "v1_PPV_95": [round(np.percentile(d_v1, 2.5), 6), round(np.percentile(d_v1, 97.5), 6)],
        "v2_PPV_95": [round(np.percentile(d_v2, 2.5), 6), round(np.percentile(d_v2, 97.5), 6)],
        "ppv_gain_95": [round(np.percentile(d_diff, 2.5), 6), round(np.percentile(d_diff, 97.5), 6)],
        "ppv_gain_median": round(float(np.median(d_diff)), 6)}

    # 5a. per-rule attribution on the 107 (category x suppression rule, removed only)
    rows = []
    for _, r in adj[adj["kept"] == 0].iterrows():
        for rule in str(r["suppressed_rules"]).split(";"):
            if rule and rule != "nan":
                rows.append({"rule": rule, "category": r["category"].upper()})
    attrib = pd.DataFrame(rows)
    if len(attrib):
        pivot = attrib.pivot_table(index="rule", columns="category", aggfunc="size", fill_value=0)
        pivot["total_removed"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total_removed", ascending=False)
        pivot.to_csv(OUT / "suppression_rule_attribution.csv")
        res["per_rule_attribution_107"] = json.loads(pivot.reset_index().to_json(orient="records"))

    # 5b. corpus-wide suppression counts (1519): how many v1 flags each rule removed
    corp = {}
    for s in note.loc[(note["v1"] == 1) & (note["improved"] == 0), "suppressed_rules"]:
        for rule in str(s).split(";"):
            if rule and rule != "nan":
                corp[rule] = corp.get(rule, 0) + 1
    res["corpus_suppression_counts_1519"] = {"v1_flagged": int(note["v1"].sum()),
                                             "v2_flagged": int(note["improved"].sum()),
                                             "removed": int(((note["v1"] == 1) & (note["improved"] == 0)).sum()),
                                             "by_rule": dict(sorted(corp.items(), key=lambda x: -x[1]))}

    (OUT / "stat_harness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
