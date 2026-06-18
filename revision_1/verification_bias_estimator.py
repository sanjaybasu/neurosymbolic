"""Two-phase, verification-bias-corrected estimator of recall / specificity / prevalence / PPV.

Phase-1 screen: the tool classifies all N pharmacy notes flag+/flag-.
Phase-2 verify: ALL flag+ verified (round 1); a seeded random sample of v1-unflagged notes verified
(Task A1); and a CENSUS of the v2-suppressed notes verified for a separate missed contraindication
(Task A2). Horvitz-Thompson inverse-probability weights recover the stratum totals, correcting the
verification bias that inflates apparent recall when only flagged notes are checked.

All adjudication counts are DERIVED from the label CSVs (none hardcoded):
  TP, FP_v1, FP_v2, n_suppressed  <- consensus_completed.csv + improved_suppression_audit.csv
The v2 unflagged universe = the v1-unflagged frame (sampled) PLUS the v2-suppressed notes (census). The
round-1 adjudication judged only each suppressed note's fired detection non-actionable and did not scan
for a separate undetected contraindication, so the suppressed notes are candidate v2 false negatives that
Task A2 adjudicates rather than assumes negative. v2 recall equals v1 recall unless a suppressed note
contains a separate missed actionable contraindication.

Confidence intervals are Wilson intervals on the missed-prevalence in each phase-2 stratum, propagated to
recall/specificity (analytic; no degenerate bootstrap). When 0 missed are found, recall is reported as a
one-sided lower bound from the Wilson upper bound on missed-prevalence, never a degenerate point interval.

Modes:
  --demo   : synthetic seeded missed-counts to self-test the arithmetic (no pharmacist labels needed).
  (default): read the filled packets; if unfilled, prints AWAITING-LABELS with the rule-of-three bound.

Output: audit/derived/verification_bias_estimate.json
"""
import os
import json, sys
from pathlib import Path
import pandas as pd

R1 = Path(os.environ.get("NEUROSYMBOLIC_REVISION1","."))
PKT = R1 / "adjudication_packet"
OUT = R1 / "audit/derived"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)
    return (max(0.0, (c-h)/d), min(1.0, (c+h)/d))


def derive_counts():
    """Derive TP, FP_v1, FP_v2, n_suppressed from the adjudication label CSVs (nothing hardcoded)."""
    cons = pd.read_csv(PKT / "consensus_completed.csv")
    cat = cons["category"].str.upper().str.strip()
    TP = int((cat == "A").sum())
    FP_v1 = int(cat.isin(["B", "C", "D"]).sum())
    aud = pd.read_csv(OUT / "improved_suppression_audit.csv")
    kept = aud["kept"] == 1
    acat = aud["category"].str.upper().str.strip()
    FP_v2 = int((kept & acat.isin(["B", "C", "D"])).sum())
    n_suppressed = int((~kept).sum())
    # internal consistency: every retained A is a true positive; no A suppressed
    assert int((kept & (acat == "A")).sum()) == TP, "retained A != TP"
    assert int((~kept & (acat == "A")).sum()) == 0, "an actionable contraindication was suppressed"
    return TP, FP_v1, FP_v2, n_suppressed


def estimate(m_unflagged, n_s, N_minus, m_suppressed, n_suppressed, counts, N):
    """Analytic two-phase estimates with Wilson-propagated CIs (handles the 0-missed case)."""
    TP, FP_v1, FP_v2, _ = counts
    w = N_minus / n_s
    pu, (pu_lo, pu_hi) = m_unflagged / n_s, wilson(m_unflagged, n_s)
    ps, (ps_lo, ps_hi) = (m_suppressed / n_suppressed if n_suppressed else 0.0), wilson(m_suppressed, n_suppressed)

    FN_v1, FN_v1_lo, FN_v1_hi = pu*N_minus, pu_lo*N_minus, pu_hi*N_minus
    FN_v2 = pu*N_minus + ps*n_suppressed
    FN_v2_lo = pu_lo*N_minus + ps_lo*n_suppressed
    FN_v2_hi = pu_hi*N_minus + ps_hi*n_suppressed
    TN_un, TN_un_lo, TN_un_hi = (1-pu)*N_minus, (1-pu_hi)*N_minus, (1-pu_lo)*N_minus

    def rec(fn):
        return TP/(TP+fn) if (TP+fn) > 0 else float("nan")

    def rec_block(fn, fn_lo, fn_hi, m):
        # recall is decreasing in FN: lower bound uses FN_hi, upper bound uses FN_lo
        lo, hi = rec(fn_hi), rec(fn_lo)
        out = {"point": round(rec(fn), 4), "ci95": [round(lo, 4), round(hi, 4)],
               "est_false_negatives": round(fn, 1)}
        if m == 0:
            out["reported_as"] = "one-sided lower bound (0 missed observed); recall >= lower"
            out["one_sided_lower_bound"] = round(lo, 4)
        return out

    spec_v1 = TN_un/(TN_un+FP_v1) if (TN_un+FP_v1) else float("nan")
    spec_v1_ci = [round(TN_un_lo/(TN_un_lo+FP_v1), 4), round(TN_un_hi/(TN_un_hi+FP_v1), 4)]
    TN_v2, TN_v2_lo, TN_v2_hi = TN_un+n_suppressed, TN_un_lo+n_suppressed, TN_un_hi+n_suppressed
    # v2 TN reduced by any suppressed-note FN; conservative point uses census FN
    TN_v2 -= ps*n_suppressed; TN_v2_lo -= ps_hi*n_suppressed; TN_v2_hi -= ps_lo*n_suppressed
    spec_v2 = TN_v2/(TN_v2+FP_v2)
    spec_v2_ci = [round(TN_v2_lo/(TN_v2_lo+FP_v2), 4), round(TN_v2_hi/(TN_v2_hi+FP_v2), 4)]

    return {
        "derived_counts": {"TP": TP, "FP_v1": FP_v1, "FP_v2": FP_v2, "n_suppressed_census": n_suppressed},
        "inputs": {"missed_unflagged_sample": m_unflagged, "n_unflagged_sample": n_s, "N_unflagged": N_minus,
                   "ht_weight": round(w, 4), "missed_suppressed_census": m_suppressed,
                   "arithmetic_check_unflagged_FN_plus_TN": round(FN_v1 + TN_un, 1)},
        "missed_prevalence_unflagged": {"point": round(pu, 4), "wilson95": [round(pu_lo, 4), round(pu_hi, 4)]},
        "recall_v1": rec_block(FN_v1, FN_v1_lo, FN_v1_hi, m_unflagged),
        "recall_v2": rec_block(FN_v2, FN_v2_lo, FN_v2_hi, m_unflagged + m_suppressed),
        "specificity_v1": {"point": round(spec_v1, 4), "ci95": spec_v1_ci},
        "specificity_v2": {"point": round(spec_v2, 4), "ci95": spec_v2_ci},
        "prevalence_actionable": round((TP+FN_v2)/N, 4),
        "ppv_v1": round(TP/(TP+FP_v1), 4), "ppv_v2": round(TP/(TP+FP_v2), 4),
    }


def main():
    design = json.load(open(PKT / "gold_standard_design.json"))
    n_s = design["strata"]["unflagged_sample"]["n"]
    N_minus = design["n_v1_unflagged"]
    n_suppressed = design["strata"]["suppressed_census"]["n"]
    N = design["n_total_pharmacy_notes"]
    counts = derive_counts()
    demo = "--demo" in sys.argv

    def col_count(path, val="Y"):
        df = pd.read_csv(PKT / path)
        c = df["contains_true_contraindication_Y_N"].astype(str).str.strip().str.upper()
        return int((c == val).sum()), int(c.isin(["Y", "N"]).sum()), len(df)

    if demo:
        out = {"mode": "DEMO (synthetic missed counts; NOT real labels)",
               **estimate(2, n_s, N_minus, 0, n_suppressed, counts, N)}
    else:
        mu, fu, tu = col_count("gold_standard_packet.csv")
        ms, fs, ts = col_count("gold_standard_suppressed_packet.csv")
        if fu < tu or fs < ts:
            out = {"mode": "AWAITING-LABELS", "unflagged_filled": fu, "unflagged_expected": tu,
                   "suppressed_filled": fs, "suppressed_expected": ts,
                   "rule_of_three_if_zero_missed": estimate(0, n_s, N_minus, 0, n_suppressed, counts, N)}
        else:
            out = {"mode": "REAL", **estimate(mu, n_s, N_minus, ms, n_suppressed, counts, N)}

    # guard: no degenerate finite-sample CI may ship
    def degenerate(block):
        ci = block.get("ci95")
        return ci and ci[0] == ci[1]
    for key in ("recall_v1", "recall_v2", "specificity_v1", "specificity_v2"):
        blk = out.get(key) or (out.get("rule_of_three_if_zero_missed", {}) or {}).get(key)
        if blk and degenerate(blk) and "one_sided_lower_bound" not in blk:
            raise AssertionError(f"degenerate CI on {key}: {blk}")

    (OUT / "verification_bias_estimate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
