"""Finalize Round-2 recall/specificity/held-out precision after the 7 discordant cases are reconciled.

Combines the two pharmacists' independent ratings (cases they already agreed on) with the reconciled
consensus for the 7 discordant cases, then computes the verification-bias-corrected estimates and the
held-out precision, and reports Cohen's kappa from the independent ratings.

Inputs (adjudication_packet/):
  gold_standard_rater_A.csv / _B.csv                       (Task A, 200 unflagged, Y/N)
  gold_standard_suppressed_rater_A.csv / _B.csv            (Task A2, 40 suppressed census, Y/N)
  heldout_rater_A.csv / _B.csv                             (Task B, 28 held-out, A/B/C/D)
  reconciliation_taskA_discordant_completed.csv           (CONSENSUS_contains_true_contraindication_Y_N)
  reconciliation_taskB_discordant_completed.csv           (CONSENSUS_category_A_B_C_D)

Output: audit/derived/verification_bias_estimate.json (REAL) + round2_final_summary.json + console table.
Run: /opt/anaconda3/bin/python3 analysis_scripts/score_round2_final.py
"""
import os
import json, sys, importlib.util
from pathlib import Path
import pandas as pd

R1 = Path(os.environ.get("NEUROSYMBOLIC_REVISION1","."))
PKT = R1 / "adjudication_packet"
OUT = R1 / "audit/derived"


def norm(s):
    return s.astype(str).str.strip().str.upper()


def kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c)/n) * (b.count(c)/n) for c in cats)
    return (po - pe)/(1 - pe) if pe < 1 else 1.0


def consensus_yn(fa, fb, recon, task):
    A = pd.read_csv(PKT / fa).set_index("case_id")
    B = pd.read_csv(PKT / fb).set_index("case_id")
    col = "contains_true_contraindication_Y_N"
    a, b = norm(A[col]), norm(B[col])
    k = kappa(a, b)
    cons = {}
    for cid in A.index:
        if a[cid] == b[cid]:
            cons[cid] = a[cid]
    if recon and (PKT / recon).exists():
        rc = pd.read_csv(PKT / recon)
        rc = rc[rc["task"] == task] if "task" in rc.columns else rc
        for _, r in rc.iterrows():
            cons[r["case_id"]] = str(r["CONSENSUS_contains_true_contraindication_Y_N"]).strip().upper()
    m = sum(1 for v in cons.values() if v == "Y")
    resolved = len(cons)
    return m, resolved, len(A), k


def main():
    spec = importlib.util.spec_from_file_location("vbe", str(R1 / "analysis_scripts/verification_bias_estimator.py"))
    vbe = importlib.util.module_from_spec(spec); spec.loader.exec_module(vbe)
    counts = vbe.derive_counts()
    design = json.load(open(PKT / "gold_standard_design.json"))
    n_s = design["strata"]["unflagged_sample"]["n"]
    N_minus = design["n_v1_unflagged"]
    n_suppressed = design["strata"]["suppressed_census"]["n"]
    N = design["n_total_pharmacy_notes"]

    reconA = "reconciliation_taskA_discordant_completed.csv"
    m_unf, res_unf, n_unf, kA = consensus_yn("gold_standard_rater_A.csv", "gold_standard_rater_B.csv", reconA, "A_unflagged")
    m_sup, res_sup, n_sup, kA2 = consensus_yn("gold_standard_suppressed_rater_A.csv", "gold_standard_suppressed_rater_B.csv", reconA, "A2_suppressed")

    # Task B held-out PPV from consensus
    HA = pd.read_csv(PKT / "heldout_rater_A.csv").set_index("case_id")
    HB = pd.read_csv(PKT / "heldout_rater_B.csv").set_index("case_id")
    ha, hb = norm(HA["category_A_B_C_D"]), norm(HB["category_A_B_C_D"])
    kB = kappa(ha, hb); kB_act = kappa(ha == "A", hb == "A")
    hcons = {cid: ha[cid] for cid in HA.index if ha[cid] == hb[cid]}
    reconB = PKT / "reconciliation_taskB_discordant_completed.csv"
    if reconB.exists():
        rcb = pd.read_csv(reconB)
        for _, r in rcb.iterrows():
            hcons[r["case_id"]] = str(r["CONSENSUS_category_A_B_C_D"]).strip().upper()
    nB = len(HA); aB = sum(1 for v in hcons.values() if v == "A")
    heldout_ppv = aB / nB if nB else None

    pending = (res_unf < n_unf) or (res_sup < n_sup) or (len(hcons) < nB)
    est = vbe.estimate(m_unf, n_s, N_minus, m_sup, n_suppressed, counts, N)

    summary = {
        "status": "AWAITING-RECONCILIATION" if pending else "FINAL",
        "kappa": {"taskA_unflagged": round(kA, 3), "taskA2_suppressed": round(kA2, 3),
                  "heldout_4cat": round(kB, 3), "heldout_actionable": round(kB_act, 3)},
        "consensus_counts": {"missed_unflagged_m": m_unf, "unflagged_resolved": f"{res_unf}/{n_unf}",
                             "missed_suppressed_m": m_sup, "suppressed_resolved": f"{res_sup}/{n_sup}",
                             "heldout_actionable": f"{aB}/{nB}", "heldout_resolved": f"{len(hcons)}/{nB}"},
        "estimates": {"recall_v1": est["recall_v1"], "recall_v2": est["recall_v2"],
                      "specificity_v1": est["specificity_v1"], "specificity_v2": est["specificity_v2"],
                      "prevalence_actionable": est["prevalence_actionable"],
                      "ppv_v1": est["ppv_v1"], "ppv_v2": est["ppv_v2"]},
        "heldout_ppv_consensus": round(heldout_ppv, 4) if heldout_ppv is not None else None,
    }
    (OUT / "round2_final_summary.json").write_text(json.dumps(summary, indent=2))
    if not pending:
        est["mode"] = "REAL (consensus)"
        (OUT / "verification_bias_estimate.json").write_text(json.dumps(est, indent=2))
    print(json.dumps(summary, indent=2))
    if pending:
        print("\nReconciliation files not yet present/complete; numbers above use rater AGREEMENT only "
              "(discordant cases unresolved). Provide reconciliation_task*_discordant_completed.csv to finalize.")


if __name__ == "__main__":
    main()
