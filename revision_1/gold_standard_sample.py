"""Random-sample gold standard for verification-bias-corrected recall / specificity / prevalence.

The original adjudication verified only TOOL-FLAGGED notes (all 107 v1 flags), which yields an unbiased
PPV but cannot estimate recall: true contraindications the tool MISSED are never examined. This is the
classic two-phase verification-bias problem. To correct it, we draw a seeded simple random sample of
the v1-UNFLAGGED notes and have pharmacists adjudicate whether each contains a true contraindication the
tool missed. With the known sampling fraction, a Horvitz-Thompson two-phase estimator recovers recall,
specificity, and prevalence with valid confidence intervals (verification_bias_estimator.py).

Because the improved tool (v2) suppresses ONLY adjudicated false positives (0 true positives removed),
v2 recall equals v1 recall by construction; the same unflagged sample therefore validates both tools.

This packet contains PHI (note excerpts) and MUST remain local (never committed to the code repository).

Outputs (adjudication_packet/):
  gold_standard_packet.csv   pharmacist packet: GS id + note excerpt + blank adjudication columns
  gold_standard_keymap.csv   GS id -> note row index, v1_flag, n_meds, n_conditions (re-linking; local)
  gold_standard_design.json  sampling frame, seed, fraction, weights, power note
"""
import os
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260617
N_SAMPLE = 200
WL = Path(os.environ.get("NEUROSYMBOLIC_ROOT","."))
REPO = WL / "packaging" / "neurosymbolic_github"
PKT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/adjudication_packet"
sys.path.append(str(REPO / "models"))
from extraction import ClinicalExtractor
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext


def main():
    notes = pd.read_csv(Path(os.environ.get("NEUROSYMBOLIC_NOTES", "data/notes.csv")), low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
    pr = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].reset_index(drop=True)
    texts = pr["text"].astype(str).tolist()

    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"),
                           rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))
    sys.path.append(str(REPO / "models"))
    import refinement as R
    flag, v2flag, nmed, ncond = [], [], [], []
    for tx in texts:
        e = ext.extract(tx)
        ctx = ClinicalContext("x", e.conditions, e.medications, [], [], {}, [], {})
        safe, viols = sym.check_contraindications(ctx, "medication_review")
        flag.append(0 if safe else 1)
        v2flag.append(1 if (not safe and R.refine(viols, tx)[0]) else 0)
        nmed.append(len(set(e.medications))); ncond.append(len(set(e.conditions)))
    flag = np.array(flag); v2flag = np.array(v2flag)
    N = len(flag)
    unflagged_idx = np.where(flag == 0)[0]               # v1-unflagged (1412): sampled stratum
    suppressed_idx = np.where((flag == 1) & (v2flag == 0))[0]  # v1-flagged, v2-suppressed (40): census
    N_minus = len(unflagged_idx)

    rng = np.random.default_rng(SEED)
    n_s = min(N_SAMPLE, N_minus)
    sampled = np.sort(rng.choice(unflagged_idx, size=n_s, replace=False))
    weight = N_minus / n_s  # Horvitz-Thompson inverse sampling weight for the unflagged stratum

    def packet_rows(indices, prefix, stratum, w):
        rows, keymap = [], []
        for j, idx in enumerate(indices, 1):
            gid = f"{prefix}{j:03d}"
            rows.append({"case_id": gid, "note_excerpt": texts[idx][:1600].replace("\r", " ").replace("\n", " "),
                         "contains_true_contraindication_Y_N": "",
                         "which_contraindication_med_with_condition": "",
                         "severity_major_moderate": "", "confidence_1to5": "", "rationale": ""})
            keymap.append({"case_id": gid, "note_row_index": int(idx), "stratum": stratum,
                           "ht_weight": w, "n_meds": int(nmed[idx]), "n_conditions": int(ncond[idx])})
        return rows, keymap

    # Task A1: random sample of v1-unflagged notes (HT-weighted)
    r1, k1 = packet_rows(sampled, "GS", "unflagged_sample", round(weight, 4))
    pd.DataFrame(r1).to_csv(PKT / "gold_standard_packet.csv", index=False)
    # Task A2: census of the v2-suppressed notes (weight 1) -- checked for a DIFFERENT missed contraindication,
    # since round 1 adjudicated only the fired (now-suppressed) detection, not the whole note
    r2, k2 = packet_rows(suppressed_idx, "SP", "suppressed_census", 1.0)
    pd.DataFrame(r2).to_csv(PKT / "gold_standard_suppressed_packet.csv", index=False)
    pd.DataFrame(k1 + k2).to_csv(PKT / "gold_standard_keymap.csv", index=False)

    design = {
        "seed": SEED, "n_total_pharmacy_notes": int(N),
        "n_v1_flagged": int(flag.sum()), "n_v1_unflagged": int(N_minus),
        "n_v2_suppressed_from_v1": int(len(suppressed_idx)),
        "strata": {
            "unflagged_sample": {"file": "gold_standard_packet.csv", "n": int(n_s),
                                 "frame": int(N_minus), "ht_weight": round(weight, 4),
                                 "sampling": "simple random sample without replacement"},
            "suppressed_census": {"file": "gold_standard_suppressed_packet.csv", "n": int(len(suppressed_idx)),
                                  "ht_weight": 1.0, "sampling": "census (all v2-suppressed notes)"}},
        "verified_flagged_stratum": "all 107 v1-flagged adjudicated in round 1 (sampling fraction 1.0)",
        "recall_logic": "v1 recall uses TP + estimated FN from the unflagged sample. v2 unflagged universe "
                        "adds the suppressed census; the round-1 adjudication judged only each suppressed "
                        "note's fired detection non-actionable and did NOT scan for a separate undetected "
                        "contraindication, so the suppressed notes are candidate v2 false negatives that this "
                        "census adjudicates rather than assumes negative. v2 recall therefore equals v1 recall "
                        "unless a suppressed note contains a separate missed actionable contraindication.",
        "power_note_rule_of_three": {
            "if_zero_FN_in_unflagged_sample": f"95% upper bound on missed-contraindication prevalence among "
                                              f"unflagged notes ~= 3/{n_s} = {round(3/n_s, 4)}",
            "recall_reported_as": "one-sided lower bound when 0 missed found (never a degenerate point CI)"},
        "instructions": "Adjudicate each note for ANY true actionable medication contraindication the tool "
                        "did not flag (GS* = unflagged sample; SP* = a note whose single flagged detection "
                        "was judged non-actionable in round 1, now checked for any OTHER missed "
                        "contraindication). Mark Y/N, name the medication+condition, severity, confidence "
                        "(1-5), rationale. Use the same actionable-contraindication standard as round 1.",
    }
    (PKT / "gold_standard_design.json").write_text(json.dumps(design, indent=2))
    print(json.dumps(design, indent=2))
    print(f"\nWROTE unflagged sample ({n_s}) + suppressed census ({len(suppressed_idx)}) + keymap + design to {PKT}")


if __name__ == "__main__":
    main()
