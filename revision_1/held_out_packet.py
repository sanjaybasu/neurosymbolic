"""Held-out (temporal) re-adjudication packet for the improved tool (v2).

Retrospective temporal split of the 1519 pharmacy reviews by dateOfEncounter:
  development period  = encounters before 2025-07-01 (failure modes that informed the 5 suppression
                        rules; 39 v2 flags).
  held-out period     = encounters on/after 2025-07-01 (28 v2 flags). The 5 rules are FROZEN (see
                        models/refinement.py); scoring the later period with the existing labels is a
                        within-sample temporal-stability check, and this packet provides the genuinely
                        out-of-sample blinded re-adjudication of the later-period flags.

The packet presents the held-out-period v2-flagged notes BLINDED (no prior category labels), reshuffled
with a fixed seed, for fresh independent dual adjudication into the same four categories used for the
original 107. Reported outputs (after labels return): held-out v2 PPV with Wilson CI, fresh Cohen kappa,
and concordance with the original-round labels.

PHI (note excerpts) -> stays local; never committed to the code repository.

Outputs (adjudication_packet/):
  heldout_v2_packet.csv     blinded packet (HO id + flagged rule(s) + extracted entities + excerpt + blank columns)
  heldout_v2_keymap.csv     HO id -> note row index, dateOfEncounter, original case_id, v2 violations (local)
  heldout_v2_design.json    split definition, seed, counts
"""
import os
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260617
CUTOFF = "2025-07-01"
WL = Path(os.environ.get("NEUROSYMBOLIC_ROOT","."))
REPO = WL / "packaging" / "neurosymbolic_github"
PKT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/adjudication_packet"
sys.path.append(str(REPO / "models"))
from extraction import ClinicalExtractor
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
import refinement as R


def main():
    notes = pd.read_csv(Path(os.environ.get("NEUROSYMBOLIC_NOTES", "data/notes.csv")), low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
    pr = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].reset_index(drop=True)
    dates = pd.to_datetime(pr["dateOfEncounter"], errors="coerce")

    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"),
                           rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))

    # original-round case_id is assigned in v1-positive order over the full corpus
    held = []
    v1_pos_counter = 0
    for i, tx in enumerate(pr["text"].astype(str)):
        e = ext.extract(tx)
        ctx = ClinicalContext("x", e.conditions, e.medications, [], [], {}, [], {})
        safe, viols = sym.check_contraindications(ctx, "medication_review")
        if safe:
            continue
        v1_pos_counter += 1
        orig_case = f"CN{v1_pos_counter:03d}"
        flagged, surviving, _ = R.refine(viols, tx)
        d = dates.iloc[i]
        if flagged and pd.notna(d) and d >= pd.Timestamp(CUTOFF):
            held.append({"note_idx": i, "date": d.date().isoformat(), "orig_case": orig_case,
                         "surviving": surviving,
                         "meds": ", ".join(sorted(set(e.medications))),
                         "conds": ", ".join(sorted(set(e.conditions))),
                         "excerpt": tx[:1700].replace("\r", " ").replace("\n", " ")})

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(held))
    rows, keymap = [], []
    for newpos, k in enumerate(order, 1):
        h = held[k]
        hid = f"HO{newpos:03d}"
        rule_fired = " ; ".join(h["surviving"])
        rows.append({"case_id": hid, "rule_fired": rule_fired,
                     "medications_extracted": h["meds"], "conditions_extracted": h["conds"],
                     "note_excerpt": h["excerpt"],
                     "category_A_B_C_D": "", "severity_major_moderate": "",
                     "dose_route_freq_would_change_call_Y_N": "", "confidence_1to5": "", "rationale": ""})
        keymap.append({"case_id": hid, "note_row_index": h["note_idx"], "dateOfEncounter": h["date"],
                       "original_case_id": h["orig_case"], "v2_violations": rule_fired})
    pd.DataFrame(rows).to_csv(PKT / "heldout_v2_packet.csv", index=False)
    pd.DataFrame(keymap).to_csv(PKT / "heldout_v2_keymap.csv", index=False)

    design = {"seed": SEED, "temporal_cutoff": CUTOFF,
              "held_out_definition": "dateOfEncounter on/after 2025-07-01",
              "n_heldout_v2_flags": len(held),
              "categories": {"A": "true actionable contraindication",
                             "B": "real but clinically appropriate / monitored combination",
                             "C": "extraction false positive (entity not truly present/active)",
                             "D": "rule false positive (entities present but combination not an actionable contraindication)"},
              "blinding": "prior-round category labels withheld; cases reshuffled with the fixed seed",
              "frozen_rules": "models/refinement.py S1-S5 (unchanged since development period)",
              "reported_after_labels": ["held-out v2 PPV = A/(A+B+C+D) with Wilson 95% CI",
                                        "fresh Cohen kappa (A vs not; 4-category)",
                                        "concordance with original-round labels"]}
    (PKT / "heldout_v2_design.json").write_text(json.dumps(design, indent=2))
    print(json.dumps(design, indent=2))
    print(f"\nWROTE held-out packet ({len(held)} v2 flags) + keymap + design to {PKT}")


if __name__ == "__main__":
    main()
