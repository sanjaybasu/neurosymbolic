"""Contraindication refinement layer (v2): suppression-only refinement of the symbolic reasoner.

The refinement layer runs AFTER SymbolicReasoner.check_contraindications and suppresses individual
fired violations that meet a clinically-principled exclusion criterion. The refined flag set is a
strict subset of the unrefined set, so the refinement can only raise precision (remove clinical false
alarms); it cannot suppress a fired detection that the adjudicators judged actionable, so it does not
lower recall for the contraindications the layer detects. Whole-note recall can fall marginally when a
suppressed note happens to contain a separate, previously undetected contraindication (measured in the
manuscript: 44.1% -> 42.7%). The heuristics are deployment-imperfect: S2's proximity test and S3's
note-level negation/resolution can, on unseen notes, suppress a genuinely co-prescribed opioid in a
substance-use-disorder context or a contraindication in an actively pregnant patient; clinician review
is therefore required before any detection is acted upon.

The five suppression rules encode established clinical knowledge, not patterns fit to any dataset:
  S1  Low-dose/antiplatelet aspirin (read from aspirin's own token; <325 mg, no non-aspirin analgesic
      NSAID present) is an antiplatelet, not an analgesic NSAID; excluded from NSAID-disease rules.
  S2  Buprenorphine/methadone treating opioid use disorder is medication-assisted treatment, and an
      opioid mentioned only in an illicit/abuse/withdrawal/historical context is the substance of the
      disorder itself; both are excluded from the opioid + substance-use-disorder rule. The rule is
      retained for opioid + benzodiazepine (respiratory-depression risk is real regardless of MAT).
  S3  A pregnancy-conditioned contraindication is retained only with positive evidence of an ACTIVE
      pregnancy (the word "pregnant", weeks of gestation/EGA, trimester, prenatal vitamins, EDD/due
      date, IUP/gestation); suppressed when pregnancy was extracted from a specialty referral
      (OB/GYN/obstetric) alone, is negated/test-negative, or is resolved (miscarriage/abortion/termination).
  S4  Albuterol + cardiovascular disease is a guideline caution (monitoring-level), not an actionable
      contraindication.
  S5  Short-course corticosteroid (burst/taper/dose-pack/<=N-day) in diabetes is guideline-directed and
      monitoring-level, not an actionable contraindication.

This module contains no patient data and loads only the medication vocabulary shipped with the repo.
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

_DATA = Path(__file__).resolve().parent.parent / "data"
_MED = {k.lower(): v for k, v in json.load(open(_DATA / "med_synonyms.json")).items()}

_ASPIRIN_TERMS = {"aspirin", "acetylsalicylic", "asa", "ecotrin", "bayer"}
_MAT_DRUGS = {"buprenorphine", "suboxone", "subutex", "sublocade", "methadone"}
_NSAID_NONASPIRIN = {t for t, c in _MED.items() if c == "nsaids" and t not in _ASPIRIN_TERMS and " " not in t}
_OPIOID_DRUGS = {t for t, c in _MED.items()
                 if c == "opioid" and t not in _MAT_DRUGS and t != "opioid" and " " not in t}

_NON_RX_OPIOID = ["illicit", "heroin", "iv drug", "ivdu", "street", "injection drug", "snort",
                  "abuse", "withdrawal", "overdose", "misuse", "recreational", "use disorder",
                  "dependence", "addiction", "addict"]
_PREG_NEG = ["not pregnant", "pregnancy test negative", "negative pregnancy", "denies pregnancy",
             "no pregnancy", "not currently pregnant", "neg upt", "upt negative", "hcg negative",
             "negative for pregnancy", "no longer pregnant"]
_PREG_RESOLVED = ["miscarriage", "miscarried", "abortion", "termination of pregnancy",
                  "terminated the pregnancy", "products of conception", "pregnancy loss",
                  "fetal demise", "spontaneous abortion", "elective termination",
                  "pregnancy was terminated", "pregnancy likely terminated"]
_SHORT_STEROID = ["pak", "dose pack", "dosepak", "taper", "burst", "5-day", "6-day", "short course",
                  "5 day", "6 day", "medrol", "10-day", "10 day"]
_ASP_DOSE_RE = re.compile(r"\b(aspirin|asa|acetylsalicylic|ecotrin)\b\D{0,15}(\d{2,4})\s*mg")
_ACTIVE_PREG_RE = re.compile(
    r"\bpregnant\b|\btrimester\b|\bprenatal\b|\bgravid|\bintrauterine pregnancy\b|\biup\b|"
    r"\bedd\b|\bdue date\b|estimated date of delivery|\d+\s*weeks?\s*(pregnant|gestation|ega|ga\b)|"
    r"\bgestational age\b|\bfetal\b|\bfetus\b")
_VIOL_RE = re.compile(r"Contraindication:\s*([a-z_]+)\s+with\s+([a-z_]+)")


def _win(text: str, term: str, r: int = 45) -> List[str]:
    t = text.lower()
    return [t[max(0, m.start()-r): m.end()+r] for m in re.finditer(r"\b" + re.escape(term) + r"\b", t)]


def _aspirin_lowdose_only(text: str) -> bool:
    t = text.lower()
    if any(re.search(r"\b" + re.escape(n) + r"\b", t) for n in _NSAID_NONASPIRIN):
        return False
    if not re.search(r"\b(aspirin|asa|acetylsalicylic|ecotrin)\b", t):
        return False
    for m in _ASP_DOSE_RE.finditer(t):
        if int(m.group(2)) >= 325:
            return False
    return True


def _prescribed_rx_opioid(text: str) -> bool:
    for term in _OPIOID_DRUGS:
        for w in _win(text, term):
            if any(x in w for x in _NON_RX_OPIOID):
                continue
            return True
    return False


def _pregnancy_active(text: str) -> bool:
    t = text.lower()
    if any(r in t for r in _PREG_RESOLVED) or any(n in t for n in _PREG_NEG):
        return False
    return bool(_ACTIVE_PREG_RE.search(t))


def _short_course_steroid(text: str) -> bool:
    return any(s in text.lower() for s in _SHORT_STEROID)


def suppression_tag(violation: str, note_text: str):
    """Return the suppression-rule tag if this violation should be suppressed, else None."""
    m = _VIOL_RE.search(violation)
    if not m:
        return None  # medication-medication interactions are never suppressed
    med, cond = m.group(1), m.group(2)
    if med == "nsaids" and _aspirin_lowdose_only(note_text):
        return "S1_lowdose_aspirin"
    if med == "opioid" and cond == "substance_use_disorder" and not _prescribed_rx_opioid(note_text):
        return "S2_mat_or_illicit_opioid"
    if cond == "pregnancy" and not _pregnancy_active(note_text):
        return "S3_pregnancy_not_active"
    if med == "albuterol" and cond == "cardiovascular_disease":
        return "S4_albuterol_cvd_caution"
    if med == "corticosteroid" and cond == "diabetes" and _short_course_steroid(note_text):
        return "S5_short_course_steroid"
    return None


def refine(violations: List[str], note_text: str) -> Tuple[bool, List[str], List[str]]:
    """Apply S1-S5 to a reasoner's violation list for one note.

    Returns (is_flagged, surviving_violations, suppression_tags).
    """
    surviving, tags = [], []
    for v in violations:
        tag = suppression_tag(v, note_text)
        if tag is None:
            surviving.append(v)
        else:
            tags.append(tag)
    return (len(surviving) > 0), surviving, sorted(set(tags))
