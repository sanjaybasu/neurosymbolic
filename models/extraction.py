"""
Clinical text extraction with scoped negation, simple temporality cues, section
awareness, medication class normalization, and multi-word phrase matching.
Designed to improve precision of symbolic checks by filtering negated or
historical mentions and aligning brand/generic names to rule vocabularies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json


SECTION_HEADERS = [
    "past medical history",
    "past surgical history",
    "family history",
    "social history",
    "problem list",
    "medication history",
    "prior medications",
    "allergies",
    "assessment",
    "plan",
]

NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies?\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bruled out\b",
    r"\bstopped\b",
    r"\bdiscontinued\b",
    r"\bno longer\b",
    r"\bnot taking\b",
    r"\bnever\b",
    r"\bno issues?\b",
    r"\bno problems?\b",
    r"\bnot currently\b",
    r"\bno concerns?\b",
]

TEMPORAL_CUES = [
    r"\bhistory of\b",
    r"\bpast medical\b",
    r"\bprevious\b",
    r"\bprior\b",
    r"\bused to\b",
    r"\bformer\b",
    r"\bformerly\b",
    r"\byears ago\b",
    r"\bmonths ago\b",
    r"\bremission\b",
    r"\bresolved\b",
    r"\bclean\b",
    r"\brecovered\b",
]

# Phrase-level cues for operational hazards (refills, prior auth, overuse)
PHRASE_MAP = {
    r"\brefill(s)? too soon\b": "med_access_issue",
    r"\bprior auth(orization)?\b": "med_access_issue",
    r"\bpa required\b": "med_access_issue",
    r"\bpa req'd\b": "med_access_issue",
    r"\bneeds? pa\b": "med_access_issue",
    r"\bran out of (my )?(meds|medication|insulin|inhaler|pills|rx)\b": "med_access_issue",
    r"\bout of (my )?(meds|medication|insulin|inhaler|pills|rx)\b": "med_access_issue",
    r"\blost (my )?(meds|medication|inhaler|pills|rx)\b": "med_access_issue",
    r"\bstolen (meds|medication|pills|rx)\b": "med_access_issue",
    r"\bcan'?t (get|fill|afford|refill) (my |his |her )?(meds|medication|rx|prescription|inhaler|insulin)\b": "med_access_issue",
    r"\brun out\b.*\b(meds|medication|rx|prescription|before)\b": "med_access_issue",
    r"\bmay run out\b": "med_access_issue",
    r"\bwon'?t (refill|fill|issue)\b": "med_access_issue",
    r"\bprescription(s)? expired\b": "med_access_issue",
    r"\bbridge fill\b": "med_access_issue",
    r"\btook (an )?extra (dose|puff|pill|tablet)\b": "med_overuse",
    r"\btook (the )?whole (bottle|pack)\b": "med_overuse",
    r"\bdouble(?:d)? (up|dos(?:e|ing))\b": "med_overuse",
    r"\boverdose\b": "med_overuse",
    r"\btook too (much|many)\b": "med_overuse",
    r"\bextra (dose|pill|tablet|puff)\b": "med_overuse",
    r"\b(severe|bad|serious)\s+withdrawal\b": "substance_use_disorder",
    r"\bstreet drugs?\b": "substance_use_disorder",
    r"\bsuicid\w*\b": "suicidal_ideation",
    r"\bkill (myself|himself|herself)\b": "suicidal_ideation",
    r"\bwant(s)? to die\b": "suicidal_ideation",
    r"\bend (my|his|her) life\b": "suicidal_ideation",
    r"\bself[- ]?harm\b": "suicidal_ideation",
    r"\bED visit\b": "ed_visit",
    r"\bemergency (room|department)\b": "ed_visit",
    r"\bwent to (the )?er\b": "ed_visit",
    r"\bhospitalized\b": "hospitalization",
    r"\badmitted\b": "hospitalization",
    r"\bdischarged?\b": "hospitalization",
    r"\bin the hospital\b": "hospitalization",
    r"\bchest (pain|tightness|pressure)\b": "chest_pain",
    r"\bshortness of breath\b": "asthma",
    r"\bbreathing (difficulty|problem|trouble)\b": "asthma",
    r"\ballergic reaction\b": "anaphylaxis_risk",
    r"\bblood pressure\b": "hypertension",
    r"\bhigh bp\b": "hypertension",
    r"\bhigh blood pressure\b": "hypertension",
    r"\bheart failure\b": "heart_failure",
    r"\bcongestive heart\b": "heart_failure",
    r"\bchronic kidney\b": "chronic_kidney_disease",
    r"\bkidney (disease|failure)\b": "chronic_kidney_disease",
    r"\brenal (failure|disease)\b": "chronic_kidney_disease",
    r"\bheart (disease|attack)\b": "cardiovascular_disease",
    r"\bcoronary artery\b": "cardiovascular_disease",
    r"\bmini[- ]?strokes?\b": "stroke",
    r"\bbrai+n bleed\b": "stroke",
    r"\btransient ischemic\b": "stroke",
    r"\bhad a fall\b": "fall_risk",
    r"\bpatient fell\b": "fall_risk",
    r"\bfell (down|at|in|on)\b": "fall_risk",
    r"\bblood sugar\b": "diabetes",
    r"\bsubstance (use|abuse)\b": "substance_use_disorder",
    r"\bfall risk\b": "fall_risk",
    r"\bliver disease\b": "liver_disease",
    r"\bbleeding disorder\b": "bleeding_disorder",
    r"\bchronic obstructive\b": "copd",
    r"\bbreast pump\b": "lactation",
    r"\bfeeling (down|blue|depressed)\b": "depression",
}


@dataclass
class ExtractionResult:
    conditions: List[str]
    medications: List[str]
    negated_mentions: List[str]
    historical_mentions: List[str]
    debug: Dict[str, List[str]]


class ClinicalExtractor:
    """Rule-based extractor with negation, temporality, and class normalization."""

    def __init__(self, vocab_dir: str) -> None:
        base = Path(vocab_dir)
        self.med_synonyms = self._load_synonyms(base / "med_synonyms.json")
        self.condition_synonyms = self._load_synonyms(base / "condition_synonyms.json")
        self.rxnorm_map = self._load_synonyms(base / "rxnorm_map.json")

        # Build multi-word lookup for conditions (sorted longest first for greedy matching)
        self._multiword_conditions = {
            k: v for k, v in self.condition_synonyms.items() if " " in k
        }
        self._multiword_meds = {
            k: v for k, v in self.med_synonyms.items() if " " in k
        }
        # Also from rxnorm_map
        self._multiword_meds.update(
            {k: v for k, v in self.rxnorm_map.items() if " " in k}
        )

    @staticmethod
    def _load_synonyms(path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        # map lowercased synonym to canonical class/id
        return {k.lower(): v for k, v in data.items()}

    def _normalize_med(self, token: str) -> str | None:
        token_l = token.lower()
        if token_l in self.rxnorm_map:
            return self.rxnorm_map[token_l]
        return self.med_synonyms.get(token_l)

    def _normalize_condition(self, token: str) -> str | None:
        return self.condition_synonyms.get(token.lower())

    def _is_negated(self, window: str) -> bool:
        return any(re.search(pat, window, flags=re.IGNORECASE) for pat in NEGATION_CUES)

    def _is_historical(self, window: str) -> bool:
        return any(re.search(pat, window, flags=re.IGNORECASE) for pat in TEMPORAL_CUES)

    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text into (section, content) pairs using simple header heuristics."""
        lower = text.lower()
        splits = []
        last_idx = 0
        last_header = "main"
        for header in SECTION_HEADERS:
            for match in re.finditer(header, lower):
                if match.start() > last_idx:
                    splits.append((last_header, text[last_idx:match.start()]))
                    last_idx = match.start()
                    last_header = header
        splits.append((last_header, text[last_idx:]))
        return splits

    @staticmethod
    def _get_sentence_window(text: str, start: int, end: int) -> str:
        """Get the sentence containing the match for scoped negation/temporality."""
        # Find sentence boundaries using common delimiters
        sent_start = start
        for delim in ".!?\n;":
            pos = text.rfind(delim, 0, start)
            if pos != -1:
                sent_start = max(sent_start, pos + 1) if pos + 1 > sent_start else sent_start
                sent_start = min(pos + 1, start)
                break
        else:
            sent_start = 0
        # Actually find the latest sentence break before start
        sent_start = 0
        for delim in ".!?\n;":
            pos = text.rfind(delim, 0, start)
            if pos != -1 and pos + 1 > sent_start:
                sent_start = pos + 1

        sent_end = len(text)
        for delim in ".!?\n;":
            pos = text.find(delim, end)
            if pos != -1 and pos < sent_end:
                sent_end = pos + 1

        return text[sent_start:sent_end].strip()

    def _check_negation_window(self, text: str, start: int, end: int) -> bool:
        """Check for negation cues within the same sentence as the match."""
        sentence = self._get_sentence_window(text, start, end)
        return self._is_negated(sentence)

    def _check_temporal_window(self, text: str, start: int, end: int) -> bool:
        """Check for temporal cues within the same sentence as the match."""
        sentence = self._get_sentence_window(text, start, end)
        return self._is_historical(sentence)

    def extract(self, text: str) -> ExtractionResult:
        conditions: List[str] = []
        medications: List[str] = []
        negated: List[str] = []
        historical: List[str] = []
        debug: Dict[str, List[str]] = {"raw_mentions": []}

        lower_text = text.lower()

        # Phrases that describe completed events -- negation does not apply
        # (e.g., "had a fall" is affirmative even if sentence contains "not")
        _AFFIRM_PHRASES = {
            r"\bhad a fall\b", r"\bpatient fell\b", r"\bfell (down|at|in|on)\b",
            r"\b(severe|bad|serious)\s+withdrawal\b", r"\bstreet drugs?\b",
            r"\ballergic reaction\b", r"\bchest (pain|tightness|pressure)\b",
            r"\bsuicid\w*\b", r"\bkill (myself|himself|herself)\b",
            r"\bwant(s)? to die\b", r"\bend (my|his|her) life\b",
            r"\bself[- ]?harm\b", r"\bfeeling (down|blue|depressed)\b",
            r"\btook (an )?extra (dose|puff|pill|tablet)\b",
            r"\btook (the )?whole (bottle|pack)\b", r"\boverdose\b",
            r"\btook too (much|many)\b",
        }

        # Phase 1: Phrase-level detection with negation/temporality checking
        for pat, cond in PHRASE_MAP.items():
            m = re.search(pat, lower_text)
            if m:
                debug["raw_mentions"].append(f"phrase:{pat}|{m.group()}")
                # Affirmative phrases skip negation check
                if pat in _AFFIRM_PHRASES:
                    is_hist = self._check_temporal_window(lower_text, m.start(), m.end())
                    if is_hist:
                        historical.append(cond)
                    else:
                        conditions.append(cond)
                else:
                    is_neg = self._check_negation_window(lower_text, m.start(), m.end())
                    is_hist = self._check_temporal_window(lower_text, m.start(), m.end())
                    if is_neg:
                        negated.append(cond)
                    elif is_hist:
                        historical.append(cond)
                    else:
                        conditions.append(cond)

        # Phase 2: Multi-word entity matching (conditions and medications)
        # Sort by length descending for greedy matching
        for phrase, canonical in sorted(
            self._multiword_conditions.items(), key=lambda x: len(x[0]), reverse=True
        ):
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                debug["raw_mentions"].append(f"mw_cond:{phrase}|{canonical}")
                is_neg = self._check_negation_window(lower_text, m.start(), m.end())
                is_hist = self._check_temporal_window(lower_text, m.start(), m.end())
                if is_neg:
                    negated.append(canonical)
                elif is_hist:
                    historical.append(canonical)
                else:
                    if canonical not in conditions:
                        conditions.append(canonical)

        for phrase, canonical in sorted(
            self._multiword_meds.items(), key=lambda x: len(x[0]), reverse=True
        ):
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                debug["raw_mentions"].append(f"mw_med:{phrase}|{canonical}")
                is_neg = self._check_negation_window(lower_text, m.start(), m.end())
                is_hist = self._check_temporal_window(lower_text, m.start(), m.end())
                if is_neg:
                    negated.append(canonical)
                elif is_hist:
                    historical.append(canonical)
                else:
                    if canonical not in medications:
                        medications.append(canonical)

        # Phase 3: Single-token matching per section
        for section, content in self._detect_sections(text):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", content)
            for i, tok in enumerate(tokens):
                norm_med = self._normalize_med(tok)
                norm_cond = self._normalize_condition(tok)
                if not norm_med and not norm_cond:
                    continue
                # Wider window: 5 tokens each side
                window_tokens = tokens[max(0, i - 5) : min(len(tokens), i + 6)]
                window = " ".join(window_tokens)
                is_neg = self._is_negated(window)
                is_hist = self._is_historical(window) or section != "main"
                if norm_med:
                    debug["raw_mentions"].append(f"med:{tok}|{window}")
                    if is_neg:
                        negated.append(norm_med)
                        continue
                    if is_hist:
                        historical.append(norm_med)
                        continue
                    if norm_med not in medications:
                        medications.append(norm_med)
                if norm_cond:
                    debug["raw_mentions"].append(f"cond:{tok}|{window}")
                    if is_neg:
                        negated.append(norm_cond)
                        continue
                    if is_hist:
                        historical.append(norm_cond)
                        continue
                    if norm_cond not in conditions:
                        conditions.append(norm_cond)

        # Phase 4: Derived operational conditions based on medication context
        if "med_access_issue" in conditions:
            if any(med in {"insulin"} for med in medications):
                conditions.append("med_access_issue_insulin")
            if any(med in {"albuterol"} for med in medications):
                conditions.append("med_access_issue_bronchodilator")
            if any(med in {"glp1"} for med in medications):
                conditions.append("med_access_issue_glp1")
            if any(med in {"immunomodulator", "biologic"} for med in medications):
                conditions.append("med_access_issue_immuno")

        return ExtractionResult(
            conditions=list(dict.fromkeys(conditions)),
            medications=list(dict.fromkeys(medications)),
            negated_mentions=list(dict.fromkeys(negated)),
            historical_mentions=list(dict.fromkeys(historical)),
            debug=debug,
        )
