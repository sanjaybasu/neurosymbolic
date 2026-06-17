"""Consistency linter: forbidden stale values, framing checks, cross-references.
Run: python3 consistency_audit.py  (exits non-zero on any error)
"""
import re, sys
from pathlib import Path

R1 = Path(__file__).resolve().parent.parent
main = (R1 / "main_revised.md").read_text()
appx = (R1 / "supplementary_revised.md").read_text()
tabs = (R1 / "tables_revised.md").read_text()
resp = (R1 / "response_to_reviewers.md").read_text()

# body of main excluding the References section (stale terms allowed in ref titles)
main_body = main.split("## References")[0]
errors, warns = [], []

# 1) FORBIDDEN stale values in current manuscript text (main body + appendix + tables)
current = main_body + "\n" + appx + "\n" + tabs
forbidden = {
    "30 contraindication-positive": "old missed count (now 64)",
    "28.0%": "old miss% (now 59.8%)",
    "35.5": "v1 miss% (now 59.8)",
    "38 of 107": "v1 missed count (now 64 of 107)",
    "95.5": "original specificity (now 89.4)",
    "SD 49.4": "old age SD (use median/IQR)",
    "30.5 years": "old mean age (use median 28.1)",
    "0.691": "old mean BERT prob on contra",
    "0.816": "old median BERT prob on contra",
    "0.58)": "v1 mean BERT prob on contra (now 0.42)",
    "+1 TP": "old OR-fusion ablation (now +0)",
    "1 additional true positive": "old OR-fusion claim",
    "70.0%": "old neural flag rate (now 41.8%)",
    "64.3%": "v1 neural flag rate (now 41.8%)",
    "66.8": "v1 OR-fusion (now 46.0)",
    "64.5": "v1 catch% (now 40.2)",
    "0.043": "old ECE (now 0.071)",
    "0.078": "old Brier (now 0.081)",
    "0.066": "v1 Brier (now 0.081)",
    "0.051": "v1 ECE (now 0.071)",
    "78.2": "v1 physician sens (now 80.1)",
    "61.7": "v1 EZ baseline catch (now 40.2)",
    "10.7%": "old physician-only TF-IDF sensitivity",
    "high-precision": "removed (PPV 36.4% is not high precision)",
}
for pat, why in forbidden.items():
    if pat in current:
        errors.append(f"STALE VALUE '{pat}' present ({why})")

# 2) framing: 'neurosymbolic' must not appear in main body (allowed only in References/ref titles)
for term in ["neurosymbolic", "Neurosymbolic"]:
    if term in main_body:
        errors.append(f"FRAMING: '{term}' appears in main body (should be reframed to hybrid; allowed only in References)")
# title must contain 'Hybrid'
title = main.splitlines()[0]
if "Hybrid" not in title and "hybrid" not in title:
    errors.append("TITLE missing 'hybrid' framing")

# 3) cross-references: every Table N referenced in main body must have a definition in tables file
refd_tables = set(re.findall(r"Table (\d)", main_body))
defined = set(re.findall(r"\*\*Table (\d)\.", tabs))
for t in sorted(refd_tables):
    if t not in defined:
        errors.append(f"CROSS-REF: Table {t} referenced in main but not defined in tables_revised.md")
# figures referenced exist
for fnum in re.findall(r"Figure (\d)", main_body):
    if f"Figure {fnum}." not in main:
        warns.append(f"Figure {fnum} referenced; confirm caption present")
# Multimedia Appendix callout present
if "Multimedia Appendix 1" not in main:
    errors.append("Missing 'Multimedia Appendix 1' callout in main text")

# 4) declarations present (comment AE)
for sec in ["## Acknowledgments", "## Conflicts of Interest", "## Funding", "## Data Availability",
            "## Author Contributions", "## Abbreviations"]:
    if sec not in main:
        errors.append(f"Missing required section: {sec}")
# GenAI disclosure (AF)
if "generative" not in main.lower():
    errors.append("Missing generative-AI disclosure (comment AF)")

# 5) references: square-bracket style, sequential, no superscript leftovers
if "<sup>" in main:
    errors.append("Superscript tag <sup> present (comment A: use [n])")
refnums = [int(n) for n in re.findall(r"^(\d+)\. ", main.split("## References")[1], flags=re.M)] if "## References" in main else []
if refnums:
    expected = list(range(1, len(refnums) + 1))
    if refnums != expected:
        errors.append(f"References not sequential 1..N: got {refnums[:5]}...{refnums[-3:]}")
    # every [n] in body within range and present
    cited = set(int(x) for x in re.findall(r"\[(\d+)(?:[,-]\d+)*\]", main_body))
    cited |= set(int(x) for grp in re.findall(r"\[([\d,\-]+)\]", main_body) for x in re.findall(r"\d+", grp))
    maxref = max(refnums)
    for c in sorted(cited):
        if c > maxref:
            errors.append(f"In-text citation [{c}] exceeds reference list size {maxref}")

# 6) P-value style (comment J): no 'p<0.05' lowercase/leading-zero patterns in main body
if re.search(r"\bp\s*[<=]\s*0\.\d", main_body):
    warns.append("Lowercase/leading-zero p-value found; JMIR uses italic P and no leading zero (e.g., P<.001)")

print(f"consistency_audit: {len(errors)} errors, {len(warns)} warnings")
for e in errors: print("  ERROR: " + e)
for w in warns: print("  WARN: " + w)
sys.exit(1 if errors else 0)
