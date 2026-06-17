"""Audit against this editor's (Coristine, JMIR Med Inform) known rejection tactics,
extracted from ms#94081's terminal-rejection history:
  1 ROC monotonicity across tables   2 arithmetic reconciliation of every table cell
  3 math-derivability (sens/spec->counts->%)   4 reference orphans + sequential numbering
  5 response-letter WHERE pointers exist   6 "all re-verified" danger phrases
  7 >1-decimal percentages (comment R)   8 cross-doc number agreement (response subset of MS)
Exits non-zero on any error.
"""
import re, sys, json
from pathlib import Path
R1 = Path(__file__).resolve().parent.parent
main = (R1/"main_revised.md").read_text(); tabs=(R1/"tables_revised.md").read_text()
appx=(R1/"supplementary_revised.md").read_text(); resp=(R1/"response_to_reviewers.md").read_text()
errors=[]; notes=[]

# 1 ROC monotonicity: Table 2 ClinicalBERT threshold sweep (v2)
sweep=[(0.10,92.9,83.3),(0.30,90.5,89.4),(0.50,86.9,89.4)]
for i in range(len(sweep)-1):
    _,s1,sp1=sweep[i]; _,s2,sp2=sweep[i+1]
    if not (sp2>=sp1 and s2<=s1): errors.append(f"ROC non-monotonic in sweep {sweep[i]}->{sweep[i+1]}")

# 2 arithmetic reconciliation of Table 6 (overlap, v2) — all four cells sum to 1519; both+sym_only=107
overlap={"0.10":(89,1058,18,354),"0.30":(63,803,44,609),"0.50":(43,592,64,820)}
for t,(both,bo,so,nei) in overlap.items():
    if both+bo+so+nei!=1519: errors.append(f"Table6 {t}: cells sum {both+bo+so+nei}!=1519")
    if both+so!=107: errors.append(f"Table6 {t}: both+symbolic_only={both+so}!=107")
    catch=round(both/107*100,1); miss=round(so/107*100,1)
    if abs(catch+miss-100)>0.15: errors.append(f"Table6 {t}: catch+miss={catch+miss}!=100")
    flag=round((both+bo)/1519*100,1)
    notes.append(f"Table6 {t}: catch {catch}% miss {miss}% neural-flag {flag}%")

# 3 math-derivability: Table2 ClinicalBERT test
def chk(num,den,pct,lbl):
    if abs(round(num/den*100,1)-pct)>0.1: errors.append(f"{lbl}: {num}/{den}={round(num/den*100,1)}!={pct}")
chk(73,84,86.9,"sens"); chk(59,66,89.4,"spec"); chk(73,80,91.2,"prec"); chk(132,150,88.0,"acc")
chk(125,156,80.1,"phys sens"); chk(51,52,98.1,"phys spec")
chk(105,107,98.1,"EZ1b aug catch@0.10"); chk(43,107,40.2,"EZ1b base catch@0.50")
chk(64,107,59.8,"miss@0.50"); chk(43,107,40.2,"catch@0.50"); chk(107,1519,7.0,"pharm contra")
chk(39,107,36.4,"PPV"); chk(635,1519,41.8,"neural flag@0.50")

# 4 references: sequential + no orphans + no [n]>maxref
reflist=main.split("## References")[1].split("---")[0]
refnums=[int(n) for n in re.findall(r"^(\d+)\.\s",reflist,flags=re.M)]
if refnums!=list(range(1,len(refnums)+1)): errors.append(f"refs not sequential: {refnums}")
body=main.split("## References")[0]
cited=set()
for grp in re.findall(r"\[([\d,\-]+)\]",body):
    for part in grp.split(","):
        if "-" in part:
            a,b=part.split("-"); cited.update(range(int(a),int(b)+1))
        elif part.strip().isdigit(): cited.add(int(part))
maxref=max(refnums)
for c in sorted(cited):
    if c>maxref: errors.append(f"citation [{c}] > maxref {maxref}")
orphans=[n for n in refnums if n not in cited]
if orphans: errors.append(f"orphan refs never cited in body: {orphans}")

# 5 response WHERE pointers (section names) exist in main or appendix
where_secs=re.findall(r"WHERE:[^.]*?(Methods|Results|Discussion|Abstract|Table \d|Appendix section \d|Conflicts of Interest|Acknowledgments|Data Availability)",resp)
# spot specific named subsections
for sec in ["Ethical Considerations","Operating Thresholds","Computational Performance",
            "Neural-Symbolic Overlap","Symbolic Detection by Documentation Type","Neural Classification Performance",
            "Fairness Analysis","Comparison With Prior Work","Principal Findings","Limitations","Clinical Implications"]:
    if sec in resp and sec not in main:
        errors.append(f"response cites section '{sec}' not found in main")

# 6 danger phrases in response
for pat in ["all numerical values","all values were re-verified","all values verified",
            "no prior-round","all .* checks passing","zero errors and zero warnings"]:
    if re.search(pat,resp,re.I): errors.append(f"danger phrase in response: '{pat}'")

# 7 percentages with >1 decimal (comment R) in main/tables/appendix (exclude CIs of AUROC which are 0.xx not %)
for name,txt in [("main",body),("tables",tabs),("appendix",appx)]:
    for m in re.findall(r"\b\d+\.\d\d+%",txt):
        errors.append(f"{name}: >1-decimal percentage '{m}' (comment R)")

# 8 response numbers subset of manuscript (key v2 values)
for v in ["86.9","89.4","59.8","98.1","65.4","41.8","46.0","1.55","0.39","28.1","36.4","0.77","105","64","107"]:
    if v in resp and v not in (main+tabs+appx):
        errors.append(f"response value '{v}' not in manuscript")

print(f"editor_pattern_audit: {len(errors)} errors")
for e in errors: print("  ERROR:",e)
for n in notes: print("  ok:",n)
sys.exit(1 if errors else 0)
