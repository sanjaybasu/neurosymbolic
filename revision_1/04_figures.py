"""Regenerate Figures 2 and 3 from canonical derived data (no PHI in figures).
Fig 2: symbolic firing rates by documentation type.
Fig 3: neural vs symbolic overlap on pharmacy reviews at threshold 0.50 (canonical 38 symbolic-only).
Fig 4 (new): note-length vs contraindication detection (mechanism for #9).
Outputs to revision_1/figures/.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R1 = Path("/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/submission/jmir/revision_1")
DER = R1 / "audit" / "derived"
FIG = R1 / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

# ---- Figure 2: symbolic firing by doc type ----
sb = json.load(open(DER / "stageB_symbolic_summary.json"))["symbolic_by_doc_type"]
labels = [s["label"] for s in sb]
contra = [s["contraindication"]["pct"] for s in sb]
risk = [s["risk_cascade"]["pct"] for s in sb]
req = [s["required_intervention"]["pct"] for s in sb]
x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x - w, contra, w, label="Contraindication", color="#b2182b")
ax.bar(x, risk, w, label="Risk amplification", color="#ef8a62")
ax.bar(x + w, req, w, label="Required intervention", color="#2166ac")
ax.set_ylabel("Firing rate, % of notes")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
ax.legend(frameon=False)
for i, s in enumerate(sb):
    ax.annotate(f"n={s['n']}", (i, -3.2), ha="center", fontsize=8, color="gray", annotation_clip=False)
    ax.annotate(f"{s['contraindication']['pct']:.1f}%", (i - w, contra[i] + 0.6), ha="center", fontsize=8)
ax.set_ylim(0, max(risk) * 1.15)
fig.tight_layout()
fig.savefig(FIG / "Figure2_symbolic_by_notetype_rev.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 2 written")

# ---- Figure 3: overlap 2x2 at 0.50 (canonical v2) ----
ov = json.load(open(DER / "canonical_v2.json"))["overlap"]["0.50"]
both, bonly, sonly, neither = ov["both"], ov["bert_only"], ov["symbolic_only"], ov["neither"]
mat = np.array([[both, sonly], [bonly, neither]])
fig, ax = plt.subplots(figsize=(6.2, 5))
im = ax.imshow(mat, cmap="YlOrRd")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Symbolic\ncontraindication", "No symbolic\ncontraindication"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Neural hazard\n(>=0.50)", "No neural hazard\n(<0.50)"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                color="white" if mat[i, j] > 250 else "black", fontsize=15, fontweight="bold")
ax.set_title("Neural vs symbolic detection on pharmacy reviews (n=1519)", fontsize=10)
fig.colorbar(im, ax=ax, label="Notes")
fig.tight_layout()
fig.savefig(FIG / "Figure3_overlap_rev.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 3 written: symbolic-only =", sonly)

# ---- Figure 4 (new): catch rate vs neural flag burden across thresholds (v2) ----
ovall = json.load(open(DER / "canonical_v2.json"))["overlap"]
ths = [0.10, 0.30, 0.50]
catch = [ovall[f"{t:.2f}"]["catch_pct"] for t in ths]
flag = [ovall[f"{t:.2f}"]["neural_flag_pct"] for t in ths]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(flag, catch, "o-", color="#2166ac")
for t, f, c in zip(ths, flag, catch):
    ax.annotate(f"thr {t:.2f}", (f, c), textcoords="offset points", xytext=(6, -10), fontsize=9)
ax.axhline(100, ls=":", color="gray")
ax.set_xlabel("Neural flagging rate, % of all pharmacy reviews")
ax.set_ylabel("Contraindications caught by neural, % of 107")
ax.set_title("Neural catch rate vs alert burden (symbolic alone: 7.0% flag, 100% catch w/ provenance)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "Figure4_catch_vs_burden_rev.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 4 written")
