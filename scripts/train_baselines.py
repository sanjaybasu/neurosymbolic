#!/usr/bin/env python3
"""Train BioBERT and PubMedBERT baselines on the same data as ClinicalBERT."""
import json, sys, numpy as np, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from transformers import AutoModel, AutoTokenizer
from transformers.utils import import_utils

_original_check = import_utils.check_torch_load_is_safe
def _compat_check():
    if import_utils.is_torch_greater_or_equal("2.6", accept_dev=True):
        return
    _original_check()
import_utils.check_torch_load_is_safe = _compat_check
import transformers.modeling_utils as modeling_utils
modeling_utils.check_torch_load_is_safe = _compat_check

REPO = Path(__file__).resolve().parent.parent
MAX_LEN = 256
BATCH_SIZE = 16

class Classifier(nn.Module):
    def __init__(self, base_model, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hs = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hs, 1))
    def forward(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(o.last_hidden_state[:, 0, :]).squeeze(-1)

class DS(Dataset):
    def __init__(self, recs, tok, ml=256):
        self.recs, self.tok, self.ml = recs, tok, ml
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        e = self.tok(r["text"], truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in e.items()}
        item["labels"] = torch.tensor(r["label"], dtype=torch.float)
        return item

def load_json(p):
    with open(p) as f: return json.load(f)

def bootstrap_ci(y_true, y_prob, fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb, pb = y_true[idx], y_prob[idx]
        if len(np.unique(yb)) < 2: continue
        try: vals.append(fn(yb, pb))
        except: continue
    if not vals: return (None, None)
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))

def main():
    train_data = load_json(REPO / "data/mixed_splits/mixed_train.json")
    val_data = load_json(REPO / "data/mixed_splits/mixed_val.json")
    rw_test = load_json(REPO / "data/mixed_splits/mixed_realworld_test.json")
    phys_test = load_json(REPO / "data/mixed_splits/mixed_physician_test.json")
    device = torch.device("cpu")
    labels_arr = np.array([r["label"] for r in train_data])
    pw = float((len(labels_arr) - labels_arr.sum()) / max(labels_arr.sum(), 1))
    results = {}
    for key, name in [("biobert", "dmis-lab/biobert-v1.1"),
                       ("pubmedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")]:
        print(f"\n{'='*60}\nTraining {key}: {name}\n{'='*60}")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        mdl = Classifier(name).to(device)
        tl = DataLoader(DS(train_data, tok, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(DS(val_data, tok, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
        opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
        best, pat = float("inf"), 0
        for ep in range(3):
            mdl.train(); tl_sum = 0.0
            for b in tl:
                opt.zero_grad()
                lo = mdl(b["input_ids"].to(device), b["attention_mask"].to(device))
                loss = crit(lo, b["labels"].to(device)); loss.backward(); opt.step()
                tl_sum += loss.item()
            mdl.eval(); vl_sum = 0.0
            with torch.no_grad():
                for b in vl:
                    lo = mdl(b["input_ids"].to(device), b["attention_mask"].to(device))
                    vl_sum += crit(lo, b["labels"].to(device)).item()
            avl = vl_sum / len(vl)
            print(f"  Epoch {ep+1}: train={tl_sum/len(tl):.4f}, val={avl:.4f}")
            if avl < best:
                best, pat = avl, 0
                torch.save(mdl.state_dict(), REPO / f"trained_models/{key}_mixed.pt")
            else:
                pat += 1
                if pat >= 3: print("  Early stopping"); break
        mdl.load_state_dict(torch.load(REPO / f"trained_models/{key}_mixed.pt", map_location=device))
        mdl.eval()
        for sn, sd in [("rw_test", rw_test), ("physician_test", phys_test)]:
            ap, al = [], []
            with torch.no_grad():
                for r in sd:
                    e = tok(r["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
                    lo = mdl(e["input_ids"].to(device), e["attention_mask"].to(device))
                    ap.append(torch.sigmoid(lo).numpy().item()); al.append(r["label"])
            yt, yp = np.array(al), np.array(ap)
            yd = (yp >= 0.5).astype(int)
            tp=int(((yt==1)&(yd==1)).sum()); fn=int(((yt==1)&(yd==0)).sum())
            tn=int(((yt==0)&(yd==0)).sum()); fp=int(((yt==0)&(yd==1)).sum())
            au = roc_auc_score(yt, yp); au_ci = bootstrap_ci(yt, yp, roc_auc_score)
            br = brier_score_loss(yt, yp)
            se = tp/(tp+fn) if (tp+fn) else 0; sp = tn/(tn+fp) if (tn+fp) else 0
            results[f"{key}_{sn}"] = {"tp":tp,"fn":fn,"tn":tn,"fp":fp,
                "sensitivity":round(se,4),"specificity":round(sp,4),
                "auroc":round(au,4),"auroc_ci":au_ci,"brier":round(br,4)}
            print(f"  {sn}: tp={tp},fn={fn},tn={tn},fp={fp}, sens={se:.4f}, spec={sp:.4f}, auroc={au:.4f} ({au_ci})")
    with open(REPO / "results/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/baseline_comparison.json")

if __name__ == "__main__":
    main()
