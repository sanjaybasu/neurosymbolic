"""
Fine-tune ClinicalBERT on mixed-source data (vignettes + real-world) with class-balanced, cost-sensitive loss.

Inputs (from build_mixed_splits.py):
  - packaging/neurosymbolic_github/data/mixed_splits/mixed_train.json
  - packaging/neurosymbolic_github/data/mixed_splits/mixed_val.json

Outputs:
  - Saves model weights to trained_models/clinicalbert_mixed.pt
  - Writes validation metrics to results/clinicalbert_mixed_metrics.json

Notes:
  - Uses class weights to upweight hazards.
  - Optional focal loss and pseudo-labeling flags can be added as needed.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import import_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Allow recent dev builds of torch (>=2.6.0.dev*) to load weights by bypassing the
# strict torch.load safety check that only accepts released versions.
_original_check = import_utils.check_torch_load_is_safe


def _compat_check():
    if import_utils.is_torch_greater_or_equal("2.6", accept_dev=True):
        return
    _original_check()


import_utils.check_torch_load_is_safe = _compat_check
# Also patch the reference imported inside modeling_utils so AutoModel uses the relaxed check.
import transformers.modeling_utils as modeling_utils
modeling_utils.check_torch_load_is_safe = _compat_check

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "data" / "mixed_splits"
MODEL_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "trained_models"
RESULTS_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "results"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3


class TextDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(rec["label"], dtype=torch.float)
        return item


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(-1)
        return logits


def load_json(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = ClinicalBERTClassifier(MODEL_NAME).to(device)

    train_data = load_json(DATA_DIR / "mixed_train.json")
    val_data = load_json(DATA_DIR / "mixed_val.json")

    train_ds = TextDataset(train_data, tokenizer)
    val_ds = TextDataset(val_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights (hazard vs benign)
    labels = np.array([r["label"] for r in train_data])
    pos_weight = torch.tensor((len(labels) - labels.sum()) / labels.sum(), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_t.tolist())

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs), threshold=0.5)
    print("Validation metrics:", metrics)

    # Save model
    torch.save(model.state_dict(), MODEL_DIR / "clinicalbert_mixed.pt")
    with open(RESULTS_DIR / "clinicalbert_mixed_metrics.json", "w") as f:
        json.dump({"val": metrics}, f, indent=2)
    print(f"Saved model to {MODEL_DIR / 'clinicalbert_mixed.pt'}")


if __name__ == "__main__":
    train()
