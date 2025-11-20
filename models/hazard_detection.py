#!/usr/bin/env python3
"""
Hazard detection module: trains a lightweight text classifier that maps
conversation prompts (plus structured context) to hazard categories.

The classifier is intentionally transparent and reproducible:
- TF–IDF bag-of-ngrams features (1–2 grams)
- Multinomial logistic regression
- Stratified cross-validation to report accuracy and macro F1

It returns probability distributions so downstream controllers can
inspect confidence; low-confidence predictions fall back to "unknown".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def _flatten_context(context: Dict) -> str:
    """Convert nested context dictionaries into a textual feature string."""
    chunks: List[str] = []

    def _walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}{k}_", v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(prefix, item)
        else:
            chunks.append(str(value).replace("_", " "))

    for key, val in (context or {}).items():
        _walk(f"{key}_", val)
    return " ".join(chunks)


@dataclass
class DetectionResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class HazardDetector:
    """Simple probabilistic classifier for hazard detection."""

    def __init__(self, threshold: float = 0.15, max_iter: int = 500) -> None:
        self.threshold = threshold
        self.pipeline: Pipeline | None = None
        self.labels_: Sequence[str] | None = None
        self.max_iter = max_iter

    @staticmethod
    def _texts_labels(scenarios) -> Tuple[List[str], List[str]]:
        texts, labels = [], []
        for s in scenarios:
            if hasattr(s, "prompt"):
                prompt = s.prompt  # type: ignore[attr-defined]
                context = s.context  # type: ignore[attr-defined]
                hazard = s.hazard_type  # type: ignore[attr-defined]
            else:
                prompt = s["prompt"]
                context = s.get("context", {})
                hazard = s["hazard_type"]
            text = f"{prompt} {_flatten_context(context)}"
            texts.append(text)
            labels.append(hazard)
        return texts, labels

    def fit(self, scenarios, cv_splits: int = 5) -> Dict[str, float]:
        texts, labels = self._texts_labels(scenarios)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")
        classifier = LogisticRegression(
            multi_class="auto",
            solver="lbfgs",
            max_iter=self.max_iter,
            class_weight="balanced",
        )
        self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])

        metrics: Dict[str, List[float]] = {"accuracy": [], "macro_f1": []}
        labels_arr = np.array(labels)
        texts_arr = np.array(texts)
        unique_classes = len(set(labels))
        splits = min(cv_splits, unique_classes, len(labels))
        if splits >= 2:
            skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(texts_arr, labels_arr):
                self.pipeline.fit(texts_arr[train_idx], labels_arr[train_idx])
                preds = self.pipeline.predict(texts_arr[test_idx])
                metrics["accuracy"].append(accuracy_score(labels_arr[test_idx], preds))
                metrics["macro_f1"].append(f1_score(labels_arr[test_idx], preds, average="macro"))

        # Fit on full dataset for downstream use
        self.pipeline.fit(texts, labels)
        self.labels_ = list(self.pipeline.classes_)
        summary = {
            "accuracy_mean": float(np.mean(metrics["accuracy"])) if metrics["accuracy"] else 1.0,
            "accuracy_std": float(np.std(metrics["accuracy"])) if metrics["accuracy"] else 0.0,
            "macro_f1_mean": float(np.mean(metrics["macro_f1"])) if metrics["macro_f1"] else 1.0,
            "macro_f1_std": float(np.std(metrics["macro_f1"])) if metrics["macro_f1"] else 0.0,
            "n_classes": unique_classes,
        }
        return summary

    def predict(self, prompt: str, context: Dict) -> DetectionResult:
        if self.pipeline is None or self.labels_ is None:
            raise RuntimeError("HazardDetector must be fitted before calling predict().")
        text = f"{prompt} {_flatten_context(context)}"
        probas = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        idx = int(np.argmax(probas))
        label = labels[idx]
        confidence = float(probas[idx])
        if confidence < self.threshold:
            label = "unknown"
        prob_dict = {label_name: float(score) for label_name, score in zip(labels, probas)}
        return DetectionResult(label=label, confidence=confidence, probabilities=prob_dict)

    def classification_report(self, scenarios) -> str:
        if self.pipeline is None:
            raise RuntimeError("Detector not fitted.")
        texts, labels = self._texts_labels(scenarios)
        preds = self.pipeline.predict(texts)
        return classification_report(labels, preds, zero_division=0)
