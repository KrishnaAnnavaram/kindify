"""Optional transformer classifier (extra `transformers`). torch and transformers load lazily.

* Inputs are cut at `max_length` tokens (default 256). `truncation_report` in `data.py` tells
  how many comments are longer.
* The head is multi-label (one sigmoid for each label), so the probabilities are comparable
  with the TF-IDF classifier.
"""

from __future__ import annotations

import numpy as np

from kindify import LABELS
from kindify.classifier import logits_to_proba


def _require():
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("install the extra: pip install 'kindify[transformers]'") from exc
    return torch, transformers


class TransformerClassifier:
    def __init__(self, model=None, tokenizer=None, model_name: str | None = None, labels=LABELS,
                 max_length: int = 256, batch_size: int = 32):
        torch, transformers = _require()
        if model is None:
            if not model_name:
                raise ValueError("give a model or a model name")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.labels = tuple(labels)
        self.max_length = max_length
        self.batch_size = batch_size
        self.multilabel = getattr(model.config, "problem_type", None) == "multi_label_classification" or \
            model.config.num_labels == len(self.labels)

    def predict_proba(self, texts) -> np.ndarray:
        torch, _ = _require()
        texts = list(texts)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt")
                logits = self.model(**enc).logits.cpu().numpy()
                out.append(logits_to_proba(logits, self.multilabel))
        return np.vstack(out) if out else np.zeros((0, len(self.labels)))


def new_model(model_name: str, labels=LABELS):  # pragma: no cover - needs a download
    """A pre-trained backbone with a fresh multi-label head, ready for fine-tuning."""
    _, transformers = _require()
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), problem_type="multi_label_classification"
    )
