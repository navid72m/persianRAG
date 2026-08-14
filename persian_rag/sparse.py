"""
A minimal BM25-style sparse vectorizer. Qdrant stores sparse vectors as
(indices, values) pairs — we need a stable term->index vocabulary that's
built once at ingestion time and reused at query time, so it's persisted
to disk alongside the parent store.
"""
import json
import math
import os
from collections import Counter

from .persian_text import tokenize

K1 = 1.5
B = 0.75


class BM25Vocab:
    def __init__(self):
        self.term_to_idx: dict[str, int] = {}
        self.doc_freq: Counter = Counter()
        self.n_docs: int = 0
        self.avg_doc_len: float = 0.0

    def fit(self, texts: list[str]) -> None:
        total_len = 0
        for text in texts:
            terms = set(tokenize(text))
            total_len += len(tokenize(text))
            for t in terms:
                self.doc_freq[t] += 1
                if t not in self.term_to_idx:
                    self.term_to_idx[t] = len(self.term_to_idx)
        self.n_docs = len(texts)
        self.avg_doc_len = total_len / max(1, self.n_docs)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        doc_len = len(tokens)
        indices, values = [], []
        for term, freq in tf.items():
            idx = self.term_to_idx.get(term)
            if idx is None:
                continue  # unseen term (query-time only) — skip, dense leg covers it
            idf = self._idf(term)
            denom = freq + K1 * (1 - B + B * doc_len / max(1.0, self.avg_doc_len))
            score = idf * (freq * (K1 + 1)) / max(1e-9, denom)
            if score > 0:
                indices.append(idx)
                values.append(float(score))
        return indices, values

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "term_to_idx": self.term_to_idx,
                "doc_freq": self.doc_freq,
                "n_docs": self.n_docs,
                "avg_doc_len": self.avg_doc_len,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BM25Vocab":
        vocab = cls()
        if not os.path.exists(path):
            return vocab
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.term_to_idx = data["term_to_idx"]
        vocab.doc_freq = Counter(data["doc_freq"])
        vocab.n_docs = data["n_docs"]
        vocab.avg_doc_len = data["avg_doc_len"]
        return vocab
