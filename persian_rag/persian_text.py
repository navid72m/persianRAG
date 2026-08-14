"""
Persian-aware normalization and tokenization, used to build the sparse
(BM25) side of hybrid retrieval. Plain whitespace/English tokenizers do a
poor job on Persian: they don't unify Arabic vs Persian character variants
(ي vs ی, ك vs ک), don't handle zero-width non-joiner correctly, and don't
strip the diacritics some PDFs embed.
"""
import re
from collections import Counter

from hazm import Normalizer, word_tokenize

_normalizer = Normalizer(
    persian_style=True,
    persian_numbers=True,
    remove_diacritics=True,
    remove_specials_chars=True,
    decrease_repeated_chars=True,
    unicodes_replacement=True,
)

_STOPWORDS = {
    "از", "به", "با", "در", "که", "را", "این", "آن", "است", "بود", "شد",
    "می", "های", "برای", "تا", "یک", "و", "یا", "هم", "بر", "اما", "اگر",
    "چون", "نیز", "شود", "کرد", "دارد", "شده", "کند",
}


def normalize(text: str) -> str:
    return _normalizer.normalize(text)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    norm = normalize(text)
    tokens = word_tokenize(norm)
    tokens = [t for t in tokens if re.search(r"\w", t)]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def bm25_term_counts(text: str) -> Counter:
    """Term frequency counter used to build a sparse vector for one chunk."""
    return Counter(tokenize(text))
