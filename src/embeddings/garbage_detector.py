import re

import numpy as np
from scipy._lib._sparse import issparse


class GarbageDetector:
    """
    Детектор мусорных и бессмысленных строк.
    Использует:
    - TF-IDF sparsity
    - длину текста
    - долю букв/слов
    - количество токенов из словаря
    """

    def __init__(self, tfidf,
                 min_length_chars=5,
                 min_length_words=2,
                 min_clean_ratio=0.3):
        self.tfidf = tfidf
        self.min_length_chars = min_length_chars
        self.min_length_words = min_length_words
        self.min_clean_ratio = min_clean_ratio

    def _clean_ratio(self, text: str) -> float:
        total = max(len(text), 1)
        alnum = sum(ch.isalnum() for ch in text)
        return alnum / total

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[a-zA-Zа-яА-Я0-9]+", text))

    def _tfidf_nnz(self, text: str) -> int:
        vec = self.tfidf.transform([text])
        if issparse(vec):
            return vec.nnz
        return np.count_nonzero(vec)

    def is_garbage(self, text: str) -> bool:
        text = text.strip()

        # 1) слишком коротко
        if len(text) < self.min_length_chars:
            return True

        # 2) мало слов
        if self._word_count(text) < self.min_length_words:
            return True

        # 3) TF-IDF слов нет
        if self._tfidf_nnz(text) == 0:
            return True

        # 4) доля букв слишком мала ("@@----====")
        if self._clean_ratio(text) < self.min_clean_ratio:
            return True

        return False
