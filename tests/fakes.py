"""Thành phần giả dùng chung cho test (không cần model, mạng hay Qdrant thật)."""
import zlib

import numpy as np

from retrieval.qparse import strip_accents


def hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    for w in strip_accents(text).split():
        v[zlib.crc32(w.encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEmbedder:
    def text_siglip(self, texts):
        return np.stack([hash_vec(t, 768) for t in texts])

    def text_bge(self, texts):
        dense = np.stack([hash_vec(t, 1024) for t in texts])
        sparse = [{zlib.crc32(w.encode()) % 100000: 1.0 for w in strip_accents(t).split()} for t in texts]
        return dense, sparse
