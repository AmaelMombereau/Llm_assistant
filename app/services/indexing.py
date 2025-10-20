import faiss
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
from ..config import settings


class VectorIndex:
def __init__(self, dim: int):
self.dim = dim
self.index = faiss.IndexFlatIP(dim)
self.store: List[str] = [] # parallel store of chunks


@staticmethod
def _norm(x: np.ndarray) -> np.ndarray:
# normalize for cosine similarity via inner product
norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
return x / norms


def add(self, vectors: np.ndarray, chunks: List[str]) -> None:
assert vectors.shape[0] == len(chunks)
vectors = self._norm(vectors.astype(np.float32))
self.index.add(vectors)
self.store.extend(chunks)


def search(self, query_vec: np.ndarray, k: int = 4) -> Tuple[List[str], List[float]]:
q = self._norm(query_vec.astype(np.float32))
scores, idx = self.index.search(q, k)
hits, sims = [], []
for i in range(idx.shape[1]):
ii = idx[0, i]
if ii == -1:
continue
hits.append(self.store[ii])
sims.append(float(scores[0, i]))
return hits, sims


def save(self, path: str):
p = Path(path)
p.mkdir(parents=True, exist_ok=True)
faiss.write_index(self.index, str(p / "index.faiss"))
(p / "store.txt").write_text("\n\n====\n\n".join(self.store), encoding="utf-8")


@classmethod
def load(cls, path: str):
p = Path(path)
index = faiss.read_index(str(p / "index.faiss"))
store_text = (p / "store.txt").read_text(encoding="utf-8")
store = store_text.split("\n\n====\n\n") if store_text else []
obj = cls(index.d)
obj.index = index
obj.store = store
return obj