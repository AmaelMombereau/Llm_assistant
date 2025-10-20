from typing import List
import os
import numpy as np
from ..config import settings
from .embedding import Embedder
from .indexing import VectorIndex


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.index_path = settings.INDEX_DIR
        self.index = None

    @staticmethod
    def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + size])
            i += size - overlap
        return chunks

    def build_index(self, texts: List[str]) -> None:
        chunks = []
        for t in texts:
            chunks.extend(self.chunk(t))
        if not chunks:
            raise ValueError("No text to index")
        vectors = self.embedder.embed(chunks)
        self.index = VectorIndex(vectors.shape[1])
        self.index.add(vectors, chunks)
        self.index.save(self.index_path)

    def ensure_index(self):
        if self.index is None:
            idx_file = os.path.join(self.index_path, "index.faiss")
            if os.path.exists(idx_file):
                self.index = VectorIndex.load(self.index_path)
            else:
                raise RuntimeError("Index not found. Ingest documents first.")

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        self.ensure_index()
        qv = self.embedder.embed([query])
        hits, _ = self.index.search(qv, k=k)
        return hits
