from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from ..config import settings


class Embedder:
def __init__(self):
self.model = SentenceTransformer(settings.EMBED_MODEL_NAME, device=settings.DEVICE)


def embed(self, texts: List[str]) -> np.ndarray:
return np.asarray(self.model.encode(texts, batch_size=settings.EMBED_BATCH_SIZE, show_progress_bar=False))