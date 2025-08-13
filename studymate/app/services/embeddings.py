from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_CACHE = {}

def get_embedding_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = get_embedding_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
