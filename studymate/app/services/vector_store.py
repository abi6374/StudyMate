from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
import pickle
from dataclasses import dataclass

@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict

class FaissVectorStore:
    def __init__(self, dim: int, store_path: Path):
        self.dim = dim
        self.store_path = store_path
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[StoredChunk] = []

    def add(self, embeddings: np.ndarray, chunk_objs: List[StoredChunk]):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunk_objs)

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[StoredChunk, float]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype('float32')
        scores, idxs = self.index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def persist(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.store_path.with_suffix('.faiss')))
        with open(self.store_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, store_path: Path):
        index = faiss.read_index(str(store_path.with_suffix('.faiss')))
        with open(store_path.with_suffix('.pkl'), 'rb') as f:
            chunks = pickle.load(f)
        obj = cls(index.d, store_path)
        obj.index = index
        obj.chunks = chunks
        return obj
