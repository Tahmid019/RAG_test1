import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}")

    def search(self, query_emb: np.ndarray, k: int = 5):
        scores, ids = self.index.search(query_emb, k)
        return ids[0], scores[0]