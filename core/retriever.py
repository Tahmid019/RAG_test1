import os
import numpy as np
from core.embeddings import Embeddings
from core.store import VectorStore

class Retriever:
    def __init__(self, config_path='models_config/config.yaml'):
        # load yaml with embed_model, data path, index
        import yaml
        cfg = yaml.safe_load(open(config_path))
        self.embedder = Embeddings(cfg['embedding_model'])
        self.store = VectorStore(cfg['dim'], cfg['index_path'])
        # load raw docs
        self.docs = open(cfg['docs_path'], encoding='utf-8').read().split("\n\n")

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q_emb = self.embedder.embed([query]).cpu().numpy()
        ids, _ = self.store.search(q_emb, k)
        return [self.docs[i] for i in ids]