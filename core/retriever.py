import os
import json
import numpy as np
from core.embeddings import Embeddings
from core.store import VectorStore

class Retriever:
    def __init__(self, config_path='models_config/config.yaml'):
        import yaml
        cfg = yaml.safe_load(open(config_path))
        
        self.embedder = Embeddings(cfg['embedding_model'])
        self.store = VectorStore(cfg['dim'], cfg['index_path'])
        with open(cfg['metadata_path'], encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.file_cache = {}
        self.docs_folder = cfg['docs_folder']

    def _get_line_from_metadata(self, file_name, line_num):
        if file_name not in self.file_cache:
            file_path = os.path.join(self.docs_folder, file_name)
            with open(file_path, encoding='utf-8') as f:
                self.file_cache[file_name] = [line.strip() for line in f if line.strip()]
        
        lines = self.file_cache[file_name]
        if line_num >= len(lines):
            return f"[Line {line_num} out of range in {file_name}]"
        return lines[line_num]


    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q_emb = self.embedder.embed([query]).cpu().numpy()  

        ids, scores = self.store.search(q_emb, k)          

        if isinstance(ids, np.ndarray):
            flat_ids = ids.flatten().tolist()
        else:
            flat_ids = [int(ids)]

        results = []
        for idx in flat_ids:
            file_name, line_num = self.metadata[idx]
            results.append(self._get_line_from_metadata(file_name, line_num))

        return results  

