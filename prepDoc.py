import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


corpus_path = 'data/docs/corpus.txt'
index_path = 'data/faiss_index/index.faiss'

with open(corpus_path, 'r', encoding='utf-8') as f:
    docs = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(docs)} documents.")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, convert_to_numpy=True).astype('float32')

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim) 

index.add(embeddings)
print("FAISS index built with shape:", embeddings.shape)

os.makedirs(os.path.dirname(index_path), exist_ok=True)
faiss.write_index(index, index_path)
print(f"Index saved to {index_path}")





