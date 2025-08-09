import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

corpus_folder = 'data/docs'
index_path = 'data/faiss_index/index.faiss'
metadata_path = 'data/faiss_index/metadata.json'

os.makedirs(os.path.dirname(index_path), exist_ok=True)

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

all_docs = []
metadata = []

for i in range(1, 6):
    file_name = f"{i}.txt"
    file_path = os.path.join(corpus_folder, file_name)

    if not os.path.isfile(file_path):
        print(f"[!] File not found: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} lines from {file_name}")

    all_docs.extend(lines)
    metadata.extend([(file_name, idx) for idx in range(len(lines))])

print("Encoding all documents...")
embeddings = model.encode(all_docs, convert_to_numpy=True).astype('float32')

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, index_path)
print(f"[+] FAISS index saved to {index_path} with shape {embeddings.shape}")

with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"[+] Metadata saved to {metadata_path} ({len(metadata)} entries)")
