
import numpy as np
import faiss
import os
import pickle
import pypdf
from sentence_transformers import SentenceTransformer
import sys
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils import EmbeddingGenerator, FaissStore, chunk_text, extract_text_from_pdfs

PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
FAISS_INDEX_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.bin')
METADATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.metadata')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def ingest_data():
    faiss_store = FaissStore.load_index(FAISS_INDEX_FILE, METADATA_FILE)

    if faiss_store is None:
        print("No index found. Starting document ingestion and indexing.")
        
        if not os.path.exists(PDF_DIRECTORY):
            print(f"Error: The directory '{PDF_DIRECTORY}' does not exist.")
            print("Please create this folder and place your PDF files inside.")
            exit()

        print("Extracting text from PDFs...")
        documents = extract_text_from_pdfs(PDF_DIRECTORY)
        if not documents:
            print("No PDF files found in the directory.")
            exit()
        
        embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)
        faiss_store = FaissStore(vector_dimension=embedding_generator.embedding_dimension)
        
        print("Chunking and generating embeddings for documents...")
        for doc_name, doc_text in documents.items():
            chunks = chunk_text(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = embedding_generator.generate(chunks)
            
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "source": doc_name,
                    "chunk_id": i,
                    "text": chunk
                })

            faiss_store.add_vectors(embeddings, chunk_metadata)
        
        faiss_store.save_index(FAISS_INDEX_FILE, METADATA_FILE)
        print("Indexing complete. Index saved to disk.")
    
    else:
        print("Existing index found and loaded successfully.")

if __name__ == "__main__":
    ingest_data()
