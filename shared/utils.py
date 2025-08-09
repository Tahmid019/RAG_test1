
import numpy as np
import faiss
import os
import pickle
import pypdf
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash"


class EmbeddingGenerator:
    def __init__(self, model_name):
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side="right"
        self.model = model
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

class FaissStore:
    def __init__(self, vector_dimension):
        self.vector_dimension = vector_dimension
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.metadata = []

    def add_vectors(self, vectors: List[List[float]], metadata_list: List[Dict]):
        numpy_vectors = np.array(vectors, dtype='float32')
        self.index.add(numpy_vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        query_vector_np = np.array([query_vector], dtype='float32')
        distances, indices = self.index.search(query_vector_np, k)
        
        results = []
        for i, index in enumerate(indices[0]):
            result_metadata = self.metadata[index]
            results.append({
                "distance": distances[0][i],
                "metadata": result_metadata
            })
        return results

    def get_llm_response(self, query: str, context: str) -> str:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        prompt = f"""You are a helpful assistant for a document repository.
Your task is to answer the user's query based *only* on the provided context.
Answer directly and concisely. If the information is not present in the context, state that you cannot answer.

Context:
{context}

Query:
{query}

Answer:
"""
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini API: {e}"

    def save_index(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    @staticmethod
    def load_index(index_path, metadata_path):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None
        
        store = FaissStore(vector_dimension=1)
        store.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            store.metadata = pickle.load(f)
        store.vector_dimension = store.index.d
        return store

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def extract_text_from_pdfs(pdf_dir: str) -> Dict[str, str]:
    all_docs = {}
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                reader = pypdf.PdfReader(filepath)
                full_text = ""
                for page_num, page in enumerate(reader.pages):
                    full_text += page.extract_text() or ""
                    full_text += f"\n\n--- Page {page_num + 1} of {filename} ---\n\n"
                all_docs[filename] = full_text
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return all_docs
