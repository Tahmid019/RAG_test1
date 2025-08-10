import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# --- Configuration ---
PDF_DIRECTORY = "./"
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_index.metadata"
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Helper Classes ---

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} with dimension {self.embedding_dimension}")

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
        print(f"Added {len(vectors)} vectors to the index.")

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

    def save_index(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index and metadata saved to {index_path} and {metadata_path}")

    @staticmethod
    def load_index(index_path, metadata_path):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None
        
        store = FaissStore(vector_dimension=1)
        store.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            store.metadata = pickle.load(f)
        store.vector_dimension = store.index.d
        print(f"Index and metadata loaded from {index_path} and {metadata_path}")
        return store

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_llm_response(query: str, context: str) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    prompt = f"""Provide me the related policy from the given context regarding the query
    
    ## Sample Query

    "46M, knee surgery, Pune, 3-month policy"


    ## Sample Response

    "Yes, knee surgery is covered under the policy."
        
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
    

@app.route('/query', methods=['POST'])
def query_endpoint():
    faiss_store = FaissStore.load_index(FAISS_INDEX_FILE, METADATA_FILE)
    if request.is_json:
        data = request.get_json()
        query_text = data.get('query') if data else None
    else:
        query_text = request.form.get('query')

    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)
    query_vector = embedding_generator.generate([query_text])[0]
    search_results = faiss_store.search(query_vector, k=3)
    context = ""
    
    for i, result in enumerate(search_results):
        source = result['metadata']['source']
        text = result['metadata']['text']
        context += f"--- Source: {source}, Chunk {result['metadata']['chunk_id']} ---\n{text}\n\n"
    
    final_answer = get_llm_response(query_text, context)

    return jsonify({"answer": final_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    


