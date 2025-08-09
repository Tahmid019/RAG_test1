from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from shared.utils import EmbeddingGenerator, FaissStore

load_dotenv()
app = Flask(__name__)

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: The GEMINI_API_KEY environment variable is not set.")
    sys.exit()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.bin')
METADATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.metadata')

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
        
        prompt = f"""You are a specialized insurance policy evaluator.
Your task is to analyze a user's query and the provided context from insurance documents.
Based *only* on the context, provide a definitive answer to the query.
Your response should be a simple "Yes" or "No" followed by a concise, one-sentence explanation.
If the context does not contain enough information to make a determination, state "The provided information is insufficient."

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

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} with dimension {self.embedding_dimension}")

    def generate(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)
faiss_store = FaissStore.load_index(FAISS_INDEX_FILE, METADATA_FILE)


# def init_rag_components():
#     global embedding_generator, faiss_store
#     print("Initializing RAG components...")
    
#     faiss_store = FaissStore.load_index(FAISS_INDEX_FILE, METADATA_FILE)
#     if faiss_store is None:
#         print("Faiss index not found. Please run the ingestion script first.")
#         sys.exit()
    
#     embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL_NAME)
#     print("RAG components initialized and ready.")

@app.route('/query', methods=['POST'])
def query_endpoint():
    # Accept JSON or form-data
    if request.is_json:
        data = request.get_json()
        query_text = data.get('query') if data else None
    else:
        query_text = request.form.get('query')

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    print(f"Received query: {query_text}")

    query_vector = embedding_generator.generate([query_text])[0]
    search_results = faiss_store.search(query_vector, k=10)

    context = ""
    for result in search_results:
        source = result['metadata']['source']
        text = result['metadata']['text']
        chunk_id = result['metadata']['chunk_id']
        context += f"--- Source: {source}, Chunk {chunk_id} ---\n{text}\n\n"

    final_answer = faiss_store.get_llm_response(query_text, context)

    return jsonify({"answer": final_answer})

if __name__ == '__main__':
    # init_rag_components()
    app.run(host='0.0.0.0', port=7860)
