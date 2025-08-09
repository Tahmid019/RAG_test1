from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from shared.utils import EmbeddingGenerator, FaissStore

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: The GEMINI_API_KEY environment variable is not set.")
    sys.exit()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.bin')
METADATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index.metadata')

embedding_generator = None
faiss_store = None

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
    data = request.get_json()
    query_text = data.get('query')

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