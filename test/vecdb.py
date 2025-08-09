import numpy as np
import faiss
import os
import pickle

class FaissStore:
    def __init__(self, vector_dimension=1536):
        self.vector_dimension = vector_dimension
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.metadata = []

    def save_vectors(self, vectors, metadata, chunks):
        numpy_vectors = np.array(vectors, dtype='float32')
        self.index.add(numpy_vectors)
        start_index = len(self.metadata)
        for i, chunk in enumerate(chunks):
            vector_id = f"{metadata['id']}_chunk_{start_index + i}"
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": start_index + i,
                "text": chunk
            }
            self.metadata.append(chunk_metadata)
        
        print(f"Successfully added {len(vectors)} vectors to the index.")
    
    def search(self, query_vector, k=5):
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

    def save_index(self, file_path="faiss_index.bin"):
        faiss.write_index(self.index, file_path)
        with open(f"{file_path}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index and metadata saved to {file_path} and {file_path}.metadata")

    @staticmethod
    def load_index(file_path="faiss_index.bin"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found at {file_path}")
        
        if not os.path.exists(f"{file_path}.metadata"):
             raise FileNotFoundError(f"Metadata file not found at {file_path}.metadata")

        store = FaissStore()
        store.index = faiss.read_index(file_path)
        with open(f"{file_path}.metadata", "rb") as f:
            store.metadata = pickle.load(f)
        print(f"Index and metadata loaded from {file_path} and {file_path}.metadata")
        return store


if __name__ == '__main__':
    VECTOR_DIMENSION = 1536
    vector_store = FaissStore(vector_dimension=VECTOR_DIMENSION)
    mock_chunks = [
        "This is the first sentence about vectors.",
        "The second sentence discusses local vector stores.",
        "A third sentence explaining how FAISS works."
    ]
    mock_vectors = [
        np.random.rand(VECTOR_DIMENSION).tolist(),
        np.random.rand(VECTOR_DIMENSION).tolist(),
        np.random.rand(VECTOR_DIMENSION).tolist()
    ]
    mock_metadata = {"id": "doc_1", "source": "local_data.txt"}
    vector_store.save_vectors(mock_vectors, mock_metadata, mock_chunks)
    query_text = "how do local vector stores function?"
    query_vector = np.random.rand(VECTOR_DIMENSION).tolist()
    search_results = vector_store.search(query_vector, k=2)
    print("\nSearch Results (top 2):")
    for result in search_results:
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['metadata']['text']}\n")
    vector_store.save_index()
    loaded_store = FaissStore.load_index()
    print("Verifying loaded index...")
    search_results_loaded = loaded_store.search(query_vector, k=1)
    print("Loaded Index Search Result:")
    print(f"Distance: {search_results_loaded[0]['distance']:.4f}")
    print(f"Text: {search_results_loaded[0]['metadata']['text']}\n")