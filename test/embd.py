from sentence_transformers import SentenceTransformer
import os

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
    
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading Hugging Face model {model_name}: {e}")

    def chunk_text_by_tokens(self, text, chunk_size, tokenizer):
    
        tokenized_input = tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokenized_input['input_ids'][0]
        
        def encode(text_to_encode):
            return tokenizer.encode(text_to_encode, add_special_tokens=False)

        def decode(tokens_to_decode):
            return tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
            
        chunks = []
        for i in range(0, len(input_ids), chunk_size):
            chunk_tokens = input_ids[i:i + chunk_size]
            chunks.append(decode(chunk_tokens))
        return chunks

    def generate_embeddings(self, chunks):
     
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings.tolist() 
    def process_text(self, text, chunk_size=800):


        chunks = self.chunk_text_by_tokens(text, chunk_size, self.model.tokenizer)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings

if __name__ == '__main__':

    text = """
    The quick brown fox jumps over the lazy dog. This is a classic sentence
    used to demonstrate typography and a complete alphabet. It contains every
    letter of the English alphabet. The fox is known for its agility and speed,
    while the dog is known for its laziness. The sentence is simple, yet effective.
    It has been used for centuries to test typewriters, keyboards, and font designs.
    The phrase is also a good test for computer vision systems, as it contains
    a variety of shapes and objects. The quick brown fox jumps over the lazy dog.
    """

    generator = EmbeddingGenerator()
    chunks, embeddings = generator.process_text(text, chunk_size=800)

    print(f"\nGenerated {len(chunks)} chunks.")
    print(f"Generated {len(embeddings)} embeddings.")
    if embeddings:
        print(f"Each embedding has a dimension of {len(embeddings[0])}.")
        print("First chunk embedding (first 10 elements):", embeddings[0][:10])

