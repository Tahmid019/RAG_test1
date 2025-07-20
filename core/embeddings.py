from transformers import AutoTokenizer, AutoModel
import torch

class Embeddings:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        # mean-pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings