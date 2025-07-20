from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, config_path='models_config/config.yaml'):
        import yaml
        cfg = yaml.safe_load(open(config_path))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['generation_model'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg['generation_model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate(self, query: str, docs: list[str]) -> str:
        prompt = "Context: " + "\n---\n".join(docs) + f"\nQuestion: {query}"
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=150)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)