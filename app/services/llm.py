from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from ..config import settings


class LLM:
def __init__(self):
self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.LLM_MODEL_NAME)
self.device = settings.DEVICE
self.model.to(self.device)


def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
with torch.no_grad():
out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
return self.tokenizer.decode(out[0], skip_special_tokens=True)