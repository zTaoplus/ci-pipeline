import os
from pathlib import Path

from transformers import BertTokenizerFast, AutoModelForMaskedLM

import torch

model_name = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

DEFAULT_PATH = str(Path(Path(__file__).parent.resolve().absolute(), "downloads", "bert"))

BASE_DIR = os.getenv("SAVE_DIR", DEFAULT_PATH)

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)

def file_abspath(base_name: str, *files):
    return str(Path(base_name, *files))


for name, param in model.named_parameters():
    # EMBEDDING
    if "embeddings.word_embeddings.weight" in name:
        torch.save(param, file_abspath(BASE_DIR, "word_emb.pt"))
        print(f"word_emb.pt file save to {BASE_DIR}")
    elif "embeddings.LayerNorm.weight" in name:
        torch.save(param, file_abspath(BASE_DIR, "layernorm_weight.pt"))
        print(f"layernorm_weight.pt file save to {BASE_DIR}")
    elif "embeddings.LayerNorm.bias" in name:
        torch.save(param, file_abspath(BASE_DIR, "layernorm_bias.pt"))
        print(f"layernorm_bias.pt file save to {BASE_DIR} ")

tk_path = file_abspath(BASE_DIR, 'bert', 'tokenizer')
tokenizer.save_pretrained(tk_path)
print(f"tokenizer files save to {tk_path}")
print("======TASK DONE======")