# embedding_info.py
from loader import load_components
from utils import ensure_outputs_folder

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

print("Embedding Size:", embedder.get_sentence_embedding_dimension())

sample = "The patient has high fever"
vec = embedder.encode(sample)
print("Embedding Sample (first 10 dims):", vec[:10])
