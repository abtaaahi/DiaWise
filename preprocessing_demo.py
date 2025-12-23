# preprocessing_demo.py
from loader import load_components
from utils import ensure_outputs_folder
import re

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

sample = df['input_text'].iloc[0]

print("Before Cleaning:", sample)
print("After Cleaning:", clean_text(sample))
