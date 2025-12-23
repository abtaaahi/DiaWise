# data_preview.py
from loader import load_components
from utils import ensure_outputs_folder

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

print("Sample Records:")
print(df.head(10))

print("\nClass Distribution:")
print(df['label'].value_counts())

df.head(10).to_csv("outputs/sample_for_thesis.csv", index=False)
