# balancing_check.py
import pandas as pd
from utils import ensure_outputs_folder

ensure_outputs_folder()

df = pd.read_csv("synthetic_data.csv")
df["label"] = df["predicted_diseases"].apply(lambda x: str(x).split(",")[0].strip())

print("Before Balancing:")
print(df['label'].value_counts())

# Simple balancing (upsample minority)
min_size = df['label'].value_counts().min()
df_balanced = pd.concat([
    g.sample(n=min_size, replace=True, random_state=42)
    for _, g in df.groupby('label')
]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Balancing:")
print(df_balanced['label'].value_counts())

df_balanced.to_csv("balanced_data.csv", index=False)
