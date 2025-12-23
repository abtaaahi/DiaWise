# loader.py
import os
import pandas as pd
import joblib
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

MODEL_DIR = "./medical_model_fast"
DATA_PATH = "./synthetic_data.csv"

def load_components():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load embedder
    embedder = SentenceTransformer(f"{MODEL_DIR}/embedder")

    # Load LightGBM model
    lgb_model = lgb.Booster(model_file=f"{MODEL_DIR}/model.txt")

    # Load label encoder + scaler
    le = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    # Ensure label exists
    df["label"] = df["predicted_diseases"].apply(lambda x: str(x).split(",")[0].strip())

    return embedder, lgb_model, le, scaler, df
