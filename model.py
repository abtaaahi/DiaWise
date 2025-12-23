import os
os.environ["STREAMLIT_WATCHDOG"] = "false"

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils import resample
from deep_translator import GoogleTranslator
from lime.lime_text import LimeTextExplainer
import re
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

MODEL_DIR = "./medical_model_fast"
DATA_PATH = "./synthetic_data.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def embed_texts(embedder, texts, batch_size=256):
    return embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def load_or_train_model():
    # Reuse trained model if exists
    if (
        os.path.exists(f"{MODEL_DIR}/model.txt")
        and os.path.exists(f"{MODEL_DIR}/label_encoder.joblib")
        and os.path.exists(f"{MODEL_DIR}/scaler.joblib")
    ):
        embedder = SentenceTransformer(f"{MODEL_DIR}/embedder")
        lgb_model = lgb.Booster(model_file=f"{MODEL_DIR}/model.txt")
        le = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
        df = pd.read_csv(DATA_PATH)
        metrics = None
        return embedder, lgb_model, le, df, metrics, scaler

    # Load and prepare dataset
    df = pd.read_csv(DATA_PATH)
    df = df[
        ["id", "input_text", "predicted_diseases", "recommendations",
         "reasoning_keywords", "lime_explainability", "language"]
    ].dropna(subset=["input_text"])
    df = df[df["input_text"].str.strip().str.len() > 3].reset_index(drop=True)

    df["label"] = df["predicted_diseases"].apply(lambda x: str(x).split(",")[0].strip())
    df["input_text"] = df["input_text"].apply(clean_text)

    # Balance dataset
    min_size = df["label"].value_counts().min()
    df_balanced = pd.concat([
        resample(g, replace=True, n_samples=min_size, random_state=42)
        for _, g in df.groupby("label")
    ])
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    X_train, X_val, y_train, y_val = train_test_split(
        df["input_text"], df["label_enc"],
        test_size=0.15, random_state=42, stratify=df["label_enc"]
    )

    embedder = SentenceTransformer(MODEL_NAME)
    X_train_emb = embed_texts(embedder, X_train.tolist())
    X_val_emb = embed_texts(embedder, X_val.tolist())

    # Scale embeddings
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_val_emb = scaler.transform(X_val_emb)

    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "boosting_type": "dart",
        "learning_rate": 0.03,
        "num_leaves": 256,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 5,
        "lambda_l1": 0.3,
        "lambda_l2": 0.3,
        "is_unbalance": True,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train_emb, label=y_train)
    val_data = lgb.Dataset(X_val_emb, label=y_val)

    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=900,
        callbacks=[
            lgb.early_stopping(stopping_rounds=70),
            lgb.log_evaluation(period=100),
        ],
    )

    probs_val = lgb_model.predict(X_val_emb)
    preds_val = np.argmax(probs_val, axis=1)

    metrics = {
        "Accuracy": round(accuracy_score(y_val, preds_val), 4),
        "F1": round(f1_score(y_val, preds_val, average="weighted"), 4),
        "LogLoss": round(log_loss(y_val, probs_val), 4),
    }

    print("\n📊 Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save model and components
    os.makedirs(MODEL_DIR, exist_ok=True)
    lgb_model.save_model(f"{MODEL_DIR}/model.txt")
    embedder.save(f"{MODEL_DIR}/embedder")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    return embedder, lgb_model, le, df, metrics, scaler


embedder, lgb_model, le, df, metrics, scaler = load_or_train_model()


def predict_proba(texts):
    emb = embed_texts(embedder, texts)
    emb = scaler.transform(emb)
    return lgb_model.predict(emb)


def predict_patient(input_text, top_k=3):
    try:
        detected_lang = GoogleTranslator(source="auto", target="en").detect(input_text)
    except Exception:
        detected_lang = "en"

    text_en = input_text
    if detected_lang != "en":
        try:
            text_en = GoogleTranslator(source=detected_lang, target="en").translate(input_text)
        except Exception:
            text_en = input_text

    probs = predict_proba([text_en])[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_diseases = le.inverse_transform(top_idx)
    top_probs = [float(probs[i]) for i in top_idx]

    uncertainty = round(-np.sum(probs * np.log(probs + 1e-12)), 3)
    return {
        "TopDiseases": dict(zip(top_diseases, top_probs)),
        "Uncertainty": uncertainty,
        "Detected_Language": detected_lang,
    }


explainer = LimeTextExplainer(class_names=list(le.classes_))

def explain_text(text, num_features=5):
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features)
    return exp.as_list()

def plot_metrics(metrics):
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values, color=["#4CAF50", "#2196F3", "#FFC107"])
    plt.title("Model Performance Metrics", fontsize=14)
    plt.ylabel("Score")
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", 
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()

plot_metrics(metrics)