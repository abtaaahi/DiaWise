# lime_demo.py
from loader import load_components
import lime.lime_text
from utils import ensure_outputs_folder
import numpy as np

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

def predict_proba(texts):
    embeds = embedder.encode(texts)
    scaled = scaler.transform(embeds)
    return lgb_model.predict(scaled)

explainer = lime.lime_text.LimeTextExplainer(class_names=le.classes_)

sample = "Patient has chest pain and breathing difficulty"

exp = explainer.explain_instance(sample, predict_proba, num_features=10)
exp.save_to_file("outputs/lime_explanation.html")
print("LIME explanation saved to outputs/lime_explanation.html")
