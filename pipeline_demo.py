# pipeline_demo.py
from loader import load_components
from utils import translate_text, ensure_outputs_folder
import numpy as np

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

text = "রোগীর জ্বর এবং মাথা ব্যথা"

print("Original Text:", text)

translated = translate_text(text)
print("Translated Text:", translated)

embed = embedder.encode([translated])
scaled = scaler.transform(embed)
proba = lgb_model.predict(scaled)[0]
pred_label = le.inverse_transform([np.argmax(proba)])[0]

print("Predicted Disease:", pred_label)
print("Confidence:", np.max(proba))
