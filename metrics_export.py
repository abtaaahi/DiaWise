# metrics_export.py
from loader import load_components
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ensure_outputs_folder

ensure_outputs_folder()
embedder, lgb_model, le, scaler, df = load_components()

# Generate embeddings for full df
X_emb = embedder.encode(df['input_text'].tolist())
X_scaled = scaler.transform(X_emb)

# Predict
pred_probs = lgb_model.predict(X_scaled)
pred_labels = np.argmax(pred_probs, axis=1)

y_true = le.transform(df['label'])

# Save classification report
report = classification_report(y_true, pred_labels, target_names=le.classes_, output_dict=True)
pd.DataFrame(report).to_csv("outputs/classification_report.csv")

# Save confusion matrix
# cm = confusion_matrix(y_true, pred_labels)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.savefig("outputs/confusion_matrix.png")
# plt.close()

cm = confusion_matrix(y_true, pred_labels, normalize="true")

plt.figure(figsize=(8, 8))
sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    xticklabels=False,
    yticklabels=False
)
plt.title("Normalized Confusion Matrix (50 Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_slide.png")
plt.close()


# Log-loss
print("Log Loss:", log_loss(y_true, pred_probs))
