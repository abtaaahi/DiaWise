import streamlit as st
import pandas as pd
from model import df, le, predict_patient, explain_text, metrics

st.set_page_config(page_title="🧠 Medical Disease Predictor", layout="wide")
st.title("🧠 Medical Disease Predictor")
st.caption("Top 3 disease predictions with reasoning and recommendations")

if metrics:
    st.subheader("📊 Model Training Metrics")
    for k, v in metrics.items():
        score_text = "Average"
        if v > 0.85: score_text = "Good"
        elif v < 0.6: score_text = "Bad"
        st.write(f"**{k}:** {v:.3f} — {score_text}")

user_input = st.text_area("Enter patient symptoms:", height=120)

if st.button("🔍 Predict"):
    if len(user_input.strip()) < 3:
        st.warning("Please enter a valid symptom description.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_patient(user_input)

        st.success("Prediction complete!")

        st.subheader("🧾 Top 3 Predicted Diseases")
        top_disease_df = pd.DataFrame(
            list(result["TopDiseases"].items()), columns=["Disease", "Probability"]
        )
        st.dataframe(top_disease_df.style.background_gradient(cmap="Greens"), use_container_width=True)

        for disease in top_disease_df["Disease"]:
            subset = df[df['predicted_diseases'].str.split(',').str[0].str.strip().str.lower() == disease.lower()]
            if not subset.empty:
                rec = subset['recommendations'].iloc[0] if pd.notna(subset['recommendations'].iloc[0]) else "No recommendation found."
                reason = subset['reasoning_keywords'].iloc[0] if pd.notna(subset['reasoning_keywords'].iloc[0]) else "N/A"
                explain = subset['lime_explainability'].iloc[0] if pd.notna(subset['lime_explainability'].iloc[0]) else "N/A"
                st.markdown(f"### 🩺 Disease: **{disease}**")
                st.caption(f"🧠 **Reasoning Keywords:** {reason}")
                st.caption(f"💡 **Recommendation:** {rec}")
                st.caption(f"🔍 **Explainability (dataset):** {explain}")

        st.markdown(f"### 🔎 Uncertainty Score: `{result['Uncertainty']}`")
