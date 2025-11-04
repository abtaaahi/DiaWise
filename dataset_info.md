After running the script, your dataset will look like this:

**Rows and size:**

* **Total rows:** 50 diseases × 500 examples each = 25,000 rows.
* **Each row** represents a single patient-like input (synthetic but realistic) describing symptoms.

**Columns and their content:**

| Column                | Description                                                                                                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                  | Unique integer identifier for each row (1…25,000).                                                                                                                                                              |
| `input_text`          | Patient symptom description in English, Bangla, Banglish, or a mix. Generated with varied phrasing, duration, and severity. Example: `"I feel very thirsty, urinate frequently, and feel fatigued for 2 days."` |
| `predicted_diseases`  | Top-3 plausible diseases, with the first being the correct one. Example: `"Diabetes Mellitus,Anemia,Thyroid Disorders"`                                                                                         |
| `probabilities`       | Probabilities associated with each predicted disease (summing ~1). Example: `"0.39,0.27,0.21"`                                                                                                                  |
| `lime_explainability` | Top 2–3 symptom keywords from the input used for model interpretability. Example: `"thirst, urinate, fatigued"`                                                                                                 |
| `reasoning_keywords`  | Predefined key symptoms/reasoning for the primary disease. Example: `"thirst; urination; fatigue; low energy; dry mouth"`                                                                                       |
| `recommendations`     | Actionable guidance for the patient. Example: `"Get blood sugar test; Stay hydrated; Eat balanced meals; Avoid sugary drinks"`                                                                                  |
| `uncertainty_score`   | Normalized uncertainty (entropy) of the predicted probabilities, range 0–1. Example: `0.91`                                                                                                                     |
| `language`            | Indicates whether the `input_text` is English (`en`), Bangla (`bn`), Banglish (`banglish`), or mixed (`mixed`).                                                                                                 |

**Key characteristics:**

* **Diverse phrasing:** Templates, paraphrases, and minor chatty prefixes/suffixes make patient statements varied.
* **Language variety:** Mix of English, Bangla Unicode, Latin-script Bangla, and mixed sentences.
* **Balanced disease coverage:** 500 examples per disease ensure roughly uniform representation.
* **Actionable and interpretable:** Each row has synthetic LIME explanations, reasoning keywords, and recommendations.
* **Uncertainty-aware:** Probabilities + normalized entropy allow training models with confidence estimation.

**Example row from the dataset:**

```csv
1,"I feel very thirsty, urinate frequently, and feel fatigued for 2 days.","Diabetes Mellitus,Anemia,Thyroid Disorders","0.39,0.27,0.21","thirst, urinate, fatigued","thirst; urination; fatigue; low energy; dry mouth","Get blood sugar test; Stay hydrated; Eat balanced meals; Avoid sugary drinks","0.91","en"
```

Essentially, your dataset now **mimics real-world patient inputs** across multiple languages, with structured outputs for **disease prediction, explainability, actionable guidance, and uncertainty** — making it suitable for training and evaluating a Human–AI collaborative decision-support model.

If you want, I can also make a **visual summary of the dataset** (distribution per language, per disease, etc.) to quickly understand its balance and diversity. Do you want me to do that?
