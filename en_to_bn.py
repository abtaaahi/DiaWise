import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

# === CONFIG ===
DATA_PATH = "synthetic_medical_dataset_50x500.csv"
SAVE_PATH = "augmented_multilingual_dataset.csv"
SOURCE_LANG = "en"
TARGET_LANG = "bn"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# Ensure 'language' column exists
if "language" not in df.columns:
    raise ValueError("❌ 'language' column not found in CSV. Add it before running.")

# === FILTER ONLY ENGLISH ROWS ===
english_rows = df[df["language"] == SOURCE_LANG].copy()
print(f"Found {len(english_rows)} English rows to translate...")

# === TRANSLATOR INIT ===
translator = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)

translated_rows = []
next_id = df["id"].max() + 1 if "id" in df.columns else 1

# === TRANSLATE ENGLISH → BANGLA ===
for _, row in tqdm(english_rows.iterrows(), total=len(english_rows)):
    try:
        # Core translations
        input_text_bn = translator.translate(row["input_text"])
        reasoning_bn = translator.translate(row["reasoning_keywords"])
        recommendations_bn = translator.translate(row["recommendations"])

        new_row = row.copy()
        new_row["id"] = next_id
        next_id += 1

        # Replace texts with translated Bangla
        new_row["input_text"] = input_text_bn
        new_row["reasoning_keywords"] = reasoning_bn
        new_row["recommendations"] = recommendations_bn
        new_row["language"] = TARGET_LANG

        translated_rows.append(new_row)

    except Exception as e:
        print(f"⚠️ Error translating row {row['id']}: {e}")
        continue

# === APPEND & SAVE ===
if translated_rows:
    translated_df = pd.DataFrame(translated_rows)
    augmented_df = pd.concat([df, translated_df], ignore_index=True)
    augmented_df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Translation complete! Added {len(translated_df)} Bangla rows.")
    print(f"💾 Saved new dataset to: {SAVE_PATH}")
else:
    print("❌ No translations added.")
