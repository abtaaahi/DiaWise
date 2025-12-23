# utils.py
from deep_translator import GoogleTranslator
import os

def translate_text(text, target="en"):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except:
        return text

def ensure_outputs_folder():
    os.makedirs("outputs", exist_ok=True)
