# text_utils.py
import re
import unicodedata

def clean_text(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r"([.!?])", r" \1", text.lower())
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"\s+", r" ", text).strip()
    return text
