# phonology/normalizer.py

import re

DIACRITICS = re.compile(r"[\u064B-\u0652]")
TATWEEL = "\u0640"
ALIF_VARIANTS = {"إ": "إ", "أ": "أ", "آ": "آ", "ى": "ى", "ئ": "ئ", "ؤ": "ؤ"}


def normalize_text(text: str) -> str:
    # Remove tatweel
    text = text.replace(TATWEEL, "")

    # Normalize Alif variants
    for form, base in ALIF_VARIANTS.items():
        text = text.replace(form, base)

    # Remove diacritics
    text = DIACRITICS.sub("", text)

    return text


def detect_shadda(char: str) -> bool:
    return "\u0651" in char


def detect_sukun(char: str) -> bool:
    return "\u0652" in char


def detect_madd(text: str) -> list:
    # Detect long vowels
    result = []
    for i, c in enumerate(text):
        if c in {"ا", "و", "ي"}:
            result.append((i, c))
    return result
