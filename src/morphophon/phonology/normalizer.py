# Arabic Text Normalizer
# Normalizes Arabic text for phonological analysis

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data re

def normalize_text(text):
    """
    Normalize Arabic text by removing unwanted characters and standardizing format

    Args:
        text (str): Input Arabic text

    Returns:
        str: Normalized Arabic text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Normalize Arabic letters
    # Replace different forms of Alif
    text = re.sub(r"[إأآا]", "ا", text)

    # Replace different forms of Taa Marboota
    text = re.sub(r"[ة]", "ه", text)

    # Replace different forms of Yaa
    text = re.sub(r"[ىي]", "ي", text)

    # Remove Tatweel (Arabic elongation)
    text = re.sub(r"ـ", "", text)

    # Remove common punctuation but keep Arabic punctuation
    text = re.sub(r'[.,;:!?()[\]{}"]', "", text)

    return text.strip()

def remove_diacritics(text):
    """
    Remove Arabic diacritics from text

    Args:
        text (str): Input Arabic text with diacritics

    Returns:
        str: Text without diacritics
    """
    # Arabic diacritics Unicode range
    diacritics = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
    return diacritics.sub("", text)

def extract_diacritics(text):
    """
    Extract diacritics from Arabic text

    Args:
        text (str): Input Arabic text

    Returns:
        list: List of diacritics found in the text
    """
    diacritics = re.findall(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", text)
    return diacritics
