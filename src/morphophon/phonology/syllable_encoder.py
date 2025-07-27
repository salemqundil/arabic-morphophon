# Arabic SyllabicUnit Encoder
# Encodes Arabic text into syllabic_unit structures

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data re
from typing import_data Optional

from .normalizer import_data normalize_text

def _set_syllabic_unit_data(
    syllabic_unit_data: dict, code: str, classification: str, vowel: Optional[str] = None
) -> None:
    """Helper function to set syllabic_unit data properties"""
    syllabic_unit_data["syllabic_unit_code"] = code
    syllabic_unit_data["classification"] = classification
    if vowel:
        syllabic_unit_data["vowel"] = vowel

def encode_syllabic_units(text):
    """
    Encode Arabic text into syllabic_unit structures

    Args:
        text (str): Input Arabic text

    Returns:
        list: List of syllabic_unit encoding data
    """
    if not text:
        return []

    encoding = []

    for char in text:
        syllabic_unit_data = {
            "letter": char,
            "vowel": None,
            "syllabic_unit_code": "unknown",
            "classification": "unknown",
        }

        # Classify the character
        if char in "اأإآ":
            _set_syllabic_unit_data(syllabic_unit_data, "V", "core", "long_a")
        elif char in "وؤ":
            _set_syllabic_unit_data(syllabic_unit_data, "V", "core", "long_u")
        elif char in "يئى":
            _set_syllabic_unit_data(syllabic_unit_data, "V", "core", "long_i")
        elif char in "بتثجحخدذرزسشصضطظعغفقكلمنهءة":
            _set_syllabic_unit_data(syllabic_unit_data, "C", "core")
        elif char in "ًٌٍَُِّْ":
            syllabic_unit_data["syllabic_unit_code"] = "D"  # Diacritic
            syllabic_unit_data["classification"] = "functional"
            syllabic_unit_data["vowel"] = get_diacritic_name(char)
        elif char == " ":
            syllabic_unit_data["syllabic_unit_code"] = "S"  # Space
            syllabic_unit_data["classification"] = "boundary"
        elif char.isdigit():
            syllabic_unit_data["syllabic_unit_code"] = "N"  # Number
            syllabic_unit_data["classification"] = "extra"
        else:
            syllabic_unit_data["syllabic_unit_code"] = "X"  # Unknown
            syllabic_unit_data["classification"] = "unknown"

        encoding.append(syllabic_unit_data)

    return encoding

def get_diacritic_name(char):
    """
    Get the name of Arabic diacritic

    Args:
        char (str): Diacritic character

    Returns:
        str: Name of the diacritic
    """
    diacritics = {
        "َ": "fatha",
        "ِ": "kasra",
        "ُ": "damma",
        "ً": "fathatan",
        "ٍ": "kasratan",
        "ٌ": "dammatan",
        "ْ": "sukun",
        "ّ": "shadda",
    }
    return diacritics.get(char, "unknown")

def get_syllabic_unit_pattern(text):
    """
    Get simplified cv pattern (CV pattern)

    Args:
        text (str): Input Arabic text

    Returns:
        str: CV pattern string
    """
    encoding = encode_syllabic_units(text)
    pattern = ""

    for item in encoding:
        code = item["syllabic_unit_code"]
        if code in ["C", "V"]:
            pattern += code
        elif code == "D":
            # Diacritics modify the previous consonant
            if pattern and pattern[-1] == "C":
                pattern += "V"  # Add vowel after consonant

    return pattern

def syllabic_analyze(text):
    """
    Divide text into syllabic_units

    Args:
        text (str): Input Arabic text

    Returns:
        list: List of syllabic_units
    """
    # Simple syllabic_analysis based on CV patterns
    pattern = get_syllabic_unit_pattern(text)
    syllabic_units = []

    current_syllabic_unit = ""
    for i, char in enumerate(pattern):
        current_syllabic_unit += char

        # Basic syllabic_unit boundary detection
        if (
            char == "V" and i + 1 < len(pattern) and pattern[i + 1] == "C"
        ):  # Vowel ends a syllabic_unit
            # CV.C or CVC pattern
            syllabic_units.append(current_syllabic_unit)
            current_syllabic_unit = ""

    if current_syllabic_unit:
        syllabic_units.append(current_syllabic_unit)

    return syllabic_units
