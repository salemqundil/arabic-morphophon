from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Phoneme:
    char: str  # Surface form: 'p', 'a', 't'
    meta: Dict[str, Any]  # Linguistic features

    def __eq__(self, other: object) -> bool:
        """Safe equality comparison with enhanced type checking"""
        if not isinstance(other, Phoneme):
            return False
        try:
            # Safe comparison of char attributes
            chars_equal = self.char == other.char
            # Safe comparison of meta dictionaries with None checking
            self_meta = self.meta if self.meta is not None else {}
            other_meta = other.meta if other.meta is not None else {}
            metas_equal = self_meta == other_meta
            return chars_equal and metas_equal
        except (AttributeError, TypeError):
            # Handle cases where attributes are missing or invalid
            return False


# @dataclass automatically provides:
# - __init__(self, char: str, meta: Dict[str, Any])
# - __repr__(self) -> str
# - Custom __eq__(self, other) -> bool with type safety
# - Type checking support

# Arabic letter classifications
CORE_LETTERS = set("جزشصضطظغ")
EXTRA_LETTERS = set("سألتمونيه")
FUNCTIONAL_LETTERS = set("بكفذجثحرخع")
WEAK_LETTERS = set("اوي")

# Arabic vowel marks and characters
ARABIC_VOWELS = set("اوي" + "َُِ" + "آأإ")  # Long and short vowels
ARABIC_CONSONANTS = set("بتثجحخدذرزسشصضطظعغفقكلمنهـ")

# Latin vowels for romanized text
LATIN_VOWELS = set("aeiouAEIOU")


def is_vowel(char: str) -> bool:
    """Determine if character is a vowel (Arabic or Latin)"""
    return char in ARABIC_VOWELS or char in LATIN_VOWELS


def get_phoneme_type(char: str) -> str:
    """Get phoneme type with Arabic script support"""
    if is_vowel(char):
        return "V"  # Vowel
    elif char in ARABIC_CONSONANTS or char.isalpha():
        return "C"  # Consonant
    else:
        return "C"  # Default to consonant for unknown characters


def classify_letter(letter: str) -> str:
    """Safe letter classification with defensive programming"""
    try:
        if letter in CORE_LETTERS:
            return "core"
        elif letter in EXTRA_LETTERS:
            return "extra"
        elif letter in FUNCTIONAL_LETTERS:
            return "functional"
        elif letter in WEAK_LETTERS:
            return "weak"
        else:
            return "unknown"
    except (TypeError, AttributeError):
        # Handle cases where letter is None or not a valid type
        return "unknown"


# Example Phoneme instances:
vowel = Phoneme(
    char="a",
    meta={
        "type": "V",  # Vowel
        "acoustic_weight": 1.0,  # Prominence
        "geminated": False,  # Not doubled
    },
)

consonant_p = Phoneme(
    char="p",
    meta={
        "type": "C",  # Consonant classification
        "acoustic_weight": 0.5,  # Lower prominence
        "geminated": False,  # Single length
    },
)

# Can add any phonological feature:
meta = {
    "type": "C",
    "manner": "fricative",  # Manner of articulation
    "place": "alveolar",  # Place of articulation
    "voiced": True,  # Voicing feature
    "acoustic_weight": 0.4,  # Perceptual weight
    "geminated": False,  # Length feature
    "syllabic": False,  # Syllable nucleus potential
}

# Example Phoneme instance for testing
example_vowel = Phoneme(
    char="a",
    meta={
        "type": "V",  # Vowel classification
        "acoustic_weight": 1.0,  # High prominence
        "geminated": False,  # Single length
    },
)
