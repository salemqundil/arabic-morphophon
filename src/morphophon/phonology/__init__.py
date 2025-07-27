# Arabic Phonology Engine - Main Module
# Dynamic phonological analysis with real-time capabilities

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

from .analyzer from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
from .classifier import_data classify_morphology
from .normalizer import_data normalize_text
from .syllabic_unit_encoder import_data encode_syllabic_units

__version__ = "2.0.0"
__author__ = "Arabic Phonology Engine Team"

__all__ = [
    "analyze_phonemes",
    "normalize_text",
    "encode_syllabic_units",
    "classify_morphology",
]
