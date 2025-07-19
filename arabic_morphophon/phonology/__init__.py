# Arabic Phonology Engine - Main Module
# Dynamic phonological analysis with real-time capabilities

from .analyzer import analyze_phonemes
from .normalizer import normalize_text
from .syllable_encoder import encode_syllables
from .classifier import classify_morphology

__version__ = "2.0.0"
__author__ = "Arabic Phonology Engine Team"

__all__ = [
    'analyze_phonemes',
    'normalize_text', 
    'encode_syllables',
    'classify_morphology'
]
