"""
Arabic Phonology Engine

A comprehensive Arabic phonology analysis system with zero-tolerance
for encoding issues and expert-level linguistic accuracy.
"""

__version__ = "1.0.0"
__author__ = "Arabic Phonology Team"

from .core.engine import ArabicPhonologyEngine
from .core.phoneme import Phoneme
from .data.phoneme_db import PhonemeDatabase

__all__ = [
    "ArabicPhonologyEngine",
    "Phoneme", 
    "PhonemeDatabase"
]
