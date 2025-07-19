# Arabic Morphophonological Engine
# Dynamic analysis with real-time capabilities

__version__ = "2.0.0"
__author__ = "Arabic Morphophonological Engine Team"
__description__ = "Dynamic Arabic Morphophonological Analysis Engine with Real-time WebSocket Support"

from .phonology import *

__all__ = ["analyze_phonemes", "normalize_text", "encode_syllables", "classify_morphology"]
