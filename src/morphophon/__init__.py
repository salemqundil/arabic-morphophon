# Arabic Morphophonological Engine
# Dynamic analysis with real-time capabilities

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

__version__ = "2.0.0"
__author__ = "Arabic Morphophonological Engine Team"
__description__ = (
    "Dynamic Arabic Morphophonological Analysis Engine with Real-time WebSocket Support"
)

from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic

__all__ = [
    "analyze_phonemes",
    "normalize_text",
    "encode_syllabic_units",
    "classify_morphology",
]
