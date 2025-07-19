"""
Modular Phoneme Mapping System
Separates phoneme definitions for easier maintenance and extensibility
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class PhonemeType(Enum):
    VOWEL = "V"
    CONSONANT = "C"


class ScriptType(Enum):
    LATIN = "Latin"
    ARABIC = "Arabic"
    MIXED = "Mixed"


@dataclass
class PhonemeDefinition:
    """Structured phoneme definition with metadata"""

    char: str
    type: PhonemeType
    acoustic_weight: float
    script: ScriptType
    articulatory_features: Dict[str, str]

    def to_mapping_dict(self) -> Dict[str, Any]:
        """Convert to the format expected by the web API"""
        return {"type": self.type.value, "acoustic_weight": self.acoustic_weight}


class PhonemeRegistry:
    """Centralized registry for phoneme definitions"""

    def __init__(self):
        self._phonemes: Dict[str, PhonemeDefinition] = {}
        self._initialize_default_phonemes()

    def _initialize_default_phonemes(self):
        """Initialize with default phoneme set"""
        # Latin vowels
        self.register_phoneme(
            "a",
            PhonemeType.VOWEL,
            1.0,
            ScriptType.LATIN,
            {"manner": "vowel", "place": "central", "voicing": "voiced"},
        )
        self.register_phoneme(
            "e",
            PhonemeType.VOWEL,
            0.9,
            ScriptType.LATIN,
            {"manner": "vowel", "place": "front", "voicing": "voiced"},
        )
        self.register_phoneme(
            "i",
            PhonemeType.VOWEL,
            0.8,
            ScriptType.LATIN,
            {"manner": "vowel", "place": "front", "voicing": "voiced"},
        )
        self.register_phoneme(
            "o",
            PhonemeType.VOWEL,
            1.1,
            ScriptType.LATIN,
            {"manner": "vowel", "place": "back", "voicing": "voiced"},
        )
        self.register_phoneme(
            "u",
            PhonemeType.VOWEL,
            0.7,
            ScriptType.LATIN,
            {"manner": "vowel", "place": "back", "voicing": "voiced"},
        )

        # Arabic vowels
        self.register_phoneme(
            "ا",
            PhonemeType.VOWEL,
            1.2,
            ScriptType.ARABIC,
            {
                "manner": "vowel",
                "place": "central",
                "voicing": "voiced",
                "length": "long",
            },
        )
        self.register_phoneme(
            "و",
            PhonemeType.VOWEL,
            1.0,
            ScriptType.ARABIC,
            {"manner": "vowel", "place": "back", "voicing": "voiced", "length": "long"},
        )
        self.register_phoneme(
            "ي",
            PhonemeType.VOWEL,
            0.9,
            ScriptType.ARABIC,
            {
                "manner": "vowel",
                "place": "front",
                "voicing": "voiced",
                "length": "long",
            },
        )

        # Latin consonants
        self.register_phoneme(
            "p",
            PhonemeType.CONSONANT,
            0.5,
            ScriptType.LATIN,
            {"manner": "stop", "place": "bilabial", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "b",
            PhonemeType.CONSONANT,
            0.5,
            ScriptType.LATIN,
            {"manner": "stop", "place": "bilabial", "voicing": "voiced"},
        )
        self.register_phoneme(
            "t",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.LATIN,
            {"manner": "stop", "place": "alveolar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "d",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.LATIN,
            {"manner": "stop", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "k",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.LATIN,
            {"manner": "stop", "place": "velar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "g",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.LATIN,
            {"manner": "stop", "place": "velar", "voicing": "voiced"},
        )

        # Arabic consonants (core set)
        self.register_phoneme(
            "ب",
            PhonemeType.CONSONANT,
            0.5,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "bilabial", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ت",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "alveolar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ث",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "dental", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ج",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.ARABIC,
            {"manner": "affricate", "place": "postalveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ح",
            PhonemeType.CONSONANT,
            0.3,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "pharyngeal", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "خ",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "uvular", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "د",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ذ",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "dental", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ر",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.ARABIC,
            {"manner": "trill", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ز",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "س",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "alveolar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ش",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "postalveolar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ص",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {
                "manner": "fricative",
                "place": "alveolar_pharyngealized",
                "voicing": "voiceless",
            },
        )
        self.register_phoneme(
            "ض",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "alveolar_pharyngealized", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ط",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.ARABIC,
            {
                "manner": "stop",
                "place": "alveolar_pharyngealized",
                "voicing": "voiceless",
            },
        )
        self.register_phoneme(
            "ظ",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {
                "manner": "fricative",
                "place": "alveolar_pharyngealized",
                "voicing": "voiced",
            },
        )
        self.register_phoneme(
            "ع",
            PhonemeType.CONSONANT,
            0.5,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "pharyngeal", "voicing": "voiced"},
        )
        self.register_phoneme(
            "غ",
            PhonemeType.CONSONANT,
            0.5,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "uvular", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ف",
            PhonemeType.CONSONANT,
            0.4,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "labiodental", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ق",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "uvular", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ك",
            PhonemeType.CONSONANT,
            0.7,
            ScriptType.ARABIC,
            {"manner": "stop", "place": "velar", "voicing": "voiceless"},
        )
        self.register_phoneme(
            "ل",
            PhonemeType.CONSONANT,
            0.6,
            ScriptType.ARABIC,
            {"manner": "lateral", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "م",
            PhonemeType.CONSONANT,
            0.8,
            ScriptType.ARABIC,
            {"manner": "nasal", "place": "bilabial", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ن",
            PhonemeType.CONSONANT,
            0.8,
            ScriptType.ARABIC,
            {"manner": "nasal", "place": "alveolar", "voicing": "voiced"},
        )
        self.register_phoneme(
            "ه",
            PhonemeType.CONSONANT,
            0.3,
            ScriptType.ARABIC,
            {"manner": "fricative", "place": "glottal", "voicing": "voiceless"},
        )

    def register_phoneme(
        self,
        char: str,
        phoneme_type: PhonemeType,
        acoustic_weight: float,
        script: ScriptType,
        articulatory_features: Dict[str, str],
    ):
        """Register a new phoneme definition"""
        self._phonemes[char] = PhonemeDefinition(
            char=char,
            type=phoneme_type,
            acoustic_weight=acoustic_weight,
            script=script,
            articulatory_features=articulatory_features,
        )

    def get_phoneme(self, char: str) -> PhonemeDefinition:
        """Get phoneme definition for character"""
        return self._phonemes.get(char)

    def has_phoneme(self, char: str) -> bool:
        """Check if character has phoneme definition"""
        return char in self._phonemes

    def get_mapping_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get phoneme mapping in web API format"""
        return {
            char: phoneme.to_mapping_dict() for char, phoneme in self._phonemes.items()
        }

    def get_phonemes_by_script(
        self, script: ScriptType
    ) -> Dict[str, PhonemeDefinition]:
        """Get all phonemes for a specific script"""
        return {
            char: phoneme
            for char, phoneme in self._phonemes.items()
            if phoneme.script == script
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        script_counts = {}
        type_counts = {}

        for phoneme in self._phonemes.values():
            script_counts[phoneme.script.value] = (
                script_counts.get(phoneme.script.value, 0) + 1
            )
            type_counts[phoneme.type.value] = type_counts.get(phoneme.type.value, 0) + 1

        return {
            "total_phonemes": len(self._phonemes),
            "by_script": script_counts,
            "by_type": type_counts,
            "supported_characters": list(self._phonemes.keys()),
        }


# Global phoneme registry instance
phoneme_registry = PhonemeRegistry()
