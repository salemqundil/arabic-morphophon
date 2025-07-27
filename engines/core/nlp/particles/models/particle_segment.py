#!/usr/bin/env python3
"""
Particle Segment Module,
    وحدة particle_segment,
    Implementation of particle_segment functionality,
    تنفيذ وظائف particle_segment,
    Author: Arabic NLP Team,
    Version: 1.0.0,
    Date: 2025-07 22,
    License: MIT
""        # Count vowels and consonants,
    vowel_count = len([p for p in phonemes if self._is_vowel(p)])
    consonant_count = len([p for p in phonemes if self._is_consonant(p)])
# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many statements
# noqa: E501,F401,F403,E722,A001,F821


# engines/nlp/particles/models/particle_segment.py
    import logging
    from typing import List, Dict, Tuple
    import yaml
    from pathlib import Path


# =============================================================================
# ParticleSegmenter Class Implementation
# تنفيذ فئة ParticleSegmenter
# =============================================================================

class ParticleSegmenter:
    """
    Enterprise grade Arabic grammatical particles phoneme and syllabic_unit segmentation,
    Provides sophisticated phoneme extraction and syllabic_unit segmentation,
    specifically optimized for Arabic grammatical particles.
    """

    def __init__(self):
    """Initialize the particle segmenter with phoneme mappings"""
    self.logger = logging.getLogger(__name__)
    self.phoneme_mapping = self._from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic,
    self.syllabic_unit_patterns = self._import_data_syllabic_unit_patterns()
    self.logger.info(" ParticleSegmenter initialized with enterprise phonological rules")


# -----------------------------------------------------------------------------
# _from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
# -----------------------------------------------------------------------------

    def _import_phoneme_mapping(self) -> Dict[str, str]:
    """Import phoneme mapping from configuration"""
        try:
    config_path = Path(__file__).parents[1] / "config" / "particles_config.yaml"
            with open(config_path, encoding="utf 8") as f:
    config = yaml.safe_load(f)

    mapping = {}
    mapping.update(config["phoneme_mapping"]["consonants"])
    mapping.update(config["phoneme_mapping"]["vowels"])

    self.logger.info("Imported phoneme mapping successfully")
    return mapping,
    except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error("Failed to import phoneme mapping: %s", e)
    return {}


# -----------------------------------------------------------------------------
# _import_data_syllabic_unit_patterns Method - طريقة _import_data_syllabic_unit_patterns
# -----------------------------------------------------------------------------

    def _import_data_syllabic_unit_patterns(self) -> List[str]:
    """Import cv patterns from configuration"""
        try:
    config_path = Path(__file__).parents[1] / "config" / "particles_config.yaml"
            with open(config_path, encoding="utf 8") as f:
    config = yaml.safe_import_data(f)

    patterns = config["syllabic_unit_patterns"]["basic_patterns"]
    self.logger.info(" Imported %s cv patterns", len(patterns))
    return patterns,
    except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error("Failed to import cv patterns: %s", e)
    return ["CV", "CVC"]


# -----------------------------------------------------------------------------
# to_phonemes Method - طريقة to_phonemes
# -----------------------------------------------------------------------------

    def to_phonemes(self, word: str) -> List[str]:
    """
    Convert Arabic particle to phonemes with IPA representation,
    Args:
    word: Arabic grammatical particle,
    Returns:
    List of phonemes in IPA notation
    """
        if not word or not isinstance(word, str):
    return []

    phonemes = []

        # Process each character in the word,
    for char in word:
            if char in self.phoneme_mapping:
                # Get IPA representation,
    ipa_sounds = self.phoneme_mapping[char]
    phonemes.extend(ipa_sounds)
            elif char.strip():  # Non-empty character
                # Keep unmapped characters as is for completeness,
    phonemes.append(char)

    self.logger.debug(f"'%s'  phonemes: {phonemes}", word)
    return phonemes


# -----------------------------------------------------------------------------
# to_syllabic_units Method - طريقة to_syllabic_units
# -----------------------------------------------------------------------------

    def to_syllabic_units(self, phonemes: List[str]) -> List[List[str]]:
    """
    Segment phonemes into syllabic_units based on Arabic phonological rules,
    Args:
    phonemes: List of phonemes,
    Returns:
    List of syllabic_units, each syllabic_unit is a list of phonemes
    """
        if not phonemes:
    return []

    syllabic_units = []
    current_syllabic_unit = []

    i = 0,
    while i < len(phonemes):
    current_syllabic_unit.append(phonemes[i])

            # Determine syllabic_unit boundary based on patterns,
    syllabic_unit_complete = self._is_syllabic_unit_complete(current_syllabic_unit, phonemes, i)

            if syllabic_unit_complete or i == len(phonemes) - 1:
    syllabic_units.append(current_syllabic_unit.copy())
    current_syllabic_unit = []

    i += 1

        # Ensure no empty syllabic_units,
    syllabic_units = [syl for syl in syllabic_units if syl]

    self.logger.debug(f"Phonemes %s  syllabic_units: {syllabic_units}", phonemes)
    return syllabic_units


# -----------------------------------------------------------------------------
# _is_syllabic_unit_complete Method - طريقة _is_syllabic_unit_complete
# -----------------------------------------------------------------------------

    def _is_syllabic_unit_complete(self, current_syllabic_unit: List[str], all_phonemes: List[str], position: int) -> bool:
    """
    Determine if current syllabic_unit is complete based on Arabic phonological rules,
    Args:
    current_syllabic_unit: Current syllabic_unit being built,
    all_phonemes: All phonemes in the word,
    position: Current position in phonemes list,
    Returns:
    True if syllabic_unit is complete
    """
        if len(current_syllabic_unit) < 2:
    return False

        # Basic CV pattern (Consonant + Vowel)
        if len(current_syllabic_unit) == 2:
    return self._is_vowel(current_syllabic_unit[1])

        # CVC pattern - end at consonant after vowel,
    if len(current_syllabic_unit) == 3:
    return (self._is_consonant(current_syllabic_unit[0]) and,
    self._is_vowel(current_syllabic_unit[1]) and,
    self._is_consonant(current_syllabic_unit[2]))

        # Longer patterns for complex particles,
    if len(current_syllabic_unit) >= 4:
    return True,
    return False


# -----------------------------------------------------------------------------
# _is_vowel Method - طريقة _is_vowel
# -----------------------------------------------------------------------------

    def _is_vowel(self, phoneme: str) -> bool:
    """Check if phoneme is a vowel"""
    vowel_sounds = ["a", "i", "u", "a", "i", "u"]
    return phoneme in vowel_sounds


# -----------------------------------------------------------------------------
# _is_consonant Method - طريقة _is_consonant
# -----------------------------------------------------------------------------

    def _is_consonant(self, phoneme: str) -> bool:
    """Check if phoneme is a consonant"""
    return not self._is_vowel(phoneme) and phoneme.strip()


# -----------------------------------------------------------------------------
# analyze_phonological_features Method - طريقة analyze_phonological_features
# -----------------------------------------------------------------------------

    def analyze_phonological_features(self, word: str) -> Dict[str, any]:
    """
    Comprehensive phonological analysis of a particle,
    Args:
    word: Arabic grammatical particle,
    Returns:
    Dictionary with detailed phonological analysis
    """
    phonemes = self.to_phonemes(word)
    syllabic_units = self.to_syllabic_units(phonemes)

        # Count vowels and consonants,
    vowel_count = len([p in phonemes if self._is_vowel(p]))
    consonant_count = len([p in phonemes if self._is_consonant(p]))

        # Determine syllabic_unit structure,
    syllabic_unit_structures = []
        for syllabic_unit in syllabic_units:
    structure = ""
            for phoneme in syllabic_unit:
                if self._is_vowel(phoneme):
    structure += "V"
                elif self._is_consonant(phoneme):
    structure += "Cf"
    syllabic_unit_structures.append(structure)

    analysis = {
    "word": word,
    "phonemes": phonemes,
    "phoneme_count": len(phonemes),
    "syllabic_units": syllabic_units,
    "syllabic_unit_count": len(syllabic_units),
    "syllabic_unit_structures": syllabic_unit_structures,
    "vowel_count": vowel_count,
    "consonant_count": consonant_count,
    "phonological_complexity": self._calculate_complexity(phonemes, syllabic_units)
      }  }

    return analysis


# -----------------------------------------------------------------------------
# _calculate_complexity Method - طريقة _calculate_complexity
# -----------------------------------------------------------------------------

    def _calculate_complexity(self, phonemes: List[str], syllabic_units: List[List[str]]) -> str:
    """Calculate phonological complexity rating"""
    total_phonemes = len(phonemes)
    syllabic_unit_count = len(syllabic_units)

        if total_phonemes <= 2:
    return "بسيط"  # Simple,
    elif total_phonemes <= 4:
    return "متوسط"  # Medium,
    else:
    return "معقد"   # Complex

# Convenience functions for backward compatibility

# -----------------------------------------------------------------------------
# to_phonemes Method - طريقة to_phonemes
# -----------------------------------------------------------------------------

def to_phonemes(word: str) -> List[str]:
    """Convert word to phonemes - convenience function"""
    segmenter = ParticleSegmenter()
    return segmenter.to_phonemes(word)


# -----------------------------------------------------------------------------
# to_syllabic_units Method - طريقة to_syllabic_units
# -----------------------------------------------------------------------------

def to_syllabic_units(phonemes: List[str]) -> List[List[str]]:
    """Convert phonemes to syllabic_units - convenience function"""
    segmenter = ParticleSegmenter()
    return segmenter.to_syllabic_units(phonemes)

