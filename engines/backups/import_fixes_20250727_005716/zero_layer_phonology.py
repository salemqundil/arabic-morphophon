#!/usr/bin/env python3
"""
Zero Layer: Phonology Engine for Arabic NLP
============================================
Enterprise-Grade Phonological Analysis and Classification System
Professional implementation following Python best practices and Arabic linguistic standards

This layer serves as the foundation for all higher-level Arabic NLP processing:
- Extracts phonemes and diacritics with linguistic precision
- Classifies phonemes as: Root (جذرية), Affixal (زائدة), Functional (وظيفية)
- Prepares phonological units for syllable construction
- Provides vector representation for machine learning pipelines

Author: Arabic NLP Expert Team
Version: 1.0.0
Date: 2025-07 23
License: MIT
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for professional codebase
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too long

import logging  # noqa: F401
import re  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from enum import Enum  # noqa: F401
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path  # noqa: F401


class PhonemeClassification(Enum):
    """Enumeration for Arabic phoneme classification types"""

    ROOT = "root"  # جذرية - Root consonants that carry lexical meaning
    AFFIXAL = "affixal"  # زائدة - Affixal morphemes (prefixes, suffixes, infixes)
    FUNCTIONAL = "functional"  # وظيفية - Functional elements (particles, connectives)
    UNKNOWN = "unknown"  # غير معروف - Unclassified or foreign elements


class HarakaClassification(Enum):
    """Enumeration for Arabic diacritical mark classification"""

    SHORT_VOWELS = "short_vowels"  # الحركات القصيرة
    LONG_VOWELS = "long_vowels"  # الحركات الطويلة
    SUKUN = "sukun"  # السكون
    SHADDA = "shadda"  # الشدة
    TANWIN = "tanwin"  # التنوين
    MADD = "madd"  # المد
    NONE = "none"  # بدون حركة


@dataclass
class PhonologicalUnit:
    """
    Represents a single phonological unit in Arabic text
    وحدة صوتية واحدة في النص العربي
    """

    phoneme: str  # الفونيم الأساسي
    haraka: str  # الحركة المرفقة
    phoneme_class: PhonemeClassification  # تصنيف الفونيم
    haraka_class: HarakaClassification  # تصنيف الحركة
    position: int  # الموقع في الكلمة
    features: Dict[str, Any] = field(default_factory=dict)  # خصائص إضافية

    def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation for serialization"""
    return {
    'phoneme': self.phoneme,
    'haraka': self.haraka,
    'phoneme_class': self.phoneme_class.value,
    'haraka_class': self.haraka_class.value,
    'position': self.position,
    'features': self.features,
    }

    def __str__(self) -> str:
    """String representation for debugging"""
    return f"{self.phoneme}{self.haraka} [{self.phoneme_class.value}]"


@dataclass
class PhonologicalAnalysis:
    """
    Complete phonological analysis result for an Arabic word
    نتيجة التحليل الصوتي الكامل للكلمة العربية
    """

    word: str  # الكلمة الأصلية
    units: List[PhonologicalUnit]  # الوحدات الصوتية
    root_phonemes: List[PhonologicalUnit]  # الفونيمات الجذرية
    affixal_phonemes: List[PhonologicalUnit]  # الفونيمات الزائدة
    functional_phonemes: List[PhonologicalUnit]  # الفونيمات الوظيفية
    statistics: Dict[str, int] = field(default_factory=dict)  # إحصائيات التحليل
    confidence: float = 0.0  # مستوى الثقة

    def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation"""
    return {
    'word': self.word,
    'units': [unit.to_dict() for unit in self.units],
    'root_phonemes': [unit.to_dict() for unit in self.root_phonemes],
    'affixal_phonemes': [unit.to_dict() for unit in self.affixal_phonemes],
    'functional_phonemes': [
    unit.to_dict() for unit in self.functional_phonemes
    ],
    'statistics': self.statistics,
    'confidence': self.confidence,
    }


class ZeroLayerPhonologyEngine:
    """
    Zero Layer: Professional Arabic Phonology Engine
    المستوى الصفر: محرك الصوتيات العربية المهني

    Enterprise-grade phonological analysis system that serves as the foundation
    for all higher level Arabic NLP processing layers.
    """

    def __init__(self, config_path: Optional[Path] = None):  # type: ignore[no-untyped def]
    """Initialize the Zero Layer Phonology Engine"""
    self.logger = logging.getLogger('ZeroLayerPhonology')
    self._setup_logging()

        # Load configuration
    self.config = self._load_config(config_path)

        # Initialize phoneme classification mappings
    self._initialize_phoneme_mappings()

        # Initialize haraka classification mappings
    self._initialize_haraka_mappings()

        # Initialize processing statistics
    self.statistics = {
    'words_processed': 0,
    'units_extracted': 0,
    'classification_accuracy': 0.0,
    }

    self.logger.info("🎯 Zero Layer Phonology Engine initialized successfully")

    def _setup_logging(self) -> None:
    """Configure logging for the engine"""
        if not self.logger.handlers:
    handler = logging.StreamHandler()
            formatter = logging.Formatter()
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    self.logger.addHandler(handler)
    self.logger.setLevel(logging.INFO)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
    """Load engine configuration"""
        default_config = {
    'enable_advanced_classification': True,
    'strict_haraka_validation': True,
    'confidence_threshold': 0.85,
    'enable_statistics': True,
    }

        if config_path and config_path.exists():
            # Load from file if provided
            # Implementation for loading JSON/YAML config
    pass

    return default_config

    def _initialize_phoneme_mappings(self) -> None:
    """Initialize Arabic phoneme classification mappings"""

        # ROOT PHONEMES (جذرية) - Core semantic-bearing consonants
    self.root_phonemes = {
            # Core stops and fricatives
    'ب',
    'ت',
    'ث',
    'ج',
    'ح',
    'خ',
    'د',
    'ذ',
    'ر',
    'ز',
    'س',
    'ش',
    'ص',
    'ض',
    'ط',
    'ظ',
    'ع',
    'غ',
    'ف',
    'ق',
    'ك',
    'ل',
    'م',
    'ن',
    'ه',
    }

        # AFFIXAL PHONEMES (زائدة) - Morphological markers
    self.affixal_phonemes = {
            # Long vowels and glides - حروف المد
    'ا',  # ألف المد - Long vowel marker
    'ي',  # ياء المد - Long vowel marker
    'و',  # واو المد - Long vowel marker
    }

        # FUNCTIONAL PHONEMES (وظيفية) - Grammatical particles
    self.functional_phonemes = {
    'ء',  # همزة القطع
    'آ',  # آ المدة
    'أ',  # همزة على ألف
    'إ',  # همزة تحت ألف
    'ة',  # تاء مربوطة
    'ى',  # ألف مقصورة
    }

        # Additional classification for advanced analysis
    self.phoneme_features = {
            # Consonant features - الخصائص الصوتية للصوامت
    'stops': {'ب', 'ت', 'د', 'ك', 'ق', 'ط', 'ض', 'ج'},  # الأصوات الانفجارية
    'fricatives': {
    'ف',
    'ث',
    'ذ',
    'س',
    'ز',
    'ش',
    'ص',
    'ظ',
    'ح',
    'خ',
    'ع',
    'غ',
    'ه',
    },  # الأصوات الاحتكاكية
    'nasals': {'م', 'ن'},  # الأصوات الأنفية
    'liquids': {'ل', 'ر'},  # الأصوات السائلة
    'glides': {
    'و',
    'ي',
    },  # الأصوات شبه الصوتية (Without ا - alif is vowel carrier)
    'pharyngealized': {'ص', 'ض', 'ط', 'ظ'},  # الأصوات المفخمة
    'emphatic': {'ص', 'ض', 'ط', 'ظ', 'ق', 'خ', 'غ'},  # الأصوات المطبقة
    'glottal': {'ء', 'ه'},  # الأصوات الحنجرية
    }

    def _initialize_haraka_mappings(self) -> None:
    """Initialize Arabic haraka (diacritics) classification mappings"""

        # Short vowels (الحركات القصيرة)
    self.short_vowels = {'َ', 'ِ', 'ُ'}

        # Long vowels (الحركات الطويلة) - contextual
    self.long_vowel_markers = {'ا', 'و', 'ي', 'ى'}

        # Sukun (السكون)
    self.sukun_markers = {'ْ'}

        # Shadda (الشدة)
    self.shadda_markers = {'ّ'}

        # Tanwin (التنوين)
    self.tanwin_markers = {'ً', 'ٌ', 'ٍ'}

        # Madd markers (علامات المد)
    self.madd_markers = {'ٰ', '~'}

        # Haraka features for linguistic analysis
    self.haraka_features = {
    'vowel_quality': {
    'َ': 'open',  # فتحة
    'ِ': 'close',  # كسرة
    'ُ': 'close',  # ضمة
    },
    'vowel_frontness': {
    'َ': 'central',  # فتحة
    'ِ': 'front',  # كسرة
    'ُ': 'back',  # ضمة
    },
    }

    def classify_phoneme()
    self, phoneme: str, context: Optional[str] = None
    ) -> PhonemeClassification:
    """
    Classify a phoneme based on its linguistic role
    تصنيف الفونيم حسب دوره اللغوي

    Args:
    phoneme: The Arabic phoneme to classify
    context: Optional word context for better classification

    Returns:
    PhonemeClassification enum value
    """
        if phoneme in self.root_phonemes:
    return PhonemeClassification.ROOT
        elif phoneme in self.affixal_phonemes:
            # Advanced contextual classification could be added here
    return PhonemeClassification.AFFIXAL
        elif phoneme in self.functional_phonemes:
    return PhonemeClassification.FUNCTIONAL
        else:
    return PhonemeClassification.UNKNOWN

    def classify_haraka(self, haraka: str) -> HarakaClassification:
    """
    Classify a haraka (diacritical mark)
    تصنيف الحركة

    Args:
    haraka: The Arabic haraka to classify

    Returns:
    HarakaClassification enum value
    """
        if haraka in self.short_vowels:
    return HarakaClassification.SHORT_VOWELS
        elif haraka in self.long_vowel_markers:
    return HarakaClassification.LONG_VOWELS
        elif haraka in self.sukun_markers:
    return HarakaClassification.SUKUN
        elif haraka in self.shadda_markers:
    return HarakaClassification.SHADDA
        elif haraka in self.tanwin_markers:
    return HarakaClassification.TANWIN
        elif haraka in self.madd_markers:
    return HarakaClassification.MADD
        else:
    return HarakaClassification.NONE

    def extract_phonemes_units(self, word: str) -> List[PhonologicalUnit]:
    """
    Extract phonological units from Arabic word
    استخراج الوحدات الصوتية من الكلمة العربية

    Args:
    word: Arabic word to analyze

    Returns:
    List of PhonologicalUnit objects
    """
    units = []
    position = 0

    i = 0
        while i < len(word):
    char = word[i]

            # Skip non Arabic characters
            if not self._is_arabic_character(char):
    i += 1
    continue

            # Extract phoneme
            if self._is_arabic_letter(char):
    phoneme = char
    haraka = ""

                # Collect following harakat
    j = i + 1
                while j < len(word) and self._is_haraka(word[j]):
    haraka += word[j]
    j += 1

                # Classify phoneme and haraka
    phoneme_class = self.classify_phoneme(phoneme, word)
    haraka_class = ()
    self.classify_haraka(haraka)
                    if haraka
                    else HarakaClassification.NONE
    )

                # Extract additional features
    features = self._extract_phoneme_features(phoneme, haraka)

                # Create phonological unit
    unit = PhonologicalUnit()
    phoneme=phoneme,
    haraka=haraka,
    phoneme_class=phoneme_class,
    haraka_class=haraka_class,
    position=position,
    features=features)

    units.append(unit)
    position += 1
    i = j
            else:
    i += 1

    return units

    def _extract_phoneme_features(self, phoneme: str, haraka: str) -> Dict[str, Any]:
    """Extract detailed phonological features"""
    features = {}

        # Consonant features
        for feature_type, phoneme_set in self.phoneme_features.items():
            if phoneme in phoneme_set:
    features[feature_type] = True

        # Haraka features
        if haraka and haraka in self.haraka_features.get('vowel_quality', {}):
    features['vowel_quality'] = self.haraka_features['vowel_quality'][haraka]

        if haraka and haraka in self.haraka_features.get('vowel_frontness', {}):
    features['vowel_frontness'] = self.haraka_features['vowel_frontness'][
    haraka
    ]

    return features

    def analyze(self, word: str) -> PhonologicalAnalysis:
    """
    Perform complete phonological analysis of Arabic word
    إجراء تحليل صوتي كامل للكلمة العربية

    Args:
    word: Arabic word to analyze

    Returns:
    PhonologicalAnalysis object with complete results
    """
    self.logger.info(f"🔬 Analyzing phonological structure of: {word}")

        # Extract phonological units
    units = self.extract_phonemes_units(word)

        # Classify units by type
    root_phonemes = [
    unit for unit in units if unit.phoneme_class == PhonemeClassification.ROOT
    ]
    affixal_phonemes = [
    unit
            for unit in units
            if unit.phoneme_class == PhonemeClassification.AFFIXAL
    ]
    functional_phonemes = [
    unit
            for unit in units
            if unit.phoneme_class == PhonemeClassification.FUNCTIONAL
    ]

        # Calculate statistics
    statistics = {
    'total_units': len(units),
    'root_count': len(root_phonemes),
    'affixal_count': len(affixal_phonemes),
    'functional_count': len(functional_phonemes),
    'harakat_count': sum(1 for unit in units if unit.haraka),
    'shadda_count': sum(1 for unit in units if 'ّ' in unit.haraka),
    }

        # Calculate confidence based on classification accuracy
    confidence = self._calculate_confidence(units, statistics)

        # Create analysis result
    analysis = PhonologicalAnalysis()
    word=word,
    units=units,
    root_phonemes=root_phonemes,
    affixal_phonemes=affixal_phonemes,
    functional_phonemes=functional_phonemes,
    statistics=statistics,
    confidence=confidence)

        # Update engine statistics
    self.statistics['words_processed'] += 1
    self.statistics['units_extracted'] += len(units)

    self.logger.info()
    f"✅ Analysis complete: {len(units)} units extracted with {confidence:.2%} confidence"
    )  # noqa: E501

    return analysis

    def _calculate_confidence()
    self, units: List[PhonologicalUnit], statistics: Dict[str, int]
    ) -> float:
    """Calculate analysis confidence score"""
        if not units:
    return 0.0

        # Factors affecting confidence
        classified_units = sum()
    1 for unit in units if unit.phoneme_class != PhonemeClassification.UNKNOWN
    )
        classification_ratio = classified_units / len(units)

        # Presence of harakat increases confidence
    harakat_ratio = statistics['harakat_count'] / len(units) if units else 0

        # Combine factors
    confidence = (classification_ratio * 0.7) + (harakat_ratio * 0.3)

    return min(confidence, 1.0)

    def _is_arabic_character(self, char: str) -> bool:
    """Check if character is Arabic"""
    return '\u0600' <= char <= '\u06ff' or '\u0750' <= char <= '\u077f'

    def _is_arabic_letter(self, char: str) -> bool:
    """Check if character is Arabic letter (not diacritic)"""
    return ()
    self._is_arabic_character(char)
    and not self._is_haraka(char)
    and char not in '؛؟،'
    )

    def _is_haraka(self, char: str) -> bool:
    """Check if character is a haraka (diacritical mark)"""
    harakat = {'َ', 'ِ', 'ُ', 'ْ', 'ّ', 'ً', 'ٌ', 'ٍ', 'ٰ', '~'}
    return char in harakat

    def get_phoneme_vector()
    self, unit: PhonologicalUnit
    ) -> Dict[str, Union[str, int, float]]:
    """
    Convert phonological unit to vector representation for ML
    تحويل الوحدة الصوتية إلى تمثيل متجه للتعلم الآلي

    Args:
    unit: PhonologicalUnit to vectorize

    Returns:
    Dictionary representation suitable for machine learning
    """
    vector = {
    'phoneme_ord': ord(unit.phoneme),
    'phoneme_class_id': list(PhonemeClassification).index(unit.phoneme_class),
    'haraka_class_id': list(HarakaClassification).index(unit.haraka_class),
    'position': unit.position,
    'has_haraka': 1 if unit.haraka else 0,
    'haraka_count': len(unit.haraka),
    }

        # Add feature flags
        for feature, value in unit.features.items():
            if isinstance(value, bool):
    vector[f'feat_{feature}'] = 1 if value else 0
            elif isinstance(value, str):
    vector[f'feat_{feature}'] = hash(value) % 1000  # Simple string encoding

    return vector

    def process_text(self, text: str) -> List[PhonologicalAnalysis]:
    """
    Process entire Arabic text (multiple words)
    معالجة النص العربي الكامل (كلمات متعددة)

    Args:
    text: Arabic text to process

    Returns:
    List of PhonologicalAnalysis for each word
    """
        # Simple word tokenization
    words = re.findall(r'[\u0600-\u06FF\u0750 \u077F]+', text)

    results = []
        for word in words:
            if word.strip():
    analysis = self.analyze(word.strip())
    results.append(analysis)

    return results

    def get_statistics(self) -> Dict[str, Any]:
    """Get engine processing statistics"""
    return self.statistics.copy()

    def reset_statistics(self) -> None:
    """Reset engine statistics"""
    self.statistics = {
    'words_processed': 0,
    'units_extracted': 0,
    'classification_accuracy': 0.0,
    }


# Example usage and testing functions
def demonstrate_zero_layer():  # type: ignore[no-untyped-def]
    """Demonstrate the Zero Layer Phonology Engine capabilities"""
    print("🎯 Zero Layer Phonology Engine Demonstration")
    print("=" * 60)

    # Initialize engine
    engine = ZeroLayerPhonologyEngine()

    # Test words with different complexity levels
    test_words = [
    "كِتَابٌ",  # Simple noun with tanwin
    "مُدَرِّسَةٌ",  # Complex noun with shadda
    "يَكْتُبُونَ",  # Verb with affixes
    "وَالْكِتَابُ",  # Word with definite article
    "مَكْتُوبٌ",  # Passive participle
    ]

    for word in test_words:
    print(f"\n📖 Analyzing: {word}")
    print(" " * 30)

    analysis = engine.analyze(word)

    print(f"Total Units: {analysis.statistics['total_units']}")
    print(f"Root Phonemes: {analysis.statistics['root_count']}")
    print(f"Affixal Phonemes: {analysis.statistics['affixal_count']}")
    print(f"Functional Phonemes: {analysis.statistics['functional_count']}")
    print(f"Confidence: {analysis.confidence:.2%}")

    print("\nPhonological Units:")
        for i, unit in enumerate(analysis.units, 1):
    print(f"  {i. {unit}}")

    print("\n📊 Engine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
    print(f"  {key: {value}}")


if __name__ == "__main__":
    demonstrate_zero_layer()

