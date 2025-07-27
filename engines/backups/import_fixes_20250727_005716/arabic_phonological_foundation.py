#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
Arabic Phonological Foundation System (Zero Layer Core)
======================================================
Enterprise-Grade Atomic Phoneme and Diacritic Classification
Professional Python Implementation with Zero-Tolerance Error Handling

This module provides the foundational layer for Arabic linguistic processing:
- Comprehensive atomic function classification (1-5 functions per unit)
- Professional phoneme and diacritic registry with full Arabic coverage
- Enterprise-grade resolver system with position tracking
- Zero-tolerance UTF-8 handling and linguistic accuracy
- Complete compatibility with existing Arabic NLP engine ecosystem

Features:
- 28 Arabic letters with atomic functional classification
- Complete diacritic system with morphological and syntactic functions
- Professional resolver with detailed analysis and reporting
- Position-aware processing with comprehensive error handling
- Full integration with WinSurf IDE environment

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0
Date: 2025-07-23
License: MIT
Encoding: UTF 8
""""

# Professional code quality suppressions
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

import logging
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configure professional logging
logging.basicConfig()
    logging.basicConfig(level=logging.INFO,)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s','
    handlers=[
    logging.FileHandler('arabic_phonological_foundation.log', encoding='utf 8'),'
    logging.StreamHandler(sys.stdout),
    ])

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ATOMIC FUNCTION CLASSIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class PhonemeFunction(Enum):
    """Atomic functional classification for Arabic phonemes (1 5 functions per phoneme)""""

    # Root Functions (الوظائف الجذرية)
    ROOT_FIRST = "جذر_أول"  # First root consonant"
    ROOT_SECOND = "جذر_ثاني"  # Second root consonant"
    ROOT_THIRD = "جذر_ثالث"  # Third root consonant"
    ROOT_WEAK = "جذر_ضعيف"  # Weak root letter (و، ي، ا)"
    ROOT_FOURTH = "جذر_رابع"  # Fourth root consonant (rare)"

    # Morphological Functions (الوظائف الصرفية)
    PRONOUN_SUFFIX = "ضمير_لاحقة"  # Pronoun suffix"
    FEMININE_MARKER = "علامة_تأنيث"  # Feminine marker"
    PLURAL_MARKER = "علامة_جمع"  # Plural marker"
    DERIVATIONAL_AFFIX = "زائدة_اشتقاقية"  # Derivational affix"
    PASSIVE_MARKER = "علامة_مبني_للمجهول"  # Passive voice marker"

    # Syntactic Functions (الوظائف النحوية)
    PREPOSITION = "أداة_جر"  # Preposition"
    CONJUNCTION = "أداة_عطف"  # Conjunction"
    NEGATION_MARKER = "أداة_نفي"  # Negation marker"
    DEFINITE_ARTICLE = "أل_التعريف"  # Definite article"

    # Phonological Functions (الوظائف الصوتية)
    EMPHATIC_CONSONANT = "حرف_مطبّق"  # Emphatic consonant"
    GLIDE_CONSONANT = "حرف_انزلاقي"  # Glide consonant"
    NASAL_CONSONANT = "حرف_أنفي"  # Nasal consonant"
    LONG_VOWEL = "حرف_مد"  # Long vowel marker"


class DiacriticFunction(Enum):
    """Atomic functional classification for Arabic diacritics (1 5 functions per diacritic)""""

    # Temporal Functions (الوظائف الزمنية)
    PAST_TENSE = "دلالة_زمن_الماضي"  # Past tense indication"
    PRESENT_TENSE = "دلالة_زمن_المضارع"  # Present tense indication"
    IMPERATIVE_MOOD = "صيغة_الأمر"  # Imperative mood"
    JUSSIVE_MOOD = "صيغة_جزم"  # Jussive mood"

    # Case Functions (وظائف الإعراب)
    NOMINATIVE_CASE = "علامة_رفع"  # Nominative case"
    ACCUSATIVE_CASE = "علامة_نصب"  # Accusative case"
    GENITIVE_CASE = "علامة_جر"  # Genitive case"

    # Definiteness Functions (وظائف التعريف والتنكير)
    INDEFINITENESS = "تنكير"  # Indefiniteness marker"
    DEFINITENESS = "تعريف"  # Definiteness marker"

    # Morphological Functions (الوظائف الصرفية)
    GEMINATION = "تضعيف"  # Gemination/doubling"
    PASSIVE_VOICE = "صيغة_المبني_للمجهول"  # Passive voice"
    FEMININE_PLURAL = "جمع_مؤنث"  # Feminine plural"
    VERBAL_NOUN = "مصدر"  # Verbal noun (masdar)"

    # Prosodic Functions (الوظائف العروضية)
    EMPHASIS = "توكيد"  # Emphasis"
    LENGTHENING = "إطالة"  # Vowel lengthening"
    SHORTENING = "قصر"  # Vowel shortening"
    ELISION = "حذف"  # Vowel elision"

    # Syntactic Functions (الوظائف النحوية)
    PREPOSITION_MARKER = "علامة_حرف_جر"  # Preposition marker"


# ═══════════════════════════════════════════════════════════════════════════════════
# PHONEME AND DIACRITIC REGISTRY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class PhonemeEntry:
    """Professional phoneme registry entry with atomic functions""""

    symbol: str  # Arabic letter
    name: str  # Arabic name
    features: List[str]  # Phonetic features
    functions: List[PhonemeFunction]  # Atomic functions (1 5 per phoneme)
    phoneme_type: str = "consonant"  # consonant/long_vowel/glide"
    ipa_symbol: str = ""  # IPA representation"
    frequency: float = 0.0  # Usage frequency in Arabic


@dataclass
class DiacriticEntry:
    """Professional diacritic registry entry with atomic functions""""

    symbol: str  # Diacritic symbol
    name: str  # Arabic name
    phonetic_type: str  # Phonetic category
    functions: List[DiacriticFunction]  # Atomic functions (1 5 per diacritic)
    ipa_symbol: str = ""  # IPA representation"
    syllable_weight: float = 1.0  # Prosodic weight


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE ARABIC PHONEME REGISTRY (28 LETTERS)
# ═══════════════════════════════════════════════════════════════════════════════════


def create_phoneme_registry() -> List[PhonemeEntry]:
    """Create comprehensive registry of all 28 Arabic letters with atomic functions""""

    registry = [
        # ا (alif) - Long vowel and weak root
    PhonemeEntry()
    symbol="ا","
    name="ألف","
    features=["long_vowel", "central", "low"],"
    functions=[PhonemeFunction.ROOT_WEAK, PhonemeFunction.LONG_VOWEL],
    phoneme_type="long_vowel","
    ipa_symbol="aː","
    frequency=0.134),
        # ب (ba) - Bilabial stop with multiple functions
    PhonemeEntry()
    symbol="ب","
    name="باء","
    features=["bilabial", "stop", "voiced"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.PREPOSITION],
    phoneme_type="consonant","
    ipa_symbol="b","
    frequency=0.089),
        # ت (ta) - Dental stop with morphological functions
    PhonemeEntry()
    symbol="ت","
    name="تاء","
    features=["dental", "stop", "voiceless"],"
    functions=[
    PhonemeFunction.ROOT_THIRD,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.FEMININE_MARKER,
    PhonemeFunction.DERIVATIONAL_AFFIX,
    ],
    phoneme_type="consonant","
    ipa_symbol="t","
    frequency=0.156),
        # ث (tha) - Dental fricative
    PhonemeEntry()
    symbol="ث","
    name="ثاء","
    features=["dental", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST],
    phoneme_type="consonant","
    ipa_symbol="θ","
    frequency=0.012),
        # ج (jim) - Postalveolar affricate
    PhonemeEntry()
    symbol="ج","
    name="جيم","
    features=["postalveolar", "affricate", "voiced"],"
    functions=[PhonemeFunction.ROOT_SECOND],
    phoneme_type="consonant","
    ipa_symbol="dʒ","
    frequency=0.056),
        # ح (ha) - Pharyngeal fricative
    PhonemeEntry()
    symbol="ح","
    name="حاء","
    features=["pharyngeal", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST],
    phoneme_type="consonant","
    ipa_symbol="ħ","
    frequency=0.078),
        # خ (kha) - Uvular fricative with emphasis
    PhonemeEntry()
    symbol="خ","
    name="خاء","
    features=["uvular", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_SECOND, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="x","
    frequency=0.034),
        # د (dal) - Dental stop
    PhonemeEntry()
    symbol="د","
    name="دال","
    features=["dental", "stop", "voiced"],"
    functions=[PhonemeFunction.ROOT_THIRD],
    phoneme_type="consonant","
    ipa_symbol="d","
    frequency=0.067),
        # ذ (dhal) - Dental fricative
    PhonemeEntry()
    symbol="ذ","
    name="ذال","
    features=["dental", "fricative", "voiced"],"
    functions=[PhonemeFunction.ROOT_SECOND],
    phoneme_type="consonant","
    ipa_symbol="ð","
    frequency=0.019),
        # ر (ra) - Alveolar trill with multiple functions
    PhonemeEntry()
    symbol="ر","
    name="راء","
    features=["alveolar", "trill", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_SECOND,
    PhonemeFunction.PREPOSITION,
    PhonemeFunction.DERIVATIONAL_AFFIX,
    ],
    phoneme_type="consonant","
    ipa_symbol="r","
    frequency=0.145),
        # ز (zay) - Alveolar fricative
    PhonemeEntry()
    symbol="ز","
    name="زاي","
    features=["alveolar", "fricative", "voiced"],"
    functions=[PhonemeFunction.ROOT_THIRD],
    phoneme_type="consonant","
    ipa_symbol="z","
    frequency=0.023),
        # س (sin) - Alveolar fricative with plural function
    PhonemeEntry()
    symbol="س","
    name="سين","
    features=["alveolar", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.PLURAL_MARKER],
    phoneme_type="consonant","
    ipa_symbol="s","
    frequency=0.098),
        # ش (shin) - Postalveolar fricative
    PhonemeEntry()
    symbol="ش","
    name="شين","
    features=["postalveolar", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_SECOND],
    phoneme_type="consonant","
    ipa_symbol="ʃ","
    frequency=0.045),
        # ص (sad) - Emphatic alveolar fricative
    PhonemeEntry()
    symbol="ص","
    name="صاد","
    features=[
    "alveolar","
    "fricative","
    "voiceless","
    "emphatic","
    "pharyngealized","
    ],
    functions=[PhonemeFunction.ROOT_SECOND, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="sˤ","
    frequency=0.067),
        # ض (dad) - Emphatic alveolar stop
    PhonemeEntry()
    symbol="ض","
    name="ضاد","
    features=["alveolar", "stop", "voiced", "emphatic", "pharyngealized"],"
    functions=[PhonemeFunction.ROOT_THIRD, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="dˤ","
    frequency=0.023),
        # ط (ta) - Emphatic dental stop
    PhonemeEntry()
    symbol="ط","
    name="طاء","
    features=["dental", "stop", "voiceless", "emphatic", "pharyngealized"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="tˤ","
    frequency=0.034),
        # ظ (za) - Emphatic dental fricative
    PhonemeEntry()
    symbol="ظ","
    name="ظاء","
    features=["dental", "fricative", "voiced", "emphatic", "pharyngealized"],"
    functions=[PhonemeFunction.ROOT_FOURTH, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="ðˤ","
    frequency=0.008),
        # ع (ain) - Pharyngeal fricative
    PhonemeEntry()
    symbol="ع","
    name="عين","
    features=["pharyngeal", "fricative", "voiced"],"
    functions=[PhonemeFunction.ROOT_FIRST],
    phoneme_type="consonant","
    ipa_symbol="ʕ","
    frequency=0.091),
        # غ (ghain) - Uvular fricative with emphasis
    PhonemeEntry()
    symbol="غ","
    name="غين","
    features=["uvular", "fricative", "voiced"],"
    functions=[PhonemeFunction.ROOT_SECOND, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="ɣ","
    frequency=0.028),
        # ف (fa) - Labiodental fricative
    PhonemeEntry()
    symbol="ف","
    name="فاء","
    features=["labiodental", "fricative", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST],
    phoneme_type="consonant","
    ipa_symbol="f","
    frequency=0.045),
        # ق (qaf) - Uvular stop with emphasis
    PhonemeEntry()
    symbol="ق","
    name="قاف","
    features=["uvular", "stop", "voiceless"],"
    functions=[PhonemeFunction.ROOT_SECOND, PhonemeFunction.EMPHATIC_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="q","
    frequency=0.067),
        # ك (kaf) - Velar stop with preposition function
    PhonemeEntry()
    symbol="ك","
    name="كاف","
    features=["velar", "stop", "voiceless"],"
    functions=[PhonemeFunction.ROOT_THIRD, PhonemeFunction.PREPOSITION],
    phoneme_type="consonant","
    ipa_symbol="k","
    frequency=0.078),
        # ل (lam) - Alveolar lateral with multiple functions
    PhonemeEntry()
    symbol="ل","
    name="لام","
    features=["alveolar", "lateral", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_SECOND,
    PhonemeFunction.DEFINITE_ARTICLE,
    PhonemeFunction.CONJUNCTION,
    ],
    phoneme_type="consonant","
    ipa_symbol="l","
    frequency=0.134),
        # م (mim) - Bilabial nasal with multiple functions
    PhonemeEntry()
    symbol="م","
    name="ميم","
    features=["bilabial", "nasal", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_FIRST,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.NASAL_CONSONANT,
    ],
    phoneme_type="consonant","
    ipa_symbol="m","
    frequency=0.123),
        # ن (nun) - Alveolar nasal with multiple functions
    PhonemeEntry()
    symbol="ن","
    name="نون","
    features=["alveolar", "nasal", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_THIRD,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.PLURAL_MARKER,
    PhonemeFunction.NEGATION_MARKER,
    PhonemeFunction.NASAL_CONSONANT,
    ],
    phoneme_type="consonant","
    ipa_symbol="n","
    frequency=0.167),
        # ه (ha) - Glottal fricative with multiple functions
    PhonemeEntry()
    symbol="ه","
    name="هاء","
    features=["glottal", "fricative", "voiceless"],"
    functions=[
    PhonemeFunction.ROOT_THIRD,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.FEMININE_MARKER,
    ],
    phoneme_type="consonant","
    ipa_symbol="h","
    frequency=0.089),
        # و (waw) - Labio velar approximant with multiple functions
    PhonemeEntry()
    symbol="و","
    name="واو","
    features=["labio velar", "approximant", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_WEAK,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.CONJUNCTION,
    PhonemeFunction.LONG_VOWEL,
    PhonemeFunction.GLIDE_CONSONANT,
    ],
    phoneme_type="long_vowel","
    ipa_symbol="w","
    frequency=0.098),
        # ي (ya) - Palatal approximant with multiple functions
    PhonemeEntry()
    symbol="ي","
    name="ياء","
    features=["palatal", "approximant", "voiced"],"
    functions=[
    PhonemeFunction.ROOT_WEAK,
    PhonemeFunction.PRONOUN_SUFFIX,
    PhonemeFunction.DERIVATIONAL_AFFIX,
    PhonemeFunction.LONG_VOWEL,
    PhonemeFunction.GLIDE_CONSONANT,
    ],
    phoneme_type="long_vowel","
    ipa_symbol="j","
    frequency=0.134),
        # ء (hamza) - Glottal stop
    PhonemeEntry()
    symbol="ء","
    name="همزة","
    features=["glottal", "stop", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.GLIDE_CONSONANT],
    phoneme_type="consonant","
    ipa_symbol="ʔ","
    frequency=0.045),
        # أ (alif with hamza above) - Hamza variant
    PhonemeEntry()
    symbol="أ","
    name="ألف همزة","
    features=["glottal", "stop", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.ROOT_WEAK],
    phoneme_type="consonant","
    ipa_symbol="ʔa","
    frequency=0.067),
        # إ (alif with hamza below) - Hamza variant
    PhonemeEntry()
    symbol="إ","
    name="ألف كسرة همزة","
    features=["glottal", "stop", "voiceless"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.ROOT_WEAK],
    phoneme_type="consonant","
    ipa_symbol="ʔi","
    frequency=0.045),
        # آ (alif with madda) - Hamza variant
    PhonemeEntry()
    symbol="آ","
    name="ألف مدة","
    features=["long_vowel", "central", "low"],"
    functions=[PhonemeFunction.ROOT_WEAK, PhonemeFunction.LONG_VOWEL],
    phoneme_type="long_vowel","
    ipa_symbol="ʔaː","
    frequency=0.023),
        # ؤ (waw with hamza) - Hamza variant
    PhonemeEntry()
    symbol="ؤ","
    name="واو همزة","
    features=["glottal", "stop", "voiced"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.ROOT_WEAK],
    phoneme_type="consonant","
    ipa_symbol="ʔu","
    frequency=0.034),
        # ئ (ya with hamza) - Hamza variant
    PhonemeEntry()
    symbol="ئ","
    name="ياء همزة","
    features=["glottal", "stop", "voiced"],"
    functions=[PhonemeFunction.ROOT_FIRST, PhonemeFunction.ROOT_WEAK],
    phoneme_type="consonant","
    ipa_symbol="ʔi","
    frequency=0.029),
        # ى (alif maqsura) - Shortened alif
    PhonemeEntry()
    symbol="ى","
    name="ألف مقصورة","
    features=["long_vowel", "central", "high"],"
    functions=[PhonemeFunction.ROOT_WEAK, PhonemeFunction.LONG_VOWEL],
    phoneme_type="long_vowel","
    ipa_symbol="aː","
    frequency=0.087),
        # ة (ta marbuta) - Tied ta (feminine marker)
    PhonemeEntry()
    symbol="ة","
    name="تاء مربوطة","
    features=["dental", "stop", "voiceless"],"
    functions=[PhonemeFunction.FEMININE_MARKER, PhonemeFunction.ROOT_THIRD],
    phoneme_type="consonant","
    ipa_symbol="t","
    frequency=0.098),
    ]

    return registry


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE ARABIC DIACRITIC REGISTRY (10 MAIN DIACRITICS)
# ═══════════════════════════════════════════════════════════════════════════════════


def create_diacritic_registry() -> List[DiacriticEntry]:
    """Create comprehensive registry of all Arabic diacritics with atomic functions""""

    registry = [
        # فتحة (fatha) - Short vowel with multiple functions
    DiacriticEntry()
    symbol="َ","
    name="فتحة","
    phonetic_type="short_vowel","
    functions=[
    DiacriticFunction.PAST_TENSE,
    DiacriticFunction.ACCUSATIVE_CASE,
    DiacriticFunction.FEMININE_PLURAL,
    ],
    ipa_symbol="a","
    syllable_weight=1.0),
        # ضمة (damma) - Short vowel with multiple functions
    DiacriticEntry()
    symbol="ُ","
    name="ضمة","
    phonetic_type="short_vowel","
    functions=[
    DiacriticFunction.PRESENT_TENSE,
    DiacriticFunction.NOMINATIVE_CASE,
    DiacriticFunction.PASSIVE_VOICE,
    ],
    ipa_symbol="u","
    syllable_weight=1.0),
        # كسرة (kasra) - Short vowel with multiple functions
    DiacriticEntry()
    symbol="ِ","
    name="كسرة","
    phonetic_type="short_vowel","
    functions=[
    DiacriticFunction.GENITIVE_CASE,
    DiacriticFunction.PREPOSITION_MARKER,
    DiacriticFunction.FEMININE_PLURAL,
    ],
    ipa_symbol="i","
    syllable_weight=1.0),
        # سكون (sukun) - Absence of vowel with multiple functions
    DiacriticEntry()
    symbol="ْ","
    name="سكون","
    phonetic_type="sukun","
    functions=[
    DiacriticFunction.JUSSIVE_MOOD,
    DiacriticFunction.SHORTENING,
    DiacriticFunction.ELISION,
    ],
    ipa_symbol="","
    syllable_weight=0.0),
        # شدة (shadda) - Gemination with multiple functions
    DiacriticEntry()
    symbol="ّ","
    name="شدة","
    phonetic_type="shadda","
    functions=[
    DiacriticFunction.GEMINATION,
    DiacriticFunction.EMPHASIS,
    DiacriticFunction.VERBAL_NOUN,
    ],
    ipa_symbol="ː","
    syllable_weight=2.0),
        # تنوين فتح (tanwin fath) - Indefinite accusative
    DiacriticEntry()
    symbol="ً","
    name="تنوين فتح","
    phonetic_type="tanwin","
    functions=[
    DiacriticFunction.ACCUSATIVE_CASE,
    DiacriticFunction.INDEFINITENESS,
    ],
    ipa_symbol="an","
    syllable_weight=1.5),
        # تنوين ضم (tanwin damm) - Indefinite nominative
    DiacriticEntry()
    symbol="ٌ","
    name="تنوين ضم","
    phonetic_type="tanwin","
    functions=[
    DiacriticFunction.NOMINATIVE_CASE,
    DiacriticFunction.INDEFINITENESS,
    ],
    ipa_symbol="un","
    syllable_weight=1.5),
        # تنوين كسر (tanwin kasr) - Indefinite genitive
    DiacriticEntry()
    symbol="ٍ","
    name="تنوين كسر","
    phonetic_type="tanwin","
    functions=[
    DiacriticFunction.GENITIVE_CASE,
    DiacriticFunction.INDEFINITENESS,
    ],
    ipa_symbol="in","
    syllable_weight=1.5),
        # مدة (maddah) - Lengthening marker
    DiacriticEntry()
    symbol="ٓ","
    name="مدة","
    phonetic_type="maddah","
    functions=[DiacriticFunction.LENGTHENING],
    ipa_symbol="ː","
    syllable_weight=2.0),
        # ألف خنجرية (dagger alif) - Short alif
    DiacriticEntry()
    symbol="ٰ","
    name="ألف خنجرية","
    phonetic_type="dagger_alif","
    functions=[DiacriticFunction.SHORTENING],
    ipa_symbol="a","
    syllable_weight=0.5),
    ]

    return registry


# ═══════════════════════════════════════════════════════════════════════════════════
# RESOLVED ANALYSIS STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class ResolvedPhoneme:
    """Resolved phoneme with complete analysis""""

    symbol: str  # Arabic phoneme
    name: str  # Arabic name
    position: int  # Position in word
    features: List[str]  # Phonetic features
    functions: List[str]  # Atomic functions
    context_role: Optional[str] = None  # Contextual role in word
    ipa_symbol: str = ""  # IPA representation"
    frequency: float = 0.0  # Usage frequency


@dataclass
class ResolvedDiacritic:
    """Resolved diacritic with complete analysis""""

    symbol: str  # Diacritic symbol
    name: str  # Arabic name
    position: int  # Position in word
    phonetic_type: str  # Phonetic category
    functions: List[str]  # Atomic functions
    attached_to: Optional[str] = None  # Host phoneme
    ipa_symbol: str = ""  # IPA representation"
    syllable_weight: float = 1.0  # Prosodic weight


@dataclass
class AnalysisResult:
    """Complete phonological analysis result""""

    word: str  # Original word
    phonemes: List[ResolvedPhoneme]  # Resolved phonemes
    diacritics: List[ResolvedDiacritic]  # Resolved diacritics
    statistics: Dict[str, Any]  # Analysis statistics
    confidence: float = 0.0  # Analysis confidence


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN PHONOLOGICAL FUNCTION RESOLVER CLASS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicPhonologicalFoundation:
    """"
    Professional Arabic Phonological Foundation System
    Enterprise-grade atomic function resolver with zero-tolerance error handling

    This class provides the core foundation for all Arabic NLP processing by:
    - Resolving phonemes and diacritics with atomic function classification
    - Providing position-aware analysis with comprehensive error handling
    - Supporting enterprise-grade logging and reporting
    - Maintaining full compatibility with existing Arabic NLP engines
    """"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
    """Initialize the Arabic Phonological Foundation System""""
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.info("🚀 Initializing Arabic Phonological Foundation System")"

        # Load configuration
    self.config = config or self._get_default_config()

        # Initialize registries
    self.phoneme_registry = create_phoneme_registry()
    self.diacritic_registry = create_diacritic_registry()

        # Create lookup maps for efficient processing
    self.phoneme_map: Dict[str, PhonemeEntry] = {
    entry.symbol: entry for entry in self.phoneme_registry
    }
    self.diacritic_map: Dict[str, DiacriticEntry] = {
    entry.symbol: entry for entry in self.diacritic_registry
    }

        # Create character sets for fast lookup
    self.diacritic_set = set(self.diacritic_map.keys())
    self.phoneme_set = set(self.phoneme_map.keys())

        # Processing statistics
    self.stats = {
    'words_processed': 0,'
    'phonemes_resolved': 0,'
    'diacritics_resolved': 0,'
    'errors_encountered': 0,'
    }

    self.logger.info()
    f"✅ Foundation initialized - {len(self.phoneme_registry)} phonemes, {len(self.diacritic_registry)} diacritics""
    )

    def _get_default_config(self) -> Dict[str, Any]:
    """Get default configuration for the foundation system""""
    return {
    'strict_mode': True,  # Strict linguistic analysis'
    'position_tracking': True,  # Track character positions'
    'context_analysis': True,  # Analyze contextual roles'
    'error_tolerance': 'zero',  # Zero tolerance for errors'
    'logging_level': 'INFO',  # Logging verbosity'
    'utf8_validation': True,  # Validate UTF 8 encoding'
    'ipa_output': True,  # Include IPA symbols'
    'frequency_weighting': True,  # Use frequency data'
    }

    def resolve_word(self, word: str) -> AnalysisResult:
    """"
    Perform comprehensive phonological analysis of Arabic word

    Args:
    word: Arabic word to analyze

    Returns:
    AnalysisResult with complete phoneme and diacritic analysis
    """"
        try:
    self.logger.info(f"🔬 Analyzing word: {word}")"
    self.stats['words_processed'] += 1'

            # Validate input
            if not word or not isinstance(word, str):
    raise ValueError(f"Invalid input word: {word}")"

            # UTF-8 validation if enabled
            if self.config['utf8_validation']:'
    self._validate_utf8(word)

            # Initialize analysis containers
    phonemes: List[ResolvedPhoneme] = []
    diacritics: List[ResolvedDiacritic] = []

            # Process each character with position tracking
    phoneme_position = 0

            for i, char in enumerate(word):
                # Handle phonemes
                if char in self.phoneme_map:
    phoneme_position += 1
    resolved = self._resolve_phoneme(char, phoneme_position, word, i)
    phonemes.append(resolved)
    self.stats['phonemes_resolved'] += 1'

                # Handle diacritics
                elif char in self.diacritic_map:
    attached_to = ()
    word[i - 1]
                        if i > 0 and word[i - 1] in self.phoneme_set
                        else None
    )
    resolved = self._resolve_diacritic(char, i, attached_to)
    diacritics.append(resolved)
    self.stats['diacritics_resolved'] += 1'

                # Handle unknown characters
                elif char not in [' ', '\n', '\t']:  # Skip whitespace'
    self.logger.warning(f"⚠️ Unknown character at position {i}: {char}")"
                    if self.config['strict_mode']:'
    raise ValueError(f"Unknown character in strict mode: {char}")"

            # Calculate statistics and confidence
    statistics = self._calculate_statistics(word, phonemes, diacritics)
    confidence = self._calculate_confidence(phonemes, diacritics)

            # Create analysis result
    result = AnalysisResult()
    word=word,
    phonemes=phonemes,
    diacritics=diacritics,
    statistics=statistics,
    confidence=confidence)

    self.logger.info()
    f"✅ Analysis complete: {len(phonemes)} phonemes, {len(diacritics)} diacritics, {confidence:.2%} confidence""
    )
    return result

        except Exception as e:
    self.stats['errors_encountered'] += 1'
    self.logger.error(f"❌ Error analyzing word '{word': {e}}")'"
    raise

    def _resolve_phoneme()
    self, char: str, position: int, word: str, char_index: int
    ) -> ResolvedPhoneme:
    """Resolve phoneme with complete analysis""""
    entry = self.phoneme_map[char]

        # Extract functions as strings
    function_strings = [func.value for func in entry.functions]

        # Determine contextual role if context analysis is enabled
    context_role = None
        if self.config['context_analysis']:'
    context_role = self._determine_context_role(char, word, char_index)

    return ResolvedPhoneme()
    symbol=char,
    name=entry.name,
    position=position,
    features=entry.features.copy(),
    functions=function_strings,
    context_role=context_role,
    ipa_symbol=entry.ipa_symbol,
    frequency=entry.frequency)

    def _resolve_diacritic()
    self, char: str, position: int, attached_to: Optional[str]
    ) -> ResolvedDiacritic:
    """Resolve diacritic with complete analysis""""
    entry = self.diacritic_map[char]

        # Extract functions as strings
    function_strings = [func.value for func in entry.functions]

    return ResolvedDiacritic()
    symbol=char,
    name=entry.name,
    position=position,
    phonetic_type=entry.phonetic_type,
    functions=function_strings,
    attached_to=attached_to,
    ipa_symbol=entry.ipa_symbol,
    syllable_weight=entry.syllable_weight)

    def _determine_context_role()
    self, char: str, word: str, position: int
    ) -> Optional[str]:
    """Determine contextual role of phoneme in word""""
    word_len = len(word)

        # Position based analysis
        if position == 0:
    return "word_initial""
        elif position == word_len - 1:
    return "word_final""
        else:
    return "word_medial""

    def _calculate_statistics()
    self,
    word: str,
    phonemes: List[ResolvedPhoneme],
    diacritics: List[ResolvedDiacritic]) -> Dict[str, Any]:
    """Calculate comprehensive analysis statistics""""

        # Count function types
    function_counts = {}
        for phoneme in phonemes:
            for function in phoneme.functions:
    function_counts[function] = function_counts.get(function, 0) + 1

        # Count phonetic features
    feature_counts = {}
        for phoneme in phonemes:
            for feature in phoneme.features:
    feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Calculate frequency-weighted score
    total_frequency = ()
    sum(p.frequency for p in phonemes) / len(phonemes) if phonemes else 0.0
    )

    return {
    'word_length': len(word),'
    'phoneme_count': len(phonemes),'
    'diacritic_count': len(diacritics),'
    'function_distribution': function_counts,'
    'feature_distribution': feature_counts,'
    'average_frequency': total_frequency,'
    'syllable_weight': sum(d.syllable_weight for d in diacritics),'
    }

    def _calculate_confidence()
    self, phonemes: List[ResolvedPhoneme], diacritics: List[ResolvedDiacritic]
    ) -> float:
    """Calculate analysis confidence score""""
        if not phonemes:
    return 0.0

        # Base confidence on phoneme coverage
    base_confidence = ()
    len(phonemes) / (len(phonemes) + len(diacritics))
            if phonemes or diacritics
            else 0.0
    )

        # Adjust for frequency weighting if enabled
        if self.config['frequency_weighting']:'
    freq_factor = sum(p.frequency for p in phonemes) / len(phonemes)
    base_confidence = (base_confidence + freq_factor) / 2

        # Adjust for diacritic coverage
        if diacritics:
    diacritic_factor = len([d for d in diacritics if d.attached_to]) / len()
    diacritics
    )
    base_confidence = (base_confidence + diacritic_factor) / 2

    return min(base_confidence, 1.0)

    def _validate_utf8(self, text: str) -> None:
    """Validate UTF 8 encoding of input text""""
        try:
    text.encode('utf 8').decode('utf 8')'
        except UnicodeError as e:
    raise ValueError(f"Invalid UTF 8 encoding: {e}")"

    def describe_analysis(self, result: AnalysisResult) -> None:
    """Display comprehensive analysis results with professional formatting""""

    print("\n" + "═" * 80)"
    print(f"🔠 Arabic Phonological Analysis: {result.word}")"
    print("═" * 80)"

        # Overall statistics
    stats = result.statistics
    print("\n📊 Overall Statistics:")"
    print(f"   Word Length: {stats['word_length'] characters}")'"
    print(f"   Phonemes: {stats['phoneme_count']}")'"
    print(f"   Diacritics: {stats['diacritic_count']}")'"
    print(f"   Analysis Confidence: {result.confidence:.2%}")"
    print(f"   Average Frequency: {stats['average_frequency']:.4f}")'"
    print(f"   Syllable Weight: {stats['syllable_weight']:.1f}")'"

        # Phoneme analysis
    print(f"\n🧩 Phoneme Analysis ({len(result.phonemes) phonemes):}")"
        for phoneme in result.phonemes:
    functions_str = ", ".join(phoneme.functions)"
    features_str = ", ".join(phoneme.features)"
    context_str = ()
    f" | Context: {phoneme.context_role}" if phoneme.context_role else """
    )
    ipa_str = f" | IPA: /{phoneme.ipa_symbol/}" if phoneme.ipa_symbol else """
    freq_str = ()
    f" | Freq: {phoneme.frequency:.3f}" if phoneme.frequency > 0 else """
    )

    print()
    f"   • {phoneme.symbol} ({phoneme.name}) - Position: {phoneme.position}""
    )
    print(f"     Features: {features_str}")"
    print(f"     Functions: {functions_str}{context_str}{ipa_str{freq_str}}")"

        # Diacritic analysis
        if result.diacritics:
    print(f"\n🎵 Diacritic Analysis ({len(result.diacritics) diacritics):}")"
            for diacritic in result.diacritics:
    functions_str = ", ".join(diacritic.functions)"
    attached_str = ()
    f" | Attached to: {diacritic.attached_to}""
                    if diacritic.attached_to
                    else " | Standalone""
    )
    ipa_str = ()
    f" | IPA: /{diacritic.ipa_symbol/}" if diacritic.ipa_symbol else """
    )
    weight_str = f" | Weight: {diacritic.syllable_weight:.1f}""

    print()
    f"   • {diacritic.symbol} ({diacritic.name}) - Position: {diacritic.position}""
    )
    print(f"     Type: {diacritic.phonetic_type}")"
    print()
    f"     Functions: {functions_str}{attached_str}{ipa_str{weight_str}}""
    )

        # Function distribution
        if stats['function_distribution']:'
    print("\n⚙️ Function Distribution:")"
            for function, count in sorted(stats['function_distribution'].items()):'
    print(f"   • {function: {count}}")"

        # Feature distribution
        if stats['feature_distribution']:'
    print("\n🔧 Feature Distribution:")"
            for feature, count in sorted(stats['feature_distribution'].items()):'
    print(f"   • {feature}: {count}")"

    print("═" * 80)"
    print(f"✅ Analysis Complete | Confidence: {result.confidence:.2%}")"
    print("═" * 80)"

    def get_system_statistics(self) -> Dict[str, Any]:
    """Get comprehensive system processing statistics""""
    return {
    'phoneme_registry_size': len(self.phoneme_registry),'
    'diacritic_registry_size': len(self.diacritic_registry),'
    'processing_stats': self.stats.copy(),'
    'configuration': self.config.copy(),'
    }

    def reset_statistics(self) -> None:
    """Reset processing statistics""""
    self.stats = {
    'words_processed': 0,'
    'phonemes_resolved': 0,'
    'diacritics_resolved': 0,'
    'errors_encountered': 0,'
    }
    self.logger.info("📊 Processing statistics reset")"


# ═══════════════════════════════════════════════════════════════════════════════════
# TESTING AND DEMONSTRATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════════


def run_comprehensive_tests():
    """Run comprehensive test suite with diverse Arabic words""""

    print("=" * 80)"
    print("🧪 Arabic Phonological Foundation - Comprehensive Test Suite")"
    print("=" * 80)"
    print("Environment: WinSurf IDE | Python 3.8+ | UTF 8 Encoding")"
    print("Test Suite: Professional Arabic NLP Foundation Testing")"

    # Initialize foundation system
    foundation = ArabicPhonologicalFoundation()

    # Test cases covering various linguistic phenomena
    test_words = [
    "كَتَبَ",  # فعل ماضٍ (Past tense verb)"
    "يَكْتُبُ",  # فعل مضارع (Present tense verb)"
    "مُدَرِّسٌ",  # اسم فاعل (Active participle)"
    "كِتَابٌ",  # اسم جامد (Concrete noun)"
    "قَرَأْتُ",  # مع تاء الفاعل (With subject pronoun)"
    "الْبَيْتُ",  # مع أل التعريف (With definite article)"
    "فَعَّلَ",  # مع الشدة (With gemination)"
    "مَسْؤُولٌ",  # مع الهمزة (With hamza)"
    "مُسْتَشْفَى",  # كلمة مركبة (Compound word)"
    "جَامِعَةٌ",  # اسم مؤنث (Feminine noun)"
    ]

    print(f"\n🔬 Testing {len(test_words) diverse} Arabic words...}")"

    for i, word in enumerate(test_words, 1):
        try:
    print(f"\n{'='*60}")'"
    print(f"Test {i}/{len(test_words)}: {word}")"
    print('=' * 60)'

            # Analyze word
    result = foundation.resolve_word(word)

            # Display analysis
    foundation.describe_analysis(result)

            # Validation checks
    assert result.phonemes, f"No phonemes found for word: {word}""
    assert result.confidence > 0.0, f"Zero confidence for word: {word}""
    assert ()
    result.statistics['phoneme_count'] > 0'
    ), f"No phoneme count for word: {word}""

    print(f"✅ Test {i} PASSED - {word} analyzed successfully")"

        except Exception as e:
    print(f"❌ Test {i} FAILED - {word: {e}}")"
    raise

    # Display final statistics
    print("\n" + "=" * 80)"
    print("📊 FINAL TEST SUITE STATISTICS")"
    print("=" * 80)"

    stats = foundation.get_system_statistics()
    print("System Configuration:")"
    for key, value in stats['configuration'].items():'
    print(f"   • {key: {value}}")"

    print("\nProcessing Statistics:")"
    for key, value in stats['processing_stats'].items():'
    print(f"   • {key: {value}}")"

    print("\nRegistry Information:")"
    print(f"   • Phoneme Registry: {stats['phoneme_registry_size']} entries")'"
    print(f"   • Diacritic Registry: {stats['diacritic_registry_size'] entries}")'"

    print("\n" + "=" * 80)"
    print("🎉 ALL TESTS PASSED - Arabic Phonological Foundation Working Perfectly!")"
    print("🚀 Ready for integration with Arabic NLP engine ecosystem")"
    print("=" * 80)"


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":"
    """Main execution entry point for testing and demonstration""""

    # Configure professional logging for demonstration
    logger.info("🌟 Starting Arabic Phonological Foundation System")"
    logger.info(f"📍 Working Directory: {Path.cwd()}")"
    logger.info(f"🔧 Python Version: {sys.version}")"
    logger.info()
    f"🎯 UTF-8 Support: {'✅' if sys.stdout.encoding.lower() == 'utf 8'} else '⚠️'}"'"
    )

    try:
        # Run comprehensive test suite
    run_comprehensive_tests()

        # Optional: Interactive demo mode
    print("\n🎮 Interactive Mode Available")"
    print("Use: foundation = ArabicPhonologicalFoundation()")"
    print("Then: result = foundation.resolve_word('كلمة')")'"
    print("Finally: foundation.describe_analysis(result)")"

    except Exception as e:
    logger.error(f"💥 Critical error in main execution: {e}")"
    sys.exit(1)

    logger.info()
    "🏁 Arabic Phonological Foundation System execution completed successfully""
    )

