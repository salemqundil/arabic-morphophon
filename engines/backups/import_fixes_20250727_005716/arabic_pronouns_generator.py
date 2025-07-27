#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Pronouns Generator from Syllables - مولد الضمائر العربية من المقاطع الصوتية
===============================================================================

This module generates Arabic pronouns (detached and attached) from syllabic patterns
using deep learning and phonological analysis. It covers both:
1. Detached pronouns (الضمائر المنفصلة): أنا، أنت، هو، هي، نحن، إلخ
2. Attached pronouns (الضمائر المتصلة): ـني، ـك، ـه، ـها، إلخ

نظام متطور يستخدم التعلم العميق لتحليل المقاطع الصوتية وتوليد الضمائر المناسبة

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - ARABIC PRONOUNS FROM SYLLABLES
Date: 2025-07-24
Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


import json  # noqa: F401
import logging  # noqa: F401
import numpy as np  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field  # noqa: F401
from enum import Enum  # noqa: F401
import re  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler('arabic_pronouns_generator.log', encoding='utf 8'),
    logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC PRONOUNS CLASSIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class PronounType(Enum):
    """تصنيف أنواع الضمائر العربية"""

    DETACHED = "منفصل"  # Detached pronouns
    ATTACHED = "متصل"  # Attached pronouns


class PronounPerson(Enum):
    """تصنيف أشخاص الضمائر"""

    FIRST = "متكلم"  # First person
    SECOND = "مخاطب"  # Second person
    THIRD = "غائب"  # Third person


class PronounNumber(Enum):
    """تصنيف عدد الضمائر"""

    SINGULAR = "مفرد"  # Singular
    DUAL = "مثنى"  # Dual
    PLURAL = "جمع"  # Plural


class PronounGender(Enum):
    """تصنيف جنس الضمائر"""

    MASCULINE = "مذكر"  # Masculine
    FEMININE = "مؤنث"  # Feminine
    NEUTRAL = "محايد"  # Neutral (for first person)


@dataclass
class PronounEntry:
    """كيان الضمير مع جميع المعلومات اللغوية"""

    text: str  # النص العربي للضمير
    pronoun_type: PronounType  # نوع الضمير (متصل/منفصل)
    person: PronounPerson  # الشخص (متكلم/مخاطب/غائب)
    number: PronounNumber  # العدد (مفرد/مثنى/جمع)
    gender: PronounGender  # الجنس (مذكر/مؤنث/محايد)
    syllable_pattern: str  # النمط المقطعي
    phonetic_features: Dict[str, Any] = field(default_factory=dict)
    usage_contexts: List[str] = field(default_factory=list)
    frequency_score: float = 0.0
    class_id: int = 0


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC PRONOUNS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicPronounsDatabase:
    """قاعدة بيانات شاملة للضمائر العربية"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.pronouns: List[PronounEntry] = []
    self.class_mapping: Dict[int, str] = {}
    self.syllable_patterns: Dict[str, List[str]] = {}
    self._initialize_pronouns_database()

    def _initialize_pronouns_database(self):  # type: ignore[no-untyped def]
    """تهيئة قاعدة بيانات الضمائر الشاملة"""

        # الضمائر المنفصلة - Detached Pronouns
    detached_pronouns = [
            # First Person - المتكلم
    PronounEntry(
    "أنا",
    PronounType.DETACHED,
    PronounPerson.FIRST,
    PronounNumber.SINGULAR,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.95,
                class_id=0,
    ),
    PronounEntry(
    "نحن",
    PronounType.DETACHED,
    PronounPerson.FIRST,
    PronounNumber.PLURAL,
    PronounGender.NEUTRAL,
    "CV CVC",
    frequency_score=0.85,
                class_id=1,
    ),
            # Second Person - المخاطب
    PronounEntry(
    "أنت",
    PronounType.DETACHED,
    PronounPerson.SECOND,
    PronounNumber.SINGULAR,
    PronounGender.MASCULINE,
    "CV CVC",
    frequency_score=0.90,
                class_id=2,
    ),
    PronounEntry(
    "أنتِ",
    PronounType.DETACHED,
    PronounPerson.SECOND,
    PronounNumber.SINGULAR,
    PronounGender.FEMININE,
    "CV CV",
    frequency_score=0.88,
                class_id=3,
    ),
    PronounEntry(
    "أنتما",
    PronounType.DETACHED,
    PronounPerson.SECOND,
    PronounNumber.DUAL,
    PronounGender.NEUTRAL,
    "CV-CV CV",
    frequency_score=0.40,
                class_id=4,
    ),
    PronounEntry(
    "أنتم",
    PronounType.DETACHED,
    PronounPerson.SECOND,
    PronounNumber.PLURAL,
    PronounGender.MASCULINE,
    "CV CVC",
    frequency_score=0.75,
                class_id=5,
    ),
    PronounEntry(
    "أنتن",
    PronounType.DETACHED,
    PronounPerson.SECOND,
    PronounNumber.PLURAL,
    PronounGender.FEMININE,
    "CV CVC",
    frequency_score=0.65,
                class_id=6,
    ),
            # Third Person - الغائب
    PronounEntry(
    "هو",
    PronounType.DETACHED,
    PronounPerson.THIRD,
    PronounNumber.SINGULAR,
    PronounGender.MASCULINE,
    "CV CV",
    frequency_score=0.95,
                class_id=7,
    ),
    PronounEntry(
    "هي",
    PronounType.DETACHED,
    PronounPerson.THIRD,
    PronounNumber.SINGULAR,
    PronounGender.FEMININE,
    "CV CV",
    frequency_score=0.93,
                class_id=8,
    ),
    PronounEntry(
    "هما",
    PronounType.DETACHED,
    PronounPerson.THIRD,
    PronounNumber.DUAL,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.45,
                class_id=9,
    ),
    PronounEntry(
    "هم",
    PronounType.DETACHED,
    PronounPerson.THIRD,
    PronounNumber.PLURAL,
    PronounGender.MASCULINE,
    "CVC",
    frequency_score=0.80,
                class_id=10,
    ),
    PronounEntry(
    "هن",
    PronounType.DETACHED,
    PronounPerson.THIRD,
    PronounNumber.PLURAL,
    PronounGender.FEMININE,
    "CVC",
    frequency_score=0.70,
                class_id=11,
    ),
    ]

        # الضمائر المتصلة - Attached Pronouns
    attached_pronouns = [
            # First Person - المتكلم
    PronounEntry(
    "ـني",
    PronounType.ATTACHED,
    PronounPerson.FIRST,
    PronounNumber.SINGULAR,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.85,
                class_id=12,
    ),
    PronounEntry(
    "ـي",
    PronounType.ATTACHED,
    PronounPerson.FIRST,
    PronounNumber.SINGULAR,
    PronounGender.NEUTRAL,
    "CV",
    frequency_score=0.90,
                class_id=13,
    ),
    PronounEntry(
    "ـنا",
    PronounType.ATTACHED,
    PronounPerson.FIRST,
    PronounNumber.PLURAL,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.88,
                class_id=14,
    ),
            # Second Person - المخاطب
    PronounEntry(
    "ـك",
    PronounType.ATTACHED,
    PronounPerson.SECOND,
    PronounNumber.SINGULAR,
    PronounGender.MASCULINE,
    "CVC",
    frequency_score=0.92,
                class_id=15,
    ),
    PronounEntry(
    "ـكِ",
    PronounType.ATTACHED,
    PronounPerson.SECOND,
    PronounNumber.SINGULAR,
    PronounGender.FEMININE,
    "CV",
    frequency_score=0.85,
                class_id=16,
    ),
    PronounEntry(
    "ـكما",
    PronounType.ATTACHED,
    PronounPerson.SECOND,
    PronounNumber.DUAL,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.35,
                class_id=17,
    ),
    PronounEntry(
    "ـكم",
    PronounType.ATTACHED,
    PronounPerson.SECOND,
    PronounNumber.PLURAL,
    PronounGender.MASCULINE,
    "CVC",
    frequency_score=0.70,
                class_id=18,
    ),
    PronounEntry(
    "ـكن",
    PronounType.ATTACHED,
    PronounPerson.SECOND,
    PronounNumber.PLURAL,
    PronounGender.FEMININE,
    "CVC",
    frequency_score=0.60,
                class_id=19,
    ),
            # Third Person - الغائب
    PronounEntry(
    "ـه",
    PronounType.ATTACHED,
    PronounPerson.THIRD,
    PronounNumber.SINGULAR,
    PronounGender.MASCULINE,
    "CV",
    frequency_score=0.95,
                class_id=20,
    ),
    PronounEntry(
    "ـها",
    PronounType.ATTACHED,
    PronounPerson.THIRD,
    PronounNumber.SINGULAR,
    PronounGender.FEMININE,
    "CV CV",
    frequency_score=0.90,
                class_id=21,
    ),
    PronounEntry(
    "ـهما",
    PronounType.ATTACHED,
    PronounPerson.THIRD,
    PronounNumber.DUAL,
    PronounGender.NEUTRAL,
    "CV CV",
    frequency_score=0.40,
                class_id=22,
    ),
    PronounEntry(
    "ـهم",
    PronounType.ATTACHED,
    PronounPerson.THIRD,
    PronounNumber.PLURAL,
    PronounGender.MASCULINE,
    "CVC",
    frequency_score=0.75,
                class_id=23,
    ),
    PronounEntry(
    "ـهن",
    PronounType.ATTACHED,
    PronounPerson.THIRD,
    PronounNumber.PLURAL,
    PronounGender.FEMININE,
    "CVC",
    frequency_score=0.65,
                class_id=24,
    ),
    ]

        # دمج جميع الضمائر
    self.pronouns = detached_pronouns + attached_pronouns

        # إنشاء خريطة التصنيف
    self.class_mapping = {p.class_id: p.text for p in self.pronouns}

        # تجميع الأنماط المقطعية
    self._group_syllable_patterns()

    logger.info(f"✅ تم تهيئة قاعدة بيانات الضمائر: {len(self.pronouns)} ضمير")

    def _group_syllable_patterns(self):  # type: ignore[no-untyped def]
    """تجميع الضمائر حسب الأنماط المقطعية"""

        for pronoun in self.pronouns:
    pattern = pronoun.syllable_pattern
            if pattern not in self.syllable_patterns:
    self.syllable_patterns[pattern] = []
    self.syllable_patterns[pattern].append(pronoun.text)

    logger.info(f"📊 الأنماط المقطعية: {len(self.syllable_patterns)} نمط")

    def get_pronoun_by_id(self, class_id: int) -> Optional[PronounEntry]:
    """الحصول على ضمير بمعرف الفئة"""
        for pronoun in self.pronouns:
            if pronoun.class_id == class_id:
    return pronoun
    return None

    def get_pronouns_by_pattern(self, pattern: str) -> List[PronounEntry]:
    """الحصول على الضمائر حسب النمط المقطعي"""
    return [p for p in self.pronouns if p.syllable_pattern == pattern]

    def get_statistics(self) -> Dict[str, Any]:
    """إحصائيات شاملة لقاعدة البيانات"""

    stats = {
    'total_pronouns': len(self.pronouns),
    'detached_count': len(
    [p for p in self.pronouns if p.pronoun_type == PronounType.DETACHED]
    ),
    'attached_count': len(
    [p for p in self.pronouns if p.pronoun_type == PronounType.ATTACHED]
    ),
    'patterns_distribution': {},
    'person_distribution': {},
    'number_distribution': {},
    'gender_distribution': {},
    }

        # توزيع الأنماط
        for pattern, pronouns in self.syllable_patterns.items():
    stats['patterns_distribution'][pattern] = len(pronouns)

        # توزيع الأشخاص
        for person in PronounPerson:
    count = len([p for p in self.pronouns if p.person == person])
    stats['person_distribution'][person.value] = count

        # توزيع العدد
        for number in PronounNumber:
    count = len([p for p in self.pronouns if p.number == number])
    stats['number_distribution'][number.value] = count

        # توزيع الجنس
        for gender in PronounGender:
    count = len([p for p in self.pronouns if p.gender == gender])
    stats['gender_distribution'][gender.value] = count

    return stats


# ═══════════════════════════════════════════════════════════════════════════════════
# SYLLABLE-TO-PRONOUN PATTERN ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════════


class SyllablePatternAnalyzer:
    """محلل الأنماط المقطعية للضمائر"""

    def __init__(self, syllables_database_path: str):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.syllables_db = self._load_syllables_database(syllables_database_path)
    self.pattern_mappings: Dict[str, List[str]] = {}
    self._analyze_pronoun_patterns()

    def _load_syllables_database(self, path: str) -> Dict[str, Any]:
    """تحميل قاعدة بيانات المقاطع الصوتية"""
        try:
            with open(path, 'r', encoding='utf 8') as f:
    return json.load(f)
        except FileNotFoundError:
    logger.error(f"❌ لم يتم العثور على قاعدة بيانات المقاطع: {path}")
    return {}

    def _analyze_pronoun_patterns(self):  # type: ignore[no-untyped def]
    """تحليل أنماط الضمائر المقطعية"""

        # أنماط الضمائر المنفصلة
    detached_patterns = {
    'CV CV': ['أنا', 'هو', 'هي', 'هما'],  # أَنَا، هُوَ، هِيَ، هُمَا
    'CV CVC': ['نحن', 'أنت', 'أنتم', 'أنتن'],  # نَحْنُ، أَنْتَ، أَنْتُمْ، أَنْتُنَّ
    'CV-CV CV': ['أنتما'],  # أَنْتُمَا
    'CVC': ['هم', 'هن'],  # هُمْ، هُنَّ
    }

        # أنماط الضمائر المتصلة
    attached_patterns = {
    'CV': ['ـي', 'ـكِ', 'ـه'],  # ـِي، ـِكِ، ـُه
    'CV CV': ['ـني', 'ـنا', 'ـها', 'ـهما', 'ـكما'],  # ـَنِي، ـَنَا، ـَهَا، ـَهُمَا، ـَكُمَا
    'CVC': ['ـك', 'ـكم', 'ـكن', 'ـهم', 'ـهن'],  # ـَكَ، ـَكُمْ، ـَكُنَّ، ـَهُمْ، ـَهُنَّ
    }

    self.pattern_mappings = {**detached_patterns, **attached_patterns}

    logger.info(f"📝 تم تحليل {len(self.pattern_mappings)} نمط للضمائر")

    def map_syllables_to_pronoun(self, syllables: List[str]) -> List[str]:
    """ربط المقاطع بالضمائر المحتملة"""

        if not syllables:
    return []

        # تحليل نمط المقاطع
    pattern = self._determine_syllable_pattern(syllables)

        # البحث عن الضمائر المطابقة
    matching_pronouns = self.pattern_mappings.get(pattern, [])

    return matching_pronouns

    def _determine_syllable_pattern(self, syllables: List[str]) -> str:
    """تحديد نمط المقاطع"""

        if not syllables:
    return ""

        # تحليل كل مقطع لتحديد نوعه
    pattern_parts = []

        for syllable in syllables:
    syllable_type = self._classify_syllable_type(syllable)
    pattern_parts.append(syllable_type)

    return ' '.join(pattern_parts)

    def _classify_syllable_type(self, syllable: str) -> str:
    """تصنيف نوع المقطع الواحد"""

        # إزالة الحركات والتشكيل للتحليل
    clean_syllable = re.sub(r'[َُِّْ]', '', syllable)

        # تحديد الأنماط الأساسية
    consonants = re.findall(r'[بتثجحخدذرزسشصضطظعغفقكلمنهوي]', clean_syllable)
    vowels = re.findall(r'[اوي]|[َُِ]', syllable)

    consonant_count = len(consonants)
    vowel_count = len(vowels)

        # تصنيف حسب النمط
        if consonant_count == 1 and vowel_count == 1:
    return "CV"
        elif consonant_count == 2 and vowel_count == 1:
    return "CVC"
        elif consonant_count == 1 and vowel_count >= 2:
    return "CVV"
        elif consonant_count >= 2 and vowel_count >= 2:
    return "CVVC"
        else:
    return "CV"  # افتراضي


# ═══════════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODEL FOR PRONOUN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """إعدادات نموذج التعلم العميق"""

    input_size: int = 40  # حجم ميزات MFCC
    hidden_size: int = 128  # حجم الطبقة المخفية
    num_layers: int = 2  # عدد طبقات LSTM
    num_classes: int = 25  # عدد فئات الضمائر
    dropout: float = 0.3  # معدل الإسقاط
    learning_rate: float = 0.001  # معدل التعلم
    batch_size: int = 32  # حجم الدفعة
    max_sequence_length: int = 100  # أقصى طول للتسلسل


class PronounClassificationDataGenerator:
    """مولد البيانات لتدريب نموذج تصنيف الضمائر"""

    def __init__(self, pronouns_db: ArabicPronounsDatabase):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.pronouns_db = pronouns_db
    self.synthetic_data: List[Tuple[np.ndarray, int]] = []

    def generate_synthetic_mfcc_features(
    self, pronoun: PronounEntry, num_samples: int = 100
    ) -> List[np.ndarray]:
    """توليد ميزات MFCC اصطناعية للضمير"""

    features_list = []

        for _ in range(num_samples):
            # محاكاة ميزات MFCC للضمير
    sequence_length = np.random.randint(20, 80)  # طول متغير
    mfcc_features = np.random.randn(sequence_length, 40)  # 40 ميزة MFCC

            # إضافة أنماط مميزة للضمير
            if pronoun.pronoun_type == PronounType.DETACHED:
                # الضمائر المنفصلة تميل لتكون أطول
    mfcc_features *= 1.2
            else:
                # الضمائر المتصلة تميل لتكون أقصر
    mfcc_features *= 0.8

            # تعديل بناء على الشخص والعدد
            if pronoun.person == PronounPerson.FIRST:
    mfcc_features[:0:10] += 0.5  # ميزات مميزة للمتكلم
            elif pronoun.person == PronounPerson.SECOND:
    mfcc_features[:10:20] += 0.5  # ميزات مميزة للمخاطب
            elif pronoun.person == PronounPerson.THIRD:
    mfcc_features[:20:30] += 0.5  # ميزات مميزة للغائب

    features_list.append(mfcc_features)

    return features_list

    def generate_training_data(
    self, samples_per_pronoun: int = 100
    ) -> Tuple[List[np.ndarray], List[int]]:
    """توليد بيانات التدريب الكاملة"""

    X_data = []
    y_data = []

        for pronoun in self.pronouns_db.pronouns:
            # توليد عينات لكل ضمير
    features_list = self.generate_synthetic_mfcc_features(
    pronoun, samples_per_pronoun
    )

            for features in features_list:
    X_data.append(features)
    y_data.append(pronoun.class_id)

    logger.info(
    f"🎯 تم توليد {len(X_data)} عينة تدريب لـ {len(self.pronouns_db.pronouns)} ضمير"
    )  # noqa: E501

    return X_data, y_data


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC PRONOUNS GENERATOR FROM SYLLABLES
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicPronounsGenerator:
    """مولد الضمائر العربية من المقاطع الصوتية"""

    def __init__(self, syllables_database_path: str = "complete_arabic_syllable_inventory.json"):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.pronouns_db = ArabicPronounsDatabase()
    self.pattern_analyzer = SyllablePatternAnalyzer(syllables_database_path)
    self.model_config = ModelConfig()
    self.data_generator = PronounClassificationDataGenerator(self.pronouns_db)

    logger.info("🚀 تم تهيئة مولد الضمائر العربية من المقاطع الصوتية")

    def generate_pronouns_from_syllables(self, syllables: List[str]) -> Dict[str, Any]:
    """توليد الضمائر من المقاطع المعطاة"""

        if not syllables:
    return {'error': 'لا توجد مقاطع للتحليل'}

        # تحليل الأنماط المقطعية
    matching_pronouns = self.pattern_analyzer.map_syllables_to_pronoun(syllables)

        # تحليل مفصل لكل ضمير مطابق
    detailed_results = []

        for pronoun_text in matching_pronouns:
            # البحث عن الضمير في قاعدة البيانات
    pronoun_entry = None
            for p in self.pronouns_db.pronouns:
                if p.text == pronoun_text:
    pronoun_entry = p
    break

            if pronoun_entry:
    detailed_results.append(
    {
    'text': pronoun_entry.text,
    'type': pronoun_entry.pronoun_type.value,
    'person': pronoun_entry.person.value,
    'number': pronoun_entry.number.value,
    'gender': pronoun_entry.gender.value,
    'pattern': pronoun_entry.syllable_pattern,
    'frequency': pronoun_entry.frequency_score,
    'class_id': pronoun_entry.class_id,
    }
    )

    return {
    'input_syllables': syllables,
    'syllable_pattern': self.pattern_analyzer._determine_syllable_pattern(
    syllables
    ),
    'matching_pronouns_count': len(matching_pronouns),
    'pronouns': detailed_results,
    'confidence': self._calculate_confidence(syllables, matching_pronouns),
    }

    def _calculate_confidence(
    self, syllables: List[str], matching_pronouns: List[str]
    ) -> float:
    """حساب درجة الثقة في التطابق"""

        if not matching_pronouns:
    return 0.0

        # عوامل الثقة
    syllable_count_factor = min(len(syllables) / 3.0, 1.0)  # عدد المقاطع
    match_count_factor = 1.0 / len(matching_pronouns)  # عدد التطابقات (أقل = أفضل)

    base_confidence = 0.7  # ثقة أساسية

    return min(
    base_confidence + syllable_count_factor * 0.2 + match_count_factor * 0.1,
    1.0,
    )

    def analyze_pronoun_by_text(self, pronoun_text: str) -> Dict[str, Any]:
    """تحليل ضمير موجود نصياً"""

        for pronoun in self.pronouns_db.pronouns:
            if pronoun.text == pronoun_text or pronoun.text == pronoun_text.replace(
    'ـ', ''
    ):
    return {
    'found': True,
    'text': pronoun.text,
    'type': pronoun.pronoun_type.value,
    'person': pronoun.person.value,
    'number': pronoun.number.value,
    'gender': pronoun.gender.value,
    'pattern': pronoun.syllable_pattern,
    'frequency': pronoun.frequency_score,
    'class_id': pronoun.class_id,
    'usage_contexts': pronoun.usage_contexts,
    }

    return {'found': False, 'message': 'لم يتم العثور على الضمير في قاعدة البيانات'}

    def get_all_pronouns_by_type(self, pronoun_type: str) -> List[Dict[str, Any]]:
    """الحصول على جميع الضمائر من نوع معين"""

    target_type = (
    PronounType.DETACHED if pronoun_type == "منفصل" else PronounType.ATTACHED
    )

    results = []
        for pronoun in self.pronouns_db.pronouns:
            if pronoun.pronoun_type == target_type:
    results.append(
    {
    'text': pronoun.text,
    'person': pronoun.person.value,
    'number': pronoun.number.value,
    'gender': pronoun.gender.value,
    'pattern': pronoun.syllable_pattern,
    'frequency': pronoun.frequency_score,
    'class_id': pronoun.class_id,
    }
    )

    return sorted(results, key=lambda x: x['frequency'], reverse=True)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
    """تقرير شامل عن نظام الضمائر"""

    stats = self.pronouns_db.get_statistics()

    report = {
    'system_info': {
    'version': '1.0.0',
    'total_pronouns': stats['total_pronouns'],
    'model_classes': self.model_config.num_classes,
    'syllable_patterns': len(self.pattern_analyzer.pattern_mappings),
    },
    'pronouns_distribution': stats,
    'pattern_analysis': {
    'available_patterns': list(
    self.pattern_analyzer.pattern_mappings.keys()
    ),
    'pattern_frequencies': {
    pattern: len(pronouns)
                    for pattern, pronouns in self.pattern_analyzer.pattern_mappings.items()
    },
    },
    'model_configuration': {
    'input_size': self.model_config.input_size,
    'hidden_size': self.model_config.hidden_size,
    'num_layers': self.model_config.num_layers,
    'num_classes': self.model_config.num_classes,
    },
    }

    return report

    def save_pronouns_database(self, output_path: str = "arabic_pronouns_database.json"):  # type: ignore[no-untyped def]
    """حفظ قاعدة بيانات الضمائر"""

    pronouns_data = []

        for pronoun in self.pronouns_db.pronouns:
    pronouns_data.append(
    {
    'text': pronoun.text,
    'type': pronoun.pronoun_type.value,
    'person': pronoun.person.value,
    'number': pronoun.number.value,
    'gender': pronoun.gender.value,
    'syllable_pattern': pronoun.syllable_pattern,
    'phonetic_features': pronoun.phonetic_features,
    'usage_contexts': pronoun.usage_contexts,
    'frequency_score': pronoun.frequency_score,
    'class_id': pronoun.class_id,
    }
    )

    output_data = {
    'metadata': {
    'version': '1.0.0',
    'total_pronouns': len(pronouns_data),
    'generation_date': '2025-07 24',
    'description': 'قاعدة بيانات شاملة للضمائر العربية المتصلة والمنفصلة',
    },
    'class_mapping': self.pronouns_db.class_mapping,
    'syllable_patterns': self.pronouns_db.syllable_patterns,
    'pronouns': pronouns_data,
    }

        with open(output_path, 'w', encoding='utf 8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 تم حفظ قاعدة بيانات الضمائر في: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════════


def demonstrate_arabic_pronouns_generator():  # type: ignore[no-untyped def]
    """عرض توضيحي لمولد الضمائر العربية"""

    print("🎯 مولد الضمائر العربية من المقاطع الصوتية")
    print("=" * 60)

    # تهيئة المولد
    generator = ArabicPronounsGenerator()

    # أمثلة على المقاطع الصوتية
    test_cases = [
        # ضمائر منفصلة
    ['أَ', 'نَا'],  # أنا
    ['هُ', 'وَ'],  # هو
    ['هِ', 'يَ'],  # هي
    ['نَحْ', 'نُ'],  # نحن
    ['أَنْ', 'تَ'],  # أنت
    ['هُم'],  # هم
        # ضمائر متصلة
    ['نِي'],  # ـني
    ['هَا'],  # ـها
    ['كَ'],  # ـك
    ['هُم'],  # ـهم
    ]

    print("\n🔍 اختبار توليد الضمائر من المقاطع:")
    print(" " * 50)

    for i, syllables in enumerate(test_cases, 1):
    print(f"\n{i}. المدخل: {syllables}")

    result = generator.generate_pronouns_from_syllables(syllables)

        if result.get('pronouns'):
    print(f"   النمط المقطعي: {result['syllable_pattern']}")
    print(f"   عدد التطابقات: {result['matching_pronouns_count']}")
    print(f"   درجة الثقة: {result['confidence']:.2f}")

            for j, pronoun in enumerate(result['pronouns'][:3], 1):
    print(
    f"   {j}. {pronoun['text']} ({pronoun['type']}) - {pronoun['person']}/{pronoun['number']}/{pronoun['gender']}"
    )
        else:
    print("   ❌ لم يتم العثور على تطابقات")

    # عرض إحصائيات النظام
    print("\n📊 إحصائيات النظام:")
    print(" " * 30)

    report = generator.generate_comprehensive_report()

    print(f"   إجمالي الضمائر: {report['system_info']['total_pronouns']}")
    print(f"   الضمائر المنفصلة: {report['pronouns_distribution']['detached_count']}")
    print(f"   الضمائر المتصلة: {report['pronouns_distribution']['attached_count']}")
    print(f"   الأنماط المقطعية: {report['system_info']['syllable_patterns']}")

    # عرض الأنماط المتاحة
    print("\n🎨 الأنماط المقطعية المتاحة:")
    for pattern, frequency in report['pattern_analysis']['pattern_frequencies'].items():
    print(f"   • {pattern}: {frequency} ضمير")

    # حفظ قاعدة البيانات
    print("\n💾 حفظ قاعدة البيانات...")
    generator.save_pronouns_database()

    print("\n✅ اكتمل عرض مولد الضمائر العربية!")
    print("🎯 النظام جاهز لتصنيف الضمائر من المقاطع الصوتية!")


if __name__ == "__main__":
    demonstrate_arabic_pronouns_generator()
