#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Demonstrative Pronouns Generation System
=============================================
نظام توليد أسماء الإشارة العربية من المقاطع الصوتية

Comprehensive system for generating Arabic demonstrative pronouns from syllable
sequences using advanced pattern recognition and morphological analysis.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - DEMONSTRATIVE PRONOUNS SYSTEM
Date: 2025-07-24
Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


import logging  # noqa: F401
import json  # noqa: F401
import yaml  # noqa: F401
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict  # noqa: F401
from enum import Enum  # noqa: F401
from pathlib import Path  # noqa: F401
import numpy as np  # noqa: F401

# إعداد نظام التسجيل
logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIVE PRONOUNS CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════════


class DemonstrativeCategory(Enum):
    """فئات أسماء الإشارة"""

    NEAR_MASCULINE_SINGULAR = "قريب_مذكر_مفرد"  # هذا
    NEAR_FEMININE_SINGULAR = "قريب_مؤنث_مفرد"  # هذه
    FAR_MASCULINE_SINGULAR = "بعيد_مذكر_مفرد"  # ذلك
    FAR_FEMININE_SINGULAR = "بعيد_مؤنث_مفرد"  # تلك
    NEAR_MASCULINE_DUAL = "قريب_مذكر_مثنى"  # هذان/هذين
    NEAR_FEMININE_DUAL = "قريب_مؤنث_مثنى"  # هاتان/هاتين
    FAR_MASCULINE_DUAL = "بعيد_مذكر_مثنى"  # ذانك/ذينك
    FAR_FEMININE_DUAL = "بعيد_مؤنث_مثنى"  # تانك/تينك
    NEAR_PLURAL = "قريب_جمع"  # هؤلاء
    FAR_PLURAL = "بعيد_جمع"  # أولئك
    LOCATIVE_NEAR = "مكاني_قريب"  # هنا/هاهنا
    LOCATIVE_FAR = "مكاني_بعيد"  # هناك/هنالك


class GrammaticalCase(Enum):
    """الحالات الإعرابية"""

    NOMINATIVE = "مرفوع"  # المبتدأ، الفاعل
    ACCUSATIVE = "منصوب"  # المفعول به، اسم كان
    GENITIVE = "مجرور"  # المضاف إليه، مجرور بحرف جر


@dataclass
class DemonstrativePronoun:
    """تمثيل اسم الإشارة العربي"""

    text: str  # النص العربي (هذا، هذه، إلخ)
    category: DemonstrativeCategory  # الفئة
    syllables: List[str]  # المقاطع الصوتية
    phonetic_features: List[str]  # الخصائص الصوتية
    grammatical_case: GrammaticalCase  # الحالة الإعرابية
    distance: str  # قريب/بعيد
    gender: str  # مذكر/مؤنث/محايد
    number: str  # مفرد/مثنى/جمع
    usage_contexts: List[str]  # سياقات الاستخدام
    frequency_score: float  # درجة التكرار (0 1)
    morphological_pattern: str  # النمط الصرفي


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIVE PRONOUNS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicDemonstrativePronounsDatabase:
    """قاعدة بيانات أسماء الإشارة العربية"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.demonstrative_pronouns: List[DemonstrativePronoun] = []
    self.syllable_patterns: Dict[str, List[str]] = {}
    self._initialize_database()

    def _initialize_database(self):  # type: ignore[no-untyped def]
    """تهيئة قاعدة بيانات أسماء الإشارة"""

    demonstratives_data = [
            # للقريب - المفرد
    {
    "text": "هذا",
    "category": DemonstrativeCategory.NEAR_MASCULINE_SINGULAR,
    "syllables": ["هَا", "ذَا"],
    "phonetic_features": ["h", "aa", "dh", "aa"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "مذكر",
    "number": "مفرد",
    "usage_contexts": ["الإشارة للقريب المذكر", "التعريف", "التخصيص"],
    "frequency_score": 0.95,
    "morphological_pattern": "CV CV",
    },
    {
    "text": "هذه",
    "category": DemonstrativeCategory.NEAR_FEMININE_SINGULAR,
    "syllables": ["هَا", "ذِهِ"],
    "phonetic_features": ["h", "aa", "dh", "i", "h", "i"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "مؤنث",
    "number": "مفرد",
    "usage_contexts": ["الإشارة للقريبة المؤنثة", "التعريف", "التخصيص"],
    "frequency_score": 0.93,
    "morphological_pattern": "CV CVC",
    },
            # للبعيد - المفرد
    {
    "text": "ذلك",
    "category": DemonstrativeCategory.FAR_MASCULINE_SINGULAR,
    "syllables": ["ذَا", "لِكَ"],
    "phonetic_features": ["dh", "aa", "l", "i", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "مذكر",
    "number": "مفرد",
    "usage_contexts": ["الإشارة للبعيد المذكر", "التفسير", "الإحالة"],
    "frequency_score": 0.91,
    "morphological_pattern": "CV CVC",
    },
    {
    "text": "تلك",
    "category": DemonstrativeCategory.FAR_FEMININE_SINGULAR,
    "syllables": ["تِل", "كَ"],
    "phonetic_features": ["t", "i", "l", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "مؤنث",
    "number": "مفرد",
    "usage_contexts": ["الإشارة للبعيدة المؤنثة", "التفسير", "الإحالة"],
    "frequency_score": 0.89,
    "morphological_pattern": "CVC CV",
    },
            # للقريب - المثنى
    {
    "text": "هذان",
    "category": DemonstrativeCategory.NEAR_MASCULINE_DUAL,
    "syllables": ["هَا", "ذَا", "نِ"],
    "phonetic_features": ["h", "aa", "dh", "aa", "n", "i"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "مذكر",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنين قريبين مذكرين", "التخصيص المزدوج"],
    "frequency_score": 0.72,
    "morphological_pattern": "CV-CV CV",
    },
    {
    "text": "هذين",
    "category": DemonstrativeCategory.NEAR_MASCULINE_DUAL,
    "syllables": ["هَا", "ذَيْ", "نِ"],
    "phonetic_features": ["h", "aa", "dh", "ay", "n", "i"],
    "grammatical_case": GrammaticalCase.ACCUSATIVE,
    "distance": "قريب",
    "gender": "مذكر",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنين قريبين مذكرين منصوب/مجرور"],
    "frequency_score": 0.68,
    "morphological_pattern": "CV-CVC CV",
    },
    {
    "text": "هاتان",
    "category": DemonstrativeCategory.NEAR_FEMININE_DUAL,
    "syllables": ["هَا", "تَا", "نِ"],
    "phonetic_features": ["h", "aa", "t", "aa", "n", "i"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "مؤنث",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنتين قريبتين مؤنثتين"],
    "frequency_score": 0.65,
    "morphological_pattern": "CV-CV CV",
    },
    {
    "text": "هاتين",
    "category": DemonstrativeCategory.NEAR_FEMININE_DUAL,
    "syllables": ["هَا", "تَيْ", "نِ"],
    "phonetic_features": ["h", "aa", "t", "ay", "n", "i"],
    "grammatical_case": GrammaticalCase.ACCUSATIVE,
    "distance": "قريب",
    "gender": "مؤنث",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنتين قريبتين مؤنثتين منصوب/مجرور"],
    "frequency_score": 0.62,
    "morphological_pattern": "CV-CVC CV",
    },
            # للبعيد - المثنى
    {
    "text": "ذانك",
    "category": DemonstrativeCategory.FAR_MASCULINE_DUAL,
    "syllables": ["ذَا", "نِ", "كَ"],
    "phonetic_features": ["dh", "aa", "n", "i", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "مذكر",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنين بعيدين مذكرين"],
    "frequency_score": 0.45,
    "morphological_pattern": "CV-CV CV",
    },
    {
    "text": "تانك",
    "category": DemonstrativeCategory.FAR_FEMININE_DUAL,
    "syllables": ["تَا", "نِ", "كَ"],
    "phonetic_features": ["t", "aa", "n", "i", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "مؤنث",
    "number": "مثنى",
    "usage_contexts": ["الإشارة لاثنتين بعيدتين مؤنثتين"],
    "frequency_score": 0.42,
    "morphological_pattern": "CV-CV CV",
    },
            # للجمع
    {
    "text": "هؤلاء",
    "category": DemonstrativeCategory.NEAR_PLURAL,
    "syllables": ["هَا", "ؤُ", "لَا", "ءِ"],
    "phonetic_features": ["h", "aa", "u", "l", "aa", "i"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "محايد",
    "number": "جمع",
    "usage_contexts": ["الإشارة لجمع قريب", "العموم القريب"],
    "frequency_score": 0.87,
    "morphological_pattern": "CV-CV-CV CV",
    },
    {
    "text": "أولئك",
    "category": DemonstrativeCategory.FAR_PLURAL,
    "syllables": ["أُو", "لَا", "ئِ", "كَ"],
    "phonetic_features": ["u", "w", "l", "aa", "i", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "محايد",
    "number": "جمع",
    "usage_contexts": ["الإشارة لجمع بعيد", "العموم البعيد"],
    "frequency_score": 0.84,
    "morphological_pattern": "CVC-CV-CV CV",
    },
            # أسماء الإشارة المكانية
    {
    "text": "هنا",
    "category": DemonstrativeCategory.LOCATIVE_NEAR,
    "syllables": ["هُ", "نَا"],
    "phonetic_features": ["h", "u", "n", "aa"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "محايد",
    "number": "مكاني",
    "usage_contexts": ["الإشارة المكانية القريبة", "تحديد المكان"],
    "frequency_score": 0.92,
    "morphological_pattern": "CV CV",
    },
    {
    "text": "هاهنا",
    "category": DemonstrativeCategory.LOCATIVE_NEAR,
    "syllables": ["هَا", "هُ", "نَا"],
    "phonetic_features": ["h", "aa", "h", "u", "n", "aa"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "قريب",
    "gender": "محايد",
    "number": "مكاني",
    "usage_contexts": ["الإشارة المكانية القريبة المؤكدة"],
    "frequency_score": 0.58,
    "morphological_pattern": "CV-CV CV",
    },
    {
    "text": "هناك",
    "category": DemonstrativeCategory.LOCATIVE_FAR,
    "syllables": ["هُ", "نَا", "كَ"],
    "phonetic_features": ["h", "u", "n", "aa", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "محايد",
    "number": "مكاني",
    "usage_contexts": ["الإشارة المكانية البعيدة", "تحديد المكان البعيد"],
    "frequency_score": 0.89,
    "morphological_pattern": "CV-CV CV",
    },
    {
    "text": "هنالك",
    "category": DemonstrativeCategory.LOCATIVE_FAR,
    "syllables": ["هُ", "نَا", "لِ", "كَ"],
    "phonetic_features": ["h", "u", "n", "aa", "l", "i", "k", "a"],
    "grammatical_case": GrammaticalCase.NOMINATIVE,
    "distance": "بعيد",
    "gender": "محايد",
    "number": "مكاني",
    "usage_contexts": ["الإشارة المكانية البعيدة المؤكدة"],
    "frequency_score": 0.73,
    "morphological_pattern": "CV-CV-CV CV",
    },
    ]

        # إنشاء كائنات أسماء الإشارة
        for data in demonstratives_data:
    demonstrative = DemonstrativePronoun(**data)
    self.demonstrative_pronouns.append(demonstrative)

        # بناء أنماط المقاطع
    self._build_syllable_patterns()

    logger.info()
    f"✅ تم تهيئة قاعدة بيانات أسماء الإشارة: {len(self.demonstrative_pronouns)} اسم إشارة"
    )  # noqa: E501

    def _build_syllable_patterns(self):  # type: ignore[no-untyped def]
    """بناء خريطة الأنماط المقطعية"""

        for demonstrative in self.demonstrative_pronouns:
    pattern = ' '.join()
    [self._get_syllable_type(syll) for syll in demonstrative.syllables]
    )

            if pattern not in self.syllable_patterns:
    self.syllable_patterns[pattern] = []

    self.syllable_patterns[pattern].append(demonstrative.text)

    logger.info(f"📊 الأنماط المقطعية: {len(self.syllable_patterns)} نمط")

    def _get_syllable_type(self, syllable: str) -> str:
    """تحديد نوع المقطع (CV, CVC, إلخ)"""

    consonants = "بتثجحخدذرزسشصضطظعغفقكلمنهويءآأإئؤة"
    vowels = "اوييةَُِْ"

    pattern = ""
        for char in syllable:
            if char in consonants:
    pattern += "C"
            elif char in vowels:
    pattern += "V"

        # تبسيط الأنماط المعقدة
        if len(pattern) > 4:
    return "COMPLEX"

    return pattern if pattern else "CV"

    def get_by_category()
    self, category: DemonstrativeCategory
    ) -> List[DemonstrativePronoun]:
    """الحصول على أسماء الإشارة حسب الفئة"""
    return [d for d in self.demonstrative_pronouns if d.category == category]

    def get_by_distance(self, distance: str) -> List[DemonstrativePronoun]:
    """الحصول على أسماء الإشارة حسب المسافة"""
    return [d for d in self.demonstrative_pronouns if d.distance == distance]

    def get_by_number(self, number: str) -> List[DemonstrativePronoun]:
    """الحصول على أسماء الإشارة حسب العدد"""
    return [d for d in self.demonstrative_pronouns if d.number == number]


# ═══════════════════════════════════════════════════════════════════════════════════
# SYLLABLE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════════


class DemonstrativeSyllableAnalyzer:
    """محلل المقاطع الصوتية لأسماء الإشارة"""

    def __init__(self, database: ArabicDemonstrativePronounsDatabase):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.database = database
    self.phoneme_weights = self._initialize_phoneme_weights()

    def _initialize_phoneme_weights(self) -> Dict[str, float]:
    """تهيئة أوزان الأصوات لتحسين التطابق"""

    return {
            # أصوات شائعة في أسماء الإشارة
    'h': 0.95,  # ه (هذا، هذه، هنا)
    'dh': 0.90,  # ذ (هذا، هذه، ذلك)
    'l': 0.85,  # ل (ذلك، هؤلاء، هنالك)
    'k': 0.80,  # ك (ذلك، تلك، هناك)
    't': 0.75,  # ت (تلك، هاتان)
    'n': 0.70,  # ن (هذان، هاتان، هنا)
    'aa': 0.88,  # ا (معظم أسماء الإشارة)
    'a': 0.82,  # َ
    'i': 0.78,  # ِ
    'u': 0.75,  # ُ
    'ay': 0.70,  # ي (في المثنى منصوب/مجرور)
    'w': 0.65,  # و (أولئك)
    }

    def analyze_syllables(self, syllables: List[str]) -> Dict[str, Any]:
    """تحليل المقاطع الصوتية"""

    analysis = {
    'syllables_count': len(syllables),
    'syllable_types': [],
    'pattern': None,
    'complexity_score': 0.0,
    'phonetic_features': [],
    'matching_candidates': [],
    }

        # تحليل كل مقطع
        for syllable in syllables:
    syll_type = self.database._get_syllable_type(syllable)
    analysis['syllable_types'].append(syll_type)

            # استخراج الخصائص الصوتية
    phonetic = self._extract_phonetic_features(syllable)
    analysis['phonetic_features'].extend(phonetic)

        # تحديد النمط
    analysis['pattern'] = ' '.join(analysis['syllable_types'])

        # حساب درجة التعقيد
    analysis['complexity_score'] = self._calculate_complexity(syllables)

        # البحث عن المرشحين المطابقين
    analysis['matching_candidates'] = self._find_matching_candidates(syllables)

    return analysis

    def _extract_phonetic_features(self, syllable: str) -> List[str]:
    """استخراج الخصائص الصوتية من المقطع"""

        # تحويل الحروف العربية إلى تمثيل صوتي
    phonetic_map = {
    'ه': 'h',
    'ذ': 'dh',
    'ل': 'l',
    'ك': 'k',
    'ت': 't',
    'ن': 'n',
    'ا': 'aa',
    'و': 'w',
    'ي': 'y',
    'ء': 'q',
    'أ': 'a',
    'إ': 'i',
    'ُ': 'u',
    'ِ': 'i',
    'َ': 'a',
    'ْ': '',
    'ة': 'h',
    'ؤ': 'u',
    'ئ': 'i',
    'آ': 'aa',
    }

    features = []
        for char in syllable:
            if char in phonetic_map:
    phonetic = phonetic_map[char]
                if phonetic:  # تجاهل الحركات الفارغة
    features.append(phonetic)

    return features

    def _calculate_complexity(self, syllables: List[str]) -> float:
    """حساب درجة تعقيد المقاطع"""

    complexity = 0.0

        # عدد المقاطع
    complexity += len(syllables) * 0.2

        # تنوع أنواع المقاطع
    types = set(self.database._get_syllable_type(s) for s in syllables)
    complexity += len(types) * 0.3

        # وجود مقاطع معقدة
        for syllable in syllables:
            if len(syllable) > 3:
    complexity += 0.5

    return min(complexity, 5.0)  # حد أقصى 5

    def _find_matching_candidates()
    self, input_syllables: List[str]
    ) -> List[Dict[str, Any]]:
    """البحث عن أسماء الإشارة المطابقة"""

    candidates = []

        for demonstrative in self.database.demonstrative_pronouns:
    similarity = self._calculate_similarity()
    input_syllables, demonstrative.syllables
    )

            if similarity > 0.3:  # حد أدنى للتشابه
    candidate = {
    'demonstrative': demonstrative.text,
    'category': demonstrative.category.value,
    'similarity': similarity,
    'syllables': demonstrative.syllables,
    'distance': demonstrative.distance,
    'gender': demonstrative.gender,
    'number': demonstrative.number,
    'frequency_score': demonstrative.frequency_score,
    }
    candidates.append(candidate)

        # ترتيب حسب التشابه
    candidates.sort(key=lambda x: x['similarity'], reverse=True)

    return candidates[:5]  # أفضل 5 مرشحين

    def _calculate_similarity()
    self, syllables1: List[str], syllables2: List[str]
    ) -> float:
    """حساب التشابه بين مجموعتين من المقاطع"""

        # التشابه في العدد
    length_similarity = 1.0 - abs(len(syllables1) - len(syllables2)) / max()
    len(syllables1), len(syllables2)
    )

        # التشابه في المحتوى
    content_similarity = 0.0
    max_length = max(len(syllables1), len(syllables2))

        for i in range(max_length):
            if i < len(syllables1) and i < len(syllables2):
                # مقارنة المقاطع المتناظرة
    syll_sim = self._syllable_similarity(syllables1[i], syllables2[i])
    content_similarity += syll_sim
            else:
                # عقوبة للمقاطع المفقودة
    content_similarity += 0.0

    content_similarity /= max_length

        # الوزن النهائي
    final_similarity = (length_similarity * 0.3) + (content_similarity * 0.7)

    return final_similarity

    def _syllable_similarity(self, syll1: str, syll2: str) -> float:
    """حساب التشابه بين مقطعين"""

        if syll1 == syll2:
    return 1.0

        # مقارنة الخصائص الصوتية
    features1 = self._extract_phonetic_features(syll1)
    features2 = self._extract_phonetic_features(syll2)

        if not features1 or not features2:
    return 0.0

        # حساب التشابه الصوتي
    common = set(features1) & set(features2)
    total = set(features1) | set(features2)

        if not total:
    return 0.0

    jaccard_similarity = len(common) / len(total)

        # تطبيق أوزان الأصوات
    weighted_similarity = 0.0
        for feature in common:
    weight = self.phoneme_weights.get(feature, 0.5)
    weighted_similarity += weight

        if features1:
    weighted_similarity /= len(features1)

        # متوسط التشابه
    return (jaccard_similarity + weighted_similarity) / 2


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicDemonstrativePronounsGenerator:
    """مولد أسماء الإشارة العربية من المقاطع الصوتية"""

    def __init__(self, config_path: Optional[str] = None):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.config = self._load_config(config_path)
    self.demonstrative_pronouns_db = ArabicDemonstrativePronounsDatabase()
    self.syllable_analyzer = DemonstrativeSyllableAnalyzer()
    self.demonstrative_pronouns_db
    )

    logger.info("🚀 تم تهيئة مولد أسماء الإشارة العربية من المقاطع الصوتية")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
    """تحميل إعدادات النظام"""

        default_config = {
    'similarity_threshold': 0.6,
    'max_candidates': 5,
    'phonetic_weight': 0.7,
    'frequency_weight': 0.3,
    'enable_fuzzy_matching': True,
    'case_sensitive': False,
    }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf 8') as f:
    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
    logger.warning(f"تعذر تحميل ملف الإعدادات: {e}")

    return default_config

    def generate_demonstrative_pronouns_from_syllables()
    self, syllables: List[str]
    ) -> Dict[str, Any]:
    """توليد أسماء الإشارة من المقاطع الصوتية"""

    logger.info(f"🔍 البحث عن أسماء الإشارة للمقاطع: {syllables}")

        if not syllables:
    return {
    'success': False,
    'error': 'قائمة المقاطع فارغة',
    'syllables': syllables,
    }

        # تحليل المقاطع
    analysis = self.syllable_analyzer.analyze_syllables(syllables)

        # تطبيق خوارزمية التطابق المتقدمة
    matches = self._advanced_matching(syllables, analysis)

        if not matches:
    return {
    'success': False,
    'error': 'لم يتم العثور على تطابق مناسب',
    'syllables': syllables,
    'analysis': analysis,
    }

        # تحديد أفضل تطابق
    best_match = self._select_best_match(matches)

    result = {
    'success': True,
    'best_match': best_match,
    'all_matches': matches,
    'syllables': syllables,
    'analysis': analysis,
    'confidence': best_match['confidence'],
    }

    logger.info()
    f"✅ تم العثور على تطابق: {best_match['demonstrative']} بثقة {best_match['confidence']:.2f}}"
    )  # noqa: E501

    return result

    def _advanced_matching()
    self, syllables: List[str], analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
    """خوارزمية التطابق المتقدمة"""

    matches = []

        # البحث المباشر في المرشحين
        for candidate in analysis['matching_candidates']:
    confidence = self._calculate_confidence(syllables, candidate)

            if confidence >= self.config['similarity_threshold']:
    match = {
    'demonstrative': candidate['demonstrative'],
    'category': candidate['category'],
    'confidence': confidence,
    'similarity': candidate['similarity'],
    'syllables': candidate['syllables'],
    'distance': candidate['distance'],
    'gender': candidate['gender'],
    'number': candidate['number'],
    'frequency_score': candidate['frequency_score'],
    'matching_method': 'direct_similarity',
    }
    matches.append(match)

        # البحث بالنمط المقطعي
    pattern_matches = self._pattern_based_matching(syllables, analysis['pattern'])
    matches.extend(pattern_matches)

        # البحث الضبابي إذا كان مفعلاً
        if self.config['enable_fuzzy_matching'] and len(matches) < 3:
    fuzzy_matches = self._fuzzy_matching(syllables)
    matches.extend(fuzzy_matches)

        # إزالة المكررات وترتيب النتائج
    unique_matches = self._deduplicate_matches(matches)
    unique_matches.sort(key=lambda x: x['confidence'], reverse=True)

    return unique_matches[: self.config['max_candidates']]

    def _calculate_confidence()
    self, syllables: List[str], candidate: Dict[str, Any]
    ) -> float:
    """حساب درجة الثقة للمرشح"""

        # مكونات الثقة
    similarity_score = candidate['similarity']
    frequency_score = candidate['frequency_score']

        # تطابق العدد (عدد المقاطع)
    length_match = 1.0 if len(syllables) == len(candidate['syllables']) else 0.7

        # الوزن النهائي
    confidence = ()
    similarity_score * self.config['phonetic_weight']
    + frequency_score * self.config['frequency_weight']
    ) * length_match

    return min(confidence, 1.0)

    def _pattern_based_matching()
    self, syllables: List[str], pattern: str
    ) -> List[Dict[str, Any]]:
    """البحث المعتمد على النمط المقطعي"""

    matches = []

        if pattern in self.demonstrative_pronouns_db.syllable_patterns:
            for demonstrative_text in self.demonstrative_pronouns_db.syllable_patterns[
    pattern
    ]:
                # البحث عن اسم الإشارة في قاعدة البيانات
    demonstrative = next()
    ()
    d
                        for d in self.demonstrative_pronouns_db.demonstrative_pronouns
                        if d.text == demonstrative_text
    ),
    None)

                if demonstrative:
    confidence = 0.75  # ثقة متوسطة للتطابق النمطي

    match = {
    'demonstrative': demonstrative.text,
    'category': demonstrative.category.value,
    'confidence': confidence,
    'similarity': 0.8,
    'syllables': demonstrative.syllables,
    'distance': demonstrative.distance,
    'gender': demonstrative.gender,
    'number': demonstrative.number,
    'frequency_score': demonstrative.frequency_score,
    'matching_method': 'pattern_based',
    }
    matches.append(match)

    return matches

    def _fuzzy_matching(self, syllables: List[str]) -> List[Dict[str, Any]]:
    """البحث الضبابي للحالات الصعبة"""

    matches = []

        # البحث مع تساهل أكبر في التشابه
        for demonstrative in self.demonstrative_pronouns_db.demonstrative_pronouns:
    similarity = self.syllable_analyzer._calculate_similarity()
    syllables, demonstrative.syllables
    )

            if similarity > 0.4:  # حد أدنى منخفض للبحث الضبابي
    confidence = similarity * 0.8  # ثقة منخفضة للبحث الضبابي

    match = {
    'demonstrative': demonstrative.text,
    'category': demonstrative.category.value,
    'confidence': confidence,
    'similarity': similarity,
    'syllables': demonstrative.syllables,
    'distance': demonstrative.distance,
    'gender': demonstrative.gender,
    'number': demonstrative.number,
    'frequency_score': demonstrative.frequency_score,
    'matching_method': 'fuzzy_matching',
    }
    matches.append(match)

    return matches

    def _deduplicate_matches()
    self, matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
    """إزالة التطابقات المكررة"""

    seen = set()
    unique_matches = []

        for match in matches:
    demonstrative = match['demonstrative']
            if demonstrative not in seen:
    seen.add(demonstrative)
    unique_matches.append(match)

    return unique_matches

    def _select_best_match(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """اختيار أفضل تطابق"""

        if not matches:
    return {}

        # الترتيب حسب الثقة ثم التكرار
    matches.sort()
    key=lambda x: (x['confidence'], x['frequency_score']), reverse=True
    )

    return matches[0]

    def get_statistics(self) -> Dict[str, Any]:
    """إحصائيات النظام"""

    stats = {
    'total_demonstratives': len()
    self.demonstrative_pronouns_db.demonstrative_pronouns
    ),
    'categories': {},
    'distances': {},
    'numbers': {},
    'genders': {},
    'syllable_patterns': len(self.demonstrative_pronouns_db.syllable_patterns),
    }

        for demonstrative in self.demonstrative_pronouns_db.demonstrative_pronouns:
            # إحصائيات الفئات
    cat = demonstrative.category.value
    stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

            # إحصائيات المسافات
    dist = demonstrative.distance
    stats['distances'][dist] = stats['distances'].get(dist, 0) + 1

            # إحصائيات الأعداد
    num = demonstrative.number
    stats['numbers'][num] = stats['numbers'].get(num, 0) + 1

            # إحصائيات الأجناس
    gen = demonstrative.gender
    stats['genders'][gen] = stats['genders'].get(gen, 0) + 1

    return stats

    def save_database(self, output_path: str = "arabic_demonstrative_pronouns_database.json"):  # type: ignore[no-untyped def]
    """حفظ قاعدة البيانات"""

    data = {
    'demonstrative_pronouns': [
    asdict(d) for d in self.demonstrative_pronouns_db.demonstrative_pronouns
    ],
    'syllable_patterns': self.demonstrative_pronouns_db.syllable_patterns,
    'statistics': self.get_statistics(),
    }

        with open(output_path, 'w', encoding='utf 8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"💾 تم حفظ قاعدة بيانات أسماء الإشارة في: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════════


def main():  # type: ignore[no-untyped def]
    """تشغيل تجريبي للنظام"""

    print("🎯 مولد أسماء الإشارة العربية من المقاطع الصوتية")
    print("=" * 50)

    # إنشاء المولد
    generator = ArabicDemonstrativePronounsGenerator()

    # اختبارات تجريبية
    test_cases = [
        # للقريب
    ["هَا", "ذَا"],  # هذا
    ["هَا", "ذِهِ"],  # هذه
    ["هَا", "ذَا", "نِ"],  # هذان
    ["هَا", "تَا", "نِ"],  # هاتان
    ["هَا", "ؤُ", "لَا", "ءِ"],  # هؤلاء
        # للبعيد
    ["ذَا", "لِكَ"],  # ذلك
    ["تِل", "كَ"],  # تلك
    ["أُو", "لَا", "ئِ", "كَ"],  # أولئك
        # المكانية
    ["هُ", "نَا"],  # هنا
    ["هُ", "نَا", "كَ"],  # هناك
        # اختبارات خاطئة
    ["بَا", "رِد"],  # غير صحيح
    ["كِ", "تَا", "ب"],  # غير صحيح
    ]

    print(f"\n🔬 تشغيل {len(test_cases)} اختبار:")
    print(" " * 40)

    successful = 0

    for i, syllables in enumerate(test_cases, 1):
    print(f"\n{i}. المقاطع: {syllables}")

    result = generator.generate_demonstrative_pronouns_from_syllables(syllables)

        if result['success']:
    best = result['best_match']
    print(f"   ✅ النتيجة: {best['demonstrative']}")
    print(f"   📊 الفئة: {best['category']}")
    print()
    f"   📍 المسافة: {best['distance']} | النوع: {best['gender']} | العدد: {best['number']}"
    )  # noqa: E501
    print(f"   🎯 الثقة: {best['confidence']:.2f}")
    successful += 1
        else:
    print(f"   ❌ فشل: {result['error']}")

    print("\n📈 النتائج النهائية:")
    print()
    f"   الناجح: {successful}/{len(test_cases)} ({successful/len(test_cases)*100:.1f%)}"
    )  # noqa: E501

    # إحصائيات النظام
    stats = generator.get_statistics()
    print("\n📊 إحصائيات النظام:")
    print(f"   إجمالي أسماء الإشارة: {stats['total_demonstratives']}")
    print(f"   الأنماط المقطعية: {stats['syllable_patterns']}")
    print(f"   الفئات: {len(stats['categories'])}")

    # حفظ قاعدة البيانات
    generator.save_database()

    print("\n✅ اكتمل العرض التوضيحي!")


if __name__ == "__main__":
    main()

