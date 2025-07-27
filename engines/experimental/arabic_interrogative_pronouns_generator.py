#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Interrogative Pronouns Generation System
==============================================
نظام توليد أسماء الاستفهام العربية من المقاطع الصوتية,
    Advanced system for generating Arabic interrogative pronouns (question words)

Author: Arabic NLP Expert Team - GitHub Copilot,
    Version: 1.0.0 - INTERROGATIVE PRONOUNS GENERATOR,
    Date: 2025-07-24,
    Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc
    import logging  # noqa: F401
    import json  # noqa: F401
    import yaml  # noqa: F401
    import numpy as np  # noqa: F401
    from datetime import datetime  # noqa: F401
    from typing import Dict, List, Any, Optional, Tuple
    from dataclasses import dataclass, asdict  # noqa: F401
    from enum import Enum  # noqa: F401
    from pathlib import Path  # noqa: F401
    import re  # noqa: F401

# إعداد نظام السجلات,
    logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# INTERROGATIVE PRONOUNS CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativeCategory(Enum):
    """فئات أسماء الاستفهام"""

    PERSON = "شخص"  # للسؤال عن الأشخاص,
    THING = "شيء"  # للسؤال عن الأشياء,
    TIME = "زمان"  # للسؤال عن الزمن,
    PLACE = "مكان"  # للسؤال عن المكان,
    MANNER = "كيفية"  # للسؤال عن الطريقة,
    QUANTITY = "كمية"  # للسؤال عن الكمية,
    CHOICE = "اختيار"  # للاختيار,
    REASON = "سبب"  # للسؤال عن السبب,
    STATE = "حال"  # للسؤال عن الحال,
    POSSESSION = "ملكية"  # للسؤال عن الملكية


@dataclass,
    class InterrogativePronoun:
    """بيانات اسم الاستفهام"""

    text: str,
    category: InterrogativeCategory,
    syllables: List[str]
    phonemes: List[str]
    frequency_score: float,
    usage_contexts: List[str]
    grammatical_cases: List[str]
    semantic_features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
    """تحويل إلى قاموس"""
    return {
    'text': self.text,
    'category': self.category.value,
    'syllables': self.syllables,
    'phonemes': self.phonemes,
    'frequency_score': self.frequency_score,
    'usage_contexts': self.usage_contexts,
    'grammatical_cases': self.grammatical_cases,
    'semantic_features': self.semantic_features,
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# INTERROGATIVE PRONOUNS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativePronounsDatabase:
    """قاعدة بيانات أسماء الاستفهام العربية"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.interrogative_pronouns: List[InterrogativePronoun] = []
    self.syllable_patterns: Dict[str, List[str]] = {}
    self.phoneme_patterns: Dict[str, List[str]] = {}
    self._initialize_database()

    def _initialize_database(self):  # type: ignore[no-untyped def]
    """تهيئة قاعدة بيانات أسماء الاستفهام"""

    interrogatives_data = [
            # أسماء الاستفهام عن الأشخاص
    {
    'text': 'مَن',
    'category': InterrogativeCategory.PERSON,
    'syllables': ['مَنْ'],
    'phonemes': ['م', 'َ', 'ن', 'ْ'],
    'frequency_score': 0.95,
    'usage_contexts': ['السؤال عن الأشخاص', 'الهوية', 'الفاعل'],
    'grammatical_cases': ['مرفوع', 'منصوب', 'مجرور'],
    'semantic_features': {
    'animacy': 'حي',
    'specificity': 'غير محدد',
    'formality': 'فصيح',
    },
    },
    {
    'text': 'مَنْ ذَا',
    'category': InterrogativeCategory.PERSON,
    'syllables': ['مَنْ', 'ذَا'],
    'phonemes': ['م', 'َ', 'ن', 'ْ', 'ذ', 'َ', 'ا'],
    'frequency_score': 0.65,
    'usage_contexts': ['السؤال عن الأشخاص بتأكيد', 'الاستفهام المؤكد'],
    'grammatical_cases': ['مرفوع', 'منصوب'],
    'semantic_features': {
    'animacy': 'حي',
    'specificity': 'غير محدد',
    'emphasis': 'مؤكد',
    },
    },
            # أسماء الاستفهام عن الأشياء
    {
    'text': 'مَا',
    'category': InterrogativeCategory.THING,
    'syllables': ['مَا'],
    'phonemes': ['م', 'َ', 'ا'],
    'frequency_score': 0.98,
    'usage_contexts': ['السؤال عن الأشياء', 'المفعول', 'التعريف'],
    'grammatical_cases': ['مرفوع', 'منصوب', 'مجرور'],
    'semantic_features': {
    'animacy': 'غير حي',
    'specificity': 'غير محدد',
    'scope': 'واسع',
    },
    },
    {
    'text': 'مَاذَا',
    'category': InterrogativeCategory.THING,
    'syllables': ['مَا', 'ذَا'],
    'phonemes': ['م', 'َ', 'ا', 'ذ', 'َ', 'ا'],
    'frequency_score': 0.85,
    'usage_contexts': ['السؤال عن الأشياء', 'المفعول المباشر'],
    'grammatical_cases': ['منصوب'],
    'semantic_features': {
    'animacy': 'غير حي',
    'specificity': 'غير محدد',
    'directness': 'مباشر',
    },
    },
            # أسماء الاستفهام عن الزمان
    {
    'text': 'مَتَى',
    'category': InterrogativeCategory.TIME,
    'syllables': ['مَ', 'تَى'],
    'phonemes': ['م', 'َ', 'ت', 'َ', 'ى'],
    'frequency_score': 0.92,
    'usage_contexts': ['السؤال عن الزمن', 'التوقيت', 'الأحداث'],
    'grammatical_cases': ['ظرف زمان'],
    'semantic_features': {
    'temporal': True,
    'specificity': 'غير محدد',
    'tense': 'مفتوح',
    },
    },
    {
    'text': 'أَيَّانَ',
    'category': InterrogativeCategory.TIME,
    'syllables': ['أَيْ', 'يَا', 'نَ'],
    'phonemes': ['أ', 'َ', 'ي', 'ْ', 'ي', 'َ', 'ا', 'ن', 'َ'],
    'frequency_score': 0.45,
    'usage_contexts': ['السؤال عن الزمن المستقبلي', 'الأحداث المهمة'],
    'grammatical_cases': ['ظرف زمان'],
    'semantic_features': {
    'temporal': True,
    'formality': 'فصيح جداً',
    'future_oriented': True,
    },
    },
            # أسماء الاستفهام عن المكان
    {
    'text': 'أَيْنَ',
    'category': InterrogativeCategory.PLACE,
    'syllables': ['أَيْ', 'نَ'],
    'phonemes': ['أ', 'َ', 'ي', 'ْ', 'ن', 'َ'],
    'frequency_score': 0.94,
    'usage_contexts': ['السؤال عن المكان', 'الموقع', 'الجهة'],
    'grammatical_cases': ['ظرف مكان'],
    'semantic_features': {
    'spatial': True,
    'specificity': 'غير محدد',
    'dimensionality': 'مكاني',
    },
    },
    {
    'text': 'أَنَّى',
    'category': InterrogativeCategory.PLACE,
    'syllables': ['أَنْ', 'نَى'],
    'phonemes': ['أ', 'َ', 'ن', 'ْ', 'ن', 'َ', 'ى'],
    'frequency_score': 0.55,
    'usage_contexts': ['السؤال عن المكان والطريقة', 'الكيفية المكانية'],
    'grammatical_cases': ['ظرف مكان', 'ظرف حال'],
    'semantic_features': {
    'spatial': True,
    'manner': True,
    'complexity': 'مركب',
    },
    },
            # أسماء الاستفهام عن الكيفية
    {
    'text': 'كَيْفَ',
    'category': InterrogativeCategory.MANNER,
    'syllables': ['كَيْ', 'فَ'],
    'phonemes': ['ك', 'َ', 'ي', 'ْ', 'ف', 'َ'],
    'frequency_score': 0.96,
    'usage_contexts': ['السؤال عن الطريقة', 'الحال', 'الكيفية'],
    'grammatical_cases': ['ظرف حال'],
    'semantic_features': {'manner': True, 'state': True, 'method': True},
    },
    {
    'text': 'كَيْفَمَا',
    'category': InterrogativeCategory.MANNER,
    'syllables': ['كَيْ', 'فَ', 'مَا'],
    'phonemes': ['ك', 'َ', 'ي', 'ْ', 'ف', 'َ', 'م', 'َ', 'ا'],
    'frequency_score': 0.35,
    'usage_contexts': ['السؤال عن أي طريقة', 'الشمولية'],
    'grammatical_cases': ['ظرف حال'],
    'semantic_features': {
    'manner': True,
    'universality': 'شامل',
    'indefiniteness': 'مطلق',
    },
    },
            # أسماء الاستفهام عن الكمية
    {
    'text': 'كَمْ',
    'category': InterrogativeCategory.QUANTITY,
    'syllables': ['كَمْ'],
    'phonemes': ['ك', 'َ', 'م', 'ْ'],
    'frequency_score': 0.89,
    'usage_contexts': ['السؤال عن العدد', 'الكمية', 'المقدار'],
    'grammatical_cases': ['مبني'],
    'semantic_features': {
    'quantitative': True,
    'numerical': True,
    'measure': True,
    },
    },
    {
    'text': 'كَأَيِّنْ',
    'category': InterrogativeCategory.QUANTITY,
    'syllables': ['كَأَيْ', 'يِنْ'],
    'phonemes': ['ك', 'َ', 'أ', 'َ', 'ي', 'ْ', 'ي', 'ِ', 'ن', 'ْ'],
    'frequency_score': 0.25,
    'usage_contexts': ['السؤال عن العدد الكثير', 'التعجب من الكثرة'],
    'grammatical_cases': ['مبني'],
    'semantic_features': {
    'quantitative': True,
    'abundance': True,
    'exclamatory': True,
    },
    },
            # أسماء الاستفهام للاختيار
    {
    'text': 'أَيّ',
    'category': InterrogativeCategory.CHOICE,
    'syllables': ['أَيّ'],
    'phonemes': ['أ', 'َ', 'ي', 'ّ'],
    'frequency_score': 0.88,
    'usage_contexts': ['السؤال عن التحديد', 'الاختيار', 'التمييز'],
    'grammatical_cases': ['معرب'],
    'semantic_features': {
    'selective': True,
    'determinative': True,
    'variable': 'متغير',
    },
    },
    {
    'text': 'أَيُّهَا',
    'category': InterrogativeCategory.CHOICE,
    'syllables': ['أَيْ', 'يُ', 'هَا'],
    'phonemes': ['أ', 'َ', 'ي', 'ْ', 'ي', 'ُ', 'ه', 'َ', 'ا'],
    'frequency_score': 0.65,
    'usage_contexts': ['النداء الاستفهامي', 'التعيين'],
    'grammatical_cases': ['منادى'],
    'semantic_features': {
    'selective': True,
    'vocative': True,
    'formal': 'رسمي',
    },
    },
            # أسماء الاستفهام عن السبب
    {
    'text': 'لِمَاذَا',
    'category': InterrogativeCategory.REASON,
    'syllables': ['لِ', 'مَا', 'ذَا'],
    'phonemes': ['ل', 'ِ', 'م', 'َ', 'ا', 'ذ', 'َ', 'ا'],
    'frequency_score': 0.93,
    'usage_contexts': ['السؤال عن السبب', 'العلة', 'الغرض'],
    'grammatical_cases': ['جار ومجرور'],
    'semantic_features': {
    'causal': True,
    'explanatory': True,
    'purpose': 'غرضي',
    },
    },
    {
    'text': 'لِمَ',
    'category': InterrogativeCategory.REASON,
    'syllables': ['لِمَ'],
    'phonemes': ['ل', 'ِ', 'م', 'َ'],
    'frequency_score': 0.75,
    'usage_contexts': ['السؤال عن السبب المختصر', 'العلة البسيطة'],
    'grammatical_cases': ['جار ومجرور'],
    'semantic_features': {
    'causal': True,
    'concise': True,
    'direct': 'مباشر',
    },
    },
            # أسماء الاستفهام عن الحال
    {
    'text': 'كَيْفَمَا',
    'category': InterrogativeCategory.STATE,
    'syllables': ['كَيْ', 'فَ', 'مَا'],
    'phonemes': ['ك', 'َ', 'ي', 'ْ', 'ف', 'َ', 'م', 'َ', 'ا'],
    'frequency_score': 0.40,
    'usage_contexts': ['السؤال عن أي حال', 'الحالة العامة'],
    'grammatical_cases': ['ظرف حال'],
    'semantic_features': {
    'state': True,
    'condition': True,
    'general': 'عام',
    },
    },
            # أسماء استفهام مركبة ومتخصصة
    {
    'text': 'أَيْنَمَا',
    'category': InterrogativeCategory.PLACE,
    'syllables': ['أَيْ', 'نَ', 'مَا'],
    'phonemes': ['أ', 'َ', 'ي', 'ْ', 'ن', 'َ', 'م', 'َ', 'ا'],
    'frequency_score': 0.45,
    'usage_contexts': ['السؤال عن أي مكان', 'المكان العام'],
    'grammatical_cases': ['ظرف مكان'],
    'semantic_features': {
    'spatial': True,
    'universal': 'شامل',
    'indefinite': 'غير محدد',
    },
    },
    {
    'text': 'مَهْمَا',
    'category': InterrogativeCategory.THING,
    'syllables': ['مَهْ', 'مَا'],
    'phonemes': ['م', 'َ', 'ه', 'ْ', 'م', 'َ', 'ا'],
    'frequency_score': 0.70,
    'usage_contexts': ['السؤال عن أي شيء', 'الشمولية'],
    'grammatical_cases': ['شرطية'],
    'semantic_features': {
    'conditional': True,
    'universal': 'شامل',
    'indefinite': 'مطلق',
    },
    },
    ]

        # إنشاء كائنات أسماء الاستفهام,
    for data in interrogatives_data:
    pronoun = InterrogativePronoun(
    text=data['text'],
    category=data['category'],
    syllables=data['syllables'],
    phonemes=data['phonemes'],
    frequency_score=data['frequency_score'],
    usage_contexts=data['usage_contexts'],
    grammatical_cases=data['grammatical_cases'],
    semantic_features=data['semantic_features'],
    )
    self.interrogative_pronouns.append(pronoun)

        # تجميع الأنماط المقطعية,
    self._build_syllable_patterns()
    self._build_phoneme_patterns()

    logger.info(
    f"✅ تم تهيئة قاعدة بيانات أسماء الاستفهام: {len(self.interrogative_pronouns)} اسم استفهام"
    )  # noqa: E501,
    def _build_syllable_patterns(self):  # type: ignore[no-untyped def]
    """بناء أنماط المقاطع"""

        for pronoun in self.interrogative_pronouns:
    pattern = " ".join(pronoun.syllables)

            if pattern not in self.syllable_patterns:
    self.syllable_patterns[pattern] = []

    self.syllable_patterns[pattern].append(pronoun.text)

    logger.info(f"📊 الأنماط المقطعية: {len(self.syllable_patterns)} نمط")

    def _build_phoneme_patterns(self):  # type: ignore[no-untyped def]
    """بناء أنماط الصوتيات"""

        for pronoun in self.interrogative_pronouns:
    pattern = " ".join(pronoun.phonemes)

            if pattern not in self.phoneme_patterns:
    self.phoneme_patterns[pattern] = []

    self.phoneme_patterns[pattern].append(pronoun.text)

    def find_by_syllables(self, syllables: List[str]) -> List[InterrogativePronoun]:
    """البحث بالمقاطع"""

    results = []
    search_pattern = " ".join(syllables)

        for pronoun in self.interrogative_pronouns:
    pronoun_pattern = " ".join(pronoun.syllables)

            # مطابقة مباشرة,
    if pronoun_pattern == search_pattern:
    results.append(pronoun)
            # مطابقة جزئية,
    elif search_pattern in pronoun_pattern or pronoun_pattern in search_pattern:
    results.append(pronoun)

    return results,
    def find_by_category(
    self, category: InterrogativeCategory
    ) -> List[InterrogativePronoun]:
    """البحث بالفئة"""

    return [p for p in self.interrogative_pronouns if p.category == category]

    def get_high_frequency_pronouns(
    self, threshold: float = 0.8
    ) -> List[InterrogativePronoun]:
    """الحصول على أسماء الاستفهام عالية التكرار"""

    return [
    p for p in self.interrogative_pronouns if p.frequency_score >= threshold
    ]


# ═══════════════════════════════════════════════════════════════════════════════════
# SYLLABLE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativeSyllableAnalyzer:
    """محلل المقاطع لأسماء الاستفهام"""

    def __init__(self, database: InterrogativePronounsDatabase):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.database = database,
    self.syllable_weights = self._calculate_syllable_weights()

    def _calculate_syllable_weights(self) -> Dict[str, float]:
    """حساب أوزان المقاطع حسب الشيوع"""

    syllable_counts = {}
    len(self.database.interrogative_pronouns)

        for pronoun in self.database.interrogative_pronouns:
            for syllable in pronoun.syllables:
    syllable_counts[syllable] = (
    syllable_counts.get(syllable, 0) + pronoun.frequency_score
    )

        # تحويل إلى أوزان,
    weights = {}
    max_count = max(syllable_counts.values()) if syllable_counts else 1,
    for syllable, count in syllable_counts.items():
    weights[syllable] = count / max_count,
    return weights,
    def analyze_syllable_pattern(self, syllables: List[str]) -> Dict[str, Any]:
    """تحليل نمط المقاطع"""

    analysis = {
    'syllable_count': len(syllables),
    'pattern_complexity': self._calculate_pattern_complexity(syllables),
    'similarity_scores': {},
    'weighted_score': self._calculate_weighted_score(syllables),
    'phonetic_features': self._extract_phonetic_features(syllables),
    }

        # حساب التشابه مع الأنماط المعروفة,
    for pronoun in self.database.interrogative_pronouns:
    similarity = self._calculate_similarity(syllables, pronoun.syllables)
    analysis['similarity_scores'][pronoun.text] = similarity,
    return analysis,
    def _calculate_pattern_complexity(self, syllables: List[str]) -> float:
    """حساب تعقيد النمط"""

    complexity = 0.0

        # عدد المقاطع,
    complexity += len(syllables) * 0.2

        # طول المقاطع,
    total_length = sum(len(syll) for syll in syllables)
    complexity += total_length * 0.1

        # تنوع المقاطع,
    unique_syllables = len(set(syllables))
        if len(syllables) > 0:
    complexity += (unique_syllables / len(syllables)) * 0.3,
    return complexity,
    def _calculate_weighted_score(self, syllables: List[str]) -> float:
    """حساب النتيجة المرجحة"""

    total_weight = 0.0,
    for syllable in syllables:
    weight = self.syllable_weights.get(syllable, 0.1)
    total_weight += weight,
    return total_weight / len(syllables) if syllables else 0.0,
    def _extract_phonetic_features(self, syllables: List[str]) -> Dict[str, Any]:
    """استخراج الخصائص الصوتية"""

    features = {
    'vowel_count': 0,
    'consonant_count': 0,
    'long_vowels': 0,
    'short_vowels': 0,
    'common_patterns': [],
    }

    vowels = ['َ', 'ُ', 'ِ', 'ا', 'و', 'ي', 'ى']
    long_vowels = ['ا', 'و', 'ي', 'ى']

        for syllable in syllables:
            for char in syllable:
                if char in vowels:
    features['vowel_count'] += 1,
    if char in long_vowels:
    features['long_vowels'] += 1,
    else:
    features['short_vowels'] += 1,
    else:
    features['consonant_count'] += 1

        # البحث عن الأنماط الشائعة,
    common_interrogative_patterns = ['مَ', 'أَ', 'كَ', 'لِ']
        for pattern in common_interrogative_patterns:
            if any(pattern in syll for syll in syllables):
    features['common_patterns'].append(pattern)

    return features,
    def _calculate_similarity(
    self, syllables1: List[str], syllables2: List[str]
    ) -> float:
    """حساب التشابه بين مجموعتي مقاطع"""

        # تحويل إلى مجموعات للمقارنة,
    set1 = set(syllables1)
    set2 = set(syllables2)

        # معامل Jaccard,
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    jaccard = intersection / union if union > 0 else 0.0

        # تعديل حسب ترتيب المقاطع,
    sequence_bonus = 0.0,
    min_len = min(len(syllables1), len(syllables2))

        for i in range(min_len):
            if syllables1[i] == syllables2[i]:
    sequence_bonus += 0.1,
    return min(jaccard + sequence_bonus, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicInterrogativePronounsGenerator:
    """مولد أسماء الاستفهام العربية من المقاطع الصوتية"""

    def __init__(self, config_path: Optional[str] = None):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.config = self._load_config(config_path)
    self.interrogative_pronouns_db = InterrogativePronounsDatabase()
    self.syllable_analyzer = InterrogativeSyllableAnalyzer(
    self.interrogative_pronouns_db
    )

    logger.info("🚀 تم تهيئة مولد أسماء الاستفهام العربية من المقاطع الصوتية")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
    """تحميل إعدادات النظام"""

        default_config = {
    'similarity_threshold': 0.6,
    'max_results': 5,
    'enable_phonetic_matching': True,
    'enable_fuzzy_matching': True,
    'frequency_weight': 0.3,
    'pattern_weight': 0.4,
    'similarity_weight': 0.3,
    }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf 8') as f:
    user_config = yaml.safe_load(f)
                default_config.update(user_config)

    return default_config,
    def generate_interrogative_pronouns_from_syllables(
    self, syllables: List[str]
    ) -> Dict[str, Any]:
    """توليد أسماء الاستفهام من المقاطع"""

    logger.info(f"🔍 البحث عن أسماء الاستفهام للمقاطع: {syllables}")

    result = {
    'input_syllables': syllables,
    'success': False,
    'candidates': [],
    'best_match': None,
    'analysis': {},
    'timestamp': datetime.now().isoformat(),
    }

        try:
            # تحليل المقاطع,
    analysis = self.syllable_analyzer.analyze_syllable_pattern(syllables)
    result['analysis'] = analysis

            # البحث المباشر بالمقاطع,
    direct_matches = self.interrogative_pronouns_db.find_by_syllables(syllables)

    candidates = []

            # إضافة المطابقات المباشرة,
    for match in direct_matches:
    confidence = 1.0  # مطابقة مباشرة,
    candidates.append(
    {
    'interrogative_pronoun': match.text,
    'category': match.category.value,
    'confidence': confidence,
    'match_type': 'direct',
    'frequency_score': match.frequency_score,
    'usage_contexts': match.usage_contexts,
    }
    )

            # البحث بالتشابه,
    if len(candidates) == 0 or self.config['enable_fuzzy_matching']:
    similarity_matches = self._find_by_similarity(syllables, analysis)
    candidates.extend(similarity_matches)

            # ترتيب المرشحين,
    candidates = self._rank_candidates(candidates, analysis)

    result['candidates'] = candidates,
    if candidates:
    result['success'] = True,
    result['best_match'] = candidates[0]

    logger.info(
    f"✅ تم العثور على {len(candidates)} مرشح. أفضل مطابقة: {candidates[0]['interrogative_pronoun']}"
    )
            else:
    logger.warning(f"❌ لم يتم العثور على مطابقات للمقاطع: {syllables}")

        except Exception as e:
    logger.error(f"❌ خطأ في معالجة المقاطع: {e}")
    result['error'] = str(e)

    return result,
    def _find_by_similarity(
    self, syllables: List[str], analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
    """البحث بالتشابه"""

    similarity_matches = []
    threshold = self.config['similarity_threshold']

        for pronoun in self.interrogative_pronouns_db.interrogative_pronouns:
    similarity = analysis['similarity_scores'].get(pronoun.text, 0.0)

            if similarity >= threshold:
    confidence = similarity * 0.8  # تقليل الثقة للمطابقات التشابهية,
    similarity_matches.append(
    {
    'interrogative_pronoun': pronoun.text,
    'category': pronoun.category.value,
    'confidence': confidence,
    'match_type': 'similarity',
    'similarity_score': similarity,
    'frequency_score': pronoun.frequency_score,
    'usage_contexts': pronoun.usage_contexts,
    }
    )

    return similarity_matches,
    def _rank_candidates(
    self, candidates: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
    """ترتيب المرشحين حسب الجودة"""

        def calculate_final_score(candidate):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    confidence = candidate.get('confidence', 0.0)
    frequency = candidate.get('frequency_score', 0.0)

            # حساب النتيجة النهائية,
    final_score = (
    confidence * self.config['similarity_weight']
    + frequency * self.config['frequency_weight']
    + analysis['weighted_score'] * self.config['pattern_weight']
    )

            # إضافة بونص للمطابقات المباشرة,
    if candidate.get('match_type') == 'direct':
    final_score += 0.2,
    candidate['final_score'] = final_score,
    return final_score

        # ترتيب المرشحين,
    sorted_candidates = sorted(candidates, key=calculate_final_score, reverse=True)

        # تحديد العدد المطلوب,
    max_results = self.config['max_results']
    return sorted_candidates[:max_results]

    def get_interrogative_by_category(
    self, category: InterrogativeCategory
    ) -> List[str]:
    """الحصول على أسماء الاستفهام حسب الفئة"""

    pronouns = self.interrogative_pronouns_db.find_by_category(category)
    return [p.text for p in pronouns]

    def get_system_statistics(self) -> Dict[str, Any]:
    """إحصائيات النظام"""

    stats = {
    'total_interrogative_pronouns': len(
    self.interrogative_pronouns_db.interrogative_pronouns
    ),
    'categories': {},
    'syllable_patterns': len(self.interrogative_pronouns_db.syllable_patterns),
    'phoneme_patterns': len(self.interrogative_pronouns_db.phoneme_patterns),
    'high_frequency_pronouns': len(
    self.interrogative_pronouns_db.get_high_frequency_pronouns()
    ),
    }

        # إحصائيات الفئات,
    for category in InterrogativeCategory:
    count = len(self.interrogative_pronouns_db.find_by_category(category))
    stats['categories'][category.value] = count,
    return stats,
    def save_database(self, output_path: str = "arabic_interrogative_pronouns_database.json"):  # type: ignore[no-untyped def]
    """حفظ قاعدة البيانات"""

    database_data = {
    'metadata': {
    'version': '1.0.0',
    'creation_date': datetime.now().isoformat(),
    'total_pronouns': len(
    self.interrogative_pronouns_db.interrogative_pronouns
    ),
    'categories': [cat.value for cat in InterrogativeCategory],
    },
    'interrogative_pronouns': [
    pronoun.to_dict()
                for pronoun in self.interrogative_pronouns_db.interrogative_pronouns
    ],
    'syllable_patterns': self.interrogative_pronouns_db.syllable_patterns,
    'phoneme_patterns': self.interrogative_pronouns_db.phoneme_patterns,
    }

        with open(output_path, 'w', encoding='utf 8') as f:
    json.dump(database_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 تم حفظ قاعدة بيانات أسماء الاستفهام في: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════


def main():  # type: ignore[no-untyped def]
    """التشغيل الرئيسي للنظام"""

    print("🧠 مولد أسماء الاستفهام العربية من المقاطع الصوتية")
    print("=" * 60)

    # إنشاء المولد,
    generator = ArabicInterrogativePronounsGenerator()

    # عرض إحصائيات النظام,
    stats = generator.get_system_statistics()
    print("\n📊 إحصائيات النظام:")
    print(f"   إجمالي أسماء الاستفهام: {stats['total_interrogative_pronouns']}")
    print(f"   الفئات: {len(stats['categories'])}")
    print(f"   الأنماط المقطعية: {stats['syllable_patterns']}")
    print(f"   أسماء الاستفهام عالية التكرار: {stats['high_frequency_pronouns']}")

    print("\n🏷️ توزيع الفئات:")
    for category, count in stats['categories'].items():
    print(f"   {category}: {count} اسم استفهام")

    # اختبارات التوليد,
    test_cases = [
    ["مَنْ"],  # مَن
    ["مَا"],  # ما
    ["مَ", "تَى"],  # متى
    ["أَيْ", "نَ"],  # أين
    ["كَيْ", "فَ"],  # كيف
    ["كَمْ"],  # كم
    ["أَيّ"],  # أي
    ["لِ", "مَا", "ذَا"],  # لماذا
    ["أَيْ", "يَا", "نَ"],  # أيان
    ["مَا", "ذَا"],  # ماذا
    ]

    print("\n🔬 اختبارات التوليد:")
    for i, syllables in enumerate(test_cases, 1):
    print(f"\n   اختبار {i}: {syllables}")

    result = generator.generate_interrogative_pronouns_from_syllables(syllables)

        if result['success']:
    best_match = result['best_match']
    print(f"   ✅ أفضل مطابقة: {best_match['interrogative_pronoun']}")
    print(f"      الفئة: {best_match['category']}")
    print(f"      الثقة: {best_match['confidence']:.3f}")
    print(f"      النوع: {best_match['match_type']}")

            if len(result['candidates']) > 1:
    print("      بدائل أخرى:")
                for j, candidate in enumerate(result['candidates'][1:3], 2):
    print(
    f"        {j}. {candidate['interrogative_pronoun']} ({candidate['confidence']:.3f})"
    )  # noqa: E501,
    else:
    print("   ❌ لم يتم العثور على مطابقات")

    # حفظ قاعدة البيانات,
    generator.save_database()

    print("\n✅ اكتمل تشغيل النظام بنجاح!")


if __name__ == "__main__":
    main()
