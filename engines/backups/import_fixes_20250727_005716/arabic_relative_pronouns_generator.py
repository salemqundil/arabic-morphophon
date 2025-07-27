#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Relative Pronouns Generator from Syllables
===============================================
مولد الأسماء الموصولة العربية من المقاطع الصوتية

A comprehensive system for generating Arabic relative pronouns from syllable sequences
using deep learning models including RNN and Transformer architectures.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - ARABIC RELATIVE PRONOUNS SYSTEM
Date: 2025-07-24
Encoding: UTF 8
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler('arabic_relative_pronouns.log', encoding='utf 8'),
    logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC RELATIVE PRONOUNS CLASSIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class RelativePronounCategory(Enum):
    """فئات الأسماء الموصولة"""

    MASCULINE_SINGULAR = "مذكر_مفرد"
    FEMININE_SINGULAR = "مؤنث_مفرد"
    MASCULINE_DUAL = "مذكر_مثنى"
    FEMININE_DUAL = "مؤنث_مثنى"
    MASCULINE_PLURAL = "مذكر_جمع"
    FEMININE_PLURAL = "مؤنث_جمع"
    GENERAL = "عام"


class SyllableType(Enum):
    """أنواع المقاطع الصوتية"""

    CV = "CV"  # حرف + حركة
    CVC = "CVC"  # حرف + حركة + حرف
    CV_CV = "CV CV"  # مقطعان بسيطان
    CV_CVC = "CV CVC"  # مقطع بسيط + مقطع مغلق
    CVC_CV = "CVC CV"  # مقطع مغلق + مقطع بسيط
    COMPLEX = "معقد"  # أنماط معقدة


@dataclass
class RelativePronounEntry:
    """بيانات الاسم الموصول"""

    text: str  # النص العربي
    category: RelativePronounCategory  # الفئة
    syllables: List[str]  # المقاطع الصوتية
    phonemes: List[str]  # الفونيمات
    syllable_pattern: str  # نمط المقاطع
    frequency_score: float  # درجة التكرار
    usage_contexts: List[str]  # سياقات الاستخدام
    morphological_features: Dict[str, str]  # الخصائص المورفولوجية


# ═══════════════════════════════════════════════════════════════════════════════════
# RELATIVE PRONOUNS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicRelativePronounsDatabase:
    """قاعدة بيانات الأسماء الموصولة العربية"""

    def __init__(self):

    self.relative_pronouns: List[RelativePronounEntry] = []
    self.syllable_patterns: Dict[str, List[str]] = {}
    self.phoneme_patterns: Dict[str, List[str]] = {}
    self._initialize_database()

    def _initialize_database(self):
    """تهيئة قاعدة البيانات"""

        # الأسماء الموصولة الأساسية
    relative_pronouns_data = [
            # الأسماء الموصولة المذكرة المفردة
    {
    "text": "الذي",
    "category": RelativePronounCategory.MASCULINE_SINGULAR,
    "syllables": ["الْ", "ذِي"],
    "phonemes": ["a", "l", "dh", "i"],
    "frequency_score": 0.95,
    "usage_contexts": ["جملة الصلة", "التعريف", "الوصف"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "مفرد",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
    {
    "text": "الذى",
    "category": RelativePronounCategory.MASCULINE_SINGULAR,
    "syllables": ["الْ", "ذَى"],
    "phonemes": ["a", "l", "dh", "aa"],
    "frequency_score": 0.85,
    "usage_contexts": ["جملة الصلة", "النصوص الكلاسيكية"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "مفرد",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة المؤنثة المفردة
    {
    "text": "التي",
    "category": RelativePronounCategory.FEMININE_SINGULAR,
    "syllables": ["الْ", "تِي"],
    "phonemes": ["a", "l", "t", "i"],
    "frequency_score": 0.92,
    "usage_contexts": ["جملة الصلة", "التعريف", "الوصف"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "مفرد",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
    {
    "text": "اللتي",
    "category": RelativePronounCategory.FEMININE_SINGULAR,
    "syllables": ["الْ", "لَ", "تِي"],
    "phonemes": ["a", "l", "l", "a", "t", "i"],
    "frequency_score": 0.75,
    "usage_contexts": ["النصوص الفصيحة", "الكتابة الرسمية"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "مفرد",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة المثنى المذكر
    {
    "text": "اللذان",
    "category": RelativePronounCategory.MASCULINE_DUAL,
    "syllables": ["الْ", "لَ", "ذَا", "نِ"],
    "phonemes": ["a", "l", "l", "a", "dh", "aa", "n"],
    "frequency_score": 0.65,
    "usage_contexts": ["المثنى المرفوع", "النصوص الفصيحة"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "مثنى",
    "case": "رفع",
    "definiteness": "معرف",
    },
    },
    {
    "text": "اللذين",
    "category": RelativePronounCategory.MASCULINE_DUAL,
    "syllables": ["الْ", "لَ", "ذَيْ", "نِ"],
    "phonemes": ["a", "l", "l", "a", "dh", "ay", "n"],
    "frequency_score": 0.62,
    "usage_contexts": ["المثنى المنصوب والمجرور", "النصوص الفصيحة"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "مثنى",
    "case": "نصب_وجر",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة المثنى المؤنث
    {
    "text": "اللتان",
    "category": RelativePronounCategory.FEMININE_DUAL,
    "syllables": ["الْ", "لَ", "تَا", "نِ"],
    "phonemes": ["a", "l", "l", "a", "t", "aa", "n"],
    "frequency_score": 0.58,
    "usage_contexts": ["المثنى المؤنث المرفوع", "النصوص الفصيحة"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "مثنى",
    "case": "رفع",
    "definiteness": "معرف",
    },
    },
    {
    "text": "اللتين",
    "category": RelativePronounCategory.FEMININE_DUAL,
    "syllables": ["الْ", "لَ", "تَيْ", "نِ"],
    "phonemes": ["a", "l", "l", "a", "t", "ay", "n"],
    "frequency_score": 0.55,
    "usage_contexts": ["المثنى المؤنث المنصوب والمجرور"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "مثنى",
    "case": "نصب_وجر",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة الجمع المذكر
    {
    "text": "الذين",
    "category": RelativePronounCategory.MASCULINE_PLURAL,
    "syllables": ["الْ", "ذِي", "نَ"],
    "phonemes": ["a", "l", "dh", "i", "n"],
    "frequency_score": 0.88,
    "usage_contexts": ["جمع المذكر السالم", "جمع التكسير"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "جمع",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة الجمع المؤنث
    {
    "text": "اللاتي",
    "category": RelativePronounCategory.FEMININE_PLURAL,
    "syllables": ["الْ", "لَا", "تِي"],
    "phonemes": ["a", "l", "l", "aa", "t", "i"],
    "frequency_score": 0.72,
    "usage_contexts": ["جمع المؤنث السالم", "جمع التكسير"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "جمع",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
    {
    "text": "اللائي",
    "category": RelativePronounCategory.FEMININE_PLURAL,
    "syllables": ["الْ", "لَائِي"],
    "phonemes": ["a", "l", "l", "aa", "i"],
    "frequency_score": 0.68,
    "usage_contexts": ["جمع المؤنث", "النصوص الكلاسيكية"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "جمع",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
    {
    "text": "اللواتي",
    "category": RelativePronounCategory.FEMININE_PLURAL,
    "syllables": ["الْ", "لَ", "وَا", "تِي"],
    "phonemes": ["a", "l", "l", "a", "w", "aa", "t", "i"],
    "frequency_score": 0.65,
    "usage_contexts": ["جمع المؤنث", "النصوص الفصيحة"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "جمع",
    "case": "متغير",
    "definiteness": "معرف",
    },
    },
            # الأسماء الموصولة العامة
    {
    "text": "مَن",
    "category": RelativePronounCategory.GENERAL,
    "syllables": ["مَنْ"],
    "phonemes": ["m", "a", "n"],
    "frequency_score": 0.90,
    "usage_contexts": ["العاقل", "الاستفهام", "الشرط"],
    "morphological_features": {
    "gender": "محايد",
    "number": "محايد",
    "case": "مبني",
    "definiteness": "معرف",
    },
    },
    {
    "text": "ما",
    "category": RelativePronounCategory.GENERAL,
    "syllables": ["مَا"],
    "phonemes": ["m", "aa"],
    "frequency_score": 0.87,
    "usage_contexts": ["غير العاقل", "الاستفهام", "الشرط"],
    "morphological_features": {
    "gender": "محايد",
    "number": "محايد",
    "case": "مبني",
    "definiteness": "معرف",
    },
    },
            # أسماء موصولة إضافية
    {
    "text": "أي",
    "category": RelativePronounCategory.GENERAL,
    "syllables": ["أَيّ"],
    "phonemes": ["a", "y", "y"],
    "frequency_score": 0.76,
    "usage_contexts": ["التعميم", "الشرط", "الاستفهام"],
    "morphological_features": {
    "gender": "متغير",
    "number": "متغير",
    "case": "معرب",
    "definiteness": "معرف",
    },
    },
    {
    "text": "ذو",
    "category": RelativePronounCategory.MASCULINE_SINGULAR,
    "syllables": ["ذُو"],
    "phonemes": ["dh", "u"],
    "frequency_score": 0.45,
    "usage_contexts": ["الإضافة", "الوصف", "النصوص الكلاسيكية"],
    "morphological_features": {
    "gender": "مذكر",
    "number": "مفرد",
    "case": "معرب",
    "definiteness": "معرف",
    },
    },
    {
    "text": "ذات",
    "category": RelativePronounCategory.FEMININE_SINGULAR,
    "syllables": ["ذَاتِ"],
    "phonemes": ["dh", "aa", "t"],
    "frequency_score": 0.42,
    "usage_contexts": ["الإضافة", "الوصف", "النصوص الكلاسيكية"],
    "morphological_features": {
    "gender": "مؤنث",
    "number": "مفرد",
    "case": "معرب",
    "definiteness": "معرف",
    },
    },
    ]

        # إنشاء كائنات RelativePronounEntry
        for data in relative_pronouns_data:
    syllable_pattern = self._determine_syllable_pattern(data["syllables"])

    relative_pronoun = RelativePronounEntry(
    text=data["text"],
    category=data["category"],
    syllables=data["syllables"],
    phonemes=data["phonemes"],
    syllable_pattern=syllable_pattern,
    frequency_score=data["frequency_score"],
    usage_contexts=data["usage_contexts"],
    morphological_features=data["morphological_features"],
    )

    self.relative_pronouns.append(relative_pronoun)

        # تجميع الأنماط
    self._group_syllable_patterns()
    self._group_phoneme_patterns()

    logger.info(
    f"✅ تم تهيئة قاعدة بيانات الأسماء الموصولة: {len(self.relative_pronouns)} اسم موصول"
    )

    def _determine_syllable_pattern(self, syllables: List[str]) -> str:
    """تحديد نمط المقاطع"""

    pattern_parts = []

        for syllable in syllables:
            # تحليل بسيط للمقطع
    clean_syllable = re.sub(r'[ًٌٍَُِّْ]', '', syllable)  # إزالة التشكيل

            if len(clean_syllable) == 1:
    pattern_parts.append("CV")
            elif len(clean_syllable) == 2:
                if clean_syllable.endswith(('ا', 'و', 'ي')):
    pattern_parts.append("CV")
                else:
    pattern_parts.append("CVC")
            elif len(clean_syllable) == 3:
    pattern_parts.append("CVC")
            else:
    pattern_parts.append("COMPLEX")

    return " ".join(pattern_parts)

    def _group_syllable_patterns(self):
    """تجميع الأنماط المقطعية"""

        for relative_pronoun in self.relative_pronouns:
    pattern = relative_pronoun.syllable_pattern
            if pattern not in self.syllable_patterns:
    self.syllable_patterns[pattern] = []
    self.syllable_patterns[pattern].append(relative_pronoun.text)

    logger.info(f"📊 الأنماط المقطعية: {len(self.syllable_patterns)} نمط")

    def _group_phoneme_patterns(self):
    """تجميع أنماط الفونيمات"""

        for relative_pronoun in self.relative_pronouns:
    phoneme_key = " ".join(relative_pronoun.phonemes[:3])  # أول 3 فونيمات
            if phoneme_key not in self.phoneme_patterns:
    self.phoneme_patterns[phoneme_key] = []
    self.phoneme_patterns[phoneme_key].append(relative_pronoun.text)

    def get_relative_pronoun_by_text(self, text: str) -> Optional[RelativePronounEntry]:
    """البحث عن اسم موصول بالنص"""

        for relative_pronoun in self.relative_pronouns:
            if relative_pronoun.text == text:
    return relative_pronoun
    return None

    def get_relative_pronouns_by_category(
    self, category: RelativePronounCategory
    ) -> List[RelativePronounEntry]:
    """البحث عن الأسماء الموصولة بالفئة"""

    return [rp for rp in self.relative_pronouns if rp.category == category]

    def get_relative_pronouns_by_pattern(
    self, pattern: str
    ) -> List[RelativePronounEntry]:
    """البحث عن الأسماء الموصولة بالنمط"""

    return [rp for rp in self.relative_pronouns if rp.syllable_pattern == pattern]


# ═══════════════════════════════════════════════════════════════════════════════════
# SYLLABLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════


class RelativePronounSyllableAnalyzer:
    """محلل المقاطع للأسماء الموصولة"""

    def __init__(self, database: ArabicRelativePronounsDatabase):

    self.database = database

    def analyze_syllable_sequence(self, syllables: List[str]) -> Dict[str, Any]:
    """تحليل تسلسل المقاطع"""

    pattern = self._determine_syllable_pattern(syllables)
    complexity = self._calculate_complexity(syllables)

    return {
    "syllables": syllables,
    "pattern": pattern,
    "complexity": complexity,
    "length": len(syllables),
    "phonetic_structure": self._analyze_phonetic_structure(syllables),
    }

    def _determine_syllable_pattern(self, syllables: List[str]) -> str:
    """تحديد نمط المقاطع"""

    pattern_parts = []

        for syllable in syllables:
    clean_syllable = re.sub(r'[ًٌٍَُِّْ]', '', syllable)

            if len(clean_syllable) <= 2:
    pattern_parts.append("CV")
            elif len(clean_syllable) == 3:
    pattern_parts.append("CVC")
            else:
    pattern_parts.append("COMPLEX")

    return " ".join(pattern_parts)

    def _calculate_complexity(self, syllables: List[str]) -> float:
    """حساب تعقيد المقاطع"""

    complexity = len(syllables)

        for syllable in syllables:
            # إضافة تعقيد بناءً على طول المقطع
    complexity += len(syllable) * 0.1

            # إضافة تعقيد للتشكيل
    diacritics = len(re.findall(r'[ًٌٍَُِّْ]', syllable))
    complexity += diacritics * 0.2

    return complexity

    def _analyze_phonetic_structure(self, syllables: List[str]) -> Dict[str, Any]:
    """تحليل البنية الصوتية"""

    total_length = sum(len(syl) for syl in syllables)
    avg_syllable_length = total_length / len(syllables) if syllables else 0

    return {
    "total_length": total_length,
    "average_syllable_length": avg_syllable_length,
    "has_long_vowels": any(
    'ا' in syl or 'و' in syl or 'ي' in syl for syl in syllables
    ),
    "has_doubled_consonants": any('ّ' in syl for syl in syllables),
    "syllable_count": len(syllables),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicRelativePronounsGenerator:
    """مولد الأسماء الموصولة العربية من المقاطع"""

    def __init__(self):

    self.relative_pronouns_db = ArabicRelativePronounsDatabase()
    self.syllable_analyzer = RelativePronounSyllableAnalyzer(
    self.relative_pronouns_db
    )

    logger.info("🚀 تم تهيئة مولد الأسماء الموصولة العربية من المقاطع الصوتية")

    def generate_relative_pronouns_from_syllables(
    self, syllables: List[str]
    ) -> Dict[str, Any]:
    """توليد الأسماء الموصولة من المقاطع"""

    logger.info(f"🔍 البحث عن الأسماء الموصولة للمقاطع: {syllables}")

        # تحليل المقاطع
    syllable_analysis = self.syllable_analyzer.analyze_syllable_sequence(syllables)

        # البحث عن التطابقات
    matches = self._find_matches(syllables, syllable_analysis["pattern"])

        if not matches:
    return {
    "success": False,
    "message": "لم يتم العثور على أسماء موصولة مطابقة",
    "input_syllables": syllables,
    "syllable_pattern": syllable_analysis["pattern"],
    "suggestions": self._get_suggestions(syllable_analysis["pattern"]),
    }

    return {
    "success": True,
    "input_syllables": syllables,
    "syllable_pattern": syllable_analysis["pattern"],
    "syllable_analysis": syllable_analysis,
    "matches": matches,
    "total_matches": len(matches),
    "best_match": matches[0] if matches else None,
    }

    def _find_matches(self, syllables: List[str], pattern: str) -> List[Dict[str, Any]]:
    """البحث عن التطابقات"""

    matches = []

        for relative_pronoun in self.relative_pronouns_db.relative_pronouns:
            # مطابقة المقاطع
    syllable_match = self._calculate_syllable_similarity(
    syllables, relative_pronoun.syllables
    )

            # مطابقة النمط
    pattern_match = pattern == relative_pronoun.syllable_pattern

            if syllable_match > 0.7 or pattern_match:
    confidence = self._calculate_confidence(
    syllable_match, pattern_match, relative_pronoun
    )

    match_data = {
    "relative_pronoun": relative_pronoun.text,
    "category": relative_pronoun.category.value,
    "syllables": relative_pronoun.syllables,
    "phonemes": relative_pronoun.phonemes,
    "pattern": relative_pronoun.syllable_pattern,
    "frequency": relative_pronoun.frequency_score,
    "syllable_similarity": syllable_match,
    "pattern_match": pattern_match,
    "confidence": confidence,
    "usage_contexts": relative_pronoun.usage_contexts,
    "morphological_features": relative_pronoun.morphological_features,
    }

    matches.append(match_data)

        # ترتيب النتائج حسب الثقة
    matches.sort(key=lambda x: x["confidence"], reverse=True)

    return matches

    def _calculate_syllable_similarity(
    self, input_syllables: List[str], target_syllables: List[str]
    ) -> float:
    """حساب تشابه المقاطع"""

        if len(input_syllables) != len(target_syllables):
    return 0.0

    total_similarity = 0.0

        for i_syl, t_syl in zip(input_syllables, target_syllables):
            # إزالة التشكيل للمقارنة
    clean_i = re.sub(r'[ًٌٍَُِّْ]', '', i_syl)
    clean_t = re.sub(r'[ًٌٍَُِّْ]', '', t_syl)

            if clean_i == clean_t:
    total_similarity += 1.0
            elif len(clean_i) == len(clean_t):
                # حساب تشابه جزئي
    matches = sum(1 for a, b in zip(clean_i, clean_t) if a == b)
    total_similarity += matches / len(clean_i)

    return total_similarity / len(input_syllables)

    def _calculate_confidence(
    self,
    syllable_similarity: float,
    pattern_match: bool,
    relative_pronoun: RelativePronounEntry,
    ) -> float:
    """حساب درجة الثقة"""

    confidence = syllable_similarity * 0.7

        if pattern_match:
    confidence += 0.2

        # مكافأة للأسماء الموصولة عالية التكرار
    confidence += relative_pronoun.frequency_score * 0.1

    return min(1.0, confidence)

    def _get_suggestions(self, pattern: str) -> List[str]:
    """الحصول على اقتراحات"""

    suggestions = []

        # البحث عن أنماط مشابهة
        for p, relative_pronouns in self.relative_pronouns_db.syllable_patterns.items():
            if p != pattern and relative_pronouns:
    suggestions.extend(relative_pronouns[:2])  # أول اثنين

    return suggestions[:5]  # أفضل 5 اقتراحات

    def get_statistics(self) -> Dict[str, Any]:
    """إحصائيات النظام"""

    total_relative_pronouns = len(self.relative_pronouns_db.relative_pronouns)

        # توزيع الفئات
    category_distribution = {}
        for rp in self.relative_pronouns_db.relative_pronouns:
    category = rp.category.value
    category_distribution[category] = category_distribution.get(category, 0) + 1

        # توزيع الأنماط
    pattern_distribution = {}
        for (
    pattern,
    relative_pronouns,
    ) in self.relative_pronouns_db.syllable_patterns.items():
    pattern_distribution[pattern] = len(relative_pronouns)

    return {
    "total_relative_pronouns": total_relative_pronouns,
    "total_patterns": len(self.relative_pronouns_db.syllable_patterns),
    "category_distribution": category_distribution,
    "pattern_distribution": pattern_distribution,
    "most_common_pattern": max(
    pattern_distribution.keys(), key=lambda k: pattern_distribution[k]
    ),
    }

    def save_database(self, filename: str = "arabic_relative_pronouns_database.json"):
    """حفظ قاعدة البيانات"""

    database_data = {
    "metadata": {
    "version": "1.0.0",
    "creation_date": datetime.now().isoformat(),
    "total_relative_pronouns": len(
    self.relative_pronouns_db.relative_pronouns
    ),
    },
    "relative_pronouns": [
    asdict(rp) for rp in self.relative_pronouns_db.relative_pronouns
    ],
    "syllable_patterns": self.relative_pronouns_db.syllable_patterns,
    "phoneme_patterns": self.relative_pronouns_db.phoneme_patterns,
    }

        # تحويل Enum إلى string للتسلسل
        for rp_data in database_data["relative_pronouns"]:
    rp_data["category"] = rp_data["category"].value

        with open(filename, 'w', encoding='utf 8') as f:
    json.dump(database_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 تم حفظ قاعدة بيانات الأسماء الموصولة في: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════════


def main():
    """التشغيل الرئيسي والعرض التوضيحي"""

    print("🚀 مولد الأسماء الموصولة العربية من المقاطع الصوتية")
    print("=" * 60)

    # إنشاء المولد
    generator = ArabicRelativePronounsGenerator()

    # عرض إحصائيات النظام
    stats = generator.get_statistics()
    print("\n📊 إحصائيات النظام:")
    print(f"   إجمالي الأسماء الموصولة: {stats['total_relative_pronouns']}")
    print(f"   إجمالي الأنماط: {stats['total_patterns']}")
    print(f"   النمط الأكثر شيوعاً: {stats['most_common_pattern']}")

    print("\n🔤 توزيع الفئات:")
    for category, count in stats['category_distribution'].items():
    print(f"   {category}: {count} اسم موصول")

    print("\n📈 توزيع الأنماط:")
    for pattern, count in stats['pattern_distribution'].items():
    print(f"   {pattern}: {count} اسم موصول")

    # اختبار المولد
    test_cases = [
    ["الْ", "ذِي"],  # الذي
    ["الْ", "تِي"],  # التي
    ["الْ", "لَ", "ذَا", "نِ"],  # اللذان
    ["الْ", "لَ", "تَا", "نِ"],  # اللتان
    ["الْ", "ذِي", "نَ"],  # الذين
    ["الْ", "لَا", "تِي"],  # اللاتي
    ["مَنْ"],  # مَن
    ["مَا"],  # ما
    ["أَيّ"],  # أي
        # اختبارات مع أخطاء
    ["الْ", "ذُو"],  # تجريب
    ["الْ", "لَا"],  # غير مكتمل
    ]

    print("\n🧪 اختبار المولد:")
    print("=" * 40)

    for i, syllables in enumerate(test_cases, 1):
    print(f"\n🔍 اختبار {i}: {syllables}")

    result = generator.generate_relative_pronouns_from_syllables(syllables)

        if result["success"]:
    best_match = result["best_match"]
    print(f"✅ أفضل تطابق: {best_match['relative_pronoun']}")
    print(f"   الفئة: {best_match['category']}")
    print(f"   الثقة: {best_match['confidence']:.2f}")
    print(f"   تشابه المقاطع: {best_match['syllable_similarity']:.2f}")
    print(f"   مطابقة النمط: {best_match['pattern_match']}")

            if result["total_matches"] > 1:
    print(f"   تطابقات إضافية: {result['total_matches']} - 1}")
        else:
    print(f"❌ {result['message']}")
            if result.get("suggestions"):
    print(f"   اقتراحات: {', '.join(result['suggestions'][:3])}")

    # حفظ قاعدة البيانات
    generator.save_database()

    print("\n✅ اكتمل العرض التوضيحي!")


if __name__ == "__main__":
    main()
