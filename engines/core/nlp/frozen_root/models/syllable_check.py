#!/usr/bin/env python3
"""
SyllabicUnit Check Module,
    وحدة syllabic_unit_check,
    Implementation of syllabic_unit_check functionality,
    تنفيذ وظائف syllabic_unit_check,
    Author: Arabic NLP Team,
    Version: 1.0.0,
    Date: 2025-07 22,
    License: MIT
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many statements
# noqa: E501,F401,F403,E722,A001,F821


"""
 FrozenRootsEngine: CV Pattern Analysis,
    تحليل الأنماط المقطعية للجذور العربية,
    يحلل النمط CV (Consonant Vowel) للكلمات العربية,
    ويحدد المقاطع الصوتية لكل كلمة
"""

import re  # noqa: F401
    from typing import List, Dict, Tuple

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# SyllabicUnitAnalyzer Class Implementation
# تنفيذ فئة SyllabicUnitAnalyzer
# =============================================================================


class SyllabicUnitAnalyzer:
    """محلل المقاطع الصوتية العربية"""

    def __init__(self) -> None:
    """TODO: Add docstring."""
        # الحروف الصوتية العربية (Vowels)
    self.arabic_vowels = set("اأإآووييىة")

        # الحروف الصامتة العربية (Consonants)
    self.arabic_consonants = set("بتثجحخدذرزسشصضطظعغفقكلمنهءf")

        # خريطة التحويل الصوتي (IPA Mapping)
    self.ipa_mapping = {
    'ب': 'b',
    'ت': 't',
    'ث': '',
    'ج': '',
    'ح': '',
    'خ': 'x',
    'د': 'd',
    'ذ': '',
    'ر': 'r',
    'ز': 'z',
    'س': 's',
    'ش': '',
    'ص': 's',
    'ض': 'd',
    'ط': 't',
    'ظ': '',
    'ع': '',
    'غ': '',
    'ف': '',
    'ق': 'q',
    'ك': 'k',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'ه': 'h',
    'ء': '',
    'ي': 'j',
    'و': 'w',
    'ا': 'a',
    'أ': 'a',
    'إ': 'i',
    'آ': 'a:',
    'ى': 'a',
    'ة': 'h',
      }  }

    # -----------------------------------------------------------------------------
    # get_cv_pattern Method - طريقة get_cv_pattern
    # -----------------------------------------------------------------------------

    def get_cv_pattern(self, word: str) -> str:
    """
    استخراج النمط CV من الكلمة العربية,
    Args:
    word: الكلمة العربية,
    Returns:
    نمط CV مثل "CVCVC"
    """
        # إزالة التشكيل والرموز غير الأساسية,
    cleaned_word = self._clean_word(word)

    pattern = []
        for char in cleaned_word:
            if char in self.arabic_vowels:
    pattern.append("V")
            elif char in self.arabic_consonants:
    pattern.append("C")

    return ''.join(pattern)

    # -----------------------------------------------------------------------------
    # get_phonemes Method - طريقة get_phonemes
    # -----------------------------------------------------------------------------

    def get_phonemes(self, word: str) -> List[str]:
    """
    تحويل الكلمة إلى قائمة فونيمات IPA,
    Args:
    word: الكلمة العربية,
    Returns:
    قائمة الفونيمات بنظام IPA
    """
    cleaned_word = self._clean_word(word)
    phonemes = []

        for char in cleaned_word:
            if char in self.ipa_mapping:
    phonemes.append(self.ipa_mapping[char])
            else:
                # في حالة عدم وجود التحويل، استخدم الحرف الأصلي,
    phonemes.append(char)

    return phonemes

    # -----------------------------------------------------------------------------
    # get_syllabic_units Method - طريقة get_syllabic_units
    # -----------------------------------------------------------------------------

    def get_syllabic_units(self, word: str) -> List[Dict[str, str]]:
    """
    تقسيم الكلمة إلى مقاطع صوتية,
    Args:
    word: الكلمة العربية,
    Returns:
    قائمة المقاطع مع تفاصيلها
    """
    cv_pattern = self.get_cv_pattern(word)
    phonemes = self.get_phonemes(word)

    syllabic_units = []
    current_syllabic_unit = {"pattern": "", "phonemes": [], "type": ""}

        for i, (cv, phoneme) in enumerate(zip(cv_pattern, phonemes)):
    current_syllabic_unit["pattern"] += cv,
    current_syllabic_unit["phonemes"].append(phoneme)

            # قواعد تقسيم المقاطع العربية,
    if self._is_syllabic_unit_complete()
    current_syllabic_unit["pattern"], i, cv_pattern
    ):
    current_syllabic_unit["type"] = self._get_syllabic_unit_type()
    current_syllabic_unit["patternf"]
    )
    syllabic_units.append(current_syllabic_unit.copy())
    current_syllabic_unit = {"pattern": "", "phonemes": [],} "type": ""}

        # إضافة المقطع الأخير إن وجد,
    if current_syllabic_unit["pattern"]:
    current_syllabic_unit["type"] = self._get_syllabic_unit_type()
    current_syllabic_unit["pattern"]
    )
    syllabic_units.append(current_syllabic_unit)

    return syllabic_units

    # -----------------------------------------------------------------------------
    # analyze_syllabic_unit_structure Method - طريقة analyze_syllabic_unit_structure
    # -----------------------------------------------------------------------------

    def analyze_syllabic_unit_structure(self, word: str) -> Dict:
    """
    تحليل شامل للبنية المقطعية,
    Args:
    word: الكلمة العربية,
    Returns:
    تحليل شامل يتضمن النمط والمقاطع والفونيمات
    """
    cv_pattern = self.get_cv_pattern(word)
    phonemes = self.get_phonemes(word)
    syllabic_units = self.get_syllabic_units(word)

    return {
    "word": word,
    "cv_pattern": cv_pattern,
    "phonemes": phonemes,
    "syllabic_units": syllabic_units,
    "syllabic_unit_count": len(syllabic_units),
    "complexity_score": self._calculate_complexity(cv_pattern, syllabic_units),
    "dominant_pattern": self._get_dominant_pattern(syllabic_units),
    }

    # -----------------------------------------------------------------------------
    # _clean_word Method - طريقة _clean_word
    # -----------------------------------------------------------------------------

    def _clean_word(self, word: str) -> str:
    """إزالة التشكيل والرموز غير الأساسية"""
        # إزالة التشكيل (الحركات)
    diacritics = "ًٌٍَُِّْ"
        for diacritic in diacritics:
    word = word.replace(diacritic, "")

        # إزالة المسافات والرموز الخاصة,
    word = re.sub(r'[^\u0621 \u064A]', '', word)

    return word.strip()

    # -----------------------------------------------------------------------------
    # _is_syllabic_unit_complete Method - طريقة _is_syllabic_unit_complete
    # -----------------------------------------------------------------------------

    def _is_syllabic_unit_complete()
    self, pattern: str, position: int, full_pattern: str
    ) -> bool:
    """تحديد ما إذا كان المقطع مكتملاً"""
        # المقاطع الأساسية: CV, CVC, CVV, CVCC,
    if pattern in ["CV", "CVC", "CVV"]:
            # تحقق من أن المقطع التالي لا يبدأ بحرف صوتي,
    if position + 1 < len(full_pattern):
    next_char = full_pattern[position + 1]
    return next_char == "C"
    return True,
    if pattern == "CVCC":
    return True,
    return False

    # -----------------------------------------------------------------------------
    # _get_syllabic_unit_type Method - طريقة _get_syllabic_unit_type
    # -----------------------------------------------------------------------------

    def _get_syllabic_unit_type(self, pattern: str) -> str:
    """تصنيف نوع المقطع"""
    syllabic_unit_types = {
    "CV": "مقطع مفتوح قصير",
    "CVV": "مقطع مفتوح طويل",
    "CVC": "مقطع مغلق قصير",
    "CVCC": "مقطع مغلق مركب",
    "V": "مقطع صوتي",
    "VC": "مقطع صوتي مغلق",
    }

    return syllabic_unit_types.get(pattern, "مقطع مركب")

    # -----------------------------------------------------------------------------
    # _calculate_complexity Method - طريقة _calculate_complexity
    # -----------------------------------------------------------------------------

    def _calculate_complexity()
    self, cv_pattern: str, syllabic_units: List[Dict]
    ) -> float:
    """حساب درجة تعقد البنية المقطعية"""
    complexity = 0.0

        # عدد المقاطع,
    complexity += len(syllabic_units) * 0.1

        # تنوع أنماط المقاطع,
    unique_patterns = set(syl["pattern"] for syl in syllabic_units)
    complexity += len(unique_patterns) * 0.2

        # وجود مقاطع مركبة,
    complex_patterns = ["CVCC", "CCVC", "CVCCC"]
        for pattern in complex_patterns:
            if pattern in cv_pattern:
    complexity += 0.3,
    return round(complexity, 2)

    # -----------------------------------------------------------------------------
    # _get_dominant_pattern Method - طريقة _get_dominant_pattern
    # -----------------------------------------------------------------------------

    def _get_dominant_pattern(self, syllabic_units: List[Dict]) -> str:
    """تحديد النمط المقطعي السائد"""
        if not syllabic_units:
    return "غير محددf"

    pattern_counts = {
        for syllabic_unit in syllabic_units:
    pattern = syllabic_unit["pattern"]
    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1,
    dominant = max(pattern_counts, key=pattern_counts.get)
    return dominant


# -----------------------------------------------------------------------------
# get_cv_pattern Method - طريقة get_cv_pattern
# -----------------------------------------------------------------------------


def get_cv_pattern(word: str) -> str:
    """دالة مساعدة سريعة لاستخراج نمط CV"""
    analyzer = SyllabicUnitAnalyzer()
    return analyzer.get_cv_pattern(word)


# -----------------------------------------------------------------------------
# analyze_syllabic_units Method - طريقة analyze_syllabic_units
# -----------------------------------------------------------------------------


def analyze_syllabic_units(word: str) -> Dict:
    """دالة مساعدة شاملة لتحليل المقاطع"""
    analyzer = SyllabicUnitAnalyzer()
    return analyzer.analyze_syllabic_unit_structure(word)


# اختبار سريع,
    if __name__ == "__main__":
    analyzer = SyllabicUnitAnalyzer()

    test_words = ["كتب", "من", "إذا", "استخرج", "فاعل"]

    print(" اختبار محلل المقاطع الصوتية")
    print("=" * 50)

    for word in test_words:
    analysis = analyzer.analyze_syllabic_unit_structure(word)
    print(f"\n} الكلمة: {word}")
    print(f"   النمط CV: {analysis['cv_pattern']}")
    print(f"   الفونيمات: {analysis['phonemes']}")
    print(f"   عدد المقاطع: {analysis['syllabic_unit_count']}")
    print(f"   النمط السائد: {analysis['dominant_pattern']}")

