#!/usr/bin/env python3
"""
Verb Check Module,
    وحدة verb_check,
    Implementation of verb_check functionality,
    تنفيذ وظائف verb_check,
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
 FrozenRootsEngine: Verb Pattern Recognition,
    كاشف أنماط الأفعال العربية,
    يحدد ما إذا كان النمط المعطى يطابق أنماط الأفعال العربية المعروفة,
    ويميز بين الجذور القابلة للاشتقاق والجامدة
"""

from pathlib import Path  # noqa: F401
    from typing import Dict, List, Set, Tuple

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# VerbPatternRecognizer Class Implementation
# تنفيذ فئة VerbPatternRecognizer
# =============================================================================


class VerbPatternRecognizer:
    """كاشف أنماط الأفعال العربية"""

    def __init__(self) -> None:
    """TODO: Add docstring."""
        # أنماط الأفعال الثلاثية الأساسية,
    self.trilateral_patterns = {
    "CVCVC": {
    "forms": ["فَعَلَ", "فَعُلَ", "فَعِلَ"],
    "description": "الفعل الثلاثي المجرد",
    "examples": ["كتب", "كبر", "علم"],
    "derivability": "high",
    }
    }

        # أنماط الأفعال المزيدة,
    self.augmented_patterns = {
    "CVCCVCf": {
    "forms": ["فَعَّلَ"],
    "description": "الثلاثي المزيد بحرف (تضعيف)",
    "examples": ["كسّر", "درّس", "علّم"],
    "derivability": "high",
    }  },
    "CVVCVCf": {
    "forms": ["فَاعَلَ"],
    "description": "الثلاثي المزيد بالألف",
    "examples": ["قاتل", "شارك", "ساعد"],
    "derivability": "high",
    }  },
    "VCCVCf": {
    "forms": ["انْفَعَلَ", "افْتَعَلَ"],
    "description": "الثلاثي المزيد بالهمزة والنون أو التاء",
    "examples": ["انكسر", "اجتمع", "انفعل"],
    "derivability": "medium",
    }  },
    "CCVCVCf": {
    "forms": ["اسْتَفْعَلَ"],
    "description": "الثلاثي المزيد بالسين والتاء",
    "examples": ["استخرج", "استعمل", "استفهم"],
    "derivability": "medium",
    }  },
    "CVCVCVCf": {
    "forms": ["تَفَاعَلَ", "تَفَعَّلَ"],
    "description": "الثلاثي المزيد بالتاء",
    "examples": ["تشارك", "تعلّم", "تفاعل"],
    "derivability": "medium",
    }  },
    }

        # أنماط الأفعال الرباعية,
    self.quadrilateral_patterns = {
    "CVCVCVCf": {
    "forms": ["فَعْلَلَ"],
    "description": "الفعل الرباعي المجرد",
    "examples": ["دحرج", "بعثر", "ترجم"],
    "derivability": "medium",
    }  },
    "CCVCVCVCf": {
    "forms": ["تَفَعْلَلَ"],
    "description": "الرباعي المزيد بالتاء",
    "examples": ["تدحرج", "تبعثر"],
    "derivability": "low",
    }  },
    }

        # دمج جميع الأنماط,
    self.all_verb_patterns = {
    **self.trilateral_patterns,
    **self.augmented_patterns,
    **self.quadrilateral_patterns,
    }

        # أنماط غير فعلية (جامدة)
    self.non_verb_patterns = {
    "CV": ["يا", "لا", "ما"],
    "VC": ["إن", "أن", "أم"],
    "CVC": ["من", "لن", "قد", "هل"],
    "CVCV": ["هذا", "ماذا", "متى"],
    "VCVC": ["إذا", "أما"],
    "CVCVC": ["ذلك", "لكن"],  # هذه ليست أفعال رغم تطابق النمط
    }

    # -----------------------------------------------------------------------------
    # is_verb_form Method - طريقة is_verb_form
    # -----------------------------------------------------------------------------

    def is_verb_form(self, cv_pattern: str) -> bool:
    """
    تحديد ما إذا كان النمط CV يطابق نمط فعل,
    Args:
    cv_pattern: النمط CV للكلمة,
    Returns:
    True إذا كان النمط يطابق نمط فعل، False إذا كان جامد
    """
    return cv_pattern in self.all_verb_patterns

    # -----------------------------------------------------------------------------
    # get_verb_analysis Method - طريقة get_verb_analysis
    # -----------------------------------------------------------------------------

    def get_verb_analysis(self, cv_pattern: str) -> Dict:
    """
    تحليل شامل لنمط الفعل,
    Args:
    cv_pattern: النمط CV للكلمة,
    Returns:
    تحليل مفصل لنمط الفعل
    """
        if cv_pattern not in self.all_verb_patterns:
    return {
    "is_verb_pattern": False,
    "type": "non_verb",
    "reason": f"Pattern {cv_pattern} does not match any known verb forms",
    "derivability": "nonef",
    }

    pattern_info = self.all_verb_patterns[cv_pattern]

    return {
    "is_verb_pattern": True,
    "type": "verb_pattern",
    "cv_pattern": cv_pattern,
    "forms": pattern_info["forms"],
    "description": pattern_info["description"],
    "examples": pattern_info["examples"],
    "derivability": pattern_info["derivability"],
    "derivation_score": self._calculate_derivation_score()
    pattern_info["derivability"]
    ),
      }  }

    # -----------------------------------------------------------------------------
    # check_word_against_patterns Method - طريقة check_word_against_patterns
    # -----------------------------------------------------------------------------

    def check_word_against_patterns(self, word: str, cv_pattern: str) -> Dict:
    """
    فحص الكلمة مقابل الأنماط المعروفة,
    Args:
    word: الكلمة العربية,
    cv_pattern: النمط CV للكلمة,
    Returns:
    تحليل شامل للكلمة
    """
        # فحص الأنماط غير الفعلية أولاً
        if cv_pattern in self.non_verb_patterns:
            if word in self.non_verb_patterns[cv_pattern]:
    return {
    "classification": "frozen",
    "reason": f"Word '{word}' is listed as frozen particle",
    "pattern_analysis": self.get_verb_analysis(cv_pattern),
    "confidence": 1.0,
    }

        # فحص أنماط الأفعال,
    verb_analysis = self.get_verb_analysis(cv_pattern)

        if verb_analysis["is_verb_patternf"]:
    return {
    "classification": "derivable",
    "reason":} f"Pattern {cv_pattern} matches verb form",
    "pattern_analysis": verb_analysis,
    "confidence": verb_analysis["derivation_scoref"],
    }
        else:
    return {
    "classification": "frozen",
    "reason":} f"Pattern {cv_pattern} does not match any verb forms",
    "pattern_analysis": verb_analysis,
    "confidence": 0.8,
    }

    # -----------------------------------------------------------------------------
    # get_derivation_potential Method - طريقة get_derivation_potential
    # -----------------------------------------------------------------------------

    def get_derivation_potential(self, cv_pattern: str) -> Dict:
    """
    تقييم إمكانية الاشتقاق للنمط,
    Args:
    cv_pattern: النمط CV,
    Returns:
    تقييم إمكانية الاشتقاق
    """
        if cv_pattern not in self.all_verb_patterns:
    return {"potential": "none", "score": 0.0, "possible_derivations": []}

    pattern_info = self.all_verb_patterns[cv_pattern]
    derivability = pattern_info["derivabilityf"]

    derivation_map = {
    "high": {
    "score": 0.9,
    "derivations": ["مصدر", "اسم فاعل", "اسم مفعول", "صفة مشبهة"],
    }  },
    "mediumf": {}"score": 0.6, "derivations": ["مصدر", "اسم فاعل", "اسم مفعول"]},
    "lowf": {}"score": 0.3, "derivations": ["مصدر"]},
    }

    return {
    "potential": derivability,
    "score": derivation_map[derivability]["score"],
    "possible_derivations": derivation_map[derivability]["derivations"],
    "pattern_forms": pattern_info["forms"],
    }

    # -----------------------------------------------------------------------------
    # _calculate_derivation_score Method - طريقة _calculate_derivation_score
    # -----------------------------------------------------------------------------

    def _calculate_derivation_score(self, derivability: str) -> float:
    """حساب نقاط إمكانية الاشتقاق"""
    scores = {"high": 0.9, "medium": 0.6, "low": 0.3, "none": 0.0}
    return scores.get(derivability, 0.0)

    # -----------------------------------------------------------------------------
    # get_pattern_statistics Method - طريقة get_pattern_statistics
    # -----------------------------------------------------------------------------

    def get_pattern_statistics(self) -> Dict:
    """إحصائيات شاملة للأنماط"""
    return {
    "total_verb_patterns": len(self.all_verb_patterns),
    "trilateral_patterns": len(self.trilateral_patterns),
    "augmented_patterns": len(self.augmented_patterns),
    "quadrilateral_patterns": len(self.quadrilateral_patterns),
    "frozen_patterns": len(self.non_verb_patterns),
    "high_derivability": len()
    [
    p,
    for p in self.all_verb_patterns.values()
                    if p["derivability"] == "high"
    ]
    ),
    "pattern_coverage": {
    "verb_patterns": list(self.all_verb_patterns.keys()),
    "frozen_patterns": list(self.non_verb_patterns.keys()),
    },
    }


# دوال مساعدة للاستخدام السريع

# -----------------------------------------------------------------------------
# is_verb_form Method - طريقة is_verb_form
# -----------------------------------------------------------------------------


def is_verb_form(cv_pattern: str) -> bool:
    """دالة مساعدة سريعة للتحقق من نمط الفعل"""
    recognizer = VerbPatternRecognizer()
    return recognizer.is_verb_form(cv_pattern)


# -----------------------------------------------------------------------------
# analyze_verb_pattern Method - طريقة analyze_verb_pattern
# -----------------------------------------------------------------------------


def analyze_verb_pattern(cv_pattern: str) -> Dict:
    """دالة مساعدة شاملة لتحليل نمط الفعل"""
    recognizer = VerbPatternRecognizer()
    return recognizer.get_verb_analysis(cv_pattern)


# -----------------------------------------------------------------------------
# check_derivation_potential Method - طريقة check_derivation_potential
# -----------------------------------------------------------------------------


def check_derivation_potential(word: str, cv_pattern: str) -> Dict:
    """دالة مساعدة لفحص إمكانية الاشتقاق"""
    recognizer = VerbPatternRecognizer()
    return recognizer.check_word_against_patterns(word, cv_pattern)


# اختبار سريع,
    if __name__ == "__main__":
    recognizer = VerbPatternRecognizer()

    test_cases = [
    ("كتب", "CVCVC"),
    ("من", "CVC"),
    ("استخرج", "CCVCVC"),
    ("إذا", "VCV"),
    ("قاتل", "CVVCVC"),
    ]

    print(" اختبار كاشف أنماط الأفعال")
    print("=" * 50)

    for word, pattern in test_cases:
    analysis = recognizer.check_word_against_patterns(word, pattern)
    print(f"\n الكلمة: {word} | النمط: {pattern}")
    print(f"   التصنيف: {analysis['classification']}")
    print(f"   السبب: {analysis['reason']}")
    print(f"   الثقة: {analysis['confidence']:.1%}")

    print("\n إحصائيات الأنماط:")
    stats = recognizer.get_pattern_statistics()
    print(f"   إجمالي أنماط الأفعال: {stats['total_verb_patterns']}")
    print(f"   الأنماط الجامدة: {stats['frozen_patterns']}")
    print(f"   عالية الاشتقاق: {stats['high_derivability']}")

