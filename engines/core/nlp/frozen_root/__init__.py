#!/usr/bin/env python3
"""
  Init   Module,
    وحدة __init__,
    Implementation of __init__ functionality,
    تنفيذ وظائف __init__,
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
 FrozenRootsEngine Package,
    حزمة محرك تصنيف الجذور الجامدة والمشتقة,
    يوفر تصنيفاً دقيقاً للجذور العربية مع:
 تحليل مقطعي شامل,
    تحليل فونيمي بنظام IPA,
    كشف أنماط الأفعال,
    واجهة REST API
"""

from .engine import FrozenRootsEngine  # noqa: F401
    from .models.classifier import (
    AdvancedRootClassifier,
    classify_root,
    is_frozen_root,
)  # noqa: F401
    from .models.syllable_check import (
    SyllabicUnitAnalyzer,
    get_cv_pattern,
    analyze_syllabic_units,
)  # noqa: F401
    from .models.verb_check import (
    VerbPatternRecognizer,
    is_verb_form,
    analyze_verb_pattern,
)  # noqa: F401

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc,
    __version__ = "1.0.0"
__author__ = "Arabic NLP Team"
__description__ = "Arabic Root Classification Engine - Frozen vs Derivable"

# تصدير الوظائف الرئيسية,
    __all__ = [
    # المحرك الرئيسي
    'FrozenRootsEngine',
    # مصنف الجذور
    'AdvancedRootClassifier',
    'classify_root',
    'is_frozen_root',
    # محلل المقاطع
    'SyllabicUnitAnalyzer',
    'get_cv_pattern',
    'analyze_syllabic_units',
    # كاشف أنماط الأفعال
    'VerbPatternRecognizer',
    'is_verb_form',
    'analyze_verb_pattern',
]

# دوال مساعدة سريعة

# -----------------------------------------------------------------------------
# quick_classify Method - طريقة quick_classify
# -----------------------------------------------------------------------------


def quick_classify(word: str) -> str:
    """تصنيف سريع للكلمة"""
    engine = FrozenRootsEngine()
    result = engine.classify(word)
    return result["classification"]


# -----------------------------------------------------------------------------
# is_word_frozen Method - طريقة is_word_frozen
# -----------------------------------------------------------------------------


def is_word_frozen(word: str) -> bool:
    """فحص سريع للجذر الجامد"""
    return quick_classify(word) == "frozen"


# -----------------------------------------------------------------------------
# get_word_pattern Method - طريقة get_word_pattern
# -----------------------------------------------------------------------------


def get_word_pattern(word: str) -> str:
    """الحصول على نمط CV للكلمة"""
    return get_cv_pattern(word)


# معلومات الحزمة,
    PACKAGE_INFO = {
    "name": "frozen_root",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
    "Arabic root classification (frozen/derivable)",
    "SyllabicUnit structure analysis",
    "IPA phoneme conversion",
    "Verb pattern recognition",
    "Batch processing support",
    "REST API integration",
    ],
    "supported_patterns": [
    "CV",
    "VC",
    "CVC",
    "CVCV",
    "CVCVC",
    "CVVCVC",
    "CVCCVC",
    "CCVCVC",
    "CVCVCVC",
    ],
    "categories": {
    "frozen": [
    "interrogative_particle",
    "negation_particle",
    "conditional_particle",
    "vocative_particle",
    "demonstrative_pronoun",
    "affirmation_particle",
    ],
    "derivable": ["trilateral_root", "augmented_root", "quadrilateral_root"],
    },
}
