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
 FrozenRootsEngine Models Package,
    حزمة نماذج محرك تصنيف الجذور الجامدة,
    تحتوي على جميع المكونات الأساسية:
- المصنف المتقدم للجذور
- محلل المقاطع الصوتية
- كاشف أنماط الأفعال
"""

from .classifier import (
    AdvancedRootClassifier,
    classify_root,
    is_frozen_root,
    get_root_confidence,
)  # noqa: F401
    from .syllable_check import (
    SyllabicUnitAnalyzer,
    get_cv_pattern,
    analyze_syllabic_units,
)  # noqa: F401
    from .verb_check import (
    VerbPatternRecognizer,
    is_verb_form,
    analyze_verb_pattern,
    check_derivation_potential,
)  # noqa: F401,
    __all__ = [
    'AdvancedRootClassifier',
    'classify_root',
    'is_frozen_root',
    'get_root_confidence',
    'SyllabicUnitAnalyzer',
    'get_cv_pattern',
    'analyze_syllabic_units',
    'VerbPatternRecognizer',
    'is_verb_form',
    'analyze_verb_pattern',
    'check_derivation_potential',
]
