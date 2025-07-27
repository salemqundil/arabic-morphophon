#!/usr/bin/env#!/usr/bin/env python3
"""
Core Arabic NLP Package
=======================

Unified Arabic NLP processing engines and interfaces.

This package provides:
- Unified engine interface (engine.py)
- Phonological processing (phonology.py)
- Morphological analysis (morphology.py)
- Inflection processing (inflection.py)
- Base engines and utilities,
    Usage:
    from core import UnifiedArabicEngine,
    engine = UnifiedArabicEngine()
    results = engine.process_text("النص العربي")
"""

from .engine import UnifiedArabicEngine, create_engine
    from .phonology import PhonologyEngine
    from .morphology import MorphologyEngine
    from .inflection import InflectionEngine

# Version info,
    __version__ = "1.0.0"
__author__ = "Arabic NLP Team"
__email__ = "contact@arabic-nlp.org"

# Public API,
    __all__ = [
    'UnifiedArabicEngine',
    'create_engine',
    'PhonologyEngine',
    'MorphologyEngine',
    'InflectionEngine',
]


# Quick access functions,
    def quick_analyze(text: str, engines: list = None):
    """Quick analysis function for simple use cases"""
    engine = create_engine()
    return engine.process_text(text, engines)


def get_version():
    """Get package version"""
    return __version__on3


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

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many statements
# noqa: E501,F401,F403,E722,A001,F821


"""
Base Engine for Arabic NLP System
"""

from .base_engine import BaseNLPEngine,
    __all__ = ['BaseNLPEngine']
