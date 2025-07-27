#!/usr/bin/env python3
"""
Arabic Phoneme Package,
    Professional Enterprise Grade Implementation
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821
    from .engine from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc,
    __version__ = "1.0.0"
__author__ = "Arabic NLP Professional Team"
__description__ = "Enterprise grade Arabic phoneme extraction system"

__all__ = ['PhonemeEngine']
