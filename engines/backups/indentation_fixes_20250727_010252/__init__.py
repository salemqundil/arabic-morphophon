#!/usr/bin/env python3
"""
Advanced Phonological Engine Package,
    Arabic NLP Mathematical Framework - Phase 1 Week 2
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821
    from .engine import PhonologicalEngine  # noqa: F401
    from .models.assimilation import AssimilationRule  # noqa: F401
    from .models.deletion import DeletionRule  # noqa: F401
    from .models.inversion import InversionRule  # noqa: F401
    from .models.rule_base import PhonoRule, ContextualRule, SequentialRule  # noqa: F401

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc,
    __version__ = "1.0.0"
__author__ = "Arabic NLP Engine Team"

__all__ = [
    'PhonologicalEngine',
    'AssimilationRule',
    'DeletionRule',
    'InversionRule',
    'PhonoRule',
    'ContextualRule',
    'SequentialRule',
]
