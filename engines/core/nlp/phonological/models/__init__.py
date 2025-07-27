#!/usr/bin/env python3
"""
Phonological Models Package,
    Mathematical framework implementations for Arabic phonological rules
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
    from .rule_base import PhonoRule, ContextualRule, SequentialRule  # noqa: F401
    from .assimilation import AssimilationRule  # noqa: F401
    from .deletion import DeletionRule  # noqa: F401
    from .inversion import InversionRule  # noqa: F401,
    __all__ = [
    'PhonoRule',
    'ContextualRule',
    'SequentialRule',
    'AssimilationRule',
    'DeletionRule',
    'InversionRule',
]
