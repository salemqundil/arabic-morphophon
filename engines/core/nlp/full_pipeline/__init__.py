#!/usr/bin/env python3
"""
 محرك المعالجة الشاملة للنصوص العربية - Full Pipeline Engine
============================================================

Features:
 تكامل جميع محركات NLP العربية في نظام موحد,
    معالجة متوازية عالية الأداء,
    تحليل شامل للأوزان والجذور والصرف والأصوات,
    واجهة ويب تفاعلية مع Flask,
    إحصائيات متقدمة ومراقبة الأداء,
    تصدير النتائج بصيغ متعددة (JSON/CSV)

Author: Arabic NLP Expert Team,
    Version: 2.0.0
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
from .engine import FullPipelineEngine, create_flask_app  # noqa: F401

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc,
__version__ = "2.0.0"
__author__ = "Arabic NLP Expert Team"
__description__ = " Comprehensive Arabic NLP Pipeline with Advanced Weight Analysis"

__all__ = ['FullPipelineEngine', 'create_flask_app']
