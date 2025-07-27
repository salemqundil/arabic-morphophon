"""
ملف الإعداد المشترك لاختبارات المحرك العربي
Shared configuration for Arabic Engine tests
"""

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


import pytest  # noqa: F401
import sys  # noqa: F401
import os  # noqa: F401
from pathlib import Path  # noqa: F401

# إضافة مسار المشروع إلى Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def arabic_test_data():  # type: ignore[no-untyped def]
    """بيانات اختبار للنصوص العربية"""
    return {
        'simple_words': ['كتاب', 'قلم', 'مدرسة', 'طالب'],
        'complex_words': ['الطالبات', 'المدرسون', 'الكتابات'],
        'phonemes': ['ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ'],
        'vowels': ['َ', 'ِ', 'ُ', 'ا', 'و', 'ي'],
        'syllables': ['با', 'تا', 'كا', 'ما'],
        'roots': [('ك', 'ت', 'ب'), ('د', 'ر', 'س'), ('ع', 'ل', 'م')],
    }


@pytest.fixture(scope="session")
def test_config():  # type: ignore[no-untyped def]
    """إعدادات الاختبار"""
    return {'timeout': 30, 'encoding': 'utf 8', 'debug': True, 'verbose': True}


@pytest.fixture
def sample_arabic_text():  # type: ignore[no-untyped-def]
    """نص عربي عينة للاختبار"""
    return "هذا نص عربي للاختبار يحتوي على كلمات مختلفة"


@pytest.fixture
def phoneme_database():  # type: ignore[no-untyped def]
    """قاعدة بيانات الأصوات للاختبار"""
    return {
        'ب': {'type': 'consonant', 'frequency': 0.85, 'position': 'any'},
        'ت': {'type': 'consonant', 'frequency': 0.75, 'position': 'any'},
        'ا': {'type': 'vowel', 'frequency': 0.95, 'position': 'any'},
        'م': {'type': 'consonant', 'frequency': 0.80, 'position': 'any'},
    }


@pytest.fixture
def morphology_rules():  # type: ignore[no-untyped-def]
    """قواعد الصرف للاختبار"""
    return {
        'prefixes': ['ال', 'و', 'ف', 'ب', 'ك', 'ل'],
        'suffixes': ['ة', 'ات', 'ون', 'ين', 'ها', 'هم'],
        'patterns': ['فعل', 'فاعل', 'مفعول', 'فعال'],
    }


@pytest.fixture
def syllable_patterns():  # type: ignore[no-untyped-def]
    """أنماط المقاطع للاختبار"""
    return {
        'CV': ['با', 'تا', 'كا'],
        'CVC': ['بات', 'كتب', 'درس'],
        'CVCC': ['كتبت', 'درست'],
        'CVVC': ['باات', 'تاان'],
    }


@pytest.fixture
def sample_fixture():  # type: ignore[no-untyped-def]
    """TODO: Add docstring."""
    return "Hello, World!"


def pytest_configure(config):  # type: ignore[no-untyped def]
    """إعداد pytest عند البدء"""
    config.addinivalue_line("markers", "arabic: اختبارات خاصة باللغة العربية")
    config.addinivalue_line("markers", "phoneme: اختبارات معالجة الأصوات")
    config.addinivalue_line("markers", "morphology: اختبارات الصرف والتحليل الصرفي")


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped def]
    """تعديل عناصر الاختبار المجمعة"""
    for item in items:
        # إضافة علامة arabic لجميع الاختبارات
        if "arabic" not in [mark.name for mark in item.iter_markers()]:
            item.add_marker(pytest.mark.arabic)


def pytest_runtest_setup(item):  # type: ignore[no-untyped def]
    """إعداد قبل تشغيل كل اختبار"""
    # يمكن إضافة منطق إعداد مخصص هنا
    pass


def pytest_runtest_teardown(item, nextitem):  # type: ignore[no-untyped def]
    """تنظيف بعد تشغيل كل اختبار"""
    # يمكن إضافة منطق تنظيف مخصص هنا
    pass
