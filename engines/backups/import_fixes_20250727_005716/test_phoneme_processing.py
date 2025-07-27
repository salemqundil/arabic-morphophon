"""
اختبارات معالجة الأصوات العربية
Arabic phoneme processing tests
"""

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


import pytest  # noqa: F401
import sys  # noqa: F401
import os  # noqa: F401
from pathlib import Path  # noqa: F401

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.phoneme
class TestArabicPhonemes:
    """فئة اختبار معالجة الأصوات العربية"""

    def test_consonant_classification(self, phoneme_database):  # type: ignore[no-untyped def]
    """اختبار تصنيف الصوامت"""
    consonants = ['ب', 'ت', 'ث', 'ج', 'ح', 'خ']

        for consonant in consonants:
            if consonant in phoneme_database:
    phoneme_info = phoneme_database[consonant]
    assert phoneme_info['type'] == 'consonant'

    def test_vowel_classification(self, phoneme_database):  # type: ignore[no-untyped-def]
    """اختبار تصنيف الصوائت"""
    vowels = ['ا', 'و', 'ي']

        for vowel in vowels:
            if vowel in phoneme_database:
    phoneme_info = phoneme_database[vowel]
    assert phoneme_info['type'] in ['vowel', 'long_vowel']

    @pytest.mark.parametrize(
    "phoneme,expected_type",
    [('ب', 'consonant'), ('ت', 'consonant'), ('ا', 'vowel'), ('م', 'consonant')],
    )
    def test_phoneme_type_detection(self, phoneme_database, phoneme, expected_type):  # type: ignore[no-untyped-def]
    """اختبار كشف نوع الصوت"""
        if phoneme in phoneme_database:
    assert phoneme_database[phoneme]['type'] == expected_type

    def test_phoneme_frequency_range(self, phoneme_database):  # type: ignore[no-untyped-def]
    """اختبار نطاق تكرار الأصوات"""
        for phoneme, info in phoneme_database.items():
    frequency = info['frequency']
    assert 0 <= frequency <= 1, f"تكرار {phoneme} خارج النطاق: {frequency}"

    def test_get_phoneme_info_function(self, phoneme_database):  # type: ignore[no-untyped def]
    """اختبار دالة الحصول على معلومات الصوت"""

        def get_phoneme_info(char: str, phoneme_db: dict) -> dict:
    """دالة للحصول على معلومات الصوت"""
            if not isinstance(char, str):
    raise ValueError("char must be a string")
            if phoneme_db is None or not isinstance(phoneme_db, dict):
    raise ValueError("phoneme_db must be a dictionary")
    return phoneme_db.get(char, {"type": "unknown", "frequency": 0.0})

        # اختبار حالة صحيحة
    result = get_phoneme_info('ب', phoneme_database)
    assert isinstance(result, dict)
    assert 'type' in result
    assert 'frequency' in result

        # اختبار حالة صوت غير موجود
    result = get_phoneme_info('x', phoneme_database)
    assert result['type'] == 'unknown'
    assert result['frequency'] == 0.0

        # اختبار الأخطاء
        with pytest.raises(ValueError, match="char must be a string"):
    get_phoneme_info(123, phoneme_database)

        with pytest.raises(ValueError, match="phoneme_db must be a dictionary"):
    get_phoneme_info('ب', None)

    def test_arabic_diacritics(self):  # type: ignore[no-untyped-def]
    """اختبار الحركات العربية"""
    diacritics = {
    'َ': 'fatha',
    'ُ': 'damma',
    'ِ': 'kasra',
    'ْ': 'sukun',
    'ً': 'tanween_fath',
    'ٌ': 'tanween_damm',
    'ٍ': 'tanween_kasr',
    }

        for diacritic, name in diacritics.items():
            # التحقق من أن الحركة قابلة للترميز
    encoded = diacritic.encode('utf 8')
    decoded = encoded.decode('utf 8')
    assert decoded == diacritic

    @pytest.mark.slow
    def test_phoneme_combinations(self):  # type: ignore[no-untyped-def]
    """اختبار تركيبات الأصوات"""
    consonants = ['ب', 'ت', 'ك', 'م']
    vowels = ['ا', 'ي', 'و']

    valid_combinations = []

        for consonant in consonants:
            for vowel in vowels:
    combination = consonant + vowel
    valid_combinations.append(combination)

    assert len(valid_combinations) == len(consonants) * len(vowels)
    assert 'با' in valid_combinations
    assert 'تي' in valid_combinations
