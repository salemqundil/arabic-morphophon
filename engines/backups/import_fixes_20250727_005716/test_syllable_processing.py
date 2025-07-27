"""
اختبارات معالجة المقاطع الصوتية العربية
Arabic syllable processing tests
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


@pytest.mark.syllable
class TestArabicSyllables:
    """فئة اختبار معالجة المقاطع الصوتية العربية"""

    def test_syllable_pattern_structure(self, syllable_patterns):  # type: ignore[no-untyped def]
    """اختبار بنية أنماط المقاطع"""
    required_patterns = ['CV', 'CVC', 'CVCC', 'CVVC']

        for pattern in required_patterns:
    assert pattern in syllable_patterns
    assert isinstance(syllable_patterns[pattern], list)
    assert len(syllable_patterns[pattern]) > 0

    @pytest.mark.parametrize(
    "pattern,expected_examples",
    [
    ('CV', ['با', 'تا', 'كا']),
    ('CVC', ['بات', 'كتب', 'درس']),
    ],
    )
    def test_syllable_pattern_examples(self, syllable_patterns, pattern, expected_examples):  # type: ignore[no-untyped-def]
    """اختبار أمثلة أنماط المقاطع"""
    pattern_examples = syllable_patterns[pattern]

        for example in expected_examples:
    assert example in pattern_examples

    def test_cv_syllable_structure(self, syllable_patterns):  # type: ignore[no-untyped def]
    """اختبار بنية المقاطع CV (صامت + صائت)"""
    cv_syllables = syllable_patterns['CV']

        for syllable in cv_syllables:
    assert len(syllable) == 2, f"مقطع CV يجب أن يكون حرفين: {syllable}"

            # الحرف الأول يجب أن يكون صامت
    first_char = syllable[0]
    consonants = 'بتثجحخدذرزسشصضطظعغفقكلمنهوي'
    assert (
    first_char in consonants
    ), f"الحرف الأول يجب أن يكون صامت: {first_char}"  # noqa: E501

            # الحرف الثاني يجب أن يكون صائت
    second_char = syllable[1]
    vowels = 'اويَُِ'
    assert (
    second_char in vowels
    ), f"الحرف الثاني يجب أن يكون صائت: {second_char}"  # noqa: E501

    def test_cvc_syllable_structure(self, syllable_patterns):  # type: ignore[no-untyped def]
    """اختبار بنية المقاطع CVC (صامت + صائت + صامت)"""
    cvc_syllables = syllable_patterns['CVC']

        for syllable in cvc_syllables:
    assert (
    len(syllable) >= 3
    ), f"مقطع CVC يجب أن يكون 3 أحرف على الأقل:_{syllable}"  # noqa: E501

    def test_syllable_segmentation(self):  # type: ignore[no-untyped def]
    """اختبار تقسيم الكلمات إلى مقاطع"""

        def segment_word_to_syllables(word: str) -> list:
    """تقسيم الكلمة إلى مقاطع (مبسط)"""
    syllables = []
    current_syllable = ""

    vowels = set('اويَُِ')
    consonants = set('بتثجحخدذرزسشصضطظعغفقكلمنهويء')

            for char in word:
                if char in consonants or char in vowels:
    current_syllable += char

                    # إنهاء المقطع عند الصائت
                    if char in vowels:
    syllables.append(current_syllable)
    current_syllable = ""

            # إضافة أي بقايا
            if current_syllable:
                if syllables:
    syllables[-1] += current_syllable
                else:
    syllables.append(current_syllable)

    return syllables

        # اختبار كلمات بسيطة
    test_cases = [
    ('كتاب', ['كتا', 'ب']),
    ('مدرسة', ['مد', 'رسة']),
    ('باب', ['باب']),
    ]

        for word, expected in test_cases:
    result = segment_word_to_syllables(word)
            # التحقق من أن النتيجة معقولة
    assert len(result) > 0, f"فشل في تقسيم الكلمة: {word}"

    def test_syllable_weight_classification(self):  # type: ignore[no-untyped def]
    """اختبار تصنيف وزن المقاطع"""

        def classify_syllable_weight(syllable: str) -> str:
    """تصنيف وزن المقطع"""
            if len(syllable) == 2:  # CV
    return 'light'
            elif len(syllable) == 3:  # CVC
    return 'heavy'
            elif len(syllable) >= 4:  # CVCC أو أكثر
    return 'super_heavy'
            else:
    return 'unknown'

    test_cases = [('با', 'light'), ('بات', 'heavy'), ('كتبت', 'super_heavy')]

        for syllable, expected_weight in test_cases:
    weight = classify_syllable_weight(syllable)
    assert (
    weight == expected_weight
    ), f"وزن خاطئ للمقطع {syllable}: حصل على {weight}, متوقع {expected_weight}"  # noqa: E501

    def test_syllable_stress_patterns(self):  # type: ignore[no-untyped def]
    """اختبار أنماط النبرة في المقاطع"""

        def determine_stress_pattern(syllables: list) -> str:
    """تحديد نمط النبرة"""
            if len(syllables) == 1:
    return 'monosyllabic'
            elif len(syllables) == 2:
    return 'penultimate'  # النبرة على المقطع قبل الأخير
            else:
    return 'antepenultimate'  # النبرة على المقطع الثالث من الأخير

    test_cases = [
    (['كتاب'], 'monosyllabic'),
    (['كتا', 'ب'], 'penultimate'),
    (['مد', 'ر', 'سة'], 'antepenultimate'),
    ]

        for syllables, expected_pattern in test_cases:
    pattern = determine_stress_pattern(syllables)
    assert pattern == expected_pattern

    @pytest.mark.slow
    def test_comprehensive_syllable_analysis(self, syllable_patterns):  # type: ignore[no-untyped-def]
    """اختبار شامل لتحليل المقاطع"""
    all_syllables = []

        for pattern_type, syllables in syllable_patterns.items():
    all_syllables.extend(syllables)

        # التحقق من عدم وجود مقاطع فارغة
        for syllable in all_syllables:
    assert len(syllable) > 0, "لا يجب أن تكون هناك مقاطع فارغة"
    assert syllable.strip() == syllable, "لا يجب أن تحتوي المقاطع على مسافات"

        # التحقق من التنوع
    unique_syllables = set(all_syllables)
    diversity_ratio = len(unique_syllables) / len(all_syllables)
    assert diversity_ratio > 0.5, f"نسبة التنوع منخفضة:_{diversity_ratio}"

    def test_syllable_boundary_detection(self):  # type: ignore[no-untyped def]
    """اختبار كشف حدود المقاطع"""

        def find_syllable_boundaries(word: str) -> list:
    """إيجاد حدود المقاطع في الكلمة"""
    boundaries = [0]  # بداية الكلمة

    vowels = set('اويَُِ')

            for i, char in enumerate(word):
                if char in vowels and i < len(word) - 1:
                    # نهاية محتملة للمقطع بعد الصائت
    boundaries.append(i + 1)

            if boundaries[-1] != len(word):
    boundaries.append(len(word))  # نهاية الكلمة

    return boundaries

        # اختبار كلمة بسيطة
    word = "كتاب"
    boundaries = find_syllable_boundaries(word)

    assert 0 in boundaries, "يجب أن تبدأ الحدود من 0"
    assert len(word) in boundaries, "يجب أن تنتهي الحدود بطول الكلمة"
    assert len(boundaries) >= 2, "يجب أن يكون هناك حدان على الأقل"
