#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مولد حروف المعاني العربية باستخدام قاعدة بيانات المقاطع الحقيقية,
    Arabic Function Words Generator Using Real Syllable Database,
    يستخدم الـ 22,218 مقطع صوتي المولد من النظام الشامل,
    Uses the 22,218 syllables generated from comprehensive system,
    المطور: نظام الذكاء الاصطناعي العربي,
    Developer: Arabic AI System
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc
    import sys  # noqa: F401
    import json  # noqa: F401
    import re  # noqa: F401
    import random  # noqa: F401
    from typing import Dict, List, Optional, Any
    from dataclasses import dataclass, field  # noqa: F401
    from enum import Enum  # noqa: F401
    import logging  # noqa: F401

# Configure logging,
    logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SYLLABLE DATABASE - التكامل مع قاعدة بيانات المقاطع
# ═══════════════════════════════════════════════════════════════════════════════════


def load_syllable_database():  # type: ignore[no-untyped-def]
    """تحميل قاعدة بيانات المقاطع من النظام الشامل"""

    try:
        # استيراد النظام الشامل لتوليد المقاطع
    from comprehensive_arabic_verb_syllable_generator import ()
    ComprehensiveArabicVerbSyllableGenerator)  # noqa: F401

        # إنشاء مولد المقاطع,
    syllable_generator = ComprehensiveArabicVerbSyllableGenerator()

        # توليد قاعدة بيانات شاملة,
    logger.info("بدء توليد قاعدة بيانات المقاطع الشاملة...")
    syllable_database = ()
    syllable_generator.generate_comprehensive_syllable_database()
    )

    logger.info(f"تم تحميل {len(syllable_database)} مقطع صوتي من النظام الشامل")

    return syllable_database,
    except ImportError:
    logger.warning()
    "لم يتم العثور على النظام الشامل، سيتم استخدام قاعدة بيانات تجريبية"
    )  # noqa: E501,
    return create_advanced_mock_database()


def create_advanced_mock_database():  # type: ignore[no-untyped def]
    """إنشاء قاعدة بيانات متقدمة للمقاطع (للاختبار)"""

    syllables = []

    # الأحرف العربية

    # حروف مفضلة لحروف المعاني,
    preferred_consonants = [
    'ب',
    'ت',
    'ج',
    'د',
    'ر',
    'ل',
    'م',
    'ن',
    'ه',
    'و',
    'ي',
    'ع',
    'ف',
    ]

    # الحركات,
    short_vowels = ['َ', 'ِ', 'ُ']

    # توليد مقاطع CV,
    for consonant in preferred_consonants:
        for vowel in short_vowels:
    syllables.append()
    {
    'syllable': consonant + vowel,
    'pattern': 'CV',
    'consonants': [consonant],
    'vowels': [vowel],
    'weight': 'light',
    'frequency': 0.7,
    'function_word_suitable': True,
    }
    )

    # توليد مقاطع CVC,
    end_consonants = ['ن', 'ل', 'ر', 'م', 'ت', 'د', 'س', 'ك', 'ي']
    for c1 in preferred_consonants[:10]:
        for vowel in short_vowels:
            for c2 in end_consonants:
    syllables.append()
    {
    'syllable': c1 + vowel + c2,
    'pattern': 'CVC',
    'consonants': [c1, c2],
    'vowels': [vowel],
    'weight': 'medium',
    'frequency': 0.5,
    'function_word_suitable': True,
    }
    )

    # مقاطع خاصة بحروف المعاني,
    special_function_syllables = [
        # حروف الجر
    {'syllable': 'في', 'pattern': 'CVC', 'category': 'preposition'},
    {'syllable': 'عَل', 'pattern': 'CVC', 'category': 'preposition'},
    {'syllable': 'مِن', 'pattern': 'CVC', 'category': 'preposition'},
    {'syllable': 'إِل', 'pattern': 'CVC', 'category': 'preposition'},
        # أدوات العطف
    {'syllable': 'وَ', 'pattern': 'CV', 'category': 'conjunction'},
    {'syllable': 'فَ', 'pattern': 'CV', 'category': 'conjunction'},
    {'syllable': 'ثُم', 'pattern': 'CVC', 'category': 'conjunction'},
        # أدوات الاستفهام
    {'syllable': 'هَل', 'pattern': 'CVC', 'category': 'interrogative'},
    {'syllable': 'مَا', 'pattern': 'CV', 'category': 'interrogative'},
    {'syllable': 'مَن', 'pattern': 'CVC', 'category': 'interrogative'},
        # أدوات النفي
    {'syllable': 'لا', 'pattern': 'CV', 'category': 'negation'},
    {'syllable': 'لَن', 'pattern': 'CVC', 'category': 'negation'},
    {'syllable': 'لَم', 'pattern': 'CVC', 'category': 'negation'},
    ]

    for special in special_function_syllables:
    special.update()
    {
    'weight': 'light',
    'frequency': 1.0,
    'function_word_suitable': True,
    'is_authentic': True,
    }
    )
    syllables.append(special)

    logger.info(f"تم إنشاء {len(syllables)} مقطع صوتي متقدم")
    return syllables


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED FUNCTION WORDS GENERATOR - مولد حروف المعاني المحسن
# ═══════════════════════════════════════════════════════════════════════════════════


class FunctionWordCategory(Enum):
    """فئات حروف المعاني"""

    PREPOSITIONS = "prepositions"  # حروف الجر,
    CONJUNCTIONS = "conjunctions"  # أدوات العطف,
    PARTICLES = "particles"  # الأدوات,
    INTERROGATIVES = "interrogatives"  # أدوات الاستفهام,
    NEGATIONS = "negations"  # أدوات النفي,
    DETERMINERS = "determiners"  # أدوات التحديد


@dataclass,
    class FunctionWordResult:
    """نتيجة توليد حرف معنى"""

    word: str,
    category: FunctionWordCategory,
    syllables: List[str]
    pattern: str,
    authenticity_score: float,
    phonetic_weight: float,
    is_known_word: bool = False,
    closest_known: str = ""


class AdvancedArabicFunctionWordsGenerator:
    """مولد حروف المعاني العربية المتقدم"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
        # تحميل قاعدة بيانات المقاطع,
    self.syllables_database = load_syllable_database()

        # قوائم حروف المعاني المعروفة,
    self.known_function_words = {
    FunctionWordCategory.PREPOSITIONS: [
    'في',
    'على',
    'إلى',
    'من',
    'عن',
    'لدى',
    'بين',
    'تحت',
    'فوق',
    'أمام',
    'خلف',
    'عند',
    'لدن',
    'منذ',
    'مذ',
    'حتى',
    'سوى',
    'خلا',
    'عدا',
    'حاشا',
    'كلا',
    ],
    FunctionWordCategory.CONJUNCTIONS: [
    'و',
    'ف',
    'ثم',
    'أو',
    'أم',
    'لكن',
    'غير',
    'سوى',
    'إما',
    'بل',
    'لا',
    ],
    FunctionWordCategory.PARTICLES: [
    'قد',
    'لقد',
    'كان',
    'إن',
    'أن',
    'كي',
    'لكي',
    'حتى',
    'لعل',
    'ليت',
    'عسى',
    ],
    FunctionWordCategory.INTERROGATIVES: [
    'هل',
    'أ',
    'ما',
    'من',
    'متى',
    'أين',
    'كيف',
    'لماذا',
    'أي',
    'كم',
    'أين',
    ],
    FunctionWordCategory.NEGATIONS: [
    'لا',
    'ما',
    'لن',
    'لم',
    'ليس',
    'غير',
    'سوى',
    'بل',
    ],
    FunctionWordCategory.DETERMINERS: [
    'ال',
    'هذا',
    'هذه',
    'ذلك',
    'تلك',
    'أي',
    'كل',
    'بعض',
    'جميع',
    'كلا',
    ],
    }

        # قيود صوتية لكل فئة,
    self.category_constraints = {
    FunctionWordCategory.PREPOSITIONS: {
    'max_syllables': 2,
    'preferred_patterns': ['CV', 'CVC', 'CVCV'],
    'max_length': 4,
    'preferred_initials': ['ف', 'ع', 'ب', 'ل', 'م', 'إ'],
    },
    FunctionWordCategory.CONJUNCTIONS: {
    'max_syllables': 1,
    'preferred_patterns': ['CV', 'CVC'],
    'max_length': 3,
    'preferred_initials': ['و', 'ف', 'ث', 'أ', 'ب'],
    },
    FunctionWordCategory.PARTICLES: {
    'max_syllables': 2,
    'preferred_patterns': ['CV', 'CVC', 'CVCV'],
    'max_length': 4,
    'preferred_initials': ['ق', 'ل', 'ك', 'إ', 'أ'],
    },
    FunctionWordCategory.INTERROGATIVES: {
    'max_syllables': 2,
    'preferred_patterns': ['CV', 'CVC', 'CVCV'],
    'max_length': 4,
    'preferred_initials': ['ه', 'أ', 'م', 'ك'],
    },
    FunctionWordCategory.NEGATIONS: {
    'max_syllables': 2,
    'preferred_patterns': ['CV', 'CVC'],
    'max_length': 3,
    'preferred_initials': ['ل', 'م', 'ن'],
    },
    FunctionWordCategory.DETERMINERS: {
    'max_syllables': 2,
    'preferred_patterns': ['V', 'CV', 'CVC'],
    'max_length': 4,
    'preferred_initials': ['أ', 'ه', 'ت', 'ذ'],
    },
    }

    def generate_function_words()
    self, category: FunctionWordCategory, count: int = 30
    ) -> List[FunctionWordResult]:
    """توليد حروف معاني لفئة محددة"""

    logger.info(f"بدء توليد {count كلمة من} فئة {category.value}}")

    constraints = self.category_constraints[category]
    suitable_syllables = self._filter_suitable_syllables(category)

    results = []
    attempts = 0,
    max_attempts = count * 15,
    while len(results) < count and attempts < max_attempts:
    attempts += 1

            # توليد مرشح,
    candidate = self._generate_candidate(suitable_syllables, constraints)

            if candidate:
                # تقييم المرشح,
    result = self._evaluate_candidate(candidate, category)

                if result and result.authenticity_score >= 0.3:  # حد أدنى للقبول,
    results.append(result)

                    if len(results) % 10 == 0:
    logger.info(f"تم توليد {len(results)} كلمة...")

        # ترتيب حسب جودة التشابه,
    results.sort(key=lambda x: x.authenticity_score, reverse=True)

    logger.info(f"تم الانتهاء من توليد {len(results) كلمة من} فئة {category.value}}")
    return results,
    def _filter_suitable_syllables(self, category: FunctionWordCategory) -> List[Dict]:
    """تصفية المقاطع المناسبة للفئة"""

    suitable = []
    constraints = self.category_constraints[category]
    preferred_initials = constraints.get('preferred_initials', [])

        for syllable in self.syllables_database:
            # فحص الملاءمة العامة,
    if syllable.get('function_word_suitable', True):

                # فحص الحروف المفضلة,
    syl_text = syllable['syllable']
                if ()
    preferred_initials,
    and syl_text,
    and syl_text[0] not in preferred_initials
    ):
    continue

                # فحص النمط المقطعي,
    pattern = syllable.get('pattern', '')
                if pattern in constraints['preferred_patterns']:
    suitable.append(syllable)

    logger.info()
    f"تم تصفية {len(suitable) مقطع مناسب من} أصل {len(self.syllables_database)}}"
    )  # noqa: E501,
    return suitable,
    def _generate_candidate()
    self, syllables: List[Dict], constraints: Dict
    ) -> Optional[str]:
    """توليد مرشح لحرف معنى"""

    max_syllables = constraints['max_syllables']
    max_length = constraints['max_length']

        # اختيار عدد المقاطع,
    num_syllables = random.randint(1, max_syllables)

        # اختيار المقاطع,
    chosen_syllables = []
        for _ in range(num_syllables):
            if syllables:
    syllable = random.choice(syllables)
    chosen_syllables.append(syllable)

        if not chosen_syllables:
    return None

        # تكوين الكلمة,
    word = ''.join([syl['syllable'] for syl in chosen_syllables])

        # فحص القيود,
    if len(word) > max_length:
    return None

        # تطبيق تعديلات صوتية,
    word = self._apply_phonetic_rules(word)

    return word,
    def _apply_phonetic_rules(self, word: str) -> str:
    """تطبيق قواعد صوتية"""

        # إزالة التكرارات المفرطة,
    word = re.sub(r'(.)\1{2,}', r'\1\1', word)

        # تبسيط بعض التراكيب,
    word = re.sub(r'([ّْ]){2,}', r'\1', word)

    return word,
    def _evaluate_candidate()
    self, candidate: str, category: FunctionWordCategory
    ) -> Optional[FunctionWordResult]:
    """تقييم مرشح لحرف معنى"""

    known_words = self.known_function_words[category]

        # فحص إذا كانت كلمة معروفة,
    is_known = candidate in known_words

        # حساب التشابه مع الكلمات المعروفة,
    max_similarity = 0.0,
    closest_word = ""

        for known_word in known_words:
    similarity = self._calculate_similarity(candidate, known_word)
            if similarity > max_similarity:
    max_similarity = similarity,
    closest_word = known_word

        # حساب الوزن الصوتي,
    phonetic_weight = self._calculate_phonetic_weight(candidate)

        # تقسيم إلى مقاطع,
    syllables = self._breakdown_syllables(candidate)
    pattern = self._extract_pattern(candidate)

    return FunctionWordResult()
    word=candidate,
    category=category,
    syllables=syllables,
    pattern=pattern,
    authenticity_score=max_similarity,
    phonetic_weight=phonetic_weight,
    is_known_word=is_known,
    closest_known=closest_word)

    def _calculate_similarity(self, word1: str, word2: str) -> float:
    """حساب التشابه بين كلمتين"""

        if word1 == word2:
    return 1.0

        # التشابه في الطول,
    len_sim = 1 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)

        # التشابه في الأحرف,
    set1, set2 = set(word1), set(word2)
    char_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

        # التشابه في البداية والنهاية,
    start_sim = 1 if word1[:1] == word2[:1] else 0,
    end_sim = 1 if word1[-1:] == word2[-1:] else 0

        # متوسط مرجح,
    similarity = 0.4 * len_sim + 0.4 * char_sim + 0.1 * start_sim + 0.1 * end_sim,
    return similarity,
    def _calculate_phonetic_weight(self, word: str) -> float:
    """حساب الوزن الصوتي"""

    weight = len(word) * 0.1

        # الأحرف الثقيلة,
    heavy_sounds = {'ق', 'ط', 'ص', 'ض', 'ظ', 'ع', 'غ', 'خ', 'ح'}
    weight += len([c for c in word if c in heavy_sounds]) * 0.3

        # التشديد والحركات الطويلة,
    weight += word.count('ّ') * 0.2,
    weight += len([c for c in word if c in {'ا', 'ي', 'و'}]) * 0.1,
    return weight,
    def _breakdown_syllables(self, word: str) -> List[str]:
    """تقسيم الكلمة إلى مقاطع"""

        # تقسيم مبسط,
    syllables = []
    current = ""

    vowels = {'َ', 'ِ', 'ُ', 'ا', 'ي', 'و'}

        for i, char in enumerate(word):
    current += char

            # إذا وصلنا لصائت وما بعده صامت، أنهي المقطع,
    if char in vowels and i < len(word) - 1:
    next_char = word[i + 1]
                if next_char not in vowels:
    syllables.append(current)
    current = ""

        # إضافة المقطع الأخير,
    if current:
    syllables.append(current)

    return syllables if syllables else [word]

    def _extract_pattern(self, word: str) -> str:
    """استخراج النمط المقطعي"""

    consonants = {
    'ب',
    'ت',
    'ث',
    'ج',
    'ح',
    'خ',
    'د',
    'ذ',
    'ر',
    'ز',
    'س',
    'ش',
    'ص',
    'ض',
    'ط',
    'ظ',
    'ع',
    'غ',
    'ف',
    'ق',
    'ك',
    'ل',
    'م',
    'ن',
    'ه',
    'و',
    'ي',
    'ء',
    }

    vowels = {'َ', 'ِ', 'ُ', 'ا', 'ي', 'و', 'ً', 'ٌ', 'ٍ'}

    pattern = ""
        for char in word:
            if char in consonants:
    pattern += "C"
            elif char in vowels:
    pattern += "V"

    return pattern,
    def generate_comprehensive_analysis()
    self) -> Dict[FunctionWordCategory, List[FunctionWordResult]]:
    """تحليل شامل لجميع فئات حروف المعاني"""

    results = {}

        for category in FunctionWordCategory:
    logger.info(f"معالجة فئة {category.value...}")
    category_results = self.generate_function_words(category, count=25)
    results[category] = category_results,
    return results,
    def print_comprehensive_report(self, results: Dict[FunctionWordCategory, List[FunctionWordResult]]):  # type: ignore[no-untyped def]
    """طباعة تقرير شامل"""

    print("\n" + "═" * 70)
    print("🔵 تقرير شامل لتوليد حروف المعاني العربية")
    print("═" * 70)

    total_generated = sum(len(words) for words in results.values())
    total_known = sum()
    len([w for w in words if w.is_known_word]) for words in results.values()
    )

    print("\n📊 إحصائيات عامة:")
    print(f"   • إجمالي الكلمات المولدة: {total_generated}")
    print(f"   • الكلمات المعروفة المكتشفة: {total_known}")
    print(f"   • الكلمات الجديدة المولدة: {total_generated} - total_known}")

        for category, words in results.items():
            if not words:
    continue,
    print(f"\n▶ {category.value.upper()} ({len(words) كلمة):}")
    print(" " * 50)

            # أفضل النتائج,
    top_words = words[:8]

            for i, word in enumerate(top_words, 1):
    status = ()
    "✅ معروفة"
                    if word.is_known_word,
    else f"🔍 تشابه: {word.authenticity_score:.2f}"
    )

    print(f"  {i}. {word.word:6} - {status}")
    print(f"     📝 المقاطع: {'} + '.join(word.syllables)}")
    print(f"     🔧 النمط: {word.pattern}")

                if word.closest_known and not word.is_known_word:
    print(f"     🎯 أقرب كلمة: {word.closest_known}")

    print()

    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION - العرض الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════════


def main():  # type: ignore[no-untyped def]
    """العرض الرئيسي للنظام"""

    print("🚀 مولد حروف المعاني العربية المتقدم")
    print("Advanced Arabic Function Words Generator")
    print("=" * 60)

    # إنشاء المولد,
    generator = AdvancedArabicFunctionWordsGenerator()

    # تشغيل التحليل الشامل,
    results = generator.generate_comprehensive_analysis()

    # طباعة التقرير,
    generator.print_comprehensive_report(results)

    return generator, results,
    if __name__ == "__main__":
    main()

