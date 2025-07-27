#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مولد حروف المعاني العربية المحسن
Enhanced Arabic Function Words Generator

يستخدم قاعدة بيانات المقاطع الصوتية لتوليد حروف المعاني العربية
Uses syllable database to generate Arabic function words

المطور: نظام الذكاء الاصطناعي العربي
Developer: Arabic AI System

التاريخ: 2025
Date: 2025
"""

import re
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# FUNCTION WORD CATEGORIES - تصنيف حروف المعاني
# ═══════════════════════════════════════════════════════════════════════════════════


class FunctionWordCategory(Enum):
    """تصنيف حروف المعاني"""

    PREPOSITIONS = "prepositions"  # حروف الجر
    CONJUNCTIONS = "conjunctions"  # أدوات العطف
    PARTICLES = "particles"  # الأدوات
    INTERROGATIVES = "interrogatives"  # أدوات الاستفهام
    NEGATIONS = "negations"  # أدوات النفي
    DETERMINERS = "determiners"  # أدوات التحديد
    CONDITIONALS = "conditionals"  # أدوات الشرط
    RELATIVE_PRONOUNS = "relative_pronouns"  # الأسماء الموصولة


@dataclass
class FunctionWordPattern:
    """نمط حرف المعاني"""

    category: FunctionWordCategory
    syllable_patterns: List[str]
    phonetic_constraints: Dict[str, Any]
    semantic_features: List[str]
    frequency_weight: float = 1.0


@dataclass
class GeneratedFunctionWord:
    """حرف معنى مولد"""

    word: str
    category: FunctionWordCategory
    pattern: str
    syllable_breakdown: List[str]
    phonetic_features: Dict[str, Any]
    similarity_score: float
    is_authentic: bool = False
    examples: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC PHONETIC ANALYZER - محلل الصوتيات العربية
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicPhoneticAnalyzer:
    """محلل الصوتيات العربية للمساعدة في توليد حروف المعاني"""

    def __init__(self):

        # الصوامت العربية
        self.consonants = {
            'ء',
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
        }

        # الصوائت والحركات
        self.vowels = {
            'short': {'َ', 'ِ', 'ُ'},  # حركات قصيرة
            'long': {'ا', 'ي', 'و'},  # حروف مد
            'tanween': {'ً', 'ٌ', 'ٍ'},  # تنوين
        }

        # علامات أخرى
        self.diacritics = {'ْ', 'ّ', 'ٰ', 'ۡ'}  # سكون  # شدة  # ألف خنجرية  # سكون صغير

        # أنماط صوتية صعبة (يجب تجنبها)
        self.difficult_patterns = [
            r'(.)\1\1',  # ثلاثة أحرف متتالية
            r'[قطصضظ][كتث]',  # تتابع صوامت ثقيلة
            r'ءء',  # همزتان متتاليتان
            r'[ّْ][ّْ]',  # سكونان أو شدتان متتاليتان
        ]

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """تحليل شامل للكلمة العربية"""

        analysis = {
            'length': len(word),
            'syllable_count': self._count_syllables(word),
            'consonant_count': self._count_consonants(word),
            'vowel_count': self._count_vowels(word),
            'initial_sound': word[0] if word else '',
            'final_sound': word[ 1] if word else '',
            'vowel_pattern': self._extract_vowel_pattern(word),
            'has_sukoon': 'ْ' in word,
            'has_shadda': 'ّ' in word,
            'has_tanween': any(t in word for t in self.vowels['tanween']),
            'phonetic_weight': self._calculate_phonetic_weight(word),
            'is_difficult': self._has_difficult_patterns(word),
        }

        return analysis

    def _count_syllables(self, word: str) -> int:
        """عد المقاطع الصوتية"""
        # تقدير بسيط: عد الحركات القصيرة والحروف المد
        vowel_count = 0
        vowel_count += len([c for c in word if c in self.vowels['short']])
        vowel_count += len([c for c in word if c in self.vowels['long']])
        vowel_count += len([c for c in word if c in self.vowels['tanween']])
        return max(1, vowel_count)

    def _count_consonants(self, word: str) -> int:
        """عد الصوامت"""
        return len([c for c in word if c in self.consonants])

    def _count_vowels(self, word: str) -> int:
        """عد الصوائت والحركات"""
        vowel_count = 0
        vowel_count += len([c for c in word if c in self.vowels['short']])
        vowel_count += len([c for c in word if c in self.vowels['long']])
        vowel_count += len([c for c in word if c in self.vowels['tanween']])
        return vowel_count

    def _extract_vowel_pattern(self, word: str) -> str:
        """استخراج نمط الحركات"""
        pattern = []
        for char in word:
            if char in self.vowels['short']:
                pattern.append('V')
            elif char in self.vowels['long']:
                pattern.append('VV')
            elif char in self.vowels['tanween']:
                pattern.append('VN')
            elif char == 'ْ':
                pattern.append('0')
        return ''.join(pattern)

    def _calculate_phonetic_weight(self, word: str) -> float:
        """حساب الوزن الصوتي للكلمة"""
        weight = 0.0

        # وزن الطول
        weight += len(word) * 0.1

        # وزن الصوامت الثقيلة
        heavy_consonants = {'ق', 'ط', 'ص', 'ض', 'ظ', 'ع', 'غ', 'خ', 'ح'}
        weight += len([c for c in word if c in heavy_consonants]) * 0.5

        # وزن الحركات الطويلة
        weight += len([c for c in word if c in self.vowels['long']]) * 0.3

        # وزن التشديد
        weight += word.count('ّ') * 0.4

        return weight

    def _has_difficult_patterns(self, word: str) -> bool:
        """فحص وجود أنماط صوتية صعبة"""
        for pattern in self.difficult_patterns:
            if re.search(pattern, word):
                return True
        return False

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """حساب التشابه الصوتي بين كلمتين"""

        # التشابه في الطول
        length_sim = 1 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)

        # التشابه في الأحرف
        set1, set2 = set(word1), set(word2)
        char_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

        # التشابه في أنماط الحركات
        pattern1 = self._extract_vowel_pattern(word1)
        pattern2 = self._extract_vowel_pattern(word2)
        pattern_sim = 1 if pattern1 == pattern2 else 0.5 if pattern1 and pattern2 else 0

        # التشابه في البداية والنهاية
        start_sim = 1 if word1[:1] == word2[:1] else 0
        end_sim = 1 if word1[-1:] == word2[-1:] else 0

        # متوسط مرجح
        similarity = ()
            0.3 * length_sim
            + 0.4 * char_sim
            + 0.2 * pattern_sim
            + 0.05 * start_sim
            + 0.05 * end_sim
        )

        return similarity


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED FUNCTION WORDS GENERATOR - مولد حروف المعاني المحسن
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedArabicFunctionWordsGenerator:
    """مولد حروف المعاني العربية المحسن باستخدام قاعدة بيانات المقاطع"""

    def __init__(self, syllables_database: Optional[List[Dict]] = None):

        self.syllables_db = syllables_database or self._create_mock_syllables()
        self.phonetic_analyzer = ArabicPhoneticAnalyzer()

        # تحميل الأنماط والقيود
        self._load_function_word_patterns()
        self._load_authentic_function_words()

        logger.info(f"تم تحميل {len(self.syllables_db)} مقطع صوتي")
        logger.info(f"تم تحميل {len(self.function_word_patterns)} نمط لحروف المعاني")

    def _create_mock_syllables(self) -> List[Dict]:
        """إنشاء قاعدة بيانات تجريبية للمقاطع"""

        mock_syllables = []

        # مقاطع CV
        consonants = [
            'ب',
            'ت',
            'ج',
            'د',
            'ر',
            'س',
            'ع',
            'ف',
            'ق',
            'ك',
            'ل',
            'م',
            'ن',
            'ه',
            'و',
            'ي',
        ]
        vowels = ['َ', 'ِ', 'ُ']

        for c in consonants:
            for v in vowels:
                mock_syllables.append()
                    {
                        'syllable': c + v,
                        'pattern': 'CV',
                        'consonants': [c],
                        'vowels': [v],
                        'weight': 'light',
                    }
                )

        # مقاطع CVC
        end_consonants = ['ن', 'ر', 'ل', 'م', 'ت', 'د', 'س', 'ك']
        for c1 in consonants[:8]:  # تحديد العدد
            for v in vowels:
                for c2 in end_consonants:
                    mock_syllables.append()
                        {
                            'syllable': c1 + v + c2,
                            'pattern': 'CVC',
                            'consonants': [c1, c2],
                            'vowels': [v],
                            'weight': 'heavy',
                        }
                    )

        # مقاطع VC للبداية
        for v in ['أَ', 'إِ', 'أُ']:
            for c in ['ل', 'ن', 'م', 'ت']:
                mock_syllables.append()
                    {
                        'syllable': v + c,
                        'pattern': 'VC',
                        'consonants': [c],
                        'vowels': [v[1:]],  # إزالة الهمزة من الحركة
                        'weight': 'medium',
                    }
                )

        return mock_syllables

    def _load_function_word_patterns(self):
        """تحميل أنماط حروف المعاني المعروفة"""

        self.function_word_patterns = {
            # حروف الجر - تميل لتكون قصيرة ومتحركة
            FunctionWordCategory.PREPOSITIONS: [
                FunctionWordPattern()
                    category=FunctionWordCategory.PREPOSITIONS,
                    syllable_patterns=['CV', 'V', 'CVC'],
                    phonetic_constraints={
                        'max_syllables': 2,
                        'preferred_initial': ['ف', 'ع', 'ب', 'ل', 'م', 'إ'],
                        'avoid_heavy_consonants': True,
                        'prefer_liquid_consonants': ['ل', 'ر', 'ن', 'م'],
                    },
                    semantic_features=['spatial', 'directional', 'locative'],
                    frequency_weight=1.0),
            ],
            # أدوات العطف - قصيرة جداً
            FunctionWordCategory.CONJUNCTIONS: [
                FunctionWordPattern()
                    category=FunctionWordCategory.CONJUNCTIONS,
                    syllable_patterns=['CV', 'V'],
                    phonetic_constraints={
                        'max_syllables': 1,
                        'preferred_initial': ['و', 'ف', 'ث', 'أ'],
                        'single_consonant_preferred': True,
                        'avoid_complex_clusters': True,
                    },
                    semantic_features=['connective', 'additive', 'sequential'],
                    frequency_weight=1.2),
            ],
            # الأدوات - متنوعة
            FunctionWordCategory.PARTICLES: [
                FunctionWordPattern()
                    category=FunctionWordCategory.PARTICLES,
                    syllable_patterns=['CV', 'CVC', 'CVCV'],
                    phonetic_constraints={
                        'max_syllables': 2,
                        'prefer_short_vowels': True,
                        'avoid_long_words': True,
                    },
                    semantic_features=['modal', 'aspectual', 'temporal'],
                    frequency_weight=0.9),
            ],
            # أدوات الاستفهام
            FunctionWordCategory.INTERROGATIVES: [
                FunctionWordPattern()
                    category=FunctionWordCategory.INTERROGATIVES,
                    syllable_patterns=['CV', 'CVC', 'CVCV'],
                    phonetic_constraints={
                        'max_syllables': 2,
                        'preferred_initial': ['ه', 'أ', 'م', 'ك'],
                        'interrogative_markers': True,
                    },
                    semantic_features=['question', 'wh word', 'polar'],
                    frequency_weight=0.8),
            ],
            # أدوات النفي
            FunctionWordCategory.NEGATIONS: [
                FunctionWordPattern()
                    category=FunctionWordCategory.NEGATIONS,
                    syllable_patterns=['CV', 'CVC'],
                    phonetic_constraints={
                        'max_syllables': 2,
                        'preferred_initial': ['ل', 'م', 'ن'],
                        'negative_markers': True,
                    },
                    semantic_features=['negative', 'denial', 'prohibition'],
                    frequency_weight=0.9),
            ],
            # أدوات التحديد
            FunctionWordCategory.DETERMINERS: [
                FunctionWordPattern()
                    category=FunctionWordCategory.DETERMINERS,
                    syllable_patterns=['V', 'CV', 'CVC'],
                    phonetic_constraints={
                        'max_syllables': 2,
                        'definite_markers': True,
                        'preferred_initial': ['أ', 'ال', 'ه'],
                    },
                    semantic_features=['definite', 'demonstrative', 'quantifier'],
                    frequency_weight=1.1),
            ],
        }

    def _load_authentic_function_words(self):
        """تحميل حروف المعاني الأصيلة للمقارنة والتشابه"""

        self.authentic_words = {
            FunctionWordCategory.PREPOSITIONS: [
                'في',
                'على',
                'إلى',
                'من',
                'عن',
                'لدى',
                'أمام',
                'خلف',
                'تحت',
                'فوق',
                'بين',
                'عند',
                'لدن',
                'منذ',
                'مذ',
                'حتى',
                'سوى',
                'خلا',
                'عدا',
                'حاشا',
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
            ],
            FunctionWordCategory.INTERROGATIVES: [
                'هل',
                'أ',
                'ما',
                'مَن',
                'متى',
                'أين',
                'كيف',
                'لماذا',
                'أي',
                'كم',
            ],
            FunctionWordCategory.NEGATIONS: [
                'لا',
                'ما',
                'لن',
                'لم',
                'ليس',
                'غير',
                'سوى',
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
            ],
        }

    def generate_function_words()
        self,
        category: FunctionWordCategory,
        count: int = 50,
        similarity_threshold: float = 0.3) -> List[GeneratedFunctionWord]:
        """توليد حروف المعاني لفئة محددة"""

        logger.info(f"بدء توليد {count} كلمة من فئة {category.value}")

        patterns = self.function_word_patterns.get(category, [])
        if not patterns:
            logger.warning(f"لا توجد أنماط محددة للفئة {category.value}")
            return []

        generated_words = []
        attempts = 0
        max_attempts = count * 10  # محاولات أكثر لضمان النوعية

        while len(generated_words) < count and attempts < max_attempts:
            attempts += 1

            # اختيار نمط عشوائي
            pattern = random.choice(patterns)

            # توليد كلمة مرشحة
            candidate = self._generate_candidate_word(pattern)

            if candidate:
                # تحليل صوتي
                phonetic_analysis = self.phonetic_analyzer.analyze_word(candidate)

                # فحص القيود
                if self._satisfies_constraints(candidate, pattern, phonetic_analysis):

                    # حساب التشابه مع الكلمات الأصيلة
                    similarity_score = self._calculate_authenticity_similarity()
                        candidate, category
                    )

                    if similarity_score >= similarity_threshold:

                        # إنشاء كائن الكلمة المولدة
                        generated_word = GeneratedFunctionWord()
                            word=candidate,
                            category=category,
                            pattern=self._extract_syllable_pattern(candidate),
                            syllable_breakdown=self._breakdown_syllables(candidate),
                            phonetic_features=phonetic_analysis,
                            similarity_score=similarity_score,
                            is_authentic=candidate
                            in self.authentic_words.get(category, []))

                        generated_words.append(generated_word)

                        if len(generated_words) % 10 == 0:
                            logger.info(f"تم توليد {len(generated_words)} كلمة")

        # ترتيب حسب نقاط التشابه
        generated_words.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"تم توليد {len(generated_words)} كلمة بنجاح من أصل {count} مطلوبة")

        return generated_words

    def _generate_candidate_word(self, pattern: FunctionWordPattern) -> Optional[str]:
        """توليد كلمة مرشحة بناءً على النمط"""

        syllable_patterns = pattern.syllable_patterns
        max_syllables = pattern.phonetic_constraints.get('max_syllables', 2)

        # اختيار عدد المقاطع
        num_syllables = random.randint(1, min(max_syllables, 2))

        # توليد المقاطع
        word_syllables = []
        for i in range(num_syllables):
            # اختيار نمط مقطع مناسب
            syllable_pattern = random.choice(syllable_patterns)

            # العثور على مقاطع تتطابق مع النمط
            matching_syllables = [
                syl
                for syl in self.syllables_db
                if syl.get('pattern') == syllable_pattern
            ]

            if matching_syllables:
                chosen_syllable = random.choice(matching_syllables)
                word_syllables.append(chosen_syllable['syllable'])

        if word_syllables:
            candidate = ''.join(word_syllables)
            return self._apply_phonetic_adjustments(candidate, pattern)

        return None

    def _apply_phonetic_adjustments()
        self, word: str, pattern: FunctionWordPattern
    ) -> str:
        """تطبيق تعديلات صوتية على الكلمة المولدة"""

        constraints = pattern.phonetic_constraints

        # تعديل الحرف الأول إذا كان مفضلاً
        preferred_initial = constraints.get('preferred_initial', [])
        if preferred_initial and word and word[0] not in preferred_initial:
            # استبدال الحرف الأول
            word = random.choice(preferred_initial) + word[1:]

        # إزالة التكرارات المتتالية
        word = re.sub(r'(.)\1+', r'\1', word)

        # تطبيق قواعد التشكيل البسيطة
        word = self._apply_simple_vocalization(word)

        return word

    def _apply_simple_vocalization(self, word: str) -> str:
        """تطبيق تشكيل بسيط ومناسب"""

        if len(word) == 1:
            # حرف واحد - إضافة حركة مناسبة
            return word + random.choice(['َ', 'ِ', 'ُ'])
        elif len(word) == 2:
            # حرفان - تشكيل متوازن
            return word[0] + random.choice(['َ', 'ِ']) + word[1]
        else:
            # أكثر من حرفين - تشكيل انتقائي
            result = word[0] + random.choice(['َ', 'ِ'])
            for i in range(1, len(word)):
                result += word[i]
                if i < len(word) - 1 and random.random() < 0.3:
                    result += random.choice(['َ', 'ِ', 'ْ'])
            return result

    def _satisfies_constraints()
        self, word: str, pattern: FunctionWordPattern, analysis: Dict[str, Any]
    ) -> bool:
        """فحص ما إذا كانت الكلمة تحقق القيود المطلوبة"""

        constraints = pattern.phonetic_constraints

        # فحص طول الكلمة
        max_syllables = constraints.get('max_syllables', 3)
        if analysis['syllable_count'] > max_syllables:
            return False

        # فحص الأنماط الصعبة
        if analysis['is_difficult']:
            return False

        # فحص الوزن الصوتي
        if analysis['phonetic_weight'] > 2.0:  # وزن ثقيل جداً
            return False

        # فحص تجنب الصوامت الثقيلة
        if constraints.get('avoid_heavy_consonants', False):
            heavy_consonants = {'ق', 'ط', 'ص', 'ض', 'ظ'}
            if any(c in word for c in heavy_consonants):
                return False

        # فحص تفضيل الصوامت السائلة
        if constraints.get('prefer_liquid_consonants', False):
            liquid_consonants = {'ل', 'ر', 'ن', 'م'}
            consonant_count = analysis['consonant_count']
            liquid_count = len([c for c in word if c in liquid_consonants])
            if consonant_count > 0 and liquid_count / consonant_count < 0.5:
                return False

        return True

    def _calculate_authenticity_similarity()
        self, word: str, category: FunctionWordCategory
    ) -> float:
        """حساب التشابه مع الكلمات الأصيلة"""

        authentic_words = self.authentic_words.get(category, [])
        if not authentic_words:
            return 0.5  # متوسط افتراضي

        # حساب أقصى تشابه مع أي كلمة أصيلة
        max_similarity = 0.0

        for auth_word in authentic_words:
            similarity = self.phonetic_analyzer.calculate_similarity(word, auth_word)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _extract_syllable_pattern(self, word: str) -> str:
        """استخراج نمط المقاطع من الكلمة"""

        # تحليل مبسط لاستخراج نمط CV
        pattern = ""
        for char in word:
            if char in self.phonetic_analyzer.consonants:
                pattern += "C"
            elif char in self.phonetic_analyzer.vowels['short']:
                pattern += "V"
            elif char in self.phonetic_analyzer.vowels['long']:
                pattern += "VV"

        return pattern

    def _breakdown_syllables(self, word: str) -> List[str]:
        """تقسيم الكلمة إلى مقاطع"""

        # تقسيم مبسط بناءً على أنماط CV
        syllables = []
        current_syllable = ""

        for i, char in enumerate(word):
            current_syllable += char

            # إذا كان الحرف التالي صامت والحالي صائت، انهِ المقطع
            if ()
                i < len(word) - 1
                and char in self.phonetic_analyzer.vowels['short']
                and word[i + 1] in self.phonetic_analyzer.consonants
            ):

                syllables.append(current_syllable)
                current_syllable = ""

        # إضافة المقطع الأخير
        if current_syllable:
            syllables.append(current_syllable)

        return syllables if syllables else [word]

    def generate_comprehensive_report()
        self, results: Dict[FunctionWordCategory, List[GeneratedFunctionWord]]
    ) -> str:
        """إنتاج تقرير شامل عن النتائج"""

        report = []
        report.append("═══════════════════════════════════════════════════════════════")
        report.append("تقرير شامل لتوليد حروف المعاني العربية")
        report.append("═══════════════════════════════════════════════════════════════")
        report.append("")

        total_generated = sum(len(words) for words in results.values())
        report.append(f"إجمالي الكلمات المولدة: {total_generated}")
        report.append("")

        for category, words in results.items():
            if not words:
                continue

            report.append(f"▶ {category.value:}")
            report.append(f"  عدد الكلمات: {len(words)}")

            # أفضل 5 كلمات
            top_words = sorted(words, key=lambda x: x.similarity_score, reverse=True)[
                :5
            ]
            report.append("  أفضل الكلمات المولدة:")

            for i, word in enumerate(top_words, 1):
                authenticity = ()
                    "✓ أصيلة"
                    if word.is_authentic
                    else f"تشابه: {word.similarity_score:.2f}"
                )
                report.append(f"    {i}. {word.word} - {authenticity}")
                report.append(f"       المقاطع: {'} + '.join(word.syllable_breakdown)}")

            report.append("")

        return "\n".join(report)


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMO AND TESTING FUNCTIONS - وظائف العرض والاختبار
# ═══════════════════════════════════════════════════════════════════════════════════


def demo_function_words_generation():
    """عرض توضيحي لتوليد حروف المعاني"""

    print("🔵 مولد حروف المعاني العربية المحسن")
    print("=" * 50)

    # إنشاء المولد
    generator = EnhancedArabicFunctionWordsGenerator()

    # توليد كلمات لكل فئة
    categories_to_test = [
        FunctionWordCategory.PREPOSITIONS,
        FunctionWordCategory.CONJUNCTIONS,
        FunctionWordCategory.PARTICLES,
        FunctionWordCategory.INTERROGATIVES,
        FunctionWordCategory.NEGATIONS,
        FunctionWordCategory.DETERMINERS,
    ]

    all_results = {}

    for category in categories_to_test:
        print(f"\n🔸 توليد {category.value}...")
        results = generator.generate_function_words(category, count=20)
        all_results[category] = results

        if results:
            print(f"تم توليد {len(results)} كلمة بنجاح")

            # عرض أفضل 3 كلمات
            top_3 = results[:3]
            for i, word in enumerate(top_3, 1):
                auth_mark = ()
                    "✓" if word.is_authentic else f"({word.similarity_score:.2f})"
                )
                print(f"  {i}. {word.word {auth_mark}}")
        else:
            print("لم يتم توليد كلمات")

    # تقرير شامل
    print("\n" + "=" * 60)
    print(generator.generate_comprehensive_report(all_results))

    return all_results


if __name__ == "__main__":
    # تشغيل العرض التوضيحي للنظام المحسن
    demo_function_words_generation()

