#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام توليد الأرقام والمفاهيم الرياضية العربية المتقدم
Advanced Arabic Mathematical Numbers and Concepts Generator

يستخدم قاعدة بيانات المقاطع الصوتية (22,218 مقطع) لتوليد:
- الأعداد الأساسية والترتيبية (واحد، أول، ثاني...)
- الكسور البسيطة والمركبة (نصف، ثلاثة أرباع...)
- العمليات الرياضية (جمع، طرح، ضرب، قسمة...)
- المفاهيم الرياضية (معادلة، متغير، مساحة...)
- مع مراعاة الخصائص الصوتية والدلالية والقواعد النحوية

المطور: نظام الذكاء الاصطناعي العربي
Developer: Arabic AI System

التاريخ: 2025
Date: 2025
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from fractions import Fraction

# Configure logging
logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL CONCEPTS CLASSIFICATION SYSTEM - نظام تصنيف المفاهيم الرياضية
# ═══════════════════════════════════════════════════════════════════════════════════


class MathConceptCategory(Enum):
    """فئات المفاهيم الرياضية"""

    NUMBER_CARDINAL = "number_cardinal"  # الأعداد الأساسية
    NUMBER_ORDINAL = "number_ordinal"  # الأعداد الترتيبية
    FRACTION_SIMPLE = "fraction_simple"  # الكسور البسيطة
    FRACTION_COMPOUND = "fraction_compound"  # الكسور المركبة
    OPERATION_BASIC = "operation_basic"  # العمليات الأساسية
    OPERATION_ADVANCED = "operation_advanced"  # العمليات المتقدمة
    CONCEPT_ARITHMETIC = "concept_arithmetic"  # المفاهيم الحسابية
    CONCEPT_ALGEBRA = "concept_algebra"  # المفاهيم الجبرية
    CONCEPT_GEOMETRY = "concept_geometry"  # المفاهيم الهندسية
    CONCEPT_STATISTICS = "concept_statistics"  # المفاهيم الإحصائية


class NumberGender(Enum):
    """جنس العدد في العربية"""

    MASCULINE = "masculine"  # مذكر
    FEMININE = "feminine"  # مؤنث


class MathPattern(Enum):
    """أنماط المفاهيم الرياضية الصوتية"""

    CV = "CV"  # صامت + صائت
    CVC = "CVC"  # صامت + صائت + صامت
    CVCV = "CVCV"  # واحد، ثلث
    CVCVC = "CVCVC"  # خمسة، ضرب
    CVVCV = "CVVCV"  # ثاني، عاشر
    CVCVCV = "CVCVCV"  # ثلاثة، قسمة
    CVCCVC = "CVCCVC"  # عشرة، جذور
    CVVCVC = "CVVCVC"  # أربعة، نسبة


@dataclass
class MathTemplate:
    """قالب المفاهيم الرياضية"""

    category: MathConceptCategory
    pattern: MathPattern
    syllable_structure: List[str]
    phonetic_constraints: Dict[str, Any]
    semantic_features: List[str]
    frequency: float = 1.0
    gender_agreement: bool = False
    numerical_range: Optional[Tuple[int, int]] = None


@dataclass
class GeneratedMathConcept:
    """مفهوم رياضي مولد"""

    term: str
    category: MathConceptCategory
    pattern: MathPattern
    syllables: List[str]
    phonetic_analysis: Dict[str, Any]
    semantic_meaning: str
    mathematical_value: Optional[Union[int, float, Fraction, str]] = None
    gender: Optional[NumberGender] = None
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    authenticity_score: float = 0.0
    examples: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC MATHEMATICAL LINGUISTICS ANALYZER - محلل اللسانيات الرياضية العربية
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicMathLinguistics:
    """محلل اللسانيات الرياضية العربية المتقدم"""

    def __init__(self):

        # تحميل قواعد الرياضيات العربية
        self._load_number_systems()
        self._load_fraction_patterns()
        self._load_operation_roots()
        self._load_concept_taxonomy()
        self._load_linguistic_rules()

    def _load_number_systems(self):
        """تحميل أنظمة الأعداد العربية"""

        # الأعداد الأساسية 0 19
        self.cardinal_numbers = {
            0: "صفر",
            1: {"m": "واحد", "f": "واحدة"},
            2: {"m": "اثنان", "f": "اثنتان"},
            3: {"m": "ثلاثة", "f": "ثلاث"},
            4: {"m": "أربعة", "f": "أربع"},
            5: {"m": "خمسة", "f": "خمس"},
            6: {"m": "ستة", "f": "ست"},
            7: {"m": "سبعة", "f": "سبع"},
            8: {"m": "ثمانية", "f": "ثمان"},
            9: {"m": "تسعة", "f": "تسع"},
            10: {"m": "عشرة", "f": "عشر"},
            11: {"m": "أحد عشر", "f": "إحدى عشرة"},
            12: {"m": "اثنا عشر", "f": "اثنتا عشرة"},
            13: {"m": "ثلاثة عشر", "f": "ثلاث عشرة"},
            14: {"m": "أربعة عشر", "f": "أربع عشرة"},
            15: {"m": "خمسة عشر", "f": "خمس عشرة"},
            16: {"m": "ستة عشر", "f": "ست عشرة"},
            17: {"m": "سبعة عشر", "f": "سبع عشرة"},
            18: {"m": "ثمانية عشر", "f": "ثمان عشرة"},
            19: {"m": "تسعة عشر", "f": "تسع عشرة"},
        }

        # العقود
        self.tens = {
            20: "عشرون",
            30: "ثلاثون",
            40: "أربعون",
            50: "خمسون",
            60: "ستون",
            70: "سبعون",
            80: "ثمانون",
            90: "تسعون",
        }

        # المئات
        self.hundreds = {
            100: "مائة",
            200: "مئتان",
            300: "ثلاثمائة",
            400: "أربعمائة",
            500: "خمسمائة",
            600: "ستمائة",
            700: "سبعمائة",
            800: "ثمانمائة",
            900: "تسعمائة",
        }

        # الأعداد الترتيبية
        self.ordinal_numbers = {
            1: {"m": "أول", "f": "أولى"},
            2: {"m": "ثاني", "f": "ثانية"},
            3: {"m": "ثالث", "f": "ثالثة"},
            4: {"m": "رابع", "f": "رابعة"},
            5: {"m": "خامس", "f": "خامسة"},
            6: {"m": "سادس", "f": "سادسة"},
            7: {"m": "سابع", "f": "سابعة"},
            8: {"m": "ثامن", "f": "ثامنة"},
            9: {"m": "تاسع", "f": "تاسعة"},
            10: {"m": "عاشر", "f": "عاشرة"},
        }

    def _load_fraction_patterns(self):
        """تحميل أنماط الكسور العربية"""

        # الكسور البسيطة (البسط = 1)
        self.simple_fractions = {
            2: "نصف",
            3: "ثلث",
            4: "ربع",
            5: "خمس",
            6: "سدس",
            7: "سبع",
            8: "ثمن",
            9: "تسع",
            10: "عشر",
        }

        # جموع الكسور للبسط > 1
        self.fraction_plurals = {
            2: "أنصاف",
            3: "أثلاث",
            4: "أرباع",
            5: "أخماس",
            6: "أسداس",
            7: "أسباع",
            8: "أثمان",
            9: "أتساع",
            10: "أعشار",
        }

        # أنماط الكسور المركبة
        self.compound_fraction_patterns = {
            "numerator_denominator": "{numerator} {denominator_plural}",
            "mixed_number": "{whole} و {numerator} {denominator_plural}",
            "decimal": "{whole} فاصلة {decimal_part}",
        }

    def _load_operation_roots(self):
        """تحميل جذور العمليات الرياضية"""

        self.operation_roots = {
            # العمليات الأساسية
            'addition': {
                'root': 'جمع',
                'derivatives': ['جمع', 'مجموع', 'إضافة', 'زيادة'],
            },
            'subtraction': {
                'root': 'طرح',
                'derivatives': ['طرح', 'نقص', 'تناقص', 'فرق'],
            },
            'multiplication': {
                'root': 'ضرب',
                'derivatives': ['ضرب', 'حاصل ضرب', 'مضاعفة'],
            },
            'division': {
                'root': 'قسم',
                'derivatives': ['قسمة', 'تقسيم', 'حاصل القسمة'],
            },
            # العمليات المتقدمة
            'power': {'root': 'أس', 'derivatives': ['أس', 'رفع للأس', 'قوة']},
            'root': {'root': 'جذر', 'derivatives': ['جذر', 'جذر تربيعي', 'جذر تكعيبي']},
            'logarithm': {'root': 'لوغ', 'derivatives': ['لوغاريتم', 'لوغاريتم طبيعي']},
            'factorial': {'root': 'مضر', 'derivatives': ['مضروب', 'عاملي']},
            # النسب والتناسب
            'ratio': {'root': 'نسب', 'derivatives': ['نسبة', 'تناسب', 'معدل']},
            'proportion': {'root': 'تناسب', 'derivatives': ['تناسب', 'تناسبية']},
            'percentage': {'root': 'مئو', 'derivatives': ['نسبة مئوية', 'بالمائة']},
        }

    def _load_concept_taxonomy(self):
        """تحميل تصنيف المفاهيم الرياضية"""

        self.concept_taxonomy = {
            # الحساب
            'arithmetic': {
                'numbers': ['عدد', 'أرقام', 'أعداد طبيعية', 'أعداد صحيحة'],
                'operations': ['عملية', 'حساب', 'احتساب'],
                'properties': ['خاصية', 'قانون', 'قاعدة'],
            },
            # الجبر
            'algebra': {
                'variables': ['متغير', 'مجهول', 'متحول'],
                'equations': ['معادلة', 'مساواة', 'نظام معادلات'],
                'functions': ['دالة', 'اقتران', 'تابع'],
                'polynomials': ['كثير حدود', 'حدة', 'معامل'],
            },
            # الهندسة
            'geometry': {
                'shapes': ['شكل', 'مضلع', 'دائرة', 'مثلث', 'مربع'],
                'measurements': ['مساحة', 'محيط', 'حجم', 'قطر', 'نصف قطر'],
                'angles': ['زاوية', 'زاوية قائمة', 'زاوية حادة', 'زاوية منفرجة'],
                'lines': ['خط', 'مستقيم', 'منحنى', 'متوازي', 'عمودي'],
            },
            # الإحصاء والاحتمالات
            'statistics': {
                'measures': ['متوسط', 'وسيط', 'منوال', 'مدى'],
                'probability': ['احتمال', 'احتمالية', 'توقع'],
                'distributions': ['توزيع', 'توزيع طبيعي', 'انحراف معياري'],
                'sampling': ['عينة', 'مجتمع', 'معاينة'],
            },
        }

    def _load_linguistic_rules(self):
        """تحميل القواعد اللسانية"""

        self.linguistic_rules = {
            # قواعد التذكير والتأنيث للأعداد
            'gender_agreement': {
                'numbers_1_2': 'agree_with_counted',  # يوافق المعدود
                'numbers_3_10': 'opposite_to_counted',  # يخالف المعدود
                'numbers_11_99': 'masculine_always',  # مذكر دائماً
                'numbers_100_plus': 'agree_with_counted',  # يوافق المعدود
            },
            # قواعد الإعراب
            'case_marking': {
                'subject': 'nominative',  # رفع
                'object': 'accusative',  # نصب
                'possession': 'genitive',  # جر
            },
            # قواعد الجمع
            'pluralization': {
                'sound_masculine': 'ون/ين',
                'sound_feminine': 'ات',
                'broken_plural': 'various_patterns',
            },
        }

    def convert_number_to_arabic()
        self, number: int, gender: NumberGender = NumberGender.MASCULINE
    ) -> str:
        """تحويل رقم إلى نص عربي"""

        if number == 0:
            return self.cardinal_numbers[0]

        gender_key = 'm' if gender == NumberGender.MASCULINE else 'f'

        # الأعداد البسيطة 1-19
        if 1 <= number <= 19:
            if isinstance(self.cardinal_numbers[number], dict):
                return self.cardinal_numbers[number][gender_key]
            else:
                return self.cardinal_numbers[number]

        # العقود 20-99
        if 20 <= number <= 99:
            tens_part = (number // 10) * 10
            units_part = number % 10

            if units_part == 0:
                return self.tens[tens_part]
            else:
                units_text = self.cardinal_numbers[units_part]
                if isinstance(units_text, dict):
                    units_text = units_text[gender_key]
                return f"{units_text و{self.tens[tens_part]}}"

        # المئات 100 999
        if 100 <= number <= 999:
            hundreds_part = (number // 100) * 100
            remainder = number % 100

            hundreds_text = self.hundreds[hundreds_part]

            if remainder == 0:
                return hundreds_text
            else:
                remainder_text = self.convert_number_to_arabic(remainder, gender)
                return f"{hundreds_text و{remainder_text}}"

        # الآلاف 1000+
        if number >= 1000:
            thousands = number // 1000
            remainder = number % 1000

            if thousands == 1:
                thousands_text = "ألف"
            elif thousands == 2:
                thousands_text = "ألفان"
            elif 3 <= thousands <= 10:
                thousands_text = f"{self.convert_number_to_arabic(thousands)} آلاف"
            else:
                thousands_text = f"{self.convert_number_to_arabic(thousands)} ألف"

            if remainder == 0:
                return thousands_text
            else:
                remainder_text = self.convert_number_to_arabic(remainder, gender)
                return f"{thousands_text و{remainder_text}}"

        return str(number)  # fallback

    def convert_ordinal_to_arabic()
        self, number: int, gender: NumberGender = NumberGender.MASCULINE
    ) -> str:
        """تحويل رقم ترتيبي إلى نص عربي"""

        gender_key = 'm' if gender == NumberGender.MASCULINE else 'f'

        if 1 <= number <= 10:
            return self.ordinal_numbers[number][gender_key]

        # للأعداد الأكبر، نكون الترتيبي من الأساسي
        base_text = self.convert_number_to_arabic(number, gender)

        # إضافة اللاحقة الترتيبية
        if gender == NumberGender.MASCULINE:
            return f"{base_text}ال"
        else:
            return f"{base_text}ة"

    def convert_fraction_to_arabic(self, numerator: int, denominator: int) -> str:
        """تحويل كسر إلى نص عربي"""

        if numerator == 1 and denominator in self.simple_fractions:
            return self.simple_fractions[denominator]

        if denominator in self.fraction_plurals:
            numerator_text = self.convert_number_to_arabic()
                numerator, NumberGender.MASCULINE
            )
            denominator_text = self.fraction_plurals[denominator]
            return f"{numerator_text} {denominator_text}"

        # للمقامات الأكبر، نستخدم صيغة "أجزاء من"
        numerator_text = self.convert_number_to_arabic()
            numerator, NumberGender.MASCULINE
        )
        denominator_text = self.convert_number_to_arabic()
            denominator, NumberGender.MASCULINE
        )

        if numerator == 1:
            return f"جزء من {denominator_text}"
        else:
            return f"{numerator_text أجزاء} من {denominator_text}}"

    def analyze_phonetic_structure(self, term: str) -> Dict[str, Any]:
        """تحليل البنية الصوتية للمصطلح الرياضي"""

        analysis = {
            'length': len(term),
            'syllable_count': self._count_syllables(term),
            'stress_pattern': self._identify_stress_pattern(term),
            'consonant_clusters': self._find_consonant_clusters(term),
            'vowel_pattern': self._extract_vowel_pattern(term),
            'phonetic_difficulty': self._assess_phonetic_difficulty(term),
            'euphony_score': self._calculate_euphony_score(term),
            'mathematical_appropriateness': self._assess_math_appropriateness(term),
        }

        return analysis

    def _count_syllables(self, term: str) -> int:
        """عد المقاطع الصوتية"""
        vowels = ['َ', 'ِ', 'ُ', 'ا', 'و', 'ي']
        return max(1, len([c for c in term if c in vowels]))

    def _identify_stress_pattern(self, term: str) -> str:
        """تحديد نمط النبرة"""
        syllable_count = self._count_syllables(term)

        if syllable_count <= 2:
            return 'ultimate'
        elif syllable_count == 3:
            return 'penultimate'
        else:
            return 'antepenultimate'

    def _find_consonant_clusters(self, term: str) -> List[str]:
        """العثور على تجمعات الصوامت"""
        clusters = []
        consonants = 'بتثجحخدذرزسشصضطظعغفقكلمنهويء'

        for i in range(len(term) - 1):
            if term[i] in consonants and term[i + 1] in consonants:
                clusters.append(term[i : i + 2])

        return clusters

    def _extract_vowel_pattern(self, term: str) -> str:
        """استخراج نمط الحركات"""
        vowels = {'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ا': 'aa', 'ي': 'ii', 'و': 'uu'}
        pattern = []

        for char in term:
            if char in vowels:
                pattern.append(vowels[char])

        return ' '.join(pattern)

    def _assess_phonetic_difficulty(self, term: str) -> float:
        """تقييم صعوبة النطق"""
        difficulty = 0.0

        # فحص التجمعات الصعبة
        clusters = self._find_consonant_clusters(term)
        difficult_clusters = ['قف', 'طع', 'حخ', 'خح']

        for cluster in clusters:
            if cluster in difficult_clusters:
                difficulty += 0.3

        # فحص الطول
        if len(term) > 8:
            difficulty += 0.2

        return min(1.0, difficulty)

    def _calculate_euphony_score(self, term: str) -> float:
        """حساب نقاط جمال الصوت"""
        euphony = 1.0

        # تنوع الأصوات
        unique_sounds = len(set(term))
        variety_bonus = min(0.3, unique_sounds / len(term) * 0.5)
        euphony += variety_bonus

        # توازن الصوامت والصوائت
        consonants = len([c for c in term if c in 'بتثجحخدذرزسشصضطظعغفقكلمنهويء'])
        vowels = len([c for c in term if c in 'َُِاويً'])

        if vowels > 0:
            balance = 1 - abs(consonants - vowels) / max(consonants, vowels)
            euphony += balance * 0.2

        # خصم الصعوبة
        difficulty = self._assess_phonetic_difficulty(term)
        euphony -= difficulty * 0.3

        return max(0.0, min(2.0, euphony))

    def _assess_math_appropriateness(self, term: str) -> float:
        """تقييم مناسبة المصطلح للرياضيات"""
        score = 0.5

        # فحص وجود جذور رياضية معروفة
        math_roots = ['عدد', 'حسب', 'قسم', 'ضرب', 'جمع', 'طرح', 'مساح', 'قيس']
        for root in math_roots:
            if root in term:
                score += 0.2
                break

        # فحص النهايات المناسبة للرياضيات
        math_endings = ['ة', 'ال', 'ي', 'ية']
        for ending in math_endings:
            if term.endswith(ending):
                score += 0.1
                break

        # فحص عدم وجود دلالات سلبية
        negative_connotations = ['موت', 'حرب', 'ضرر', 'فقر']
        for neg in negative_connotations:
            if neg in term:
                score -= 0.3

        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════════════
# ADVANCED MATHEMATICAL CONCEPTS GENERATOR - مولد المفاهيم الرياضية المتقدم
# ═══════════════════════════════════════════════════════════════════════════════════


class AdvancedArabicMathGenerator:
    """مولد المفاهيم الرياضية العربية المتقدم"""

    def __init__(self, syllables_database: Optional[List[Dict]] = None):

        self.syllables_db = syllables_database or self._load_syllables_database()
        self.linguistics = ArabicMathLinguistics()

        # تحميل قوالب المفاهيم الرياضية
        self._load_math_templates()

        logger.info(f"تم تحميل {len(self.syllables_db)} مقطع صوتي")
        logger.info(f"تم تحميل {len(self.math_templates)} قالب رياضي")

    def _load_syllables_database(self) -> List[Dict]:
        """تحميل قاعدة بيانات المقاطع"""

        try:
            # محاولة تحميل من النظام الشامل
            from comprehensive_arabic_verb_syllable_generator import ()
                ComprehensiveArabicVerbSyllableGenerator)

            syllable_generator = ComprehensiveArabicVerbSyllableGenerator()
            logger.info("تحميل قاعدة بيانات المقاطع من النظام الشامل...")

            syllable_database = ()
                syllable_generator.generate_comprehensive_syllable_database()
            )

            logger.info(f"تم تحميل {len(syllable_database)} مقطع من النظام الشامل")
            return syllable_database

        except ImportError:
            logger.warning("النظام الشامل غير متاح، استخدام قاعدة بيانات محسنة")
            return self._create_enhanced_math_syllable_database()

    def _create_enhanced_math_syllable_database(self) -> List[Dict]:
        """إنشاء قاعدة بيانات مقاطع محسنة للرياضيات"""

        syllables = []

        # الأحرف العربية مع تصنيفها
        consonants = {
            'common': [
                'ب',
                'ت',
                'ج',
                'د',
                'ر',
                'س',
                'ع',
                'ف',
                'ك',
                'ل',
                'م',
                'ن',
                'ه',
                'و',
                'ي',
            ],
            'emphatic': ['ص', 'ض', 'ط', 'ظ'],
            'mathematical': [
                'ح',
                'خ',
                'ذ',
                'ز',
                'ش',
                'غ',
                'ق',
            ],  # أحرف شائعة في الرياضيات
        }

        # الحركات
        vowels = {'short': ['َ', 'ِ', 'ُ'], 'long': ['ا', 'ي', 'و']}

        # توليد مقاطع للأعداد
        number_syllables = [
            # مقاطع الأعداد الأساسية
            {'syllable': 'واح', 'pattern': 'CVC', 'math_type': 'number', 'value': 1},
            {'syllable': 'اث', 'pattern': 'VC', 'math_type': 'number', 'value': 2},
            {'syllable': 'ثلا', 'pattern': 'CCV', 'math_type': 'number', 'value': 3},
            {'syllable': 'أر', 'pattern': 'VC', 'math_type': 'number', 'value': 4},
            {'syllable': 'خم', 'pattern': 'CVC', 'math_type': 'number', 'value': 5},
            {'syllable': 'ست', 'pattern': 'CVC', 'math_type': 'number', 'value': 6},
            {'syllable': 'سب', 'pattern': 'CVC', 'math_type': 'number', 'value': 7},
            {'syllable': 'ثما', 'pattern': 'CCV', 'math_type': 'number', 'value': 8},
            {'syllable': 'تس', 'pattern': 'CVC', 'math_type': 'number', 'value': 9},
            {'syllable': 'عش', 'pattern': 'CVC', 'math_type': 'number', 'value': 10},
            # مقاطع الكسور
            {
                'syllable': 'نص',
                'pattern': 'CVC',
                'math_type': 'fraction',
                'value': (1, 2),
            },
            {
                'syllable': 'ثل',
                'pattern': 'CVC',
                'math_type': 'fraction',
                'value': (1, 3),
            },
            {
                'syllable': 'رب',
                'pattern': 'CVC',
                'math_type': 'fraction',
                'value': (1, 4),
            },
            {
                'syllable': 'خم',
                'pattern': 'CVC',
                'math_type': 'fraction',
                'value': (1, 5),
            },
            # مقاطع العمليات
            {
                'syllable': 'جم',
                'pattern': 'CVC',
                'math_type': 'operation',
                'operation': 'addition',
            },
            {
                'syllable': 'طر',
                'pattern': 'CVC',
                'math_type': 'operation',
                'operation': 'subtraction',
            },
            {
                'syllable': 'ضر',
                'pattern': 'CVC',
                'math_type': 'operation',
                'operation': 'multiplication',
            },
            {
                'syllable': 'قس',
                'pattern': 'CVC',
                'math_type': 'operation',
                'operation': 'division',
            },
        ]

        syllables.extend(number_syllables)

        # توليد مقاطع عامة محسنة للرياضيات
        for category, cons_list in consonants.items():
            for consonant in cons_list[:10]:
                for vowel in vowels['short']:
                    syllables.append()
                        {
                            'syllable': consonant + vowel,
                            'pattern': 'CV',
                            'consonants': [consonant],
                            'vowels': [vowel],
                            'consonant_type': category,
                            'weight': 'light',
                            'math_suitable': True,
                            'frequency': 0.8 if category == 'common' else 0.6,
                        }
                    )

        # مقاطع خاصة للمفاهيم الرياضية
        math_specific_syllables = [
            # المفاهيم الحسابية
            {'syllable': 'عد', 'pattern': 'CVC', 'concept': 'number'},
            {'syllable': 'حس', 'pattern': 'CVC', 'concept': 'calculation'},
            {'syllable': 'قي', 'pattern': 'CV', 'concept': 'measurement'},
            # المفاهيم الجبرية
            {'syllable': 'معا', 'pattern': 'CCV', 'concept': 'equation'},
            {'syllable': 'متغ', 'pattern': 'CVC', 'concept': 'variable'},
            {'syllable': 'دا', 'pattern': 'CV', 'concept': 'function'},
            # المفاهيم الهندسية
            {'syllable': 'مسا', 'pattern': 'CCV', 'concept': 'area'},
            {'syllable': 'محي', 'pattern': 'CCV', 'concept': 'perimeter'},
            {'syllable': 'حج', 'pattern': 'CVC', 'concept': 'volume'},
            # المفاهيم الإحصائية
            {'syllable': 'متو', 'pattern': 'CCV', 'concept': 'average'},
            {'syllable': 'احت', 'pattern': 'VCC', 'concept': 'probability'},
            {'syllable': 'توز', 'pattern': 'CVC', 'concept': 'distribution'},
        ]

        for special in math_specific_syllables:
            special.update()
                {
                    'weight': 'medium',
                    'frequency': 1.0,
                    'math_suitable': True,
                    'is_authentic': True,
                }
            )
            syllables.append(special)

        logger.info(f"تم إنشاء {len(syllables)} مقطع صوتي محسن للرياضيات")
        return syllables

    def _load_math_templates(self):
        """تحميل قوالب المفاهيم الرياضية"""

        self.math_templates = [
            # قوالب الأعداد الأساسية
            MathTemplate()
                category=MathConceptCategory.NUMBER_CARDINAL,
                pattern=MathPattern.CVCV,
                syllable_structure=['CVC', 'V'],
                phonetic_constraints={
                    'stress': 'ultimate',
                    'length': 'short_to_medium',
                    'difficulty': 'low',
                },
                semantic_features=['countable', 'quantitative', 'basic'],
                frequency=1.0,
                gender_agreement=True,
                numerical_range=(1, 1000)),
            # قوالب الأعداد الترتيبية
            MathTemplate()
                category=MathConceptCategory.NUMBER_ORDINAL,
                pattern=MathPattern.CVVCV,
                syllable_structure=['CVV', 'CV'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium',
                    'difficulty': 'low',
                },
                semantic_features=['ordinal', 'sequential', 'positional'],
                frequency=0.8,
                gender_agreement=True,
                numerical_range=(1, 100)),
            # قوالب الكسور البسيطة
            MathTemplate()
                category=MathConceptCategory.FRACTION_SIMPLE,
                pattern=MathPattern.CVC,
                syllable_structure=['CVC'],
                phonetic_constraints={
                    'stress': 'ultimate',
                    'length': 'short',
                    'difficulty': 'low',
                },
                semantic_features=['fractional', 'unit', 'part_of_whole'],
                frequency=1.0,
                numerical_range=(2, 10)),
            # قوالب الكسور المركبة
            MathTemplate()
                category=MathConceptCategory.FRACTION_COMPOUND,
                pattern=MathPattern.CVCVCV,
                syllable_structure=['CV', 'CV', 'CV'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium_to_long',
                    'difficulty': 'medium',
                },
                semantic_features=['fractional', 'compound', 'complex'],
                frequency=0.6),
            # قوالب العمليات الأساسية
            MathTemplate()
                category=MathConceptCategory.OPERATION_BASIC,
                pattern=MathPattern.CVC,
                syllable_structure=['CVC'],
                phonetic_constraints={
                    'stress': 'ultimate',
                    'length': 'short',
                    'difficulty': 'low',
                },
                semantic_features=['operational', 'active', 'transformative'],
                frequency=1.0),
            # قوالب العمليات المتقدمة
            MathTemplate()
                category=MathConceptCategory.OPERATION_ADVANCED,
                pattern=MathPattern.CVVCVC,
                syllable_structure=['CVV', 'CVC'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium',
                    'difficulty': 'medium',
                },
                semantic_features=['operational', 'complex', 'mathematical'],
                frequency=0.7),
            # قوالب المفاهيم الحسابية
            MathTemplate()
                category=MathConceptCategory.CONCEPT_ARITHMETIC,
                pattern=MathPattern.CVCV,
                syllable_structure=['CV', 'CV'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium',
                    'difficulty': 'low',
                },
                semantic_features=['conceptual', 'arithmetic', 'basic'],
                frequency=0.8),
            # قوالب المفاهيم الجبرية
            MathTemplate()
                category=MathConceptCategory.CONCEPT_ALGEBRA,
                pattern=MathPattern.CVCCVC,
                syllable_structure=['CVC', 'CVC'],
                phonetic_constraints={
                    'stress': 'ultimate',
                    'length': 'medium',
                    'difficulty': 'medium',
                },
                semantic_features=['algebraic', 'abstract', 'formal'],
                frequency=0.6),
            # قوالب المفاهيم الهندسية
            MathTemplate()
                category=MathConceptCategory.CONCEPT_GEOMETRY,
                pattern=MathPattern.CVCVCV,
                syllable_structure=['CV', 'CV', 'CV'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium_to_long',
                    'difficulty': 'low',
                },
                semantic_features=['geometric', 'spatial', 'visual'],
                frequency=0.7),
            # قوالب المفاهيم الإحصائية
            MathTemplate()
                category=MathConceptCategory.CONCEPT_STATISTICS,
                pattern=MathPattern.CVVCV,
                syllable_structure=['CVV', 'CV'],
                phonetic_constraints={
                    'stress': 'penultimate',
                    'length': 'medium',
                    'difficulty': 'medium',
                },
                semantic_features=['statistical', 'probabilistic', 'analytical'],
                frequency=0.5),
        ]

    def generate_number_concept()
        self,
        number: int,
        concept_type: str = 'cardinal',
        gender: NumberGender = NumberGender.MASCULINE) -> GeneratedMathConcept:
        """توليد مفهوم رقمي"""

        if concept_type == 'cardinal':
            arabic_text = self.linguistics.convert_number_to_arabic(number, gender)
            category = MathConceptCategory.NUMBER_CARDINAL
            meaning = f"العدد الأساسي {number}"

        elif concept_type == 'ordinal':
            arabic_text = self.linguistics.convert_ordinal_to_arabic(number, gender)
            category = MathConceptCategory.NUMBER_ORDINAL
            meaning = f"العدد الترتيبي {number}"

        else:
            raise ValueError(f"نوع المفهوم غير مدعوم: {concept_type}")

        # تحليل المقاطع
        syllables = self._extract_syllables_from_text(arabic_text)

        # تحليل صوتي
        phonetic_analysis = self.linguistics.analyze_phonetic_structure(arabic_text)

        # تحديد النمط
        pattern = self._determine_pattern_from_syllables(syllables)

        # إنشاء أمثلة
        examples = self._generate_number_examples(number, concept_type, gender)

        return GeneratedMathConcept()
            term=arabic_text,
            category=category,
            pattern=pattern,
            syllables=syllables,
            phonetic_analysis=phonetic_analysis,
            semantic_meaning=meaning,
            mathematical_value=number,
            gender=gender,
            linguistic_features={
                'type': concept_type,
                'agreement_rules': self._get_agreement_rules(number),
                'case_variations': self._get_case_variations(arabic_text, gender),
            },
            authenticity_score=1.0,  # الأعداد التقليدية أصيلة تماماً
            examples=examples)

    def generate_fraction_concept()
        self, numerator: int, denominator: int
    ) -> GeneratedMathConcept:
        """توليد مفهوم كسر"""

        arabic_text = self.linguistics.convert_fraction_to_arabic()
            numerator, denominator
        )

        # تحديد نوع الكسر
        if numerator == 1:
            category = MathConceptCategory.FRACTION_SIMPLE
            meaning = f"الكسر البسيط {numerator/{denominator}}"
        else:
            category = MathConceptCategory.FRACTION_COMPOUND
            meaning = f"الكسر المركب {numerator/{denominator}}"

        # تحليل المقاطع
        syllables = self._extract_syllables_from_text(arabic_text)

        # تحليل صوتي
        phonetic_analysis = self.linguistics.analyze_phonetic_structure(arabic_text)

        # تحديد النمط
        pattern = self._determine_pattern_from_syllables(syllables)

        # إنشاء أمثلة
        examples = self._generate_fraction_examples(numerator, denominator)

        return GeneratedMathConcept()
            term=arabic_text,
            category=category,
            pattern=pattern,
            syllables=syllables,
            phonetic_analysis=phonetic_analysis,
            semantic_meaning=meaning,
            mathematical_value=Fraction(numerator, denominator),
            linguistic_features={
                'fraction_type': 'simple' if numerator == 1 else 'compound',
                'unit_fraction': numerator == 1,
                'decimal_equivalent': float(Fraction(numerator, denominator)),
            },
            authenticity_score=1.0,
            examples=examples)

    def generate_operation_concept(self, operation: str) -> GeneratedMathConcept:
        """توليد مفهوم عملية رياضية"""

        if operation not in self.linguistics.operation_roots:
            raise ValueError(f"العملية غير مدعومة: {operation}")

        operation_data = self.linguistics.operation_roots[operation]
        main_term = operation_data['derivatives'][0]  # المصطلح الرئيسي

        # تحديد الفئة
        basic_operations = ['addition', 'subtraction', 'multiplication', 'division']
        if operation in basic_operations:
            category = MathConceptCategory.OPERATION_BASIC
        else:
            category = MathConceptCategory.OPERATION_ADVANCED

        # تحليل المقاطع
        syllables = self._extract_syllables_from_text(main_term)

        # تحليل صوتي
        phonetic_analysis = self.linguistics.analyze_phonetic_structure(main_term)

        # تحديد النمط
        pattern = self._determine_pattern_from_syllables(syllables)

        # إنشاء أمثلة
        examples = self._generate_operation_examples(operation)

        return GeneratedMathConcept()
            term=main_term,
            category=category,
            pattern=pattern,
            syllables=syllables,
            phonetic_analysis=phonetic_analysis,
            semantic_meaning=f"العملية الرياضية: {main_term}",
            mathematical_value=operation,
            linguistic_features={
                'operation_type': operation,
                'derivatives': operation_data['derivatives'],
                'root': operation_data['root'],
                'category': 'basic' if operation in basic_operations else 'advanced',
            },
            authenticity_score=1.0,
            examples=examples)

    def generate_concept_term()
        self, concept_type: str, domain: str
    ) -> GeneratedMathConcept:
        """توليد مصطلح مفهومي"""

        if domain not in self.linguistics.concept_taxonomy:
            raise ValueError(f"المجال غير مدعوم: {domain}")

        domain_concepts = self.linguistics.concept_taxonomy[domain]

        if concept_type not in domain_concepts:
            raise ValueError(f"نوع المفهوم غير متاح في {domain: {concept_type}}")

        # اختيار مصطلح عشوائي من النوع المطلوب
        available_terms = domain_concepts[concept_type]
        term = random.choice(available_terms)

        # تحديد الفئة بناءً على المجال
        category_mapping = {
            'arithmetic': MathConceptCategory.CONCEPT_ARITHMETIC,
            'algebra': MathConceptCategory.CONCEPT_ALGEBRA,
            'geometry': MathConceptCategory.CONCEPT_GEOMETRY,
            'statistics': MathConceptCategory.CONCEPT_STATISTICS,
        }

        category = category_mapping[domain]

        # تحليل المقاطع
        syllables = self._extract_syllables_from_text(term)

        # تحليل صوتي
        phonetic_analysis = self.linguistics.analyze_phonetic_structure(term)

        # تحديد النمط
        pattern = self._determine_pattern_from_syllables(syllables)

        # إنشاء أمثلة
        examples = self._generate_concept_examples(term, domain, concept_type)

        return GeneratedMathConcept()
            term=term,
            category=category,
            pattern=pattern,
            syllables=syllables,
            phonetic_analysis=phonetic_analysis,
            semantic_meaning=f"مفهوم {concept_type} في {domain}}",
            mathematical_value=None,
            linguistic_features={
                'domain': domain,
                'concept_type': concept_type,
                'related_terms': [t for t in available_terms if t != term],
            },
            authenticity_score=1.0,
            examples=examples)

    def generate_comprehensive_math_concepts()
        self, count: int = 50
    ) -> List[GeneratedMathConcept]:
        """توليد مجموعة شاملة من المفاهيم الرياضية"""

        concepts = []

        logger.info("بدء توليد المفاهيم الرياضية الشاملة...")

        # 1. الأعداد الأساسية (1 20)
        logger.info("توليد الأعداد الأساسية...")
        for num in range(1, 21):
            for gender in [NumberGender.MASCULINE, NumberGender.FEMININE]:
                concept = self.generate_number_concept(num, 'cardinal', gender)
                concepts.append(concept)

        # 2. الأعداد الترتيبية (1-10)
        logger.info("توليد الأعداد الترتيبية...")
        for num in range(1, 11):
            for gender in [NumberGender.MASCULINE, NumberGender.FEMININE]:
                concept = self.generate_number_concept(num, 'ordinal', gender)
                concepts.append(concept)

        # 3. الكسور البسيطة
        logger.info("توليد الكسور البسيطة...")
        for denom in range(2, 11):
            concept = self.generate_fraction_concept(1, denom)
            concepts.append(concept)

        # 4. الكسور المركبة
        logger.info("توليد الكسور المركبة...")
        compound_fractions = [(2, 3), (3, 4), (2, 5), (3, 5), (4, 5), (5, 6), (7, 8)]
        for num, denom in compound_fractions:
            concept = self.generate_fraction_concept(num, denom)
            concepts.append(concept)

        # 5. العمليات الأساسية
        logger.info("توليد العمليات الأساسية...")
        basic_operations = ['addition', 'subtraction', 'multiplication', 'division']
        for operation in basic_operations:
            concept = self.generate_operation_concept(operation)
            concepts.append(concept)

        # 6. العمليات المتقدمة
        logger.info("توليد العمليات المتقدمة...")
        advanced_operations = ['power', 'root', 'logarithm', 'factorial']
        for operation in advanced_operations:
            concept = self.generate_operation_concept(operation)
            concepts.append(concept)

        # 7. المفاهيم الحسابية
        logger.info("توليد المفاهيم الحسابية...")
        arithmetic_concepts = [
            ('numbers', 'arithmetic'),
            ('operations', 'arithmetic'),
            ('properties', 'arithmetic'),
        ]
        for concept_type, domain in arithmetic_concepts:
            concept = self.generate_concept_term(concept_type, domain)
            concepts.append(concept)

        # 8. المفاهيم الجبرية
        logger.info("توليد المفاهيم الجبرية...")
        algebra_concepts = [
            ('variables', 'algebra'),
            ('equations', 'algebra'),
            ('functions', 'algebra'),
            ('polynomials', 'algebra'),
        ]
        for concept_type, domain in algebra_concepts:
            concept = self.generate_concept_term(concept_type, domain)
            concepts.append(concept)

        # 9. المفاهيم الهندسية
        logger.info("توليد المفاهيم الهندسية...")
        geometry_concepts = [
            ('shapes', 'geometry'),
            ('measurements', 'geometry'),
            ('angles', 'geometry'),
            ('lines', 'geometry'),
        ]
        for concept_type, domain in geometry_concepts:
            concept = self.generate_concept_term(concept_type, domain)
            concepts.append(concept)

        # 10. المفاهيم الإحصائية
        logger.info("توليد المفاهيم الإحصائية...")
        statistics_concepts = [
            ('measures', 'statistics'),
            ('probability', 'statistics'),
            ('distributions', 'statistics'),
        ]
        for concept_type, domain in statistics_concepts:
            concept = self.generate_concept_term(concept_type, domain)
            concepts.append(concept)

        # تقليم القائمة حسب العدد المطلوب
        if len(concepts) > count:
            concepts = concepts[:count]

        logger.info(f"تم توليد {len(concepts)} مفهوم رياضي بنجاح")

        return concepts

    def _extract_syllables_from_text(self, text: str) -> List[str]:
        """استخراج المقاطع من النص"""

        # تطبيق خوارزمية تقسيم المقاطع العربية
        syllables = []

        # إزالة المسافات والتنوين
        clean_text = re.sub(r'[ًٌٍَُِّْ\s]', '', text)

        # خوارزمية تقسيم مبسطة
        i = 0
        current_syllable = ""

        vowels = set('اويَُِ')
        consonants = set('بتثجحخدذرزسشصضطظعغفقكلمنه')

        while i < len(clean_text):
            char = clean_text[i]
            current_syllable += char

            # إذا وصلنا لحرف علة، نكمل حتى نجد صامت أو نهاية
            if char in vowels:
                if i + 1 < len(clean_text) and clean_text[i + 1] in consonants:
                    # نضيف الصامت التالي إذا لم يكن يبدأ مقطع جديد
                    next_syllable_start = i + 2
                    if next_syllable_start < len(clean_text):
                        current_syllable += clean_text[i + 1]
                        i += 1

                syllables.append(current_syllable)
                current_syllable = ""

            i += 1

        # إضافة أي بقايا
        if current_syllable:
            if syllables:
                syllables[-1] += current_syllable
            else:
                syllables.append(current_syllable)

        return syllables or [text]

    def _determine_pattern_from_syllables(self, syllables: List[str]) -> MathPattern:
        """تحديد النمط من المقاطع"""

        if not syllables:
            return MathPattern.CV

        # تحليل المقطع الأول لتحديد النمط
        first_syllable = syllables[0]

        vowels = set('اويَُِ')
        consonants = set('بتثجحخدذرزسشصضطظعغفقكلمنه')

        pattern_string = ""
        for char in first_syllable:
            if char in consonants:
                pattern_string += "C"
            elif char in vowels:
                pattern_string += "V"

        # تطبيق قوانين النمط
        if pattern_string == "CV":
            return MathPattern.CV
        elif pattern_string == "CVC":
            return MathPattern.CVC
        elif pattern_string == "CVCV":
            return MathPattern.CVCV
        elif pattern_string == "CVCVC":
            return MathPattern.CVCVC
        elif pattern_string in ["CVVC", "CVVV"]:
            return MathPattern.CVVCV
        elif pattern_string in ["CVCVCV", "CVVCVC"]:
            return MathPattern.CVCVCV
        else:
            return MathPattern.CVC  # نمط افتراضي

    def _get_agreement_rules(self, number: int) -> Dict[str, str]:
        """الحصول على قواعد التوافق للعدد"""

        if number in [1, 2]:
            return {
                'gender_rule': 'agree_with_counted',
                'case_rule': 'follows_counted',
                'description': 'يوافق المعدود في التذكير والتأنيث',
            }
        elif 3 <= number <= 10:
            return {
                'gender_rule': 'opposite_to_counted',
                'case_rule': 'genitive_plural',
                'description': 'يخالف المعدود في التذكير والتأنيث',
            }
        elif 11 <= number <= 99:
            return {
                'gender_rule': 'masculine_always',
                'case_rule': 'accusative_singular',
                'description': 'مذكر دائماً مع نصب المعدود',
            }
        else:
            return {
                'gender_rule': 'agree_with_counted',
                'case_rule': 'follows_counted',
                'description': 'يوافق المعدود في حالات خاصة',
            }

    def _get_case_variations(self, term: str, gender: NumberGender) -> Dict[str, str]:
        """الحصول على تصريفات الإعراب"""

        # هذه دالة مبسطة لتصريفات الإعراب
        variations = {
            'nominative': term,
            'accusative': term,
            'genitive': term,
        }  # الرفع  # النصب  # الجر

        # تطبيق بعض القواعد البسيطة
        if term.endswith('ة'):
            variations['accusative'] = term[: 1] + 'ة'
            variations['genitive'] = term[: 1] + 'ة'
        elif term.endswith('ان'):
            variations['accusative'] = term[: 2] + 'ين'
            variations['genitive'] = term[: 2] + 'ين'

        return variations

    def _generate_number_examples()
        self, number: int, concept_type: str, gender: NumberGender
    ) -> List[str]:
        """توليد أمثلة للأعداد"""

        examples = []

        if concept_type == 'cardinal':
            examples = [
                f"لديه {self.linguistics.convert_number_to_arabic(number,} gender) كتاب}",
                f"عدد الطلاب {self.linguistics.convert_number_to_arabic(number, gender)}",
                f"مجموع العدد {self.linguistics.convert_number_to_arabic(number, gender)}",
            ]
        else:  # ordinal
            examples = [
                f"هذا هو اليوم {self.linguistics.convert_ordinal_to_arabic(number, gender)}",
                f"في المرتبة {self.linguistics.convert_ordinal_to_arabic(number, gender)}",
                f"الفصل {self.linguistics.convert_ordinal_to_arabic(number, gender)}",
            ]

        return examples

    def _generate_fraction_examples()
        self, numerator: int, denominator: int
    ) -> List[str]:
        """توليد أمثلة للكسور"""

        fraction_text = self.linguistics.convert_fraction_to_arabic()
            numerator, denominator
        )

        examples = [
            f"أكل {fraction_text} من التفاحة",
            f"قطع مسافة {fraction_text} من الطريق",
            f"حل {fraction_text} من المسائل",
        ]

        return examples

    def _generate_operation_examples(self, operation: str) -> List[str]:
        """توليد أمثلة للعمليات"""

        operation_data = self.linguistics.operation_roots[operation]
        main_term = operation_data['derivatives'][0]

        examples = [
            f"عملية {main_term} الأرقام",
            f"نتيجة {main_term} العددين",
            f"تطبيق {main_term} في الحساب",
        ]

        # أمثلة خاصة لكل عملية
        if operation == 'addition':
            examples.append("٣ + ٢ = ٥ (جمع ثلاثة واثنين)")
        elif operation == 'subtraction':
            examples.append("٥ - ٢ = ٣ (طرح اثنين من خمسة)")
        elif operation == 'multiplication':
            examples.append("٣ × ٢ = ٦ (ضرب ثلاثة في اثنين)")
        elif operation == 'division':
            examples.append("٦ ÷ ٢ = ٣ (قسمة ستة على اثنين)")

        return examples

    def _generate_concept_examples()
        self, term: str, domain: str, concept_type: str
    ) -> List[str]:
        """توليد أمثلة للمفاهيم"""

        examples = [
            f"دراسة {term} في الرياضيات}",
            f"تطبيق مفهوم {term}",
            f"فهم {term} بشكل صحيح",
        ]

        # أمثلة خاصة حسب المجال
        if domain == 'geometry':
            examples.append(f"رسم {term} على الورق")
        elif domain == 'algebra':
            examples.append(f"حل {term} بالطرق الجبرية")
        elif domain == 'statistics':
            examples.append(f"حساب {term} للبيانات")

        return examples


# ═══════════════════════════════════════════════════════════════════════════════════
# TESTING AND VALIDATION SYSTEM - نظام الاختبار والتحقق
# ═══════════════════════════════════════════════════════════════════════════════════


def test_math_generator():
    """اختبار مولد المفاهيم الرياضية"""

    print("🔢 اختبار مولد المفاهيم الرياضية العربية")
    print("=" * 60)

    # إنشاء المولد
    generator = AdvancedArabicMathGenerator()

    # اختبار الأعداد الأساسية
    print("\n📊 1. اختبار الأعداد الأساسية:")
    for num in [1, 5, 10, 15, 20]:
        for gender in [NumberGender.MASCULINE, NumberGender.FEMININE]:
            concept = generator.generate_number_concept(num, 'cardinal', gender)
            gender_name = "مذكر" if gender == NumberGender.MASCULINE else "مؤنث"
            print(f"   العدد {num} ({gender_name}): {concept.term}")
            print(f"   المقاطع: {'} - '.join(concept.syllables)}")
            print(f"   المعنى: {concept.semantic_meaning}")
            print()

    # اختبار الأعداد الترتيبية
    print("\n🔢 2. اختبار الأعداد الترتيبية:")
    for num in [1, 3, 5, 10]:
        concept = generator.generate_number_concept()
            num, 'ordinal', NumberGender.MASCULINE
        )
        print(f"   الترتيبي {num}: {concept.term}")
        print(f"   المقاطع: {'} - '.join(concept.syllables)}")
        print()

    # اختبار الكسور
    print("\n🍰 3. اختبار الكسور:")
    fractions = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    for num, denom in fractions:
        concept = generator.generate_fraction_concept(num, denom)
        print(f"   الكسر {num}/{denom}: {concept.term}")
        print(f"   القيمة العشرية: {float(concept.mathematical_value):.3f}")
        print()

    # اختبار العمليات
    print("\n⚙️ 4. اختبار العمليات الرياضية:")
    operations = [
        'addition',
        'subtraction',
        'multiplication',
        'division',
        'power',
        'root',
    ]
    for operation in operations:
        concept = generator.generate_operation_concept(operation)
        print(f"   {operation}: {concept.term}")
        print(f"   المشتقات: {', '.join(concept.linguistic_features['derivatives'])}")
        print()

    # اختبار المفاهيم
    print("\n🧮 5. اختبار المفاهيم الرياضية:")
    test_concepts = [
        ('numbers', 'arithmetic'),
        ('variables', 'algebra'),
        ('shapes', 'geometry'),
        ('probability', 'statistics'),
    ]

    for concept_type, domain in test_concepts:
        concept = generator.generate_concept_term(concept_type, domain)
        print(f"   {domain}/{concept_type}: {concept.term}")
        print(f"   المعنى: {concept.semantic_meaning}")
        print()

    # اختبار التوليد الشامل
    print("\n🎯 6. اختبار التوليد الشامل:")
    comprehensive_concepts = generator.generate_comprehensive_math_concepts(20)

    print(f"   تم توليد {len(comprehensive_concepts)} مفهوم رياضي")

    # إحصائيات حسب الفئة
    category_stats = {}
    for concept in comprehensive_concepts:
        category = concept.category.value
        category_stats[category] = category_stats.get(category, 0) + 1

    print("\n   إحصائيات حسب الفئة:")
    for category, count in category_stats.items():
        print(f"     {category}: {count} مفهوم")

    # عرض عينة من النتائج
    print("\n   عينة من المفاهيم المولدة:")
    for i, concept in enumerate(comprehensive_concepts[:10]):
        print(f"     {i+1}. {concept.term} ({concept.category.value})")

    print("\n✅ انتهاء الاختبار بنجاح!")


if __name__ == "__main__":
    test_math_generator()

