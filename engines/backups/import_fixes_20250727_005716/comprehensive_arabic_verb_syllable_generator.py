#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Arabic Verb and Source Syllable Pattern Generator
==============================================================
مولد أنماط المقاطع الصوتية الشاملة للأفعال والمصادر العربية

يغطي:
- الأفعال المجردة (ثلاثية ورباعية)
- الأفعال المزيدة (قياسية وغير قياسية)
- المصادر القياسية والسماعية
- الظواهر الفونولوجية (إدغام، إعلال، إبدال)
- المقاطع المعقدة (CV, CVC, CVCC, CVV, CVVC, CVVCV)

Author: GitHub Copilot Arabic NLP Expert
Version: 3.0.0 - COMPREHENSIVE VERB SYSTEM
Date: 2025-07-26
Encoding: UTF 8
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import re
import json
from collections import Counter
import logging

# Configure logging
logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC VERB MORPHOLOGY SYSTEM - نظام صرف الأفعال العربية
# ═══════════════════════════════════════════════════════════════════════════════════


class VerbType(Enum):
    """أنواع الأفعال العربية"""

    TRILATERAL_SIMPLE = "trilateral_simple"  # ثلاثي مجرد
    QUADRILATERAL_SIMPLE = "quadrilateral_simple"  # رباعي مجرد
    TRILATERAL_AUGMENTED = "trilateral_augmented"  # ثلاثي مزيد
    QUADRILATERAL_AUGMENTED = "quadrilateral_augmented"  # رباعي مزيد


class SyllableType(Enum):
    """أنواع المقاطع الصوتية"""

    V = "V"  # صائت منفرد
    CV = "CV"  # صامت + صائت
    CVC = "CVC"  # صامت + صائت + صامت
    CVV = "CVV"  # صامت + صائت طويل
    CVVC = "CVVC"  # صامت + صائت طويل + صامت
    CVCC = "CVCC"  # صامت + صائت + صامتان
    CCV = "CCV"  # صامتان + صائت (نادر في العربية)
    CVCCC = "CVCCC"  # صامت + صائت + ثلاثة صوامت (في نهاية الكلمة)
    CVVCV = "CVVCV"  # نمط مركب
    CVVCVC = "CVVCVC"  # نمط مركب


@dataclass
class VerbForm:
    """صيغة الفعل"""

    form_number: str  # رقم الصيغة (I, II, III, ...)
    form_name: str  # اسم الصيغة
    pattern: str  # الوزن الصرفي
    meaning: str  # المعنى العام
    syllable_pattern: List[str]  # نمط المقاطع
    morphemes: List[str]  # المورفيمات


@dataclass
class SourcePattern:
    """نمط المصدر"""

    source_word: str  # المصدر
    verb_form: str  # صيغة الفعل
    syllable_pattern: str  # نمط المقاطع
    is_standard: bool  # قياسي أم سماعي
    phonological_features: List[str]  # الخصائص الصوتية


class ArabicVerbMorphologySystem:
    """نظام صرف الأفعال العربية الشامل"""

    def __init__(self):

        # الفونيمات الأساسية
    self.root_consonants = [
    'ك',
    'ت',
    'ب',
    'س',
    'ل',
    'م',
    'ن',
    'ه',
    'ر',
    'ج',
    'د',
    'ع',
    ]
    self.all_consonants = [
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
    ]

        # الصوائت والحركات
    self.short_vowels = ['َ', 'ِ', 'ُ']
    self.long_vowels = ['ا', 'ي', 'و']
    self.diacritics = ['ً', 'ٌ', 'ٍ', 'ْ', 'ّ', 'ٰ']

        # تحميل أنماط الأفعال والمصادر
    self.verb_forms = self._load_verb_forms()
    self.source_patterns = self._load_source_patterns()
    self.phonological_rules = self._load_phonological_rules()

    logger.info("تم تهيئة نظام صرف الأفعال العربية الشامل")

    def _load_verb_forms(self) -> Dict[str, VerbForm]:
    """تحميل صيغ الأفعال العربية"""
    return {
            # الثلاثي المجرد
    "I": VerbForm()
                form_number="I",
                form_name="الثلاثي المجرد",
    pattern="فَعَلَ",
    meaning="basic_action",
    syllable_pattern=["CV", "CV", "CV"],  # فَ-عَ لَ
    morphemes=["ف", "ع", "ل"]),
            # الثلاثي المزيد
    "II": VerbForm()
                form_number="II",
                form_name="فَعَّلَ",
    pattern="فَعَّلَ",
    meaning="intensive/causative",
    syllable_pattern=["CV", "CVC", "CV"],  # فَ-عَّ لَ
    morphemes=["ف", "عّ", "ل"]),
    "III": VerbForm()
                form_number="III",
                form_name="فَاعَلَ",
    pattern="فَاعَلَ",
    meaning="reciprocal/attempt",
    syllable_pattern=["CV", "CV", "CV", "CV"],  # فَا-عَ لَ
    morphemes=["ف", "ا", "ع", "ل"]),
    "IV": VerbForm()
                form_number="IV",
                form_name="أَفْعَلَ",
    pattern="أَفْعَلَ",
    meaning="causative",
    syllable_pattern=["V", "CVC", "CV"],  # أَ-فْع لَ
    morphemes=["أ", "فع", "ل"]),
    "V": VerbForm()
                form_number="V",
                form_name="تَفَعَّلَ",
    pattern="تَفَعَّلَ",
    meaning="reflexive",
    syllable_pattern=["CV", "CV", "CVC", "CV"],  # تَ-فَ-عَّ لَ
    morphemes=["ت", "ف", "عّ", "ل"]),
    "VI": VerbForm()
                form_number="VI",
                form_name="تَفَاعَلَ",
    pattern="تَفَاعَلَ",
    meaning="mutual_action",
    syllable_pattern=["CV", "CV", "CV", "CV", "CV"],  # تَ-فَا-عَ لَ
    morphemes=["ت", "ف", "ا", "ع", "ل"]),
    "VII": VerbForm()
                form_number="VII",
                form_name="انْفَعَلَ",
    pattern="انْفَعَلَ",
    meaning="passive/reflexive",
    syllable_pattern=["V", "CVC", "CV", "CV"],  # ان-فَع لَ
    morphemes=["ان", "فع", "ل"]),
    "VIII": VerbForm()
                form_number="VIII",
                form_name="افْتَعَلَ",
    pattern="افْتَعَلَ",
    meaning="reflexive",
    syllable_pattern=["VC", "CV", "CV", "CV"],  # اف-تَع لَ
    morphemes=["اف", "ت", "ع", "ل"]),
    "IX": VerbForm()
                form_number="IX",
                form_name="افْعَلَّ",
    pattern="افْعَلَّ",
    meaning="colors/defects",
    syllable_pattern=["VC", "CV", "CVC"],  # اف-عَ لَّ
    morphemes=["اف", "ع", "لّ"]),
    "X": VerbForm()
                form_number="X",
                form_name="اسْتَفْعَلَ",
    pattern="اسْتَفْعَلَ",
    meaning="seeking/requesting",
    syllable_pattern=["VC", "CV", "CVC", "CV"],  # اس-تَ-فْع لَ
    morphemes=["است", "فع", "ل"]),
            # الرباعي المجرد
    "Q1": VerbForm()
                form_number="Q1",
                form_name="فَعْلَلَ",
    pattern="فَعْلَلَ",
    meaning="quadrilateral_basic",
    syllable_pattern=["CVC", "CV", "CV"],  # فَعْ-لَ لَ
    morphemes=["فع", "ل", "ل"]),
            # الرباعي المزيد
    "Q2": VerbForm()
                form_number="Q2",
                form_name="تَفَعْلَلَ",
    pattern="تَفَعْلَلَ",
    meaning="quadrilateral_reflexive",
    syllable_pattern=["CV", "CVC", "CV", "CV"],  # تَ-فَعْ-لَ لَ
    morphemes=["ت", "فع", "ل", "ل"]),
    "Q3": VerbForm()
                form_number="Q3",
                form_name="افْعَنْلَلَ",
    pattern="افْعَنْلَلَ",
    meaning="quadrilateral_augmented",
    syllable_pattern=["VC", "CV", "CVC", "CV", "CV"],  # اف-عَ-نْل لَ
    morphemes=["اف", "ع", "نل", "ل"]),
    }

    def _load_source_patterns(self) -> Dict[str, List[SourcePattern]]:
    """تحميل أنماط المصادر"""
    return {
            # مصادر الثلاثي المجرد
    "I": [
    SourcePattern("فَعْل", "I", "CVC", True, ["مقطع_مغلق"]),
    SourcePattern("فُعُول", "I", "CV CVC", True, ["مقطع_مركب"]),
    SourcePattern("فِعَال", "I", "CV CVC", True, ["صائت_طويل"]),
    SourcePattern("فَعَالَة", "I", "CV-CV CV", True, ["تاء_التأنيث"]),
    SourcePattern("مَفْعَل", "I", "CVC CV", True, ["ميم_زائدة"]),
    ],
            # مصادر الثلاثي المزيد
    "II": [
    SourcePattern("تَفْعِيل", "II", "CVC CVVC", True, ["تنشيط", "مقطع_طويل"]),
    SourcePattern("تَفْعِلَة", "II", "CVC-CV CV", True, ["تاء_التأنيث"]),
    ],
    "III": [
    SourcePattern("مُفَاعَلَة", "III", "CV-CV-CV CV", True, ["مشاركة"]),
    SourcePattern("فِعَال", "III", "CV CVC", True, ["مختصر"]),
    ],
    "IV": [
    SourcePattern("إِفْعَال", "IV", "VC CVC", True, ["همزة_وصل"]),
    ],
    "V": [
    SourcePattern("تَفَعُّل", "V", "CV-CV CVC", True, ["تشديد"]),
    ],
    "VI": [
    SourcePattern("تَفَاعُل", "VI", "CV-CV-CV CVC", True, ["تبادل"]),
    ],
    "VII": [
    SourcePattern("انْفِعَال", "VII", "VC-CV CVC", True, ["انفعال"]),
    ],
    "VIII": [
    SourcePattern("افْتِعَال", "VIII", "VC-CV CVC", True, ["اكتساب"]),
    ],
    "IX": [
    SourcePattern("افْعِلَال", "IX", "VC-CV CVC", True, ["ألوان_وعيوب"]),
    ],
    "X": [
    SourcePattern("اسْتِفْعَال", "X", "VC-CVC CVC", True, ["طلب"]),
    ],
            # مصادر الرباعي
    "Q1": [
    SourcePattern("فَعْلَلَة", "Q1", "CVC-CV CV", True, ["تاء_التأنيث"]),
    SourcePattern("فِعْلَال", "Q1", "CVC CVC", True, ["مقطع_مضاعف"]),
    ],
    "Q2": [
    SourcePattern("تَفَعْلُل", "Q2", "CV-CVC CVC", True, ["تدرج"]),
    ],
            # مصادر سماعية (غير قياسية)
    "irregular": [
    SourcePattern("مَجِيء", "I", "CV CVVC", False, ["همزة_نهائية", "ياء_مد"]),
    SourcePattern("وُضُوء", "I", "CV-CV CVC", False, ["واو_ضمة", "همزة"]),
    SourcePattern("سُؤَال", "I", "CV-CV CVC", False, ["همزة_متوسطة"]),
    SourcePattern("بَنَاء", "I", "CV-CV CVC", False, ["همزة_ممدودة"]),
    ],
    }

    def _load_phonological_rules(self) -> Dict[str, Any]:
    """تحميل القواعد الصوتية"""
    return {
    'assimilation': {
                # الإدغام
    'consonant_clusters': {
    'تد': 'دد',  # التاء تُدغم في الدال
    'تز': 'زز',  # التاء تُدغم في الزاي
    'نب': 'مب',  # النون تقلب ميماً قبل الباء
    'نم': 'مم',  # النون تُدغم في الميم
    },
    'vowel_harmony': {
    'ُو': 'ُو',  # الضمة مع الواو
    'ِي': 'ِي',  # الكسرة مع الياء
    'َا': 'َا',  # الفتحة مع الألف
    },
    },
    'weakening': {
                # الإعلال
    'waw_alif': {
    'قول': 'قال',  # قوَل  > قال
    'نوم': 'نام',  # نوَم  > نام
    },
    'ya_alif': {
    'بيع': 'باع',  # بيَع  > باع
    'سير': 'سار',  # سيَر  > سار
    },
    },
    'epenthesis': {
                # الإشباع وكسر التقاء الساكنين
    'consonant_clusters': {
    'CC': 'CiC',  # إدخال كسرة بين ساكنين
    },
    'word_initial': {
    'CC': 'iCC',  # همزة وصل في البداية
    },
    },
    'metathesis': {
                # القلب المكاني
    'specific_contexts': {
    'اصطبر': 'اصبر',  # قلب التاء والصاد
    'ادّارك': 'ادرك',  # قلب التاء والدال
    }
    },
    }

    def generate_verb_syllable_patterns()
    self, root: List[str], verb_form: str, include_pronouns: bool = False
    ) -> Dict[str, Any]:
    """
    توليد أنماط المقاطع الصوتية للفعل

    Args:
    root: الجذر (مثل ['ك', 'ت', 'ب'])
    verb_form: صيغة الفعل (I, II, III, ...)
    include_pronouns: تشمل الضمائر المتصلة

    Returns:
    Dict: التحليل الشامل للفعل
    """

        if verb_form not in self.verb_forms:
    raise ValueError(f"صيغة الفعل غير مدعومة: {verb_form}")

        form_data = self.verb_forms[verb_form]

        # بناء الفعل من الجذر والوزن
    verb_word = self._construct_verb(root, form_data)

        # تطبيق القواعد الصوتية
    phonologically_adjusted = self._apply_phonological_rules(verb_word, root)

        # تحليل المقاطع
    syllable_analysis = self._analyze_syllables(phonologically_adjusted)

        # إضافة الضمائر إذا طُلبت
        if include_pronouns:
    pronoun_variants = self._add_pronoun_variants(phonologically_adjusted)
        else:
    pronoun_variants = []

    return {
    'root': root,
    'verb_form': verb_form,
    'pattern': form_data.pattern,
    'constructed_verb': verb_word,
    'phonologically_adjusted': phonologically_adjusted,
    'syllable_structure': syllable_analysis['patterns'],
    'syllable_count': syllable_analysis['count'],
    'complexity_score': syllable_analysis['complexity'],
    'phonological_processes': syllable_analysis['processes'],
    'pronoun_variants': pronoun_variants,
    'morphological_analysis': {
    'morphemes': form_data.morphemes,
    'meaning': form_data.meaning,
    'type': self._classify_verb_type(verb_form),
    },
    }

    def generate_source_syllable_patterns()
    self, root: List[str], verb_form: str, source_type: str = "standard"
    ) -> List[Dict[str, Any]]:
    """
    توليد أنماط المقاطع الصوتية للمصادر

    Args:
    root: الجذر
    verb_form: صيغة الفعل
    source_type: نوع المصدر (standard, irregular)

    Returns:
    List[Dict]: قائمة المصادر مع تحليلها
    """

        if verb_form not in self.source_patterns and source_type != "irregular":
    raise ValueError(f"مصادر الصيغة غير مدعومة: {verb_form}")

    sources_data = self.source_patterns.get(verb_form, [])
        if source_type == "irregular":
    sources_data.extend(self.source_patterns.get("irregular", []))

    results = []

        for source_pattern in sources_data:
            # بناء المصدر من الجذر
    source_word = self._construct_source(root, source_pattern)

            # تطبيق القواعد الصوتية
    phonologically_adjusted = self._apply_phonological_rules(source_word, root)

            # تحليل المقاطع
    syllable_analysis = self._analyze_syllables(phonologically_adjusted)

            # تحليل الخصائص الصوتية
    phonological_features = self._analyze_phonological_features()
    phonologically_adjusted, source_pattern.phonological_features
    )

    results.append()
    {
    'root': root,
    'verb_form': verb_form,
    'source_word': source_word,
    'phonologically_adjusted': phonologically_adjusted,
    'syllable_pattern': source_pattern.syllable_pattern,
    'analyzed_patterns': syllable_analysis['patterns'],
    'syllable_count': syllable_analysis['count'],
    'is_standard': source_pattern.is_standard,
    'phonological_features': phonological_features,
    'complexity_score': syllable_analysis['complexity'],
    'morphological_analysis': {
    'source_type': ()
    'قياسي' if source_pattern.is_standard else 'سماعي'
    ),
    'semantic_field': self._determine_semantic_field(verb_form),
    },
    }
    )

    return results

    def _construct_verb(self, root: List[str], form_data: VerbForm) -> str:
    """بناء الفعل من الجذر والوزن"""

    pattern = form_data.pattern

        # استبدال رموز الجذر في الوزن
        if len(root) == 3:  # جذر ثلاثي
    result = ()
    pattern.replace('ف', root[0])
    .replace('ع', root[1])
    .replace('ل', root[2])
    )
        elif len(root) == 4:  # جذر رباعي
            # نحتاج لنمط خاص للرباعي
            if form_data.form_number.startswith('Q'):
    result = ()
    pattern.replace('ف', root[0])
    .replace('ع', root[1])
    .replace('ل', root[2])
    .replace('ل', root[3])
    )
            else:
    result = pattern  # استخدام النمط كما هو للأوزان الخاصة
        else:
    raise ValueError(f"طول الجذر غير مدعوم: {len(root)}")

    return result

    def _construct_source(self, root: List[str], source_pattern: SourcePattern) -> str:
    """بناء المصدر من الجذر والنمط"""

    pattern = source_pattern.source_word

        # استبدال رموز الجذر في نمط المصدر
        if len(root) == 3:
    result = ()
    pattern.replace('ف', root[0])
    .replace('ع', root[1])
    .replace('ل', root[2])
    )
        elif len(root) == 4:
            # للرباعي
    result = ()
    pattern.replace('ف', root[0])
    .replace('ع', root[1])
    .replace('ل', root[2])
    .replace('ل', root[3])
    )
        else:
    result = pattern

    return result

    def _apply_phonological_rules(self, word: str, root: List[str]) -> str:
    """تطبيق القواعد الصوتية"""

    result = word
    applied_processes = []

        # تطبيق الإدغام
        for cluster, replacement in self.phonological_rules['assimilation'][
    'consonant_clusters'
    ].items():
            if cluster in result:
    result = result.replace(cluster, replacement)
    applied_processes.append(f"إدغام: {cluster } > {replacement}}")

        # تطبيق الإعلال
        for original, changed in self.phonological_rules['weakening'][
    'waw_alif'
    ].items():
            if any(char in original for char in root):
                # تطبيق الإعلال حسب السياق
                if 'و' in word and 'َ' in word:
    result = result.replace('وَ', 'ا')
    applied_processes.append("إعلال: وَ  > ا")

        # معالجة التقاء الساكنين
    result = self._resolve_consonant_clusters(result)

    return result

    def _resolve_consonant_clusters(self, word: str) -> str:
    """حل التقاء الساكنين"""

        # خوارزمية مبسطة لكسر التقاء الساكنين
    result = word

        # البحث عن أنماط التقاء الساكنين وإدخال حركة
    consonant_patterns = re.findall(r'[بتثجحخدذرزسشصضطظعغفقكلمنهوي]{2,}', result)

        for pattern in consonant_patterns:
            if len(pattern) >= 2:
                # إدخال كسرة بين الصوامت الساكنة
    modified = pattern[0] + 'ِ' + pattern[1:]
    result = result.replace(pattern, modified, 1)

    return result

    def _analyze_syllables(self, word: str) -> Dict[str, Any]:
    """تحليل المقاطع الصوتية للكلمة"""

        # تقسيم الكلمة إلى مقاطع
    syllables = self._segment_into_syllables(word)

        # تحديد نمط كل مقطع
    patterns = []
        for syllable in syllables:
    pattern = self._determine_syllable_pattern(syllable)
    patterns.append(pattern)

        # حساب درجة التعقيد
    complexity = self._calculate_syllable_complexity(patterns)

        # تحديد العمليات الصوتية المطبقة
    processes = self._identify_phonological_processes(syllables)

    return {
    'syllables': syllables,
    'patterns': patterns,
    'count': len(syllables),
    'complexity': complexity,
    'processes': processes,
    }

    def _segment_into_syllables(self, word: str) -> List[str]:
    """تقسيم الكلمة إلى مقاطع"""

    syllables = []
    current_syllable = ""

    i = 0
        while i < len(word):
    char = word[i]

            # إضافة الحرف الحالي
    current_syllable += char

            # التحقق من نهاية المقطع
            if self._is_vowel(char) or char in self.short_vowels:
                # نواة المقطع موجودة، ننظر للحرف التالي
                if i + 1 < len(word):
    next_char = word[i + 1]
                    if self._is_consonant(next_char):
                        # صامت بعد الصائت
                        if i + 2 < len(word) and self._is_vowel(word[i + 2]):
                            # صامت + صائت بعدها  > نهاية المقطع الحالي
    syllables.append(current_syllable)
    current_syllable = ""
                        else:
                            # صامت في نهاية الكلمة -> نضيفه للمقطع
                            if i + 1 == len(word) - 1:
    current_syllable += next_char
    syllables.append(current_syllable)
    current_syllable = ""
    i += 1  # تخطي الحرف المضاف
                else:
                    # نهاية الكلمة
    syllables.append(current_syllable)
    current_syllable = ""

    i += 1

        # إضافة المقطع الأخير إذا بقي شيء
        if current_syllable:
    syllables.append(current_syllable)

    return syllables

    def _determine_syllable_pattern(self, syllable: str) -> str:
    """تحديد نمط المقطع الصوتي"""

    pattern = ""

        for char in syllable:
            if self._is_consonant(char):
    pattern += "C"
            elif self._is_vowel(char) or char in self.short_vowels:
    pattern += "V"
            elif char in self.diacritics:
                # التعامل مع علامات التشكيل
                if char == 'ّ':  # الشدة
    pattern += "C"  # تضعيف الصامت
                # باقي العلامات لا تؤثر على النمط الأساسي

    return pattern

    def _is_consonant(self, char: str) -> bool:
    """تحديد ما إذا كان الحرف صامتاً"""
    return char in self.all_consonants

    def _is_vowel(self, char: str) -> bool:
    """تحديد ما إذا كان الحرف صائتاً"""
    return char in self.long_vowels or char in self.short_vowels

    def _calculate_syllable_complexity(self, patterns: List[str]) -> float:
    """حساب درجة تعقيد المقاطع"""

    complexity_weights = {
    'V': 1.0,
    'CV': 1.2,
    'CVC': 1.5,
    'CVV': 1.8,
    'CVVC': 2.0,
    'CVCC': 2.5,
    'CCV': 3.0,
    'CVCCC': 3.5,
    }

    total_complexity = sum()
    complexity_weights.get(pattern, 1.0) for pattern in patterns
    )
    average_complexity = total_complexity / len(patterns) if patterns else 0.0

    return round(average_complexity, 2)

    def _identify_phonological_processes(self, syllables: List[str]) -> List[str]:
    """تحديد العمليات الصوتية المطبقة"""

    processes = []

        for syllable in syllables:
            # البحث عن الشدة (الإدغام)
            if 'ّ' in syllable:
    processes.append("إدغام")

            # البحث عن المد
            if any(vowel in syllable for vowel in self.long_vowels):
    processes.append("مد_صوتي")

            # البحث عن التنوين
            if any(diac in syllable for diac in ['ً', 'ٌ', 'ٍ']):
    processes.append("تنوين")

    return list(set(processes))  # إزالة التكرار

    def _add_pronoun_variants(self, verb: str) -> List[Dict[str, Any]]:
    """إضافة متغيرات الضمائر المتصلة"""

    pronouns = {
    'ون': {'type': 'plural_masculine', 'meaning': 'they_masc'},
    'ين': {'type': 'plural_feminine', 'meaning': 'they_fem'},
    'ت': {'type': 'second_person', 'meaning': 'you'},
    'نا': {'type': 'first_person_plural', 'meaning': 'we'},
    'ها': {'type': 'attached_feminine', 'meaning': 'her/it_fem'},
    'ه': {'type': 'attached_masculine', 'meaning': 'him/it_masc'},
    }

    variants = []

        for pronoun, data in pronouns.items():
            # إضافة الضمير للفعل
    extended_verb = verb + pronoun

            # تحليل المقاطع الجديدة
    syllable_analysis = self._analyze_syllables(extended_verb)

    variants.append()
    {
    'verb_with_pronoun': extended_verb,
    'pronoun': pronoun,
    'pronoun_type': data['type'],
    'meaning': data['meaning'],
    'syllable_patterns': syllable_analysis['patterns'],
    'syllable_count': syllable_analysis['count'],
    'complexity_score': syllable_analysis['complexity'],
    }
    )

    return variants

    def _classify_verb_type(self, verb_form: str) -> str:
    """تصنيف نوع الفعل"""

        if verb_form == "I":
    return "ثلاثي_مجرد"
        elif verb_form in ["II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]:
    return "ثلاثي_مزيد"
        elif verb_form == "Q1":
    return "رباعي_مجرد"
        elif verb_form.startswith("Q"):
    return "رباعي_مزيد"
        else:
    return "غير_محدد"

    def _analyze_phonological_features()
    self, word: str, base_features: List[str]
    ) -> List[str]:
    """تحليل الخصائص الصوتية"""

    features = base_features.copy()

        # البحث عن الهمزة
        if 'ء' in word:
    features.append("همزة")

        # البحث عن الحروف المفخمة
    emphatic_letters = ['ص', 'ض', 'ط', 'ظ']
        if any(letter in word for letter in emphatic_letters):
    features.append("تفخيم")

        # البحث عن حروف القلقلة
    qalqala_letters = ['ق', 'ط', 'ب', 'ج', 'د']
        if any(letter in word for letter in qalqala_letters):
    features.append("قلقلة")

        # البحث عن المد
        if any(vowel in word for vowel in self.long_vowels):
    features.append("مد_طبيعي")

    return features

    def _determine_semantic_field(self, verb_form: str) -> str:
    """تحديد المجال الدلالي"""

    semantic_fields = {
    "I": "الأفعال_الأساسية",
    "II": "التكثير_والسببية",
    "III": "المشاركة_والمحاولة",
    "IV": "السببية_والتعدية",
    "V": "التدرج_والانفعال",
    "VI": "المشاركة_المتبادلة",
    "VII": "الانفعال_والتأثر",
    "VIII": "الاكتساب_والطلب",
    "IX": "الألوان_والعيوب",
    "X": "الطلب_والاستدعاء",
    "Q1": "الحركة_والصوت",
    "Q2": "التدرج_الرباعي",
    }

    return semantic_fields.get(verb_form, "غير_محدد")

    def generate_comprehensive_analysis()
    self,
    roots: List[List[str]],
    verb_forms: List[str] = None,
    include_sources: bool = True,
    include_pronouns: bool = False) -> Dict[str, Any]:
    """
    توليد تحليل شامل لمجموعة من الجذور والأفعال

    Args:
    roots: قائمة الجذور
    verb_forms: قائمة صيغ الأفعال (إذا لم تُحدد، ستُستخدم جميع الصيغ)
    include_sources: تشمل تحليل المصادر
    include_pronouns: تشمل الضمائر المتصلة

    Returns:
    Dict: التحليل الشامل
    """

        if verb_forms is None:
    verb_forms = list(self.verb_forms.keys())

    results = {
    'total_combinations': 0,
    'verb_analysis': [],
    'source_analysis': [],
    'statistics': {
    'syllable_patterns': Counter(),
    'complexity_distribution': [],
    'phonological_processes': Counter(),
    'verb_types': Counter(),
    },
    'coverage_analysis': {
    'covered_patterns': set(),
    'new_patterns': set(),
    'complexity_range': {'min': float('inf'), 'max': 0},
    },
    }

    logger.info(f"بدء التحليل الشامل لـ {len(roots)} جذر و {len(verb_forms)} صيغة")

        for root in roots:
            for verb_form in verb_forms:
                try:
                    # تحليل الفعل
    verb_analysis = self.generate_verb_syllable_patterns()
    root, verb_form, include_pronouns
    )
    results['verb_analysis'].append(verb_analysis)

                    # تحديث الإحصائيات
    self._update_statistics(results['statistics'], verb_analysis)
    self._update_coverage_analysis()
    results['coverage_analysis'], verb_analysis
    )

                    # تحليل المصادر
                    if include_sources and verb_form in self.source_patterns:
    source_analyses = self.generate_source_syllable_patterns()
    root, verb_form
    )
    results['source_analysis'].extend(source_analyses)

                        for source_analysis in source_analyses:
    self._update_statistics()
    results['statistics'], source_analysis, is_source=True
    )

    results['total_combinations'] += 1

                except Exception as e:
    logger.warning()
    f"خطأ في تحليل الجذر {root} مع الصيغة {verb_form: {str(e)}}"
    )

        # حساب الإحصائيات النهائية
    results['final_statistics'] = self._calculate_final_statistics(results)

    logger.info(f"اكتمل التحليل الشامل: {results['total_combinations']} توافيق")

    return results

    def _update_statistics(self, stats: Dict, analysis: Dict, is_source: bool = False):
    """تحديث الإحصائيات"""

        # تحديث أنماط المقاطع
    patterns = ()
    analysis.get('syllable_structure', [])
            if not is_source
            else analysis.get('analyzed_patterns', [])
    )
        for pattern in patterns:
    stats['syllable_patterns'][pattern] += 1

        # تحديث توزيع التعقيد
    complexity = analysis.get('complexity_score', 0)
    stats['complexity_distribution'].append(complexity)

        # تحديث العمليات الصوتية
    processes = analysis.get('phonological_processes', [])
        for process in processes:
    stats['phonological_processes'][process] += 1

        # تحديث أنواع الأفعال
        if not is_source:
    verb_type = analysis.get('morphological_analysis', {}).get()
    'type', 'غير_محدد'
    )
    stats['verb_types'][verb_type] += 1

    def _update_coverage_analysis(self, coverage: Dict, analysis: Dict):
    """تحديث تحليل التغطية"""

    patterns = analysis.get('syllable_structure', [])
        for pattern in patterns:
    coverage['covered_patterns'].add(pattern)

    complexity = analysis.get('complexity_score', 0)
    coverage['complexity_range']['min'] = min()
    coverage['complexity_range']['min'], complexity
    )
    coverage['complexity_range']['max'] = max()
    coverage['complexity_range']['max'], complexity
    )

    def _calculate_final_statistics(self, results: Dict) -> Dict:
    """حساب الإحصائيات النهائية"""

    stats = results['statistics']

    return {
    'total_unique_patterns': len(stats['syllable_patterns']),
    'most_common_patterns': stats['syllable_patterns'].most_common(10),
    'average_complexity': ()
    sum(stats['complexity_distribution'])
    / len(stats['complexity_distribution'])
                if stats['complexity_distribution']
                else 0
    ),
    'complexity_range': {
    'min': ()
    min(stats['complexity_distribution'])
                    if stats['complexity_distribution']
                    else 0
    ),
    'max': ()
    max(stats['complexity_distribution'])
                    if stats['complexity_distribution']
                    else 0
    ),
    },
    'most_common_processes': stats['phonological_processes'].most_common(5),
    'verb_type_distribution': dict(stats['verb_types']),
    'coverage_percentage': len(results['coverage_analysis']['covered_patterns'])
    / 14
    * 100,  # من أصل 14 نمط مقطعي
    }

    def export_analysis_report()
    self, results: Dict, filename: str = "arabic_verb_analysis_report.json"
    ):
    """تصدير تقرير التحليل"""

        # تحويل Counter و set إلى تنسيقات قابلة للتسلسل
    exportable_results = self._make_serializable(results)

        with open(filename, 'w', encoding='utf 8') as f:
    json.dump(exportable_results, f, ensure_ascii=False, indent=2)

    logger.info(f"تم تصدير تقرير التحليل إلى {filename}")

    def _make_serializable(self, obj):
    """تحويل الكائنات إلى تنسيق قابل للتسلسل"""

        if isinstance(obj, dict):
    return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
    return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Counter):
    return dict(obj)
        elif isinstance(obj, set):
    return list(obj)
        else:
    return obj


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING - العرض التوضيحي والاختبار
# ═══════════════════════════════════════════════════════════════════════════════════


def main():
    """التشغيل الرئيسي للنظام"""

    print("🔤 نظام توليد أنماط المقاطع الصوتية للأفعال والمصادر العربية")
    print("=" * 80)

    # إنشاء النظام
    system = ArabicVerbMorphologySystem()

    # جذور للاختبار
    test_roots = [
    ['ك', 'ت', 'ب'],  # كتب
    ['د', 'ر', 'س'],  # درس
    ['ع', 'ل', 'م'],  # علم
    ['س', 'أ', 'ل'],  # سأل (مع همزة)
    ['ق', 'و', 'ل'],  # قول (معتل)
    ['د', 'ح', 'ر', 'ج'],  # دحرج (رباعي)
    ]

    # صيغ للاختبار
    test_forms = ['I', 'II', 'IV', 'V', 'X', 'Q1']

    print("🔬 اختبار الأفعال:")
    print(" " * 40)

    # اختبار فعل واحد مفصل
    sample_analysis = system.generate_verb_syllable_patterns()
    ['ك', 'ت', 'ب'], 'X', include_pronouns=True
    )

    print(f"📝 مثال تفصيلي: {sample_analysis['constructed_verb']}")
    print(f"   الجذر: {'} - '.join(sample_analysis['root'])}")
    print(f"   الصيغة: {sample_analysis['verb_form']} ({sample_analysis['pattern'])}")
    print(f"   المقاطع: {'} - '.join(sample_analysis['syllable_structure'])}")
    print(f"   درجة التعقيد: {sample_analysis['complexity_score']}")
    print()
    f"   العمليات الصوتية: {', '.join(sample_analysis['phonological_processes'])}"
    )

    if sample_analysis['pronoun_variants']:
    print("   🔸 متغيرات مع الضمائر:")
        for variant in sample_analysis['pronoun_variants'][:3]:  # أول 3 فقط
    print()
    f"      {variant['verb_with_pronoun']} ({variant['pronoun_type']}) - {' '.join(variant['syllable_patterns'])}"
    )

    print("\n🔬 اختبار المصادر:")
    print(" " * 40)

    # اختبار المصادر
    source_analyses = system.generate_source_syllable_patterns(['ع', 'ل', 'م'], 'II')

    for source in source_analyses:
    print(f"📚 المصدر: {source['source_word']}")
    print(f"   النمط: {source['syllable_pattern']}")
    print(f"   المقاطع المحللة: {'} - '.join(source['analyzed_patterns'])}")
    print(f"   النوع: {source['morphological_analysis']['source_type']}")
    print(f"   الخصائص: {', '.join(source['phonological_features'])}")
    print()

    print("🔬 التحليل الشامل:")
    print(" " * 40)

    # التحليل الشامل
    comprehensive_results = system.generate_comprehensive_analysis()
    test_roots[:3],  # أول 3 جذور فقط للسرعة
    test_forms[:4],  # أول 4 صيغ
    include_sources=True,
    include_pronouns=False)

    print(f"📊 إجمالي التوافيق: {comprehensive_results['total_combinations']}")
    print()
    f"🎯 الأنماط المتفردة: {comprehensive_results['final_statistics']['total_unique_patterns']}"
    )
    print()
    f"📈 متوسط التعقيد: {comprehensive_results['final_statistics']['average_complexity']:.2f}"
    )
    print()
    f"📋 نسبة التغطية: {comprehensive_results['final_statistics']['coverage_percentage']:.1f}%"
    )

    print("\n🏆 الأنماط الأكثر شيوعاً:")
    for pattern, count in comprehensive_results['final_statistics'][
    'most_common_patterns'
    ]:
    print(f"   {pattern}: {count} مرة")

    print("\n🔧 العمليات الصوتية الأكثر تطبيقاً:")
    for process, count in comprehensive_results['final_statistics'][
    'most_common_processes'
    ]:
    print(f"   {process}: {count} مرة")

    # تصدير التقرير
    system.export_analysis_report(comprehensive_results)

    print("\n✅ تم إكمال التحليل الشامل وتصدير التقرير!")
    print("📄 ملف التقرير: arabic_verb_analysis_report.json")


if __name__ == "__main__":
    main()

