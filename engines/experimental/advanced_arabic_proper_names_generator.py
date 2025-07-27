#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام توليد أسماء الأعلام العربية المتقدم,
    Advanced Arabic Proper Names Generator,
    يستخدم قاعدة بيانات المقاطع الصوتية (22,218 مقطع) لتوليد:
- أسماء الأشخاص (ذكور وإناث)
- أسماء الأماكن (مدن، دول، معالم طبيعية)
- مع مراعاة الخصائص الصوتية والدلالية والثقافية,
    المطور: نظام الذكاء الاصطناعي العربي,
    Developer: Arabic AI System,
    التاريخ: 2025,
    Date: 2025
"""

import re
    import random
    from typing import Dict, List, Optional, Any
    from dataclasses import dataclass, field
    from enum import Enum
    import logging

# Configure logging,
    logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# PROPER NAMES CLASSIFICATION SYSTEM - نظام تصنيف أسماء الأعلام
# ═══════════════════════════════════════════════════════════════════════════════════


class ProperNameCategory(Enum):
    """فئات أسماء الأعلام"""

    PERSON_MALE = "person_male"  # أسماء الذكور,
    PERSON_FEMALE = "person_female"  # أسماء الإناث,
    PLACE_CITY = "place_city"  # أسماء المدن,
    PLACE_COUNTRY = "place_country"  # أسماء الدول,
    PLACE_NATURAL = "place_natural"  # المعالم الطبيعية,
    PLACE_REGION = "place_region"  # أسماء المناطق,
    class NamePattern(Enum):
    """أنماط الأسماء الصوتية"""

    CV = "CV"  # صامت + صائت,
    CVC = "CVC"  # صامت + صائت + صامت,
    CVCV = "CVCV"  # صامت + صائت + صامت + صائت,
    CVCVC = "CVCVC"  # فعلان، محمد,
    CVVCV = "CVVCV"  # فاعل، سامي,
    CVCVCV = "CVCVCV"  # فعلة، سميرة,
    CVCCVC = "CVCCVC"  # فعلان، عثمان,
    CVVCVC = "CVVCVC"  # فاعلة، عائشة


@dataclass,
    class NameTemplate:
    """قالب أسماء"""

    category: ProperNameCategory,
    pattern: NamePattern,
    syllable_structure: List[str]
    phonetic_constraints: Dict[str, Any]
    semantic_features: List[str]
    frequency: float = 1.0,
    cultural_significance: str = "common"


@dataclass,
    class GeneratedName:
    """اسم مولد"""

    name: str,
    category: ProperNameCategory,
    pattern: NamePattern,
    syllables: List[str]
    phonetic_analysis: Dict[str, Any]
    semantic_meaning: str,
    cultural_context: str,
    authenticity_score: float,
    historical_template: Optional[str] = None,
    examples: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC ONOMASTICS ANALYZER - محلل علم الأسماء العربية
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicOnomastics:
    """محلل علم الأسماء العربية المتقدم"""

    def __init__(self):

        # تحميل قواعد علم الأسماء,
    self._load_name_roots()
    self._load_semantic_patterns()
    self._load_phonetic_rules()
    self._load_cultural_templates()

    def _load_name_roots(self):
    """تحميل جذور الأسماء العربية"""

    self.name_roots = {
            # جذور أسماء الذكور
    'male_roots': {
    'حمد': {
    'meaning': 'الشكر والثناء',
    'derivatives': ['أحمد', 'محمد', 'حمدان', 'حامد'],
    },
    'عبد': {
    'meaning': 'العبادة والخضوع',
    'derivatives': ['عبدالله', 'عبدالرحمن', 'عبدالعزيز'],
    },
    'نصر': {
    'meaning': 'الانتصار والغلبة',
    'derivatives': ['نصر', 'ناصر', 'نصير', 'منصور'],
    },
    'كرم': {
    'meaning': 'الجود والسخاء',
    'derivatives': ['كريم', 'أكرم', 'كرام'],
    },
    'علم': {
    'meaning': 'المعرفة والحكمة',
    'derivatives': ['علي', 'أعلم', 'عالم', 'علام'],
    },
    'صبر': {
    'meaning': 'التحمل والثبات',
    'derivatives': ['صابر', 'صبور', 'مصبور'],
    },
    'شجع': {
    'meaning': 'الإقدام والبسالة',
    'derivatives': ['شجاع', 'شجعان', 'أشجع'],
    },
    'سلم': {
    'meaning': 'الأمان والسكينة',
    'derivatives': ['سالم', 'سليم', 'مسلم', 'سلامة'],
    },
    },
            # جذور أسماء الإناث
    'female_roots': {
    'فطم': {
    'meaning': 'الذكاء والحكمة',
    'derivatives': ['فاطمة', 'فطوم', 'فطيمة'],
    },
    'عيش': {
    'meaning': 'الحياة والسعادة',
    'derivatives': ['عائشة', 'عيشة', 'معيشة'],
    },
    'خدج': {'meaning': 'البكر والطهارة', 'derivatives': ['خديجة', 'خادجة']},
    'زين': {
    'meaning': 'الجمال والحسن',
    'derivatives': ['زينب', 'زينة', 'زين'],
    },
    'أمن': {
    'meaning': 'الأمان والطمأنينة',
    'derivatives': ['آمنة', 'أمينة', 'أمان'],
    },
    'رحم': {
    'meaning': 'الرأفة والحنان',
    'derivatives': ['رحمة', 'راحمة', 'رحيمة'],
    },
    'سعد': {
    'meaning': 'الفرح والبهجة',
    'derivatives': ['سعاد', 'سعدة', 'سعيدة'],
    },
    'صفو': {
    'meaning': 'النقاء والصفاء',
    'derivatives': ['صفية', 'صافية', 'صفاء'],
    },
    },
            # جذور أسماء الأماكن
    'place_roots': {
    'قدس': {
    'meaning': 'الطهارة والتقديس',
    'derivatives': ['القدس', 'المقدس'],
    },
    'شرق': {
    'meaning': 'الاتجاه الشرقي',
    'derivatives': ['الشرق', 'مشرق', 'شرقية'],
    },
    'نجد': {
    'meaning': 'المرتفع من الأرض',
    'derivatives': ['نجد', 'النجود'],
    },
    'حجز': {
    'meaning': 'المانع والحاجز',
    'derivatives': ['الحجاز', 'حجازية'],
    },
    'يمن': {'meaning': 'البركة واليمن', 'derivatives': ['اليمن', 'يمنية']},
    'رفح': {'meaning': 'الرفعة والعلو', 'derivatives': ['رفح', 'الرافحة']},
    'بصر': {
    'meaning': 'الإبصار والنظر',
    'derivatives': ['البصرة', 'بصراوية'],
    },
    'كوف': {
    'meaning': 'التجمع والاجتماع',
    'derivatives': ['الكوفة', 'كوفية'],
    },
    },
    }

    def _load_semantic_patterns(self):
    """تحميل أنماط المعاني الدلالية"""

    self.semantic_patterns = {
    'theophoric': {  # أسماء الثناء والتعبد
    'patterns': [
    'عبد + {divine_name}',
    '{virtue} + الدين',
    '{virtue} + الله',
    ],
    'divine_names': [
    'الرحمن',
    'الرحيم',
    'الكريم',
    'الرؤوف',
    'الودود',
    'الصبور',
    ],
    'virtues': ['نور', 'بهاء', 'جمال', 'كمال', 'صلاح', 'فلاح'],
    },
    'descriptive': {  # أسماء وصفية
    'male_descriptors': ['شجاع', 'كريم', 'حكيم', 'رؤوف', 'صبور', 'حليم'],
    'female_descriptors': [
    'جميلة',
    'حسناء',
    'رقيقة',
    'لطيفة',
    'رحيمة',
    'حنونة',
    ],
    'place_descriptors': [
    'الجديدة',
    'القديمة',
    'الكبرى',
    'الصغرى',
    'العليا',
    'السفلى',
    ],
    },
    'nature_based': {  # أسماء مستوحاة من الطبيعة
    'natural_elements': ['نهر', 'بحر', 'جبل', 'وادي', 'صحراء', 'واحة'],
    'celestial': ['نجم', 'قمر', 'شمس', 'كوكب', 'فجر', 'ضحى'],
    'plants': ['وردة', 'ياسمين', 'زهرة', 'نرجس', 'ريحان', 'أزهار'],
    },
    }

    def _load_phonetic_rules(self):
    """تحميل قواعد النطق الصوتي"""

    self.phonetic_rules = {
    'consonant_clusters': {
    'allowed': ['نت', 'نك', 'مب', 'لج', 'رس', 'شر', 'قت'],
    'difficult': ['قف', 'طع', 'حخ', 'خح', 'ظص', 'ضط'],
    'forbidden': ['ءء', 'ججع', 'ححخ'],
    },
    'vowel_patterns': {
    'male_endings': ['َ', 'ِ', 'ُ', 'ان', 'ين'],  # مفتوحة أو مكسورة
    'female_endings': ['ة', 'اء', 'ى', 'ان'],  # تاء مربوطة، ألف ممدودة
    'place_endings': ['ة', 'ية', 'ان', 'ستان'],  # تاء مربوطة، ياء النسبة
    },
    'stress_patterns': {
    'penultimate_stress': [
    'CV-CV CV',
    'CVC-CV CV',
    ],  # النبرة على ما قبل الأخير
    'ultimate_stress': ['CV-CV CVC', 'CV-CVC CVC'],  # النبرة على الأخير
    'antepenultimate_stress': [
    'CV-CV-CV CV'
    ],  # النبرة على ما قبل ما قبل الأخير
    },
    }

    def _load_cultural_templates(self):
    """تحميل القوالب الثقافية"""

    self.cultural_templates = {
    'classical_arabic': {
    'male_patterns': ['فاعل', 'فعيل', 'فعال', 'فعلان', 'مفعول'],
    'female_patterns': ['فاعلة', 'فعيلة', 'فعال', 'فعلى', 'مفعولة'],
    'examples': {
    'فاعل': ['عامر', 'سامر', 'كامل', 'ناصر'],
    'فعيل': ['كريم', 'حليم', 'رحيم', 'عليم'],
    },
    },
    'geographical_patterns': {
    'arabian_peninsula': ['نجد', 'حجاز', 'تهامة', 'عسير'],
    'mesopotamian': ['بغداد', 'البصرة', 'الكوفة', 'سامراء'],
    'levantine': ['دمشق', 'حلب', 'حمص', 'اللاذقية'],
    'maghrebi': ['فاس', 'مراكش', 'تونس', 'القيروان'],
    },
    'tribal_names': {
    'noble_tribes': ['قريش', 'هاشم', 'أمية', 'تميم'],
    'geographical_tribes': ['حجازي', 'نجدي', 'شامي', 'عراقي'],
    },
    }

    def derive_meaning(self, name: str, category: ProperNameCategory) -> str:
    """استخراج معنى الاسم"""

        # البحث في جذور الأسماء,
    if category in [ProperNameCategory.PERSON_MALE]:
            for root, info in self.name_roots['male_roots'].items():
                if root in name or any(deriv in name for deriv in info['derivatives']):
    return info['meaning']

        elif category in [ProperNameCategory.PERSON_FEMALE]:
            for root, info in self.name_roots['female_roots'].items():
                if root in name or any(deriv in name for deriv in info['derivatives']):
    return info['meaning']

        elif category.value.startswith('place_'):
            for root, info in self.name_roots['place_roots'].items():
                if root in name or any(deriv in name for deriv in info['derivatives']):
    return info['meaning']

        # تحليل الأنماط الدلالية,
    if 'عبد' in name:
    return 'التعبد والخضوع لله'
        elif name.endswith('ية'):
    return 'النسبة والانتماء'
        elif name.endswith('ان'):
    return 'المكان أو الزمان'

    return 'اسم عربي أصيل'

    def analyze_phonetic_structure(self, name: str) -> Dict[str, Any]:
    """تحليل البنية الصوتية للاسم"""

    analysis = {
    'length': len(name),
    'syllable_count': self._count_syllables(name),
    'stress_pattern': self._identify_stress_pattern(name),
    'consonant_clusters': self._find_consonant_clusters(name),
    'vowel_pattern': self._extract_vowel_pattern(name),
    'phonetic_difficulty': self._assess_phonetic_difficulty(name),
    'euphony_score': self._calculate_euphony_score(name),
    }

    return analysis,
    def _count_syllables(self, name: str) -> int:
    """عد المقاطع الصوتية"""
        # تقدير بسيط بناءً على الحركات,
    vowels = ['َ', 'ِ', 'ُ', 'ا', 'و', 'ي']
    return max(1, len([c for c in name if c in vowels]))

    def _identify_stress_pattern(self, name: str) -> str:
    """تحديد نمط النبرة"""
    syllable_count = self._count_syllables(name)

        if syllable_count <= 2:
    return 'ultimate'  # النبرة على المقطع الأخير,
    elif syllable_count == 3:
    return 'penultimate'  # النبرة على ما قبل الأخير,
    else:
    return 'antepenultimate'  # النبرة على ما قبل ما قبل الأخير,
    def _find_consonant_clusters(self, name: str) -> List[str]:
    """العثور على تجمعات الصوامت"""
    clusters = []
    consonants = 'بتثجحخدذرزسشصضطظعغفقكلمنهويء'

        for i in range(len(name) - 1):
            if name[i] in consonants and name[i + 1] in consonants:
    clusters.append(name[i : i + 2])

    return clusters,
    def _extract_vowel_pattern(self, name: str) -> str:
    """استخراج نمط الحركات"""
    vowels = {'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ا': 'aa', 'ي': 'ii', 'و': 'uu'}
    pattern = []

        for char in name:
            if char in vowels:
    pattern.append(vowels[char])

    return ' '.join(pattern)

    def _assess_phonetic_difficulty(self, name: str) -> float:
    """تقييم صعوبة النطق"""
    difficulty = 0.0

        # فحص التجمعات الصعبة,
    clusters = self._find_consonant_clusters(name)
        for cluster in clusters:
            if cluster in self.phonetic_rules['consonant_clusters']['difficult']:
    difficulty += 0.3,
    elif cluster in self.phonetic_rules['consonant_clusters']['forbidden']:
    difficulty += 0.5

        # فحص الطول,
    if len(name) > 10:
    difficulty += 0.2,
    return min(1.0, difficulty)

    def _calculate_euphony_score(self, name: str) -> float:
    """حساب نقاط جمال الصوت"""
    euphony = 1.0

        # تنوع الأصوات,
    unique_sounds = len(set(name))
    variety_bonus = min(0.3, unique_sounds / len(name) * 0.5)
    euphony += variety_bonus

        # توازن الصوامت والصوائت,
    consonants = len([c for c in name if c in 'بتثجحخدذرزسشصضطظعغفقكلمنهويء'])
    vowels = len([c for c in name if c in 'َُِاويً'])

        if vowels > 0:
    balance = 1 - abs(consonants - vowels) / max(consonants, vowels)
    euphony += balance * 0.2

        # خصم الصعوبة,
    difficulty = self._assess_phonetic_difficulty(name)
    euphony -= difficulty * 0.3,
    return max(0.0, min(2.0, euphony))

    def has_negative_connotation(self, name: str) -> bool:
    """فحص الدلالات السلبية"""
    negative_roots = {
    'موت',
    'قتل',
    'حرب',
    'مرض',
    'فقر',
    'حزن',
    'ضعف',
    'ذل',
    'خسر',
    'هزم',
    'كسر',
    'تعب',
    'ضيق',
    'ظلم',
    'غضب',
    }

        for root in negative_roots:
            if root in name:
    return True,
    return False,
    def suggest_similar_authentic_names()
    self, name: str, category: ProperNameCategory
    ) -> List[str]:
    """اقتراح أسماء أصيلة مشابهة"""

    authentic_names = {
    ProperNameCategory.PERSON_MALE: [
    'محمد',
    'أحمد',
    'علي',
    'حسن',
    'حسين',
    'عبدالله',
    'عبدالرحمن',
    'عمر',
    'عثمان',
    'خالد',
    'سعد',
    'فيصل',
    'عبدالعزيز',
    ],
    ProperNameCategory.PERSON_FEMALE: [
    'فاطمة',
    'عائشة',
    'خديجة',
    'زينب',
    'رقية',
    'أم كلثوم',
    'صفية',
    'مريم',
    'آمنة',
    'سارة',
    'ليلى',
    'سعاد',
    'نورا',
    ],
    ProperNameCategory.PLACE_CITY: [
    'مكة',
    'المدينة',
    'الرياض',
    'جدة',
    'الدمام',
    'الطائف',
    'أبها',
    'القاهرة',
    'دمشق',
    'بغداد',
    'بيروت',
    'تونس',
    'الرباط',
    ],
    }

    candidates = authentic_names.get(category, [])
    similar_names = []

        for candidate in candidates:
            # حساب التشابه الصوتي,
    similarity = self._calculate_phonetic_similarity(name, candidate)
            if similarity > 0.5:
    similar_names.append((candidate, similarity))

        # ترتيب حسب التشابه وإرجاع أفضل 5,
    similar_names.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in similar_names[:5]]

    def _calculate_phonetic_similarity(self, name1: str, name2: str) -> float:
    """حساب التشابه الصوتي"""
        # التشابه في الطول,
    len_sim = 1 - abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)

        # التشابه في الأحرف,
    set1, set2 = set(name1), set(name2)
    char_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

        # التشابه في البداية والنهاية,
    start_sim = 1 if name1[:1] == name2[:1] else 0,
    end_sim = 1 if name1[-1:] == name2[-1:] else 0

        # متوسط مرجح,
    similarity = 0.3 * len_sim + 0.4 * char_sim + 0.15 * start_sim + 0.15 * end_sim,
    return similarity


# ═══════════════════════════════════════════════════════════════════════════════════
# ADVANCED PROPER NAMES GENERATOR - مولد أسماء الأعلام المتقدم
# ═══════════════════════════════════════════════════════════════════════════════════


class AdvancedArabicProperNamesGenerator:
    """مولد أسماء الأعلام العربية المتقدم"""

    def __init__(self, syllables_database: Optional[List[Dict]] = None):

    self.syllables_db = syllables_database or self._load_syllables_database()
    self.onomastics = ArabicOnomastics()

        # تحميل قوالب الأسماء,
    self._load_name_templates()

    logger.info(f"تم تحميل {len(self.syllables_db)} مقطع صوتي")
    logger.info(f"تم تحميل {len(self.name_templates)} قالب اسم")

    def _load_syllables_database(self) -> List[Dict]:
    """تحميل قاعدة بيانات المقاطع"""

        try:
            # محاولة تحميل من النظام الشامل
    from comprehensive_arabic_verb_syllable_generator import ()
    ComprehensiveArabicVerbSyllableGenerator)

    syllable_generator = ComprehensiveArabicVerbSyllableGenerator()
    logger.info("تحميل قاعدة بيانات المقاطع من النظام الشامل...")

            # توليد قاعدة البيانات الشاملة,
    syllable_database = ()
    syllable_generator.generate_comprehensive_syllable_database()
    )

    logger.info(f"تم تحميل {len(syllable_database)} مقطع من النظام الشامل")
    return syllable_database,
    except ImportError:
    logger.warning("النظام الشامل غير متاح، استخدام قاعدة بيانات متقدمة")
    return self._create_enhanced_syllable_database()

    def _create_enhanced_syllable_database(self) -> List[Dict]:
    """إنشاء قاعدة بيانات مقاطع محسنة للأسماء"""

    syllables = []

        # الأحرف العربية مع تصنيفها,
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
    'uvular': ['ق', 'غ', 'خ'],
    'pharyngeal': ['ح', 'ع'],
    'fricative': ['ث', 'ذ', 'ش', 'ز'],
    'glottal': ['ء', 'ه'],
    }

        # الحركات مع تصنيفها,
    vowels = {
    'short': ['َ', 'ِ', 'ُ'],
    'long': ['ا', 'ي', 'و'],
    'diphthongs': ['اي', 'او', 'وي'],
    }

        # توليد مقاطع CV,
    for category, cons_list in consonants.items():
            for consonant in cons_list:
                for vowel in vowels['short']:
    syllables.append()
    {
    'syllable': consonant + vowel,
    'pattern': 'CV',
    'consonants': [consonant],
    'vowels': [vowel],
    'consonant_type': category,
    'weight': 'light',
    'name_suitable': True,
    'frequency': 0.8 if category == 'common' else 0.5,
    }
    )

        # توليد مقاطع CVC,
    end_consonants = ['ن', 'ل', 'ر', 'م', 'ت', 'د', 'س', 'ك', 'ي', 'ب']
        for category, cons_list in consonants.items():
            for c1 in cons_list[:8]:  # تحديد العدد,
    for vowel in vowels['short']:
                    for c2 in end_consonants:
    syllables.append()
    {
    'syllable': c1 + vowel + c2,
    'pattern': 'CVC',
    'consonants': [c1, c2],
    'vowels': [vowel],
    'consonant_type': category,
    'weight': 'medium',
    'name_suitable': True,
    'frequency': 0.6 if category == 'common' else 0.3,
    }
    )

        # توليد مقاطع طويلة CVV,
    for category, cons_list in consonants.items():
            for consonant in cons_list[:10]:
                for long_vowel in vowels['long']:
    syllables.append()
    {
    'syllable': consonant + long_vowel,
    'pattern': 'CVV',
    'consonants': [consonant],
    'vowels': [long_vowel],
    'consonant_type': category,
    'weight': 'heavy',
    'name_suitable': True,
    'frequency': 0.4 if category == 'common' else 0.2,
    }
    )

        # مقاطع خاصة بالأسماء,
    name_specific_syllables = [
            # مقاطع أسماء ذكور
    {
    'syllable': 'مح',
    'pattern': 'CVC',
    'name_type': 'male',
    'meaning': 'محو/طهارة',
    },
    {
    'syllable': 'أح',
    'pattern': 'CVC',
    'name_type': 'male',
    'meaning': 'الحمد',
    },
    {
    'syllable': 'عبد',
    'pattern': 'CVCC',
    'name_type': 'male',
    'meaning': 'العبادة',
    },
    {
    'syllable': 'خا',
    'pattern': 'CV',
    'name_type': 'male',
    'meaning': 'الخير',
    },
    {
    'syllable': 'نا',
    'pattern': 'CV',
    'name_type': 'male',
    'meaning': 'النيل',
    },
            # مقاطع أسماء إناث
    {
    'syllable': 'فا',
    'pattern': 'CV',
    'name_type': 'female',
    'meaning': 'العظمة',
    },
    {
    'syllable': 'عا',
    'pattern': 'CV',
    'name_type': 'female',
    'meaning': 'الحياة',
    },
    {
    'syllable': 'زي',
    'pattern': 'CV',
    'name_type': 'female',
    'meaning': 'الزينة',
    },
    {
    'syllable': 'خدي',
    'pattern': 'CCV',
    'name_type': 'female',
    'meaning': 'البكارة',
    },
    {
    'syllable': 'مة',
    'pattern': 'CV',
    'name_type': 'female',
    'meaning': 'التأنيث',
    },
            # مقاطع أسماء أماكن
    {
    'syllable': 'مك',
    'pattern': 'CVC',
    'name_type': 'place',
    'meaning': 'المكان المقدس',
    },
    {
    'syllable': 'بغ',
    'pattern': 'CVC',
    'name_type': 'place',
    'meaning': 'العطاء',
    },
    {
    'syllable': 'دم',
    'pattern': 'CVC',
    'name_type': 'place',
    'meaning': 'القدم/العراقة',
    },
    {
    'syllable': 'قد',
    'pattern': 'CVC',
    'name_type': 'place',
    'meaning': 'التقديس',
    },
    ]

        for special in name_specific_syllables:
    special.update()
    {
    'weight': 'light',
    'frequency': 1.0,
    'name_suitable': True,
    'is_authentic': True,
    }
    )
    syllables.append(special)

    logger.info(f"تم إنشاء {len(syllables)} مقطع صوتي محسن للأسماء")
    return syllables,
    def _load_name_templates(self):
    """تحميل قوالب الأسماء"""

    self.name_templates = {
            # قوالب أسماء الذكور,
    ProperNameCategory.PERSON_MALE: [
    NameTemplate()
    category=ProperNameCategory.PERSON_MALE,
    pattern=NamePattern.CVCVC,
    syllable_structure=['CVC', 'CVC'],
    phonetic_constraints={
    'max_syllables': 3,
    'min_syllables': 2,
    'preferred_endings': ['ِ', 'َ', 'ان', 'ين'],
    'avoid_feminine_endings': True,
    },
    semantic_features=['masculine', 'strength', 'honor'],
    frequency=1.0,
    cultural_significance='classical'),
    NameTemplate()
    category=ProperNameCategory.PERSON_MALE,
    pattern=NamePattern.CVVCV,
    syllable_structure=['CVV', 'CV'],
    phonetic_constraints={
    'max_syllables': 3,
    'preferred_patterns': ['فاعل', 'كاتب', 'عامر'],
    'long_vowel_position': 'medial',
    },
    semantic_features=['descriptive', 'active', 'professional'],
    frequency=0.8,
    cultural_significance='descriptive'),
    ],
            # قوالب أسماء الإناث,
    ProperNameCategory.PERSON_FEMALE: [
    NameTemplate()
    category=ProperNameCategory.PERSON_FEMALE,
    pattern=NamePattern.CVCVCV,
    syllable_structure=['CV', 'CV', 'CV'],
    phonetic_constraints={
    'max_syllables': 4,
    'min_syllables': 2,
    'required_endings': ['ة', 'اء', 'ى'],
    'feminine_markers': True,
    },
    semantic_features=['feminine', 'beauty', 'grace'],
    frequency=1.0,
    cultural_significance='traditional'),
    NameTemplate()
    category=ProperNameCategory.PERSON_FEMALE,
    pattern=NamePattern.CVVCVC,
    syllable_structure=['CVV', 'CVC'],
    phonetic_constraints={
    'max_syllables': 3,
    'preferred_patterns': ['فاعلة', 'كاتبة'],
    'long_vowel_position': 'initial',
    },
    semantic_features=['descriptive', 'noble', 'virtue'],
    frequency=0.7,
    cultural_significance='aristocratic'),
    ],
            # قوالب أسماء المدن,
    ProperNameCategory.PLACE_CITY: [
    NameTemplate()
    category=ProperNameCategory.PLACE_CITY,
    pattern=NamePattern.CVCVC,
    syllable_structure=['CVC', 'CVC'],
    phonetic_constraints={
    'max_syllables': 3,
    'geographical_indicators': True,
    'common_city_endings': ['ة', 'اد', 'ان'],
    },
    semantic_features=['urban', 'settlement', 'commerce'],
    frequency=1.0,
    cultural_significance='geographical'),
    NameTemplate()
    category=ProperNameCategory.PLACE_CITY,
    pattern=NamePattern.CVVCV,
    syllable_structure=['CVV', 'CV'],
    phonetic_constraints={
    'max_syllables': 3,
    'historical_patterns': True,
    },
    semantic_features=['ancient', 'cultural', 'trading'],
    frequency=0.6,
    cultural_significance='historical'),
    ],
            # قوالب أسماء الدول,
    ProperNameCategory.PLACE_COUNTRY: [
    NameTemplate()
    category=ProperNameCategory.PLACE_COUNTRY,
    pattern=NamePattern.CVCVCV,
    syllable_structure=['CV', 'CV', 'CV'],
    phonetic_constraints={
    'max_syllables': 4,
    'min_syllables': 3,
    'country_suffixes': ['ستان', 'ية', 'ان'],
    'formal_tone': True,
    },
    semantic_features=['nation', 'sovereignty', 'territory'],
    frequency=1.0,
    cultural_significance='political'),
    ],
            # قوالب المعالم الطبيعية,
    ProperNameCategory.PLACE_NATURAL: [
    NameTemplate()
    category=ProperNameCategory.PLACE_NATURAL,
    pattern=NamePattern.CVCV,
    syllable_structure=['CV', 'CV'],
    phonetic_constraints={
    'max_syllables': 3,
    'nature_indicators': True,
    'descriptive_elements': ['وادي', 'جبل', 'نهر'],
    },
    semantic_features=['natural', 'landscape', 'geographical'],
    frequency=1.0,
    cultural_significance='environmental'),
    ],
    }

    def generate_names()
    self,
    category: ProperNameCategory,
    count: int = 20,
    specific_meaning: Optional[str] = None) -> List[GeneratedName]:
    """توليد أسماء أعلام لفئة محددة"""

    logger.info(f"بدء توليد {count} اسم من فئة {category.value}")

    templates = self.name_templates.get(category, [])
        if not templates:
    logger.warning(f"لا توجد قوالب للفئة {category.value}")
    return []

    generated_names = []
    attempts = 0,
    max_attempts = count * 15,
    while len(generated_names) < count and attempts < max_attempts:
    attempts += 1

            # اختيار قالب عشوائي,
    template = random.choice(templates)

            # توليد اسم مرشح,
    candidate_name = self._generate_candidate_name(template)

            if candidate_name:
                # تحليل وتقييم الاسم,
    generated_name = self._evaluate_and_create_name()
    candidate_name, template, specific_meaning
    )

                if generated_name and generated_name.authenticity_score >= 0.4:
                    # تجنب التكرار,
    if not any()
    gn.name == generated_name.name for gn in generated_names
    ):
    generated_names.append(generated_name)

                        if len(generated_names) % 5 == 0:
    logger.info(f"تم توليد {len(generated_names)} اسم...")

        # ترتيب حسب جودة الأسماء,
    generated_names.sort(key=lambda x: x.authenticity_score, reverse=True)

    logger.info()
    f"تم الانتهاء من توليد {len(generated_names)} اسم من فئة {category.value}"
    )
    return generated_names,
    def _generate_candidate_name(self, template: NameTemplate) -> Optional[str]:
    """توليد اسم مرشح بناءً على القالب"""

    syllable_structure = template.syllable_structure

        # اختيار المقاطع المناسبة,
    name_syllables = []
        for i, syllable_pattern in enumerate(syllable_structure):

            # تصفية المقاطع المناسبة,
    suitable_syllables = self._filter_suitable_syllables()
    syllable_pattern, template, i == 0, i == len(syllable_structure) - 1
    )

            if not suitable_syllables:
    return None,
    chosen_syllable = random.choice(suitable_syllables)
    name_syllables.append(chosen_syllable['syllable'])

        if not name_syllables:
    return None

        # تجميع الاسم,
    candidate_name = ''.join(name_syllables)

        # تطبيق قواعد تاريخية وثقافية,
    candidate_name = self._apply_cultural_modifications(candidate_name, template)

        # تطبيق قواعد صوتية,
    candidate_name = self._apply_phonetic_rules(candidate_name, template)

    return candidate_name,
    def _filter_suitable_syllables()
    self, pattern: str, template: NameTemplate, is_first: bool, is_last: bool
    ) -> List[Dict]:
    """تصفية المقاطع المناسبة للنمط والقالب"""

    suitable = []
    constraints = template.phonetic_constraints,
    for syllable in self.syllables_db:
            # فحص النمط الأساسي,
    if syllable.get('pattern') != pattern:
    continue

            # فحص الملاءمة للأسماء,
    if not syllable.get('name_suitable', True):
    continue

            # قيود الموقع (أول/أخير)
            if is_first and 'preferred_initials' in constraints:
    syl_text = syllable['syllable']
                if syl_text and syl_text[0] not in constraints['preferred_initials']:
    continue,
    if is_last and 'required_endings' in constraints:
    syl_text = syllable['syllable']
                if not any()
    syl_text.endswith(ending)
                    for ending in constraints['required_endings']
    ):
    continue

            # قيود النوع (ذكر/أنثى)
            if template.category == ProperNameCategory.PERSON_FEMALE:
                if constraints.get('feminine_markers') and is_last:
    syl_text = syllable['syllable']
                    if not any()
    syl_text.endswith(marker) for marker in ['ة', 'اء', 'ى', 'ان']
    ):
    continue,
    elif template.category == ProperNameCategory.PERSON_MALE:
                if constraints.get('avoid_feminine_endings') and is_last:
    syl_text = syllable['syllable']
                    if any(syl_text.endswith(marker) for marker in ['ة', 'اء']):
    continue

            # قيود الأماكن,
    elif template.category.value.startswith('place_'):
                if 'geographical_indicators' in constraints and is_last:
                    # تفضيل نهايات جغرافية,
    pass

    suitable.append(syllable)

    return suitable,
    def _apply_cultural_modifications(self, name: str, template: NameTemplate) -> str:
    """تطبيق تعديلات ثقافية على الاسم"""

        # تطبيق القوالب التاريخية,
    if ()
    template.category == ProperNameCategory.PERSON_MALE,
    and random.random() < 0.3
    ):
            # أسماء الثناء (عبد + صفة إلهية)
            if len(name) <= 4:
    divine_names = ['الرحمن', 'الرحيم', 'الكريم', 'الودود']
    divine_name = random.choice(divine_names)
    return f"عبد{divine_name}"

            # أسماء مركبة دينية,
    elif random.random() < 0.5:
    religious_prefixes = ['نور', 'بهاء', 'جمال']
    religious_suffixes = ['الدين', 'الإسلام']
                if random.random() < 0.5:
    prefix = random.choice(religious_prefixes)
    suffix = random.choice(religious_suffixes)
    return f"{prefix {suffix}}"

        # تطبيق لواحق الأماكن,
    elif ()
    template.category == ProperNameCategory.PLACE_CITY and random.random() < 0.4
    ):
    city_suffixes = ['ية', 'ان', 'اباد']
    suffix = random.choice(city_suffixes)
    return name + suffix,
    elif ()
    template.category == ProperNameCategory.PLACE_COUNTRY,
    and random.random() < 0.6
    ):
    country_suffixes = ['ستان', 'ية', 'ان']
    suffix = random.choice(country_suffixes)
    return name + suffix,
    elif ()
    template.category == ProperNameCategory.PLACE_NATURAL,
    and random.random() < 0.5
    ):
    nature_prefixes = ['وادي', 'جبل', 'نهر', 'بحر']
    prefix = random.choice(nature_prefixes)
    return f"{prefix {name}}"

    return name,
    def _apply_phonetic_rules(self, name: str, template: NameTemplate) -> str:
    """تطبيق قواعد صوتية للتحسين"""

        # إزالة التكرارات المفرطة,
    name = re.sub(r'(.)\1{2,}', r'\1\1', name)

        # تبسيط التجمعات الصعبة,
    difficult_clusters = ['قف', 'طع', 'حخ', 'خح']
        for cluster in difficult_clusters:
            if cluster in name:
                # استبدال بتجمع أسهل,
    easier_alternatives = {'قف': 'قد', 'طع': 'طر', 'حخ': 'حر', 'خح': 'خر'}
    name = name.replace()
    cluster, easier_alternatives.get(cluster, cluster[0])
    )

        # تحسين التدفق الصوتي,
    name = self._improve_euphony(name)

    return name,
    def _improve_euphony(self, name: str) -> str:
    """تحسين جمال الصوت"""

        # إضافة حركات للتوضيح,
    if len(name) >= 3 and not any(c in name for c in 'َُِاويً'):
            # إضافة حركة في الوسط,
    mid_pos = len(name) // 2,
    vowels = ['َ', 'ِ', 'ُ']
    name = name[:mid_pos] + random.choice(vowels) + name[mid_pos:]

    return name,
    def _evaluate_and_create_name()
    self, name: str, template: NameTemplate, specific_meaning: Optional[str] = None
    ) -> Optional[GeneratedName]:
    """تقييم وإنشاء كائن الاسم المولد"""

        # التحقق من القيود الأساسية,
    if not self._validate_basic_constraints(name, template):
    return None

        # فحص الدلالات السلبية,
    if self.onomastics.has_negative_connotation(name):
    return None

        # تحليل صوتي شامل,
    phonetic_analysis = self.onomastics.analyze_phonetic_structure(name)

        # فحص الصعوبة الصوتية,
    if phonetic_analysis['phonetic_difficulty'] > 0.6:
    return None

        # استخراج المعنى,
    semantic_meaning = self.onomastics.derive_meaning(name, template.category)
        if specific_meaning and specific_meaning not in semantic_meaning:
    semantic_meaning = f"{semantic_meaning} - {specific_meaning}}"

        # حساب نقاط الأصالة,
    authenticity_score = self._calculate_authenticity_score()
    name, template, phonetic_analysis
    )

        # إنشاء كائن الاسم,
    generated_name = GeneratedName()
    name=name,
    category=template.category,
    pattern=template.pattern,
    syllables=self._breakdown_name_syllables(name),
    phonetic_analysis=phonetic_analysis,
    semantic_meaning=semantic_meaning,
    cultural_context=template.cultural_significance,
    authenticity_score=authenticity_score,
    historical_template=self._identify_historical_template(name),
    examples=self.onomastics.suggest_similar_authentic_names()
    name, template.category
    ))

    return generated_name,
    def _validate_basic_constraints(self, name: str, template: NameTemplate) -> bool:
    """التحقق من القيود الأساسية"""

    constraints = template.phonetic_constraints

        # فحص الطول,
    if len(name) < 2 or len(len(name) -> 15) > 15:
    return False

        # فحص عدد المقاطع,
    syllable_count = self.onomastics._count_syllables(name)
    max_syllables = constraints.get('max_syllables', 4)
    min_syllables = constraints.get('min_syllables', 1)

        if syllable_count > max_syllables or syllable_count < min_syllables:
    return False

        # فحص النهايات المطلوبة,
    if 'required_endings' in constraints:
            if not any()
    name.endswith(ending) for ending in constraints['required_endings']
    ):
    return False,
    return True,
    def _calculate_authenticity_score()
    self, name: str, template: NameTemplate, phonetic_analysis: Dict[str, Any]
    ) -> float:
    """حساب نقاط الأصالة والجودة"""

    score = 0.5  # نقطة البداية

        # نقاط الجمال الصوتي,
    euphony_score = phonetic_analysis.get('euphony_score', 1.0)
    score += euphony_score * 0.3

        # نقاط التشابه مع الأسماء الأصيلة,
    similar_names = self.onomastics.suggest_similar_authentic_names()
    name, template.category
    )
        if similar_names:
    similarity_bonus = len(similar_names) * 0.1,
    score += min(0.3, similarity_bonus)

        # نقاط السهولة الصوتية,
    difficulty = phonetic_analysis.get('phonetic_difficulty', 0.0)
    score += (1.0 - difficulty) * 0.2

        # نقاط الأهمية الثقافية,
    if template.cultural_significance in ['classical', 'traditional']:
    score += 0.1,
    elif template.cultural_significance in ['historical', 'religious']:
    score += 0.15

        # نقاط التوازن الصوتي,
    if phonetic_analysis.get('syllable_count', 1) in [2, 3]:  # طول مثالي,
    score += 0.1,
    return min(1.0, max(0.0, score))

    def _breakdown_name_syllables(self, name: str) -> List[str]:
    """تقسيم الاسم إلى مقاطع"""

    syllables = []
    current_syllable = ""

    vowels = {'َ', 'ِ', 'ُ', 'ا', 'ي', 'و', 'ً', 'ٌ', 'ٍ'}
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

        for i, char in enumerate(name):
    current_syllable += char

            # إنهاء المقطع عند الوصول لصائت يليه صامت,
    if char in vowels and i < len(name) - 1:
    next_char = name[i + 1]
                if next_char in consonants:
    syllables.append(current_syllable)
    current_syllable = ""

        # إضافة المقطع الأخير,
    if current_syllable:
    syllables.append(current_syllable)

    return syllables if syllables else [name]

    def _identify_historical_template(self, name: str) -> Optional[str]:
    """تحديد القالب التاريخي للاسم"""

        if name.startswith('عبد'):
    return 'theophoric'  # أسماء الثناء,
    elif ' الدين' in name or ' الإسلام' in name:
    return 'religious_compound'  # مركبات دينية,
    elif name.startswith(('وادي', 'جبل', 'نهر', 'بحر')):
    return 'geographical_descriptive'  # وصفية جغرافية,
    elif name.endswith(('ستان', 'اباد')):
    return 'persian_influence'  # تأثير فارسي,
    elif re.search(r'(ان|ين|ية)$', name):
    return 'nisba_form'  # صيغة النسب,
    return None,
    def generate_by_meaning()
    self, meaning: str, category: ProperNameCategory, count: int = 10
    ) -> List[GeneratedName]:
    """توليد أسماء بمعنى محدد"""

    logger.info(f"توليد أسماء بمعنى '{meaning' من} فئة {category.value}}")

        # البحث عن جذور مناسبة للمعنى,
    relevant_roots = self._find_meaning_related_roots(meaning, category)

        if not relevant_roots:
    logger.warning(f"لم يتم العثور على جذور مناسبة للمعنى '{meaning}")
    return self.generate_names(category, count, meaning)

        # توليد أسماء مستهدفة,
    targeted_names = []
    attempts = 0,
    max_attempts = count * 20,
    while len(targeted_names) < count and attempts < max_attempts:
    attempts += 1

            # اختيار جذر مناسب,
    root = random.choice(relevant_roots)

            # توليد اسم بناءً على الجذر,
    targeted_name = self._generate_name_from_root(root, category, meaning)

            if targeted_name and targeted_name.authenticity_score >= 0.5:
                # تجنب التكرار,
    if not any(tn.name == targeted_name.name for tn in targeted_names):
    targeted_names.append(targeted_name)

    logger.info(f"تم توليد {len(targeted_names)} اسم بمعنى '{meaning}")
    return targeted_names,
    def _find_meaning_related_roots()
    self, meaning: str, category: ProperNameCategory
    ) -> List[Dict]:
    """البحث عن جذور متعلقة بالمعنى"""

        # خريطة المعاني للجذور,
    meaning_map = {
    'الشجاعة': ['شجع', 'بطل', 'قوي', 'عز'],
    'الحكمة': ['حكم', 'علم', 'فهم', 'عقل'],
    'الجمال': ['جمل', 'حسن', 'زين', 'بهي'],
    'الرحمة': ['رحم', 'رأف', 'حنن', 'عطف'],
    'القوة': ['قوي', 'عز', 'غلب', 'قدر'],
    'السلام': ['سلم', 'أمن', 'طمن', 'سكن'],
    'النور': ['نور', 'ضوء', 'شرق', 'أشرق'],
    'الماء': ['مو', 'نهر', 'بحر', 'عين'],
    'الجبل': ['جبل', 'طود', 'علو', 'رفع'],
    }

    relevant_roots = []

        # البحث المباشر,
    for concept, roots in meaning_map.items():
            if meaning in concept or concept in meaning:
                for root in roots:
    relevant_roots.append({'root': root, 'meaning': concept})

        # البحث في قاعدة بيانات الجذور,
    if category == ProperNameCategory.PERSON_MALE:
            for root, info in self.onomastics.name_roots['male_roots'].items():
                if meaning in info['meaning'] or any()
    meaning in deriv for deriv in info['derivatives']
    ):
    relevant_roots.append({'root': root, 'meaning': info['meaning']})

        elif category == ProperNameCategory.PERSON_FEMALE:
            for root, info in self.onomastics.name_roots['female_roots'].items():
                if meaning in info['meaning'] or any()
    meaning in deriv for deriv in info['derivatives']
    ):
    relevant_roots.append({'root': root, 'meaning': info['meaning']})

    return relevant_roots,
    def _generate_name_from_root()
    self, root_info: Dict, category: ProperNameCategory, meaning: str
    ) -> Optional[GeneratedName]:
    """توليد اسم من جذر محدد"""

    root = root_info['root']

        # اختيار قالب مناسب,
    templates = self.name_templates.get(category, [])
        if not templates:
    return None,
    template = random.choice(templates)

        # البحث عن مقاطع تحتوي على الجذر,
    root_syllables = [
    syl,
    for syl in self.syllables_db,
    if root in syl.get('syllable', '')
    or any(root_char in syl.get('syllable', '') for root_char in root)
    ]

        if not root_syllables:
    return None

        # بناء اسم باستخدام الجذر,
    chosen_root_syllable = random.choice(root_syllables)

        # إكمال الاسم بمقاطع مكملة,
    remaining_structure = ()
    template.syllable_structure[1:]
            if len(template.syllable_structure) > 1,
    else []
    )

    name_syllables = [chosen_root_syllable['syllable']]

        for syllable_pattern in remaining_structure:
    suitable_syllables = self._filter_suitable_syllables()
    syllable_pattern,
    template,
    False,
    syllable_pattern == remaining_structure[ 1])

            if suitable_syllables:
    chosen_syllable = random.choice(suitable_syllables)
    name_syllables.append(chosen_syllable['syllable'])

        # تجميع الاسم,
    candidate_name = ''.join(name_syllables)

        # تطبيق التعديلات,
    candidate_name = self._apply_cultural_modifications(candidate_name, template)
    candidate_name = self._apply_phonetic_rules(candidate_name, template)

        # تقييم الاسم,
    return self._evaluate_and_create_name(candidate_name, template, meaning)

    def generate_comprehensive_analysis()
    self) -> Dict[ProperNameCategory, List[GeneratedName]]:
    """تحليل شامل لجميع فئات الأسماء"""

    results = {}

    categories_to_analyze = [
    ProperNameCategory.PERSON_MALE,
    ProperNameCategory.PERSON_FEMALE,
    ProperNameCategory.PLACE_CITY,
    ProperNameCategory.PLACE_COUNTRY,
    ProperNameCategory.PLACE_NATURAL,
    ]

        for category in categories_to_analyze:
    logger.info(f"تحليل فئة {category.value...}")
    category_results = self.generate_names(category, count=15)
    results[category] = category_results,
    return results,
    def print_comprehensive_report()
    self, results: Dict[ProperNameCategory, List[GeneratedName]]
    ):
    """طباعة تقرير شامل للنتائج"""

    print("\n" + "═" * 80)
    print("🎯 تقرير شامل لتوليد أسماء الأعلام العربية")
    print("═" * 80)

    total_generated = sum(len(names) for names in results.values())
    total_authentic = sum()
    len([n for n in names if n.authenticity_score > 0.7])
            for names in results.values()
    )

    print("\n📊 إحصائيات عامة:")
    print(f"   • إجمالي الأسماء المولدة: {total_generated}")
    print(f"   • الأسماء عالية الجودة: {total_authentic}")
    print(f"   • معدل الجودة: {total_authentic/total_generated*100:.1f}%")

        for category, names in results.items():
            if not names:
    continue,
    print(f"\n▶ {category.value.upper().replace('_',} ' ')} ({len(names) اسم):}")
    print(" " * 60)

            # تصنيف حسب الجودة,
    high_quality = [n for n in names if n.authenticity_score > 0.7]
    medium_quality = [n for n in names if 0.5 <= n.authenticity_score <= 0.7]

    print(f"   🥇 عالي الجودة: {len(high_quality)}")
    print(f"   🥈 متوسط الجودة: {len(medium_quality)}")

            # أفضل الأسماء,
    top_names = sorted(names, key=lambda x: x.authenticity_score, reverse=True)[
    :8
    ]

    print("\n   🌟 أفضل الأسماء المولدة:")
            for i, name in enumerate(top_names, 1):
    quality_indicator = ()
    "🥇"
                    if name.authenticity_score > 0.7,
    else "🥈" if name.authenticity_score > 0.5 else "🥉"
    )

    print()
    f"      {i}. {name.name:12} {quality_indicator} جودة: {name.authenticity_score:.2f}"
    )
    print(f"         📝 المقاطع: {'} + '.join(name.syllables)}")
    print(f"         🔤 النمط: {name.pattern.value}")
    print(f"         💭 المعنى: {name.semantic_meaning}")

                if name.historical_template:
    print(f"         🏛️  القالب: {name.historical_template}")

                if name.examples:
    print(f"         🎯 أسماء مشابهة: {', '.join(name.examples[:3])}")

    print()

    print("═" * 80)


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMO AND TESTING FUNCTIONS - وظائف العرض والاختبار
# ═══════════════════════════════════════════════════════════════════════════════════


def demo_proper_names_generation():
    """عرض توضيحي شامل لتوليد أسماء الأعلام"""

    print("🚀 مولد أسماء الأعلام العربية المتقدم")
    print("Advanced Arabic Proper Names Generator")
    print("=" * 70)

    # إنشاء المولد,
    generator = AdvancedArabicProperNamesGenerator()

    # تشغيل التحليل الشامل,
    results = generator.generate_comprehensive_analysis()

    # طباعة التقرير الشامل,
    generator.print_comprehensive_report(results)

    # عرض أمثلة على التوليد بمعاني محددة,
    print("\n" + "═" * 70)
    print("🎯 أمثلة على التوليد بمعاني محددة")
    print("═" * 70)

    specific_meanings = [
    ('الشجاعة', ProperNameCategory.PERSON_MALE),
    ('الجمال', ProperNameCategory.PERSON_FEMALE),
    ('الماء', ProperNameCategory.PLACE_NATURAL),
    ]

    for meaning, category in specific_meanings:
    print(f"\n🔍 توليد أسماء بمعنى '{meaning}' - فئة {category.value}:")

    targeted_names = generator.generate_by_meaning(meaning, category, 5)

        for i, name in enumerate(targeted_names, 1):
    print(f"   {i}. {name.name} - جودة: {name.authenticity_score:.2f}")
    print(f"      معنى: {name.semantic_meaning}")

    return generator, results,
    if __name__ == "__main__":
    # تشغيل العرض التوضيحي,
    demo_proper_names_generation()

