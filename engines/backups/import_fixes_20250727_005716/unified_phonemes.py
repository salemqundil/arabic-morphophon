#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED ARABIC PHONEME SYSTEM
Single source of truth for all Arabic phonemes and diacritics
This replaces ALL other phonology systems in the project
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


from dataclasses import dataclass  # noqa: F401
from typing import Dict, List, Optional
from enum import Enum  # noqa: F401


class PhonemeType(Enum):
    """نوع الفونيم"""

    CONSONANT = "consonant"  # صامت
    VOWEL = "vowel"  # صائت
    DIACRITIC = "diacritic"  # حركة


class Place(Enum):
    """مكان النطق"""

    BILABIAL = "bilabial"  # شفوي
    LABIODENTAL = "labiodental"  # شفوي أسناني
    DENTAL = "dental"  # أسناني
    ALVEOLAR = "alveolar"  # لثوي
    POSTALVEOLAR = "postalveolar"  # ما بعد لثوي
    PALATAL = "palatal"  # غاري
    VELAR = "velar"  # طبقي
    UVULAR = "uvular"  # لهوي
    PHARYNGEAL = "pharyngeal"  # بلعومي
    GLOTTAL = "glottal"  # حنجري
    LABIOVELAR = "labiovelar"  # شفوي طبقي


class Manner(Enum):
    """طريقة النطق"""

    STOP = "stop"  # انسدادي
    FRICATIVE = "fricative"  # احتكاكي
    AFFRICATE = "affricate"  # انسدادي احتكاكي
    NASAL = "nasal"  # أنفي
    LATERAL = "lateral"  # جانبي
    TRILL = "trill"  # مكرر
    APPROXIMANT = "approximant"  # تقريبي


@dataclass
class Phoneme:
    """تعريف الفونيم الموحد"""

    arabic_char: str  # الحرف العربي
    ipa: str  # الرمز الصوتي الدولي
    phoneme_type: PhonemeType  # نوع الفونيم
    place: Optional[Place] = None  # مكان النطق
    manner: Optional[Manner] = None  # طريقة النطق
    voiced: Optional[bool] = None  # مجهور/مهموس
    emphatic: Optional[bool] = None  # مفخم/مرقق
    long: Optional[bool] = None  # طويل/قصير (للأصوات)
    frequency: float = 1.0  # تكرار الاستخدام


class UnifiedArabicPhonemes:
    """النظام الموحد للفونيمات العربية"""

    def __init__(self):  # type: ignore[no-untyped def]
    """تهيئة النظام الموحد"""
    self._initialize_phonemes()
    self._initialize_diacritics()
    self._create_lookup_tables()

    def _initialize_phonemes(self):  # type: ignore[no-untyped def]
    """تهيئة قائمة الفونيمات الأساسية"""

        # الحروف الساكنة - Consonants
    self.consonants = [
            # انسدادية - Stops
    Phoneme(
    "ب",
    "b",
    PhonemeType.CONSONANT,
    Place.BILABIAL,
    Manner.STOP,
    True,
    False,
    frequency=0.8,
    ),
    Phoneme(
    "ت",
    "t",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.STOP,
    False,
    False,
    frequency=0.9,
    ),
    Phoneme(
    "ط",
    "tˤ",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.STOP,
    False,
    True,
    frequency=0.6,
    ),
    Phoneme(
    "د",
    "d",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.STOP,
    True,
    False,
    frequency=0.7,
    ),
    Phoneme(
    "ض",
    "dˤ",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.STOP,
    True,
    True,
    frequency=0.3,
    ),
    Phoneme(
    "ك",
    "k",
    PhonemeType.CONSONANT,
    Place.VELAR,
    Manner.STOP,
    False,
    False,
    frequency=0.8,
    ),
    Phoneme(
    "ق",
    "q",
    PhonemeType.CONSONANT,
    Place.UVULAR,
    Manner.STOP,
    False,
    False,
    frequency=0.5,
    ),
    Phoneme(
    "ء",
    "ʔ",
    PhonemeType.CONSONANT,
    Place.GLOTTAL,
    Manner.STOP,
    False,
    False,
    frequency=0.4,
    ),
            # احتكاكية - Fricatives
    Phoneme(
    "ف",
    "f",
    PhonemeType.CONSONANT,
    Place.LABIODENTAL,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.6,
    ),
    Phoneme(
    "ث",
    "θ",
    PhonemeType.CONSONANT,
    Place.DENTAL,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.3,
    ),
    Phoneme(
    "ذ",
    "ð",
    PhonemeType.CONSONANT,
    Place.DENTAL,
    Manner.FRICATIVE,
    True,
    False,
    frequency=0.4,
    ),
    Phoneme(
    "س",
    "s",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.8,
    ),
    Phoneme(
    "ز",
    "z",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.FRICATIVE,
    True,
    False,
    frequency=0.5,
    ),
    Phoneme(
    "ص",
    "sˤ",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.FRICATIVE,
    False,
    True,
    frequency=0.5,
    ),
    Phoneme(
    "ظ",
    "ðˤ",
    PhonemeType.CONSONANT,
    Place.DENTAL,
    Manner.FRICATIVE,
    True,
    True,
    frequency=0.2,
    ),
    Phoneme(
    "ش",
    "ʃ",
    PhonemeType.CONSONANT,
    Place.POSTALVEOLAR,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.6,
    ),
    Phoneme(
    "خ",
    "x",
    PhonemeType.CONSONANT,
    Place.UVULAR,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.4,
    ),
    Phoneme(
    "غ",
    "ɣ",
    PhonemeType.CONSONANT,
    Place.UVULAR,
    Manner.FRICATIVE,
    True,
    False,
    frequency=0.3,
    ),
    Phoneme(
    "ح",
    "ħ",
    PhonemeType.CONSONANT,
    Place.PHARYNGEAL,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.6,
    ),
    Phoneme(
    "ع",
    "ʕ",
    PhonemeType.CONSONANT,
    Place.PHARYNGEAL,
    Manner.FRICATIVE,
    True,
    False,
    frequency=0.5,
    ),
    Phoneme(
    "ه",
    "h",
    PhonemeType.CONSONANT,
    Place.GLOTTAL,
    Manner.FRICATIVE,
    False,
    False,
    frequency=0.7,
    ),
            # انسدادية احتكاكية - Affricates
    Phoneme(
    "ج",
    "dʒ",
    PhonemeType.CONSONANT,
    Place.POSTALVEOLAR,
    Manner.AFFRICATE,
    True,
    False,
    frequency=0.6,
    ),
            # أنفية - Nasals
    Phoneme(
    "م",
    "m",
    PhonemeType.CONSONANT,
    Place.BILABIAL,
    Manner.NASAL,
    True,
    False,
    frequency=0.9,
    ),
    Phoneme(
    "ن",
    "n",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.NASAL,
    True,
    False,
    frequency=1.0,
    ),
            # جانبية - Laterals
    Phoneme(
    "ل",
    "l",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.LATERAL,
    True,
    False,
    frequency=1.0,
    ),
            # مكررة - Trills
    Phoneme(
    "ر",
    "r",
    PhonemeType.CONSONANT,
    Place.ALVEOLAR,
    Manner.TRILL,
    True,
    False,
    frequency=0.9,
    ),
            # تقريبية - Approximants
    Phoneme(
    "و",
    "w",
    PhonemeType.CONSONANT,
    Place.LABIOVELAR,
    Manner.APPROXIMANT,
    True,
    False,
    frequency=0.8,
    ),
    Phoneme(
    "ي",
    "j",
    PhonemeType.CONSONANT,
    Place.PALATAL,
    Manner.APPROXIMANT,
    True,
    False,
    frequency=0.8,
    ),
    ]

        # الأصوات - Vowels
    self.vowels = [
            # أصوات قصيرة - Short vowels
    Phoneme("َ", "a", PhonemeType.VOWEL, long=False, frequency=1.0),  # فتحة
    Phoneme("ِ", "i", PhonemeType.VOWEL, long=False, frequency=0.9),  # كسرة
    Phoneme("ُ", "u", PhonemeType.VOWEL, long=False, frequency=0.8),  # ضمة
            # أصوات طويلة - Long vowels
    Phoneme("ا", "aː", PhonemeType.VOWEL, long=True, frequency=1.0),  # ألف
    Phoneme("ي", "iː", PhonemeType.VOWEL, long=True, frequency=0.8),  # ياء مد
    Phoneme("و", "uː", PhonemeType.VOWEL, long=True, frequency=0.7),  # واو مد
            # أصوات مركبة - Diphthongs
    Phoneme("أي", "aj", PhonemeType.VOWEL, long=False, frequency=0.3),  # أي
    Phoneme("أو", "aw", PhonemeType.VOWEL, long=False, frequency=0.3),  # أو
    ]

    def _initialize_diacritics(self):  # type: ignore[no-untyped def]
    """تهيئة الحركات والعلامات"""

    self.diacritics = [
            # حركات أساسية - Basic diacritics
    Phoneme("َ", "a", PhonemeType.DIACRITIC, frequency=1.0),  # فتحة
    Phoneme("ِ", "i", PhonemeType.DIACRITIC, frequency=0.9),  # كسرة
    Phoneme("ُ", "u", PhonemeType.DIACRITIC, frequency=0.8),  # ضمة
    Phoneme("ْ", "", PhonemeType.DIACRITIC, frequency=0.7),  # سكون
    Phoneme("ً", "an", PhonemeType.DIACRITIC, frequency=0.6),  # تنوين فتح
    Phoneme("ٍ", "in", PhonemeType.DIACRITIC, frequency=0.5),  # تنوين كسر
    Phoneme("ٌ", "un", PhonemeType.DIACRITIC, frequency=0.4),  # تنوين ضم
    Phoneme("ّ", "", PhonemeType.DIACRITIC, frequency=0.8),  # شدة
    Phoneme("ٰ", "aː", PhonemeType.DIACRITIC, frequency=0.2),  # ألف خنجرية
    Phoneme("ٱ", "a", PhonemeType.DIACRITIC, frequency=0.3),  # ألف وصل
    ]

    def _create_lookup_tables(self):  # type: ignore[no-untyped def]
    """إنشاء جداول البحث السريع"""

        # جمع كل الفونيمات
    all_phonemes = self.consonants + self.vowels + self.diacritics

        # جدول البحث بالحرف العربي
    self.char_to_phoneme = {p.arabic_char: p for p in all_phonemes}

        # جدول البحث بالـ IPA
    self.ipa_to_phoneme = {p.ipa: p for p in all_phonemes}

        # جدول الخصائص
    self.emphatic_consonants = {
    p.arabic_char for p in self.consonants if p.emphatic
    }
    self.voiced_consonants = {p.arabic_char for p in self.consonants if p.voiced}
    self.fricatives = {
    p.arabic_char for p in self.consonants if p.manner == Manner.FRICATIVE
    }
    self.stops = {p.arabic_char for p in self.consonants if p.manner == Manner.STOP}

        # الحروف الشمسية والقمرية
    self.sun_letters = {
    "ت",
    "ث",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ل",
    "ن",
    }
    self.moon_letters = {
    "ا",
    "ب",
    "ج",
    "ح",
    "خ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ك",
    "م",
    "ه",
    "و",
    "ي",
    }

    def get_phoneme(self, char: str) -> Optional[Phoneme]:
    """الحصول على الفونيم للحرف المعطى"""
    return self.char_to_phoneme.get(char)

    def get_phoneme_by_ipa(self, ipa: str) -> Optional[Phoneme]:
    """الحصول على الفونيم بواسطة IPA"""
    return self.ipa_to_phoneme.get(ipa)

    def is_emphatic(self, char: str) -> bool:
    """فحص إذا كان الحرف مفخماً"""
    return char in self.emphatic_consonants

    def is_voiced(self, char: str) -> bool:
    """فحص إذا كان الحرف مجهوراً"""
    return char in self.voiced_consonants

    def is_sun_letter(self, char: str) -> bool:
    """فحص إذا كان حرفاً شمسياً"""
    return char in self.sun_letters

    def is_moon_letter(self, char: str) -> bool:
    """فحص إذا كان حرفاً قمرياً"""
    return char in self.moon_letters

    def extract_phonemes(self, text: str) -> List[Phoneme]:
    """استخراج الفونيمات من النص"""
    phonemes = []
        for char in text:
    phoneme = self.get_phoneme(char)
            if phoneme:
    phonemes.append(phoneme)
    return phonemes

    def get_phonetic_features(self, char: str) -> Dict[str, any]:
    """الحصول على الخصائص الصوتية للحرف"""
    phoneme = self.get_phoneme(char)
        if not phoneme:
    return {}

    return {
    "arabic_char": phoneme.arabic_char,
    "ipa": phoneme.ipa,
    "type": phoneme.phoneme_type.value,
    "place": phoneme.place.value if phoneme.place else None,
    "manner": phoneme.manner.value if phoneme.manner else None,
    "voiced": phoneme.voiced,
    "emphatic": phoneme.emphatic,
    "long": phoneme.long,
    "frequency": phoneme.frequency,
    }

    def analyze_emphatic_spreading(self, text: str) -> List[bool]:
    """تحليل انتشار التفخيم"""
    chars = list(text)
    spreading = [False] * len(chars)

        for i, char in enumerate(chars):
            if self.is_emphatic(char):
                # انتشار التفخيم للأمام والخلف
    start = max(0, i - 2)
    end = min(len(chars), i + 3)
                for j in range(start, end):
    spreading[j] = True

    return spreading

    def get_syllable_structure(self, text: str) -> List[str]:
    """تحليل البنية المقطعية CV"""
    structure = []
        for char in text:
    phoneme = self.get_phoneme(char)
            if phoneme:
                if phoneme.phoneme_type == PhonemeType.CONSONANT:
    structure.append("C")
                elif phoneme.phoneme_type == PhonemeType.VOWEL:
    structure.append("V")
    return structure


# إنشاء المثيل الموحد العالمي
UNIFIED_PHONEMES = UnifiedArabicPhonemes()


def get_unified_phonemes() -> UnifiedArabicPhonemes:
    """الحصول على النظام الموحد للفونيمات"""
    return UNIFIED_PHONEMES


# دوال مساعدة سريعة
def get_phoneme(char: str) -> Optional[Phoneme]:
    """دالة سريعة للحصول على الفونيم"""
    return UNIFIED_PHONEMES.get_phoneme(char)


def is_emphatic(char: str) -> bool:
    """دالة سريعة للفحص عن التفخيم"""
    return UNIFIED_PHONEMES.is_emphatic(char)


def extract_phonemes(text: str) -> List[Phoneme]:
    """دالة سريعة لاستخراج الفونيمات"""
    return UNIFIED_PHONEMES.extract_phonemes(text)


def get_phonetic_features(char: str) -> Dict[str, any]:
    """دالة سريعة للحصول على الخصائص الصوتية"""
    return UNIFIED_PHONEMES.get_phonetic_features(char)


if __name__ == "__main__":
    # اختبار النظام الموحد
    print("🔤 النظام الموحد للفونيمات العربية")
    print("=" * 40)

    test_text = "كتاب"
    print(f"📝 النص: {test_text}")

    phonemes = extract_phonemes(test_text)
    print(f"🔤 الفونيمات: {[p.arabic_char for p} in phonemes]}")
    print(f"🎵 IPA: {[p.ipa for p} in phonemes]}")

    for char in test_text:
    features = get_phonetic_features(char)
        if features:
    print(f"   {char}: {features}")

    spreading = UNIFIED_PHONEMES.analyze_emphatic_spreading(test_text)
    print(f"🎯 انتشار التفخيم: {spreading}")

    structure = UNIFIED_PHONEMES.get_syllable_structure(test_text)
    print(f"🏗️ البنية المقطعية: {structure}")

    print("\n✅ النظام الموحد جاهز للاستخدام!")
