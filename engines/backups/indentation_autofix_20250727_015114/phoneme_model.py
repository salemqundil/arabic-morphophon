"""
Phoneme Model
نموذج الفونيم

Represents Arabic phonemes and their phonological analysis.
Based on analysis of existing phoneme analyzer functions.
"""

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


from dataclasses import dataclass  # noqa: F401
from enum import Enum  # noqa: F401
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod  # noqa: F401


class PhonemeType(Enum):
    """أنواع الفونيمات"""

    CONSONANT = "consonant"
    SHORT_VOWEL = "short_vowel"
    LONG_VOWEL = "long_vowel"
    TANWEEN = "tanween"
    SUKUN = "sukun"
    GEMINATION = "gemination"
    SPACE = "space"
    DIGIT = "digit"
    UNKNOWN = "unknown"


class PlaceOfArticulation(Enum):
    """مخارج الحروف"""

    BILABIAL = "bilabial"  # شفوي
    DENTAL = "dental"  # أسناني
    ALVEOLAR = "alveolar"  # لثوي
    POST_ALVEOLAR = "post alveolar"  # ما بعد لثوي
    PALATAL = "palatal"  # غاري
    VELAR = "velar"  # طبقي
    UVULAR = "uvular"  # لهوي
    PHARYNGEAL = "pharyngeal"  # حلقي
    GLOTTAL = "glottal"  # حنجري


class MannerOfArticulation(Enum):
    """طرائق النطق"""

    STOP = "stop"  # انفجاري
    FRICATIVE = "fricative"  # احتكاكي
    AFFRICATE = "affricate"  # انفجاري احتكاكي
    NASAL = "nasal"  # أنفي
    LATERAL = "lateral"  # جانبي
    TRILL = "trill"  # مكرر
    TAP = "tap"  # نقري
    APPROXIMANT = "approximant"  # تقريبي


class VoicingType(Enum):
    """الجهر والهمس"""

    VOICED = "voiced"  # مجهور
    VOICELESS = "voiceless"  # مهموس


class MorphClass(Enum):
    """التصنيف الصرفي"""

    CORE = "core"  # أساسي
    EXTRA = "extra"  # زائد
    FUNCTIONAL = "functional"  # وظيفي
    WEAK = "weak"  # ضعيف
    BOUNDARY = "boundary"  # حدودي
    UNKNOWN = "unknown"  # مجهول


@dataclass
class PhonemeModel:
    """
    Model representing an Arabic phoneme
    نموذج الفونيم العربي

    Based on the analysis functions in phonology/analyzer.py
    """

    # Core properties
    character: str  # الحرف
    phoneme_type: PhonemeType  # نوع الفونيم
    morph_class: MorphClass = MorphClass.UNKNOWN  # التصنيف الصرفي

    # Consonant properties
    place: Optional[PlaceOfArticulation] = None  # مخرج الحرف
    manner_primary: Optional[MannerOfArticulation] = None  # الطريقة الأساسية
    voicing: Optional[VoicingType] = None  # الجهر/الهمس
    emphatic: bool = False  # إطباق

    # Vowel properties
    name: Optional[str] = None  # اسم الحركة
    quality: Optional[str] = None  # نوعية الحركة
    length: Optional[str] = None  # طول الحركة

    def is_consonant(self) -> bool:
        """Check if phoneme is a consonant"""
        return self.phoneme_type == PhonemeType.CONSONANT

    def is_vowel(self) -> bool:
        """Check if phoneme is a vowel"""
        return self.phoneme_type in [
            PhonemeType.SHORT_VOWEL,
            PhonemeType.LONG_VOWEL,
            PhonemeType.TANWEEN,
        ]

    def is_diacritic(self) -> bool:
        """Check if phoneme is a diacritic"""
        return self.phoneme_type in [
            PhonemeType.SHORT_VOWEL,
            PhonemeType.TANWEEN,
            PhonemeType.SUKUN,
            PhonemeType.GEMINATION,
        ]

    def is_emphatic(self) -> bool:
        """Check if consonant is emphatic"""
        return self.emphatic and self.is_consonant()

    def is_weak_letter(self) -> bool:
        """Check if letter is weak (حروف العلة)"""
        weak_letters = {'و', 'ي', 'ا', 'ء'}
        return self.character in weak_letters

    def is_sun_letter(self) -> bool:
        """Check if letter is a sun letter (حروف شمسية)"""
        sun_letters = {
            'ت',
            'ث',
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
            'ل',
            'ن',
        }
        return self.character in sun_letters

    def is_moon_letter(self) -> bool:
        """Check if letter is a moon letter (حروف قمرية)"""
        return not self.is_sun_letter() and self.is_consonant()

    def get_phonetic_features(self) -> Dict:
        """Get all phonetic features as dictionary"""
        features = {
            "character": self.character,
            "type": self.phoneme_type.value,
            "morph_class": self.morph_class.value,
        }

        if self.is_consonant():
            features.update(
                {
                    "place": self.place.value if self.place else None,
                    "manner_primary": (
                        self.manner_primary.value if self.manner_primary else None
                    ),
                    "voicing": self.voicing.value if self.voicing else None,
                    "emphatic": self.emphatic,
                }
            )

        if self.is_vowel():
            features.update(
                {"name": self.name, "quality": self.quality, "length": self.length}
            )

        return features

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return self.get_phonetic_features()

    @classmethod
    def from_dict(cls, data: Dict) -> 'PhonemeModel':
        """Create instance from dictionary"""
        phoneme_type = PhonemeType(data.get("type", "unknown"))
        morph_class = MorphClass(data.get("morph_class", "unknown"))

        # Parse enums safely
        place = None
        if data.get("place"):
            try:
                place = PlaceOfArticulation(data["place"])
            except ValueError:
                pass

        manner = None
        if data.get("manner_primary"):
            try:
                manner = MannerOfArticulation(data["manner_primary"])
            except ValueError:
                pass

        voicing = None
        if data.get("voicing"):
            try:
                voicing = VoicingType(data["voicing"])
            except ValueError:
                pass

        return cls(
            character=data["character"],
            phoneme_type=phoneme_type,
            morph_class=morph_class,
            place=place,
            manner_primary=manner,
            voicing=voicing,
            emphatic=data.get("emphatic", False),
            name=data.get("name"),
            quality=data.get("quality"),
            length=data.get("length"),
        )

    def __str__(self) -> str:
        """TODO: Add docstring."""
        return f"Phoneme({self.character})"

    def __repr__(self) -> str:
        """TODO: Add docstring."""
        return f"PhonemeModel(character='{self.character}', type={self.phoneme_type.value})"


@dataclass
class PhonemeAnalysisModel:
    """
    Model representing phonological analysis of text
    نموذج التحليل الصوتي للنص

    Based on the analyze_phonemes function output structure
    """

    # Analysis results
    phonemes: List[Tuple[str, PhonemeModel]]  # قائمة الفونيمات المحللة
    total_characters: int = 0  # إجمالي الأحرف
    consonant_count: int = 0  # عدد الأحرف الصحيحة
    vowel_count: int = 0  # عدد الحركات
    emphatic_count: int = 0  # عدد أحرف الإطباق
    weak_letter_count: int = 0  # عدد حروف العلة

    def __post_init__(self):  # type: ignore[no-untyped def]
        """Calculate statistics after initialization"""
        self.total_characters = len(self.phonemes)
        self.consonant_count = sum(1 for _, p in self.phonemes if p.is_consonant())
        self.vowel_count = sum(1 for _, p in self.phonemes if p.is_vowel())
        self.emphatic_count = sum(1 for _, p in self.phonemes if p.is_emphatic())
        self.weak_letter_count = sum(1 for _, p in self.phonemes if p.is_weak_letter())

    def get_characters(self) -> List[str]:
        """Get list of characters"""
        return [char for char, _ in self.phonemes]

    def get_phoneme_models(self) -> List[PhonemeModel]:
        """Get list of phoneme models"""
        return [phoneme for _, phoneme in self.phonemes]

    def get_consonants(self) -> List[Tuple[str, PhonemeModel]]:
        """Get only consonants"""
        return [
            (char, phoneme) for char, phoneme in self.phonemes if phoneme.is_consonant()
        ]

    def get_vowels(self) -> List[Tuple[str, PhonemeModel]]:
        """Get only vowels"""
        return [
            (char, phoneme) for char, phoneme in self.phonemes if phoneme.is_vowel()
        ]

    def get_emphatic_phonemes(self) -> List[Tuple[str, PhonemeModel]]:
        """Get emphatic phonemes"""
        return [
            (char, phoneme) for char, phoneme in self.phonemes if phoneme.is_emphatic()
        ]

    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        return {
            "total_characters": self.total_characters,
            "consonant_count": self.consonant_count,
            "vowel_count": self.vowel_count,
            "emphatic_count": self.emphatic_count,
            "weak_letter_count": self.weak_letter_count,
            "consonant_ratio": self.consonant_count / max(self.total_characters, 1),
            "vowel_ratio": self.vowel_count / max(self.total_characters, 1),
            "emphatic_ratio": self.emphatic_count / max(self.total_characters, 1),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "phonemes": [(char, phoneme.to_dict()) for char, phoneme in self.phonemes],
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_analysis_result(
        cls, analysis_result: List[Tuple[str, Dict]]
    ) -> 'PhonemeAnalysisModel':
        """Create from the output of analyze_phonemes function"""
        phonemes = []
        for char, phoneme_data in analysis_result:
            phoneme_model = PhonemeModel.from_dict(phoneme_data)
            phonemes.append((char, phoneme_model))

        return cls(phonemes=phonemes)


class PhonemeAnalyzerInterface(ABC):
    """
    Interface for phoneme analysis operations
    واجهة عمليات تحليل الفونيمات
    """

    @abstractmethod
    def analyze_phonemes(self, text: str) -> PhonemeAnalysisModel:
        """Analyze text into phonemes"""
        pass

    @abstractmethod
    def get_phoneme_features(self, character: str) -> Optional[PhonemeModel]:
        """Get phonological features for a character"""
        pass

    @abstractmethod
    def is_emphatic(self, character: str) -> bool:
        """Check if character is emphatic"""
        pass

    @abstractmethod
    def get_consonant_type(self, character: str) -> Optional[str]:
        """Get consonant type"""
        pass
