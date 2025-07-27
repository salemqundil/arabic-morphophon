"""
Morphology Model
نموذج الصرف

Represents Arabic morphological structures and analysis.
Based on analysis of existing morphological functions.
"""

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


from dataclasses import dataclass, field  # noqa: F401
from enum import Enum  # noqa: F401
from typing import Dict, List, Optional, Set
from abc import ABC, abstractmethod  # noqa: F401


class MorphemeType(Enum):
    """أنواع المورفيمات"""

    ROOT = "root"  # جذر
    PREFIX = "prefix"  # سابقة
    SUFFIX = "suffix"  # لاحقة
    INFIX = "infix"  # داخلية
    PATTERN = "pattern"  # وزن/قالب
    STEM = "stem"  # أصل الكلمة


class PatternType(Enum):
    """أنواع الأوزان الصرفية"""

    VERB_FORM_I = "فعل"  # الوزن الأول
    VERB_FORM_II = "فعّل"  # الوزن الثاني
    VERB_FORM_III = "فاعل"  # الوزن الثالث
    VERB_FORM_IV = "أفعل"  # الوزن الرابع
    VERB_FORM_V = "تفعّل"  # الوزن الخامس
    VERB_FORM_VI = "تفاعل"  # الوزن السادس
    VERB_FORM_VII = "انفعل"  # الوزن السابع
    VERB_FORM_VIII = "افتعل"  # الوزن الثامن
    VERB_FORM_IX = "افعلّ"  # الوزن التاسع
    VERB_FORM_X = "استفعل"  # الوزن العاشر

    NOUN_MAFUL = "مفعول"  # اسم مفعول
    NOUN_FAIL = "فاعل"  # اسم فاعل
    NOUN_MIFAL = "مفعل"  # اسم آلة
    NOUN_MASDAR = "مصدر"  # مصدر

    ADJECTIVE_FAIL = "فاعل"  # صفة على وزن فاعل
    ADJECTIVE_MAFUL = "مفعول"  # صفة على وزن مفعول


class InflectionType(Enum):
    """أنواع التصريف"""

    PERFECT = "ماضي"  # فعل ماضي
    IMPERFECT = "مضارع"  # فعل مضارع
    IMPERATIVE = "أمر"  # فعل أمر

    NOMINATIVE = "مرفوع"  # مرفوع
    ACCUSATIVE = "منصوب"  # منصوب
    GENITIVE = "مجرور"  # مجرور

    MASCULINE = "مذكر"  # مذكر
    FEMININE = "مؤنث"  # مؤنث

    SINGULAR = "مفرد"  # مفرد
    DUAL = "مثنى"  # مثنى
    PLURAL = "جمع"  # جمع


class WordType(Enum):
    """أنواع الكلمات"""

    VERB = "فعل"  # فعل
    NOUN = "اسم"  # اسم
    ADJECTIVE = "صفة"  # صفة
    PRONOUN = "ضمير"  # ضمير
    PARTICLE = "حرف"  # حرف
    ADVERB = "ظرف"  # ظرف
    PREPOSITION = "حرف جر"  # حرف جر
    CONJUNCTION = "حرف عطف"  # حرف عطف
    INTERJECTION = "تعجب"  # تعجب
    UNKNOWN = "مجهول"  # مجهول


@dataclass
class MorphemeModel:
    """
    Model representing a morpheme (smallest meaningful unit)
    نموذج المورفيم (أصغر وحدة دلالية)
    """

    # Core properties
    text: str  # النص
    morpheme_type: MorphemeType  # نوع المورفيم
    meaning: Optional[str] = None  # المعنى
    function: Optional[str] = None  # الوظيفة النحوية

    # Position information
    start_position: int = 0  # موضع البداية
    end_position: int = 0  # موضع النهاية

    # Morphological properties
    is_productive: bool = True  # هل هو منتج
    frequency: int = 0  # تكرار الاستخدام

    def get_length(self) -> int:
        """Get morpheme length"""
        return len(self.text)

    def is_bound(self) -> bool:
        """Check if morpheme is bound (prefix/suffix/infix)"""
        return self.morpheme_type in [
            MorphemeType.PREFIX,
            MorphemeType.SUFFIX,
            MorphemeType.INFIX,
        ]

    def is_free(self) -> bool:
        """Check if morpheme is free (can stand alone)"""
        return self.morpheme_type in [MorphemeType.ROOT, MorphemeType.STEM]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "text": self.text,
            "morpheme_type": self.morpheme_type.value,
            "meaning": self.meaning,
            "function": self.function,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "length": self.get_length(),
            "is_bound": self.is_bound(),
            "is_free": self.is_free(),
            "is_productive": self.is_productive,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MorphemeModel':
        """Create instance from dictionary"""
        morpheme_type = MorphemeType(data.get("morpheme_type", "root"))

        return cls()
            text=data["text"],
            morpheme_type=morpheme_type,
            meaning=data.get("meaning"),
            function=data.get("function"),
            start_position=data.get("start_position", 0),
            end_position=data.get("end_position", 0),
            is_productive=data.get("is_productive", True),
            frequency=data.get("frequency", 0))

    def __str__(self) -> str:
        """TODO: Add docstring."""
        return f"Morpheme({self.text})"

    def __repr__(self) -> str:
        """TODO: Add docstring."""
        return f"MorphemeModel(text='{self.text}', type={self.morpheme_type.value})"


@dataclass
class MorphologyModel:
    """
    Model representing morphological analysis of a word
    نموذج التحليل الصرفي للكلمة
    """

    # Core properties
    word: str  # الكلمة الأصلية
    root: Optional[str] = None  # الجذر
    pattern: Optional[PatternType] = None  # الوزن الصرفي
    stem: Optional[str] = None  # الأصل

    # Morphemes
    morphemes: List[MorphemeModel] = field(default_factory=list)  # المورفيمات
    prefixes: List[str] = field(default_factory=list)  # السوابق
    suffixes: List[str] = field(default_factory=list)  # اللواحق
    infixes: List[str] = field(default_factory=list)  # الداخليات

    # Grammatical information
    part_of_speech: Optional[str] = None  # نوع الكلمة
    inflection: Set[InflectionType] = field(default_factory=set)  # التصريف

    # Analysis metadata
    confidence: float = 0.0  # مستوى الثقة
    analysis_method: Optional[str] = None  # طريقة التحليل

    def add_morpheme(self, morpheme: MorphemeModel):  # type: ignore[no-untyped def]
        """Add a morpheme to the analysis"""
        self.morphemes.append(morpheme)

        # Update type-specific lists
        if morpheme.morpheme_type == MorphemeType.PREFIX:
            self.prefixes.append(morpheme.text)
        elif morpheme.morpheme_type == MorphemeType.SUFFIX:
            self.suffixes.append(morpheme.text)
        elif morpheme.morpheme_type == MorphemeType.INFIX:
            self.infixes.append(morpheme.text)
        elif morpheme.morpheme_type == MorphemeType.ROOT:
            self.root = morpheme.text
        elif morpheme.morpheme_type == MorphemeType.STEM:
            self.stem = morpheme.text

    def get_morpheme_count(self) -> int:
        """Get total number of morphemes"""
        return len(self.morphemes)

    def get_prefix_count(self) -> int:
        """Get number of prefixes"""
        return len(self.prefixes)

    def get_suffix_count(self) -> int:
        """Get number of suffixes"""
        return len(self.suffixes)

    def has_root(self) -> bool:
        """Check if analysis includes a root"""
        return self.root is not None

    def has_pattern(self) -> bool:
        """Check if analysis includes a pattern"""
        return self.pattern is not None

    def is_derived(self) -> bool:
        """Check if word is morphologically derived"""
        return len(len(self.morphemes)  > 1) > 1

    def is_simple(self) -> bool:
        """Check if word is morphologically simple"""
        return len(self.morphemes) == 1

    def get_inflection_features(self) -> List[str]:
        """Get list of inflection features"""
        return [infl.value for infl in self.inflection]

    def add_inflection(self, inflection: InflectionType):  # type: ignore[no-untyped def]
        """Add an inflection feature"""
        self.inflection.add(inflection)

    def remove_inflection(self, inflection: InflectionType):  # type: ignore[no-untyped def]
        """Remove an inflection feature"""
        self.inflection.discard(inflection)

    def get_morphological_complexity(self) -> float:
        """Calculate morphological complexity score"""
        base_score = len(self.morphemes)
        pattern_bonus = 0.5 if self.has_pattern() else 0
        inflection_bonus = len(self.inflection) * 0.2

        return base_score + pattern_bonus + inflection_bonus

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "word": self.word,
            "root": self.root,
            "pattern": self.pattern.value if self.pattern else None,
            "stem": self.stem,
            "morphemes": [morpheme.to_dict() for morpheme in self.morphemes],
            "prefixes": self.prefixes,
            "suffixes": self.suffixes,
            "infixes": self.infixes,
            "part_of_speech": self.part_of_speech,
            "inflection": self.get_inflection_features(),
            "confidence": self.confidence,
            "analysis_method": self.analysis_method,
            "morpheme_count": self.get_morpheme_count(),
            "is_derived": self.is_derived(),
            "morphological_complexity": self.get_morphological_complexity(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MorphologyModel':
        """Create instance from dictionary"""
        # Parse pattern
        pattern = None
        if data.get("pattern"):
            try:
                pattern = PatternType(data["pattern"])
            except ValueError:
                pass

        # Parse inflection
        inflection = set()
        for infl_str in data.get("inflection", []):
            try:
                inflection.add(InflectionType(infl_str))
            except ValueError:
                pass

        # Parse morphemes
        morphemes = []
        for morpheme_data in data.get("morphemes", []):
            morphemes.append(MorphemeModel.from_dict(morpheme_data))

        return cls()
            word=data["word"],
            root=data.get("root"),
            pattern=pattern,
            stem=data.get("stem"),
            morphemes=morphemes,
            prefixes=data.get("prefixes", []),
            suffixes=data.get("suffixes", []),
            infixes=data.get("infixes", []),
            part_of_speech=data.get("part_of_speech"),
            inflection=inflection,
            confidence=data.get("confidence", 0.0),
            analysis_method=data.get("analysis_method"))

    def __str__(self) -> str:
        """TODO: Add docstring."""
        return f"MorphologyModel({self.word})"

    def __repr__(self) -> str:
        """TODO: Add docstring."""
        return f"MorphologyModel(word='{self.word}', morphemes={len(self.morphemes)})"


class MorphologicalAnalyzerInterface(ABC):
    """
    Interface for morphological analysis operations
    واجهة عمليات التحليل الصرفي
    """

    @abstractmethod
    def analyze_word(self, word: str) -> MorphologyModel:
        """Analyze a single word morphologically"""
        pass

    @abstractmethod
    def extract_root(self, word: str) -> Optional[str]:
        """Extract root from word"""
        pass

    @abstractmethod
    def identify_pattern(self, word: str) -> Optional[PatternType]:
        """Identify morphological pattern"""
        pass

    @abstractmethod
    def segment_morphemes(self, word: str) -> List[MorphemeModel]:
        """Segment word into morphemes"""
        pass

    @abstractmethod
    def get_inflection_features(self, word: str) -> Set[InflectionType]:
        """Get inflection features of word"""
        pass

