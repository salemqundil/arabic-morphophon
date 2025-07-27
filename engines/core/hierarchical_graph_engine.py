#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 نظام هرمي شبكي للتحليل اللغوي العربي المتطور
====================================================
Hierarchical Graph Engine for Advanced Arabic NLP Analysis,
    التدفق: فونيم → حركة → مقطع → تركيب صرفي → وزن → اشتقاق → جذر → نوع → وظيفة نحوية → معنى
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too long
    import logging
    import networkx as nx
    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field
    from typing import Dict, List, Any, Optional, Tuple, Union
    from enum import Enum

# إعداد السجلات,
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== أنواع البيانات الأساسية ==============


class AnalysisLevel(Enum):
    """مستويات التحليل في النظام الهرمي"""

    PHONEME_HARAKAH = 1,
    SYLLABLE_PATTERN = 2,
    MORPHEME_MAPPER = 3,
    WEIGHT_INFERENCE = 4,
    WORD_CLASSIFIER = 5,
    SEMANTIC_ROLE = 6,
    WORD_TRACER = 7,
    TRACE_GRAPH = 7


@dataclass,
    class EngineOutput:
    """مخرجات موحدة لجميع المحركات"""

    level: AnalysisLevel,
    vector: List[float]
    graph_node: Dict[str, Any]
    confidence: float,
    processing_time: float,
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass,
    class PhonemeHarakahData:
    """بيانات الفونيمات والحركات"""

    phonemes: List[str]
    harakaat: List[str]
    positions: List[int]
    ipa_representation: str,
    stress_markers: List[bool]
    lengthening: List[bool]
    shadda: List[bool]


@dataclass,
    class SyllablePatternData:
    """بيانات أنماط المقاطع"""

    syllabic_units: List[str]
    cv_patterns: List[str]
    stress_positions: List[int]
    syllable_types: List[str]
    prosodic_weights: List[float]


@dataclass,
    class MorphemeMapperData:
    """بيانات الخريطة الصرفية"""

    root: str,
    pattern: str,
    prefixes: List[str]
    suffixes: List[str]
    stem: str,
    morpheme_boundaries: List[int]


@dataclass,
    class WeightInferenceData:
    """بيانات استنتاج الوزن"""

    morphological_weight: str,
    derivation_pattern: str,
    derivation_type: str,
    pattern_confidence: float


@dataclass,
    class WordClassifierData:
    """بيانات تصنيف الكلمة"""

    word_type: str  # اسم/فعل/حرف,
    morphological_type: str  # جامد/مشتق,
    inflection_type: str  # مبني/معرب,
    grammatical_features: Dict[str, str]


@dataclass,
    class SemanticRoleData:
    """بيانات الدور الدلالي"""

    syntactic_role: str  # فاعل/مفعول/خبر,
    semantic_role: str  # Agent/Patient/Theme,
    dependency_relations: List[str]
    thematic_role: str


# ============== الواجهة الأساسية للمحركات ==============


class BaseHierarchicalEngine(ABC):
    """الواجهة الأساسية لجميع المحركات الهرمية"""

    def __init__(self, level: AnalysisLevel):

    self.level = level,
    self.graph = nx.DiGraph()

    @abstractmethod,
    def process()
    self, input_data: Any, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """معالجة البيانات وإرجاع النتائج"""
    pass

    @abstractmethod,
    def generate_vector(self, analysis_data: Any) -> List[float]:
    """توليد المتجه الرقمي"""
    pass

    @abstractmethod,
    def create_graph_node(self, analysis_data: Any) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة"""
    pass


# ============== المحرك الأول: UnifiedPhonemeHarakahEngine ==============


class UnifiedPhonemeHarakahEngine(BaseHierarchicalEngine):
    """محرك استخراج الفونيمات والحركات بدقة IPA - يستخدم النظام الموحد"""

    def __init__(self):

    super().__init__(AnalysisLevel.PHONEME_HARAKAH)
        # Import unified phoneme system
    from unified_phonemes import UnifiedArabicPhonemes,
    self.unified_phonemes = UnifiedArabicPhonemes()
    self.ipa_mapping = self._create_ipa_mapping_from_unified()
    self.harakah_patterns = self._create_harakah_patterns_from_unified()

    def _create_ipa_mapping_from_unified(self) -> Dict[str, str]:
    """إنشاء خريطة IPA من النظام الموحد"""
    mapping = {}

        # Add consonants,
    for phoneme in self.unified_phonemes.consonants:
    mapping[phoneme.arabic_char] = phoneme.ipa

        # Add vowels,
    for phoneme in self.unified_phonemes.vowels:
    mapping[phoneme.arabic_char] = phoneme.ipa

        # Add diacritics,
    for diacritic in self.unified_phonemes.diacritics:
    mapping[diacritic.arabic_char] = diacritic.ipa,
    return mapping,
    def _create_harakah_patterns_from_unified(self) -> Dict[str, Dict]:
    """إنشاء أنماط الحركات من النظام الموحد"""
    patterns = {}

        # Add diacritic patterns from unified system,
    for diacritic in self.unified_phonemes.diacritics:
    patterns[diacritic.arabic_char] = {
    "type": diacritic.name,
    "ipa": diacritic.ipa,
    "description": diacritic.description,
    }

    return patterns,
    def process()
    self, input_data: str, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """معالجة الكلمة واستخراج الفونيمات والحركات"""
    start_time = time.time()

    analysis_data = self._extract_phonemes_harakaat(input_data)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.95,  # ثابت مؤقتاً
    processing_time=processing_time,
    metadata={
    "input_word": input_data,
    "ipa_length": len(analysis_data.ipa_representation),
    "analysis_data": analysis_data,
    })

    def _extract_phonemes_harakaat(self, word: str) -> PhonemeHarakahData:
    """استخراج الفونيمات والحركات من الكلمة"""
    phonemes = []
    harakaat = []
    positions = []
    stress_markers = []
    lengthening = []
    shadda = []
    ipa_parts = []

    chars = list(word)
    position = 0,
    for i, char in enumerate(chars):
            if char in self.ipa_mapping:
                # فونيم,
    phonemes.append(char)
    positions.append(position)
    ipa_parts.append(self.ipa_mapping[char])

                # فحص الحركة التالية,
    has_harakah = False,
    has_shadda_mark = False,
    has_lengthening_mark = False,
    if i + 1 < len(chars) and chars[i + 1] in self.harakah_patterns:
    next_char = chars[i + 1]
    harakah_info = self.harakah_patterns[next_char]

    harakaat.append(next_char)
    ipa_parts.append(harakah_info["ipa"])
    has_harakah = True,
    if harakah_info["type"] == "shadda":
    has_shadda_mark = True,
    if harakah_info["length"] in ["long", "gemination"]:
    has_lengthening_mark = True,
    if not has_harakah:
    harakaat.append("")

    stress_markers.append(False)  # سيتم تحديدها لاحقاً
    lengthening.append(has_lengthening_mark)
    shadda.append(has_shadda_mark)

    position += 1,
    return PhonemeHarakahData()
    phonemes=phonemes,
    harakaat=harakaat,
    positions=positions,
    ipa_representation="".join(ipa_parts),
    stress_markers=stress_markers,
    lengthening=lengthening,
    shadda=shadda)

    def generate_vector(self, analysis_data: PhonemeHarakahData) -> List[float]:
    """توليد المتجه الرقمي للفونيمات والحركات"""
    vector = []

        # متجه الفونيمات (one-hot encoding)
    phoneme_vector = [0.0] * len(self.ipa_mapping)
        for phoneme in analysis_data.phonemes:
            if phoneme in self.ipa_mapping:
    idx = list(self.ipa_mapping.keys()).index(phoneme)
    phoneme_vector[idx] = 1.0,
    vector.extend(phoneme_vector)

        # متجه الحركات,
    harakah_vector = [0.0] * len(self.harakah_patterns)
        for harakah in analysis_data.harakaat:
            if harakah and harakah in self.harakah_patterns:
    idx = list(self.harakah_patterns.keys()).index(harakah)
    harakah_vector[idx] = 1.0,
    vector.extend(harakah_vector)

        # معلومات إضافية,
    vector.extend()
    [
    len(analysis_data.phonemes),  # عدد الفونيمات,
    sum(analysis_data.stress_markers),  # عدد النبرات,
    sum(analysis_data.lengthening),  # عدد التطويلات,
    sum(analysis_data.shadda),  # عدد الشدات,
    len(analysis_data.ipa_representation),  # طول التمثيل IPA
    ]
    )

    return vector,
    def create_graph_node(self, analysis_data: PhonemeHarakahData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للفونيمات والحركات"""
    return {
    "type": "phoneme_harakah",
    "level": 1,
    "phonemes": analysis_data.phonemes,
    "harakaat": analysis_data.harakaat,
    "ipa": analysis_data.ipa_representation,
    "features": {
    "has_stress": any(analysis_data.stress_markers),
    "has_lengthening": any(analysis_data.lengthening),
    "has_shadda": any(analysis_data.shadda),
    "phoneme_count": len(analysis_data.phonemes),
    },
    }


# ============== المحرك الثاني: SyllablePatternEngine ==============


class SyllablePatternEngine(BaseHierarchicalEngine):
    """محرك توليد مقاطع CV/CVC وتحليل النبر"""

    def __init__(self):

    super().__init__(AnalysisLevel.SYLLABLE_PATTERN)
    self.syllable_patterns = {
    "CV": {"weight": 1, "type": "light"},
    "CVC": {"weight": 2, "type": "heavy"},
    "CVCC": {"weight": 3, "type": "super_heavy"},
    "CVV": {"weight": 2, "type": "heavy"},
    "CVVC": {"weight": 3, "type": "super_heavy"},
    }

    def process()
    self, input_data: PhonemeHarakahData, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """معالجة الفونيمات لتوليد المقاطع"""
    start_time = time.time()

    analysis_data = self._segment_syllabic_units(input_data)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.90,
    processing_time=processing_time,
    metadata={
    "syllable_count": len(analysis_data.syllabic_units),
    "analysis_data": analysis_data,
    })

    def _segment_syllabic_units()
    self, phoneme_data: PhonemeHarakahData
    ) -> SyllablePatternData:
    """تقسيم الفونيمات إلى مقاطع"""
    syllabic_units = []
    cv_patterns = []
    stress_positions = []
    syllable_types = []
    prosodic_weights = []

        # خوارزمية تقسيم المقاطع المبسطة,
    current_syllable = ""
    current_cv = ""

        for i, (phoneme, harakah) in enumerate()
    zip(phoneme_data.phonemes, phoneme_data.harakaat)
    ):
            # إضافة الصامت,
    current_syllable += phoneme,
    current_cv += "C"

            # فحص الحركة,
    if harakah and harakah != "ْ":  # ليس سكون,
    current_syllable += harakah,
    current_cv += "V"

                # إذا كان هناك صامت تالي، قد ينتهي المقطع,
    if i + 1 < len(phoneme_data.phonemes):
    next_harakah = ()
    phoneme_data.harakaat[i + 1]
                        if i + 1 < len(phoneme_data.harakaat)
                        else ""
    )
                    if next_harakah == "ْ" or not next_harakah:  # سكون أو نهاية
                        # المقطع كامل,
    syllabic_units.append(current_syllable)
    cv_patterns.append(current_cv)

                        # تحديد نوع المقطع والوزن,
    pattern_info = self.syllable_patterns.get()
    current_cv, {"weight": 1, "type": "light"}
    )
    syllable_types.append(pattern_info["type"])
    prosodic_weights.append(pattern_info["weight"])

                        # إعادة تعيين,
    current_syllable = ""
    current_cv = ""

        # إضافة المقطع الأخير إن وجد,
    if current_syllable:
    syllabic_units.append(current_syllable)
    cv_patterns.append(current_cv)
    pattern_info = self.syllable_patterns.get()
    current_cv, {"weight": 1, "type": "light"}
    )
    syllable_types.append(pattern_info["type"])
    prosodic_weights.append(pattern_info["weight"])

        # تحديد مواقع النبر (مبسط)
        if len(syllabic_units) > 1:
    stress_positions = [len(syllabic_units) - 2]  # النبر على الثاني من الآخر,
    else:
    stress_positions = [0]

    return SyllablePatternData()
    syllabic_units=syllabic_units,
    cv_patterns=cv_patterns,
    stress_positions=stress_positions,
    syllable_types=syllable_types,
    prosodic_weights=prosodic_weights)

    def generate_vector(self, analysis_data: SyllablePatternData) -> List[float]:
    """توليد المتجه الرقمي للمقاطع"""
    vector = []

        # معلومات المقاطع,
    vector.extend()
    [
    len(analysis_data.syllabic_units),  # عدد المقاطع,
    sum(analysis_data.prosodic_weights),  # الوزن الإجمالي,
    len(analysis_data.stress_positions),  # عدد النبرات
    ()
    analysis_data.prosodic_weights.index()
    max(analysis_data.prosodic_weights)
    )
                    if analysis_data.prosodic_weights,
    else 0
    ),  # موقع أثقل مقطع
    ]
    )

        # أنماط CV (one hot للأنماط الشائعة)
    common_patterns = ["CV", "CVC", "CVV", "CVCC", "CVVC"]
        for pattern in common_patterns:
    vector.append(float(analysis_data.cv_patterns.count(pattern)))

        # نسب أنواع المقاطع,
    total_syllabic_units = len(analysis_data.syllabic_units)
        if total_syllabic_units > 0:
    light_ratio = ()
    analysis_data.syllable_types.count("light") / total_syllabic_units
    )
    heavy_ratio = ()
    analysis_data.syllable_types.count("heavy") / total_syllabic_units
    )
    super_heavy_ratio = ()
    analysis_data.syllable_types.count("super_heavy") / total_syllabic_units
    )
        else:
    light_ratio = heavy_ratio = super_heavy_ratio = 0.0,
    vector.extend([light_ratio, heavy_ratio, super_heavy_ratio])

    return vector,
    def create_graph_node(self, analysis_data: SyllablePatternData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للمقاطع"""
    return {
    "type": "syllable_pattern",
    "level": 2,
    "syllabic_units": analysis_data.syllabic_units,
    "cv_patterns": analysis_data.cv_patterns,
    "stress_positions": analysis_data.stress_positions,
    "features": {
    "syllable_count": len(analysis_data.syllabic_units),
    "total_weight": sum(analysis_data.prosodic_weights),
    "dominant_pattern": ()
    max()
    set(analysis_data.cv_patterns),
    key=analysis_data.cv_patterns.count)
                    if analysis_data.cv_patterns,
    else None
    ),
    "has_stress": len(len(analysis_data.stress_positions)  > 0) > 0,
    },
    }


# ============== المحرك الثالث: MorphemeMapperEngine ==============


class MorphemeMapperEngine(BaseHierarchicalEngine):
    """محرك تحليل البنية الداخلية للكلمة إلى جذر/زوائد"""

    def __init__(self):

    super().__init__(AnalysisLevel.MORPHEME_MAPPER)
    self.root_database = self._import_data_root_database()
    self.morphological_patterns = self._import_data_morphological_patterns()

    def _import_data_root_database(self) -> Dict[str, Dict]:
    """تحميل قاعدة بيانات الجذور"""
    return {
    "كتب": {"meaning": "write", "type": "trilateral", "class": "verbal"},
    "درس": {"meaning": "study", "type": "trilateral", "class": "verbal"},
    "علم": {"meaning": "know", "type": "trilateral", "class": "verbal"},
    "جمل": {"meaning": "beauty", "type": "trilateral", "class": "nominal"},
    "طلب": {"meaning": "request", "type": "trilateral", "class": "verbal"},
    }

    def _import_data_morphological_patterns(self) -> Dict[str, Dict]:
    """تحميل الأوزان الصرفية"""
    return {
    "فعل": {"type": "verb", "pattern": "CVC", "derivation": "base"},
    "فاعل": {"type": "noun", "pattern": "CVCVC", "derivation": "agent"},
    "مفعول": {"type": "noun", "pattern": "MVCVVC", "derivation": "object"},
    "فعيل": {
    "type": "adjective",
    "pattern": "CVCVVC",
    "derivation": "intensive",
    },
    "افعال": {"type": "noun", "pattern": "VCCVC", "derivation": "plural"},
    }

    def process()
    self, input_data: SyllablePatternData, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """تحليل البنية الصرفية للكلمة"""
    start_time = time.time()

        # استخراج الكلمة الأصلية من البيانات,
    word = ()
    getattr(input_data, "original_word", "")
            if hasattr(input_data, "original_word")
            else ""
    )

    analysis_data = self._analyze_morpheme_structure(word, input_data)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.85,
    processing_time=processing_time,
    metadata={
    "root_found": bool(analysis_data.root),
    "pattern_matched": bool(analysis_data.pattern),
    "analysis_data": analysis_data,
    })

    def _analyze_morpheme_structure()
    self, word: str, syllable_data: SyllablePatternData
    ) -> MorphemeMapperData:
    """تحليل البنية الصرفية"""
        # استخراج الجذر (خوارزمية مبسطة)
    root = self._extract_root(word)

        # تحديد الوزن,
    pattern = self._determine_pattern(word, root)

        # تحليل الزوائد,
    prefixes, stem, suffixes = self._analyze_affixes(word, root)

        # تحديد حدود المورفيمات,
    morpheme_boundaries = self._find_morpheme_boundaries()
    word, prefixes, stem, suffixes
    )

    return MorphemeMapperData()
    root=root,
    pattern=pattern,
    prefixes=prefixes,
    suffixes=suffixes,
    stem=stem,
    morpheme_boundaries=morpheme_boundaries)

    def _extract_root(self, word: str) -> str:
    """استخراج الجذر من الكلمة"""
        # إزالة أداة التعريف,
    clean_word = word,
    if clean_word.startswith("ال"):
    clean_word = clean_word[2:]

        # إزالة اللواحق الشائعة,
    suffixes = ["ة", "ات", "ان", "ين", "ون", "ها", "هم", "كم"]
        for suffix in suffixes:
            if clean_word.endswith(suffix):
    clean_word = clean_word[:  len(suffix)]
    break

        # استخراج الصوامت (جذر مبسط)
    consonants = []
        for char in clean_word:
            if char not in "اويِّ َُ ً ٌ ٍ ْ":  # ليس حركة أو حرف علة,
    consonants.append(char)

    potential_root = "".join(consonants[:3])  # أخذ أول 3 صوامت

        # البحث في قاعدة البيانات,
    if potential_root in self.root_database:
    return potential_root,
    return potential_root  # إرجاع الجذر المستخرج حتى لو لم يوجد في القاعدة,
    def _determine_pattern(self, word: str, root: str) -> str:
    """تحديد الوزن الصرفي"""
        # خوارزمية مبسطة لتحديد الوزن,
    if word.startswith("م") and word.endswith("ول"):
    return "مفعول"
        elif len(word) == len(root) + 1:
    return "فاعل"
        elif len(word) == len(root):
    return "فعل"
        else:
    return "unknown"

    def _analyze_affixes()
    self, word: str, root: str
    ) -> Tuple[List[str], str, List[str]]:
    """تحليل البادئات والجذع واللواحق"""
    prefixes = []
    suffixes = []
    stem = word

        # البادئات الشائعة,
    if stem.startswith("ال"):
    prefixes.append("ال")
    stem = stem[2:]
        elif stem.startswith("و"):
    prefixes.append("و")
    stem = stem[1:]

        # اللواحق الشائعة,
    if stem.endswith("ة"):
    suffixes.append("ة")
    stem = stem[: 1]
        elif stem.endswith("ان"):
    suffixes.append("ان")
    stem = stem[:-2]

    return prefixes, stem, suffixes,
    def _find_morpheme_boundaries()
    self, word: str, prefixes: List[str], stem: str, suffixes: List[str]
    ) -> List[int]:
    """تحديد حدود المورفيمات"""
    boundaries = [0]  # بداية الكلمة,
    current_pos = 0

        # حدود البادئات,
    for prefix in prefixes:
    current_pos += len(prefix)
    boundaries.append(current_pos)

        # حد الجذع,
    current_pos += len(stem)
    boundaries.append(current_pos)

        # حدود اللواحق,
    for suffix in suffixes:
    current_pos += len(suffix)
    boundaries.append(current_pos)

    return boundaries,
    def generate_vector(self, analysis_data: MorphemeMapperData) -> List[float]:
    """توليد المتجه الرقمي للبنية الصرفية"""
    vector = []

        # معلومات الجذر,
    vector.extend()
    [
    len(analysis_data.root),  # طول الجذر
    ()
    1.0 if analysis_data.root in self.root_database else 0.0
    ),  # وجود في قاعدة البيانات,
    len(analysis_data.prefixes),  # عدد البادئات,
    len(analysis_data.suffixes),  # عدد اللواحق,
    len(analysis_data.stem),  # طول الجذع,
    len(analysis_data.morpheme_boundaries),  # عدد حدود المورفيمات
    ]
    )

        # نوع الوزن (one hot)
    pattern_types = ["فعل", "فاعل", "مفعول", "فعيل", "افعال", "unknown"]
        for ptype in pattern_types:
    vector.append(1.0 if analysis_data.pattern == ptype else 0.0)

    return vector,
    def create_graph_node(self, analysis_data: MorphemeMapperData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للبنية الصرفية"""
    return {
    "type": "morpheme_mapper",
    "level": 3,
    "root": analysis_data.root,
    "pattern": analysis_data.pattern,
    "morphemes": {
    "prefixes": analysis_data.prefixes,
    "stem": analysis_data.stem,
    "suffixes": analysis_data.suffixes,
    },
    "features": {
    "morpheme_count": len(analysis_data.prefixes)
    + 1
    + len(analysis_data.suffixes),
    "root_type": self.root_database.get(analysis_data.root, {}).get()
    "type", "unknown"
    ),
    "has_prefixes": len(len(analysis_data.prefixes) -> 0) > 0,
    "has_suffixes": len(len(analysis_data.suffixes) -> 0) > 0,
    },
    }


# ============== المحرك الرابع: WeightInferenceEngine ==============


class WeightInferenceEngine(BaseHierarchicalEngine):
    """محرك استنتاج الوزن الصرفي والعروضي"""

    def __init__(self):

    super().__init__(AnalysisLevel.WEIGHT_INFERENCE)
    self.morphological_weights = self._import_data_morphological_weights()
    self.prosodic_patterns = self._import_data_prosodic_patterns()

    def _import_data_morphological_weights(self) -> Dict[str, Dict]:
    """تحميل الأوزان الصرفية"""
    return {
    "فَعَلَ": {
    "type": "verb",
    "tense": "past",
    "pattern": "CaCaC",
    "syllabic_units": 2,
    },
    "يَفْعَلُ": {
    "type": "verb",
    "tense": "present",
    "pattern": "yaCCaC",
    "syllabic_units": 3,
    },
    "فاعِل": {
    "type": "noun",
    "derivation": "agent",
    "pattern": "CaCiC",
    "syllabic_units": 2,
    },
    "مَفْعُول": {
    "type": "noun",
    "derivation": "object",
    "pattern": "maCCuC",
    "syllabic_units": 3,
    },
    "فَعيل": {
    "type": "adjective",
    "intensification": True,
    "pattern": "CaCiC",
    "syllabic_units": 2,
    },
    }

    def _import_data_prosodic_patterns(self) -> Dict[str, str]:
    """تحميل الأنماط العروضية"""
    return {
    "فَعَلَ": "-- (قصير قصير)",
    "يَفْعَلُ": "--- (قصير-ساكن قصير)",
    "فاعِل": "-/ (قصير طويل)",
    "مَفْعُول": "/-- (طويل-قصير قصير)",
    }

    def process()
    self, input_data: MorphemeMapperData, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """تحليل الوزن الصرفي والعروضي"""
    start_time = time.time()

    analysis_data = self._infer_weight_patterns(input_data)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.82,
    processing_time=processing_time,
    metadata={
    "weight_found": bool(analysis_data.morphological_weight),
    "analysis_data": analysis_data,
    })

    def _infer_weight_patterns()
    self, morpheme_data: MorphemeMapperData
    ) -> WeightInferenceData:
    """استنتاج أنماط الوزن"""
        # تحديد الوزن الصرفي بناءً على البنية,
    morphological_weight = self._determine_morphological_weight(morpheme_data)

        # تحديد النمط العروضي,
    prosodic_pattern = self.prosodic_patterns.get(morphological_weight, "unknown")

        # تحديد نوع الاشتقاق,
    derivation_type = self._determine_derivation_type()
    morpheme_data, morphological_weight
    )

        # حساب الثقة,
    confidence = self._calculate_weight_confidence()
    morpheme_data, morphological_weight
    )

    return WeightInferenceData()
    morphological_weight=morphological_weight,
    derivation_pattern=prosodic_pattern,
    derivation_type=derivation_type,
    pattern_confidence=confidence)

    def _determine_morphological_weight(self, morpheme_data: MorphemeMapperData) -> str:
    """تحديد الوزن الصرفي"""
        # خوارزمية متقدمة لتحديد الوزن,
    pattern = morpheme_data.pattern,
    root = morpheme_data.root,
    if pattern in self.morphological_weights:
    return pattern

        # تحليل تقريبي بناءً على البنية,
    if len(morpheme_data.stem) == len(root):
    return "فَعَلَ"
        elif len(morpheme_data.stem) == len(root) + 1:
    return "فاعِل"
        elif len(morpheme_data.stem) == len(root) + 2:
    return "مَفْعُول"
        else:
    return "unknown"

    def _determine_derivation_type()
    self, morpheme_data: MorphemeMapperData, weight: str
    ) -> str:
    """تحديد نوع الاشتقاق"""
    weight_info = self.morphological_weights.get(weight, {})

        if weight_info.get("type") == "verb":
    return "verbal"
        elif weight_info.get("derivation"):
    return weight_info["derivation"]
        else:
    return "base"

    def _calculate_weight_confidence()
    self, morpheme_data: MorphemeMapperData, weight: str
    ) -> float:
    """حساب ثقة تحديد الوزن"""
    confidence = 0.5  # قيمة أساسية

        # زيادة الثقة إذا كان الجذر معروف,
    if morpheme_data.root in ["كتب", "درس", "علم", "جمل", "طلب"]:
    confidence += 0.3

        # زيادة الثقة إذا كان الوزن معروف,
    if weight in self.morphological_weights:
    confidence += 0.2,
    return min(confidence, 1.0)

    def generate_vector(self, analysis_data: WeightInferenceData) -> List[float]:
    """توليد المتجه الرقمي للوزن"""
    vector = []

        # تشفير الوزن الصرفي (one hot)
    weight_types = ["فَعَلَ", "يَفْعَلُ", "فاعِل", "مَفْعُول", "فَعيل", "unknown"]
        for weight in weight_types:
    vector.append(1.0 if analysis_data.morphological_weight == weight else 0.0)

        # تشفير نوع الاشتقاق,
    derivation_types = ["verbal", "agent", "object", "base", "unknown"]
        for dtype in derivation_types:
    vector.append(1.0 if analysis_data.derivation_type == dtype else 0.0)

        # الثقة والمعلومات الإضافية,
    vector.extend()
    [
    analysis_data.pattern_confidence,
    1.0 if analysis_data.derivation_pattern != "unknown" else 0.0,
    ]
    )

    return vector,
    def create_graph_node(self, analysis_data: WeightInferenceData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للوزن"""
    return {
    "type": "weight_inference",
    "level": 4,
    "morphological_weight": analysis_data.morphological_weight,
    "derivation_pattern": analysis_data.derivation_pattern,
    "derivation_type": analysis_data.derivation_type,
    "features": {
    "confidence": analysis_data.pattern_confidence,
    "has_prosodic_pattern": analysis_data.derivation_pattern != "unknown",
    "is_derived": analysis_data.derivation_type != "base",
    },
    }


# ============== المحرك الخامس: WordClassifierEngine ==============


class WordClassifierEngine(BaseHierarchicalEngine):
    """محرك تصنيف الكلمات نحوياً وصرفياً"""

    def __init__(self):

    super().__init__(AnalysisLevel.WORD_CLASSIFIER)
    self.word_classification_rules = self._import_data_classification_rules()
    self.grammatical_features = self._import_data_grammatical_features()

    def _import_data_classification_rules(self) -> Dict[str, List[str]]:
    """تحميل قواعد التصنيف"""
    return {
    "verb_indicators": ["يَفْعَلُ", "فَعَلَ", "افْعَلْ"],
    "noun_indicators": ["فاعِل", "مَفْعُول", "فَعيل"],
    "particle_indicators": ["في", "من", "إلى", "على", "عن"],
    }

    def _import_data_grammatical_features(self) -> Dict[str, List[str]]:
    """تحميل الخصائص النحوية"""
    return {
    "case": ["nominative", "accusative", "genitive"],
    "number": ["singular", "dual", "plural"],
    "gender": ["masculine", "feminine"],
    "definiteness": ["definite", "indefinite"],
    "tense": ["past", "present", "imperative"],
    }

    def process()
    self, input_data: WeightInferenceData, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """تصنيف الكلمة نحوياً وصرفياً"""
    start_time = time.time()

    analysis_data = self._classify_word(input_data)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.87,
    processing_time=processing_time,
    metadata={
    "word_type": analysis_data.word_type,
    "analysis_data": analysis_data,
    })

    def _classify_word(self, weight_data: WeightInferenceData) -> WordClassifierData:
    """تصنيف الكلمة"""
        # تحديد نوع الكلمة الرئيسي,
    word_type = self._determine_word_type(weight_data)

        # تحديد النوع الصرفي,
    morphological_type = self._determine_morphological_type(weight_data)

        # تحديد نوع الإعراب,
    inflection_type = self._determine_inflection_type(word_type)

        # تحديد الخصائص النحوية,
    grammatical_features = self._extract_grammatical_features()
    weight_data, word_type
    )

    return WordClassifierData()
    word_type=word_type,
    morphological_type=morphological_type,
    inflection_type=inflection_type,
    grammatical_features=grammatical_features)

    def _determine_word_type(self, weight_data: WeightInferenceData) -> str:
    """تحديد نوع الكلمة (اسم/فعل/حرف)"""
    weight = weight_data.morphological_weight

        # قواعد التصنيف الأساسية,
    if weight in ["فَعَلَ", "يَفْعَلُ", "افْعَلْ"]:
    return "verb"
        elif weight in ["فاعِل", "مَفْعُول", "فَعيل"]:
    return "noun"
        elif weight_data.derivation_type == "verbal":
    return "verb"
        else:
    return "noun"  # افتراضي,
    def _determine_morphological_type(self, weight_data: WeightInferenceData) -> str:
    """تحديد النوع الصرفي (جامد/مشتق)"""
        if weight_data.derivation_type in ["agent", "object", "verbal"]:
    return "derived"
        else:
    return "primitive"

    def _determine_inflection_type(self, word_type: str) -> str:
    """تحديد نوع الإعراب (مبني/معرب)"""
        if word_type == "verb":
    return "inflected"  # الأفعال معربة (غالباً)
        elif word_type == "noun":
    return "inflected"  # الأسماء معربة (غالباً)
        else:
    return "indeclinable"  # الحروف مبنية,
    def _extract_grammatical_features()
    self, weight_data: WeightInferenceData, word_type: str
    ) -> Dict[str, str]:
    """استخراج الخصائص النحوية"""
    features = {}

        if word_type == "noun":
    features.update()
    {
    "case": "nominative",  # افتراضي
    "number": "singular",  # افتراضي
    "gender": "masculine",  # افتراضي
    "definiteness": "indefinite",  # افتراضي
    }
    )
        elif word_type == "verb":
    features.update()
    {
    "tense": self._infer_tense(weight_data.morphological_weight),
    "person": "third",  # افتراضي
    "number": "singular",  # افتراضي
    "gender": "masculine",  # افتراضي
    }
    )

    return features,
    def _infer_tense(self, weight: str) -> str:
    """استنتاج الزمن من الوزن"""
        if weight == "فَعَلَ":
    return "past"
        elif weight == "يَفْعَلُ":
    return "present"
        elif weight == "افْعَلْ":
    return "imperative"
        else:
    return "unknown"

    def generate_vector(self, analysis_data: WordClassifierData) -> List[float]:
    """توليد المتجه الرقمي للتصنيف"""
    vector = []

        # تشفير نوع الكلمة,
    word_types = ["noun", "verb", "particle"]
        for wtype in word_types:
    vector.append(1.0 if analysis_data.word_type == wtype else 0.0)

        # تشفير النوع الصرفي,
    morph_types = ["primitive", "derived"]
        for mtype in morph_types:
    vector.append(1.0 if analysis_data.morphological_type == mtype else 0.0)

        # تشفير نوع الإعراب,
    infl_types = ["inflected", "indeclinable"]
        for itype in infl_types:
    vector.append(1.0 if analysis_data.inflection_type == itype else 0.0)

        # الخصائص النحوية (encoding مبسط)
    vector.extend()
    [
    1.0 if "case" in analysis_data.grammatical_features else 0.0,
    1.0 if "tense" in analysis_data.grammatical_features else 0.0,
    1.0 if "number" in analysis_data.grammatical_features else 0.0,
    1.0 if "gender" in analysis_data.grammatical_features else 0.0,
    ]
    )

    return vector,
    def create_graph_node(self, analysis_data: WordClassifierData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للتصنيف"""
    return {
    "type": "word_classifier",
    "level": 5,
    "word_type": analysis_data.word_type,
    "morphological_type": analysis_data.morphological_type,
    "inflection_type": analysis_data.inflection_type,
    "grammatical_features": analysis_data.grammatical_features,
    "features": {
    "is_derived": analysis_data.morphological_type == "derived",
    "is_inflected": analysis_data.inflection_type == "inflected",
    "feature_count": len(analysis_data.grammatical_features),
    },
    }


# ============== المحرك السادس: SemanticRoleEngine ==============


class SemanticRoleEngine(BaseHierarchicalEngine):
    """محرك تحليل الأدوار الدلالية والوظائف النحوية"""

    def __init__(self):

    super().__init__(AnalysisLevel.SEMANTIC_ROLE)
    self.semantic_roles = self._import_data_semantic_roles()
    self.argument_structures = self._import_data_argument_structures()

    def _import_data_semantic_roles(self) -> Dict[str, Dict]:
    """تحميل الأدوار الدلالية"""
    return {
    "agent": {
    "description": "الفاعل",
    "animacy": "animate",
    "volitionality": "volitional",
    },
    "patient": {
    "description": "المفعول به",
    "animacy": "any",
    "volitionality": "non_volitional",
    },
    "theme": {
    "description": "الموضوع",
    "animacy": "any",
    "volitionality": "non_volitional",
    },
    "location": {
    "description": "المكان",
    "animacy": "inanimate",
    "volitionality": "non_volitional",
    },
    "time": {
    "description": "الزمان",
    "animacy": "inanimate",
    "volitionality": "non_volitional",
    },
    "instrument": {
    "description": "الآلة",
    "animacy": "inanimate",
    "volitionality": "non_volitional",
    },
    }

    def _import_data_argument_structures(self) -> Dict[str, List[str]]:
    """تحميل بنى الحجج"""
    return {
    "transitive": ["agent", "patient"],
    "intransitive": ["agent"],
    "ditransitive": ["agent", "patient", "recipient"],
    "locative": ["agent", "theme", "location"],
    "temporal": ["agent", "theme", "time"],
    }

    def process()
    self, input_data: WordClassifierData, context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """تحليل الأدوار الدلالية"""
    start_time = time.time()

    analysis_data = self._analyze_semantic_roles(input_data, context)
    vector = self.generate_vector(analysis_data)
    graph_node = self.create_graph_node(analysis_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.75,
    processing_time=processing_time,
    metadata={
    "semantic_role": analysis_data.semantic_role,
    "analysis_data": analysis_data,
    })

    def _analyze_semantic_roles()
    self, word_data: WordClassifierData, context: Optional[Dict[str, Any]]
    ) -> SemanticRoleData:
    """تحليل الأدوار الدلالية للكلمة"""
        # تحديد الدور الدلالي الأساسي,
    semantic_role = self._determine_primary_role(word_data)

        # تحديد بنية الحجج,
    argument_structure = self._determine_argument_structure()
    word_data, semantic_role
    )

        # تحديد الأدوار الموضوعية,
    thematic_roles = self._extract_thematic_roles(word_data, semantic_role)

        # تحديد الخصائص الدلالية,
    self._extract_semantic_features(word_data, semantic_role)

        # تحديد المعنى السياقي,
    self._infer_contextual_meaning(word_data, context)

    return SemanticRoleData()
    syntactic_role=semantic_role,  # الدور النحوي,
    semantic_role=semantic_role,  # الدور الدلالي,
    dependency_relations=argument_structure,  # العلاقات التابعة,
    thematic_role=()
    thematic_roles[0] if thematic_roles else semantic_role
    ),  # الدور الموضوعي الرئيسي
    )

    def _determine_primary_role(self, word_data: WordClassifierData) -> str:
    """تحديد الدور الدلالي الأساسي"""
        if word_data.word_type == "verb":
    return "predicate"
        elif word_data.word_type == "noun":
            # تحديد الدور بناءً على الخصائص النحوية,
    case = word_data.grammatical_features.get("case", "nominative")
            if case == "nominative":
    return "agent"
            elif case == "accusative":
    return "patient"
            elif case == "genitive":
    return "modifier"
            else:
    return "argument"
        else:
    return "modifier"

    def _determine_argument_structure()
    self, word_data: WordClassifierData, role: str
    ) -> List[str]:
    """تحديد بنية الحجج"""
        if word_data.word_type == "verb":
            # تحديد نوع الفعل (لازم/متعد/متعد إلى مفعولين)
            if word_data.morphological_type == "derived":
    return self.argument_structures.get("transitive", ["agent", "patient"])
            else:
    return self.argument_structures.get("intransitive", ["agent"])
        else:
    return [role]

    def _extract_thematic_roles()
    self, word_data: WordClassifierData, primary_role: str
    ) -> List[str]:
    """استخراج الأدوار الموضوعية"""
    thematic_roles = [primary_role]

        # إضافة أدوار ثانوية بناءً على السياق,
    if word_data.word_type == "noun":
            if "definiteness" in word_data.grammatical_features:
                if word_data.grammatical_features["definiteness"] == "definite":
    thematic_roles.append("specific")
                else:
    thematic_roles.append("generic")

    return thematic_roles,
    def _extract_semantic_features()
    self, word_data: WordClassifierData, role: str
    ) -> Dict[str, str]:
    """استخراج الخصائص الدلالية"""
    features = {}

        # خصائص أساسية,
    role_info = self.semantic_roles.get(role, {})
    features.update(role_info)

        # خصائص إضافية من البيانات النحوية,
    if "gender" in word_data.grammatical_features:
    features["gender"] = word_data.grammatical_features["gender"]

        if "number" in word_data.grammatical_features:
    features["number"] = word_data.grammatical_features["number"]

    return features,
    def _infer_contextual_meaning()
    self, word_data: WordClassifierData, context: Optional[Dict[str, Any]]
    ) -> str:
    """استنتاج المعنى السياقي"""
        if context and "sentence_context" in context:
            # تحليل مبسط للسياق,
    return "contextual"
        else:
    return "literal"

    def generate_vector(self, analysis_data: SemanticRoleData) -> List[float]:
    """توليد المتجه الرقمي للأدوار الدلالية"""
    vector = []

        # تشفير الدور الدلالي الأساسي,
    primary_roles = ["agent", "patient", "predicate", "modifier", "argument"]
        for role in primary_roles:
    vector.append(1.0 if analysis_data.semantic_role == role else 0.0)

        # تشفير العلاقات التابعة,
    vector.extend()
    [
    len(analysis_data.dependency_relations),
    1.0 if "agent" in analysis_data.dependency_relations else 0.0,
    1.0 if "patient" in analysis_data.dependency_relations else 0.0,
    ]
    )

        # تشفير الدور الموضوعي,
    vector.extend()
    [
    ()
    1.0,
    if analysis_data.thematic_role == analysis_data.semantic_role,
    else 0.0
    ),
    ()
    1.0,
    if analysis_data.syntactic_role == analysis_data.semantic_role,
    else 0.0
    ),
    ]
    )

        # معلومات إضافية,
    vector.extend()
    [
    1.0 if analysis_data.thematic_role != "unknown" else 0.0,
    len(analysis_data.dependency_relations),
    ]
    )

    return vector,
    def create_graph_node(self, analysis_data: SemanticRoleData) -> Dict[str, Any]:
    """إنشاء عقدة الشبكة للأدوار الدلالية"""
    return {
    "type": "semantic_role",
    "level": 6,
    "semantic_role": analysis_data.semantic_role,
    "syntactic_role": analysis_data.syntactic_role,
    "thematic_role": analysis_data.thematic_role,
    "dependency_relations": analysis_data.dependency_relations,
    "features": {
    "is_argument": analysis_data.semantic_role,
    in ["agent", "patient", "theme"],
    "is_modifier": analysis_data.semantic_role,
    in ["modifier", "location", "time"],
    "has_dependencies": len(len(analysis_data.dependency_relations) -> 0) > 0,
    "role_consistency": analysis_data.semantic_role
    == analysis_data.syntactic_role,
    },
    }


# ============== المحرك السابع: WordTraceGraph ==============


class WordTraceGraph(BaseHierarchicalEngine):
    """محرك تتبع الكلمة النهائي - يجمع جميع التحليلات في رسم بياني موحد"""

    def __init__(self):

    super().__init__(AnalysisLevel.WORD_TRACER)
    self.trace_graph = nx.DiGraph()

    def process()
    self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> EngineOutput:
    """إنشاء تتبع شامل للكلمة"""
    start_time = time.time()

        # تجميع جميع النتائج من المستويات السابقة,
    trace_data = self._create_word_trace(input_data)
    vector = self.generate_vector(trace_data)
    graph_node = self.create_graph_node(trace_data)

    processing_time = time.time() - start_time,
    return EngineOutput()
    level=self.level,
    vector=vector,
    graph_node=graph_node,
    confidence=0.95,
    processing_time=processing_time,
    metadata={"trace_complete": True, "total_levels": len(input_data)})

    def _create_word_trace(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
    """إنشاء تتبع شامل للكلمة"""
    trace = {
    "word": all_results.get("original_word", ""),
    "analysis_levels": {},
    "connections": [],
    "final_analysis": {},
    }

        # جمع النتائج من جميع المستويات,
    for level_name, result in all_results.items():
            if isinstance(result, dict) and "vector" in result:
    trace["analysis_levels"][level_name] = {
    "vector": result["vector"],
    "confidence": result.get("confidence", 0.0),
    "processing_time": result.get("processing_time", 0.0),
    "features": result.get("graph_node", {}).get("features", {}),
    }

        # إنشاء الروابط بين المستويات,
    trace["connections"] = self._create_level_connections(trace["analysis_levels"])

        # التحليل النهائي المجمع,
    trace["final_analysis"] = self._synthesize_final_analysis()
    trace["analysis_levels"]
    )

    return trace,
    def _create_level_connections(self, levels: Dict[str, Any]) -> List[Dict[str, str]]:
    """إنشاء روابط بين مستويات التحليل"""
    connections = []
    level_sequence = [
    "phoneme_harakah",
    "syllable_pattern",
    "morpheme_mapper",
    "weight_inference",
    "word_classifier",
    "semantic_role",
    ]

        for i in range(len(level_sequence) - 1):
    current_level = level_sequence[i]
    next_level = level_sequence[i + 1]

            if current_level in levels and next_level in levels:
    connections.append()
    {
    "from": current_level,
    "to": next_level,
    "relationship": "feeds_into",
    "strength": min()
    levels[current_level]["confidence"],
    levels[next_level]["confidence"]),
    }
    )

    return connections,
    def _synthesize_final_analysis(self, levels: Dict[str, Any]) -> Dict[str, Any]:
    """تجميع التحليل النهائي"""
    synthesis = {
    "overall_confidence": 0.0,
    "total_processing_time": 0.0,
    "analysis_completeness": 0.0,
    "key_features": {},
    "linguistic_summary": {},
    }

        # حساب الثقة الإجمالية,
    confidences = [level["confidence"] for level in levels.values()]
    synthesis["overall_confidence"] = ()
    sum(confidences) / len(confidences) if confidences else 0.0
    )

        # إجمالي وقت المعالجة,
    synthesis["total_processing_time"] = sum()
    level["processing_time"] for level in levels.values()
    )

        # نسبة اكتمال التحليل,
    expected_levels = 6  # عدد المحركات المتوقعة,
    synthesis["analysis_completeness"] = len(levels) / expected_levels

        # الخصائص الرئيسية,
    synthesis["key_features"] = self._extract_key_features(levels)

        # الملخص اللغوي,
    synthesis["linguistic_summary"] = self._create_linguistic_summary(levels)

    return synthesis,
    def _extract_key_features(self, levels: Dict[str, Any]) -> Dict[str, Any]:
    """استخراج الخصائص الرئيسية"""
    features = {}

        # من المستوى الصوتي,
    if "phoneme_harakah" in levels:
    features["phonetic"] = levels["phoneme_harakah"]["features"]

        # من المستوى الصرفي,
    if "morpheme_mapper" in levels:
    features["morphological"] = levels["morpheme_mapper"]["features"]

        # من المستوى النحوي,
    if "word_classifier" in levels:
    features["syntactic"] = levels["word_classifier"]["features"]

        # من المستوى الدلالي,
    if "semantic_role" in levels:
    features["semantic"] = levels["semantic_role"]["features"]

    return features,
    def _create_linguistic_summary(self, levels: Dict[str, Any]) -> Dict[str, str]:
    """إنشاء ملخص لغوي"""
    summary = {}

        # تحديد نوع الكلمة,
    if "word_classifier" in levels:
    word_type = levels["word_classifier"]["features"].get()
    "word_type", "unknown"
    )
    summary["word_type"] = word_type

        # تحديد الجذر إن وجد,
    if "morpheme_mapper" in levels:
    has_root = levels["morpheme_mapper"]["features"].get("has_root", False)
    summary["has_root"] = "yes" if has_root else "no"

        # تحديد الوزن الصرفي,
    if "weight_inference" in levels:
    has_pattern = levels["weight_inference"]["features"].get()
    "has_prosodic_pattern", False
    )
    summary["has_morphological_pattern"] = "yes" if has_pattern else "no"

        # تحديد الدور الدلالي,
    if "semantic_role" in levels:
    is_argument = levels["semantic_role"]["features"].get("is_argument", False)
    summary["semantic_role_type"] = "argument" if is_argument else "modifier"

    return summary,
    def generate_vector(self, analysis_data: Dict[str, Any]) -> List[float]:
    """توليد المتجه النهائي الشامل"""
    vector = []

        # متجهات من جميع المستويات,
    for level_name in [
    "phoneme_harakah",
    "syllable_pattern",
    "morpheme_mapper",
    "weight_inference",
    "word_classifier",
    "semantic_role",
    ]:
            if level_name in analysis_data["analysis_levels"]:
    level_vector = analysis_data["analysis_levels"][level_name]["vector"]
    vector.extend(level_vector)
            else:
    vector.extend([0.0] * 10)  # متجه فارغ للمستوى المفقود

        # معلومات التحليل النهائي,
    final_analysis = analysis_data["final_analysis"]
    vector.extend()
    [
    final_analysis["overall_confidence"],
    final_analysis["analysis_completeness"],
    final_analysis["total_processing_time"],
    len(analysis_data["connections"]),
    ]
    )

    return vector,
    def create_graph_node(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """إنشاء العقدة النهائية للشبكة"""
    return {
    "type": "word_trace",
    "level": 7,
    "word": analysis_data["word"],
    "analysis_levels": list(analysis_data["analysis_levels"].keys()),
    "connections": analysis_data["connections"],
    "final_analysis": analysis_data["final_analysis"],
    "features": {
    "is_complete": analysis_data["final_analysis"]["analysis_completeness"]
    >= 0.8,
    "high_confidence": analysis_data["final_analysis"]["overall_confidence"]
    >= 0.8,
    "all_levels_present": len(analysis_data["analysis_levels"]) >= 6,
    "processing_efficient": analysis_data["final_analysis"][
    "total_processing_time"
    ]
    < 1.0,
    },
    }


# ============== نظام التحكم الرئيسي ==============


class HierarchicalGraphSystem:
    """النظام الرئيسي للتحليل الهرمي الشبكي"""

    def __init__(self):

    self.engines = {
    AnalysisLevel.PHONEME_HARAKAH: UnifiedPhonemeHarakahEngine(),
    AnalysisLevel.SYLLABLE_PATTERN: SyllablePatternEngine(),
    AnalysisLevel.MORPHEME_MAPPER: MorphemeMapperEngine(),
    AnalysisLevel.WEIGHT_INFERENCE: WeightInferenceEngine(),
    AnalysisLevel.WORD_CLASSIFIER: WordClassifierEngine(),
    AnalysisLevel.SEMANTIC_ROLE: SemanticRoleEngine(),
    AnalysisLevel.WORD_TRACER: WordTraceGraph(),
    }
    self.global_graph = nx.DiGraph()

    def analyze_word(self, word: str) -> Dict[str, Any]:
    """تحليل شامل للكلمة عبر جميع المستويات"""
    results: Dict[str, Any] = {"original_word": word}

        # المستوى 1: تحليل الفونيمات والحركات,
    phoneme_result = self.engines[AnalysisLevel.PHONEME_HARAKAH].process(word)
    results["phoneme_harakah"] = {
    "vector": phoneme_result.vector,
    "confidence": phoneme_result.confidence,
    "processing_time": phoneme_result.processing_time,
    "graph_node": phoneme_result.graph_node,
    }

        # المستوى 2: تحليل المقاطع,
    syllable_result = self.engines[AnalysisLevel.SYLLABLE_PATTERN].process()
    phoneme_result.metadata.get("analysis_data")
    )
    results["syllable_pattern"] = {
    "vector": syllable_result.vector,
    "confidence": syllable_result.confidence,
    "processing_time": syllable_result.processing_time,
    "graph_node": syllable_result.graph_node,
    }

        # المستوى 3: تحليل البنية الصرفية,
    morpheme_result = self.engines[AnalysisLevel.MORPHEME_MAPPER].process()
    syllable_result.metadata.get("analysis_data")
    )
    results["morpheme_mapper"] = {
    "vector": morpheme_result.vector,
    "confidence": morpheme_result.confidence,
    "processing_time": morpheme_result.processing_time,
    "graph_node": morpheme_result.graph_node,
    }

        # المستوى 4: استنتاج الوزن,
    weight_result = self.engines[AnalysisLevel.WEIGHT_INFERENCE].process()
    morpheme_result.metadata.get("analysis_data")
    )
    results["weight_inference"] = {
    "vector": weight_result.vector,
    "confidence": weight_result.confidence,
    "processing_time": weight_result.processing_time,
    "graph_node": weight_result.graph_node,
    }

        # المستوى 5: تصنيف الكلمة,
    classifier_result = self.engines[AnalysisLevel.WORD_CLASSIFIER].process()
    weight_result.metadata.get("analysis_data")
    )
    results["word_classifier"] = {
    "vector": classifier_result.vector,
    "confidence": classifier_result.confidence,
    "processing_time": classifier_result.processing_time,
    "graph_node": classifier_result.graph_node,
    }

        # المستوى 6: الأدوار الدلالية,
    semantic_result = self.engines[AnalysisLevel.SEMANTIC_ROLE].process()
            classifier_result.metadata.get("analysis_data")
    )
    results["semantic_role"] = {
    "vector": semantic_result.vector,
    "confidence": semantic_result.confidence,
    "processing_time": semantic_result.processing_time,
    "graph_node": semantic_result.graph_node,
    }

        # المستوى 7: تتبع الكلمة النهائي,
    trace_result = self.engines[AnalysisLevel.WORD_TRACER].process(results)
    results["word_tracer"] = {
    "vector": trace_result.vector,
    "confidence": trace_result.confidence,
    "processing_time": trace_result.processing_time,
    "graph_node": trace_result.graph_node,
    }

        # بناء الشبكة المتكاملة,
    self._build_integrated_graph(word, results)

    return results,
    def _build_integrated_graph(self, word: str, results: Dict[str, Any]):
    """بناء الشبكة المتكاملة للكلمة"""
        # إضافة عقدة الكلمة الرئيسية,
    self.global_graph.add_node(f"word_{word}", type="root", word=word)

        # ربط النتائج,
    for level, result in results.items():
    node_id = f"{level}_{word}"
    self.global_graph.add_node(node_id, **result.graph_node)
    self.global_graph.add_edge()
    f"word_{word}", node_id, weight=result.confidence
    )

    def export_graph(self, format_type: str = "json") -> Union[str, Dict]:
    """تصدير الشبكة بصيغ مختلفة"""
        if format_type == "json":
    return nx.node_link_data(self.global_graph)
        elif format_type == "graphml":
    return nx.generate_graphml(self.global_graph)
        else:
    return {"error": "Unsupported format"}


def main():
    """دالة اختبار النظام"""
    print("🧠 اختبار النظام الهرمي الشبكي للتحليل اللغوي العربي")
    print("=" * 60)

    system = HierarchicalGraphSystem()

    test_words = ["كتاب", "مدرسة", "يكتب"]

    for word in test_words:
    print(f"\n🔍 تحليل كلمة: {word}")
    print(" " * 30)

    results = system.analyze_word(word)

        for level, result in results.items():
    print()
    f"   {level}: {len(result.vector)} أبعاد، ثقة: {result.confidence:.2f}"
    )

        # تصدير الشبكة,
    graph_data = system.export_graph()
    print(f"   🌐 عقد الشبكة: {len(graph_data['nodes'])}")


if __name__ == "__main__":
    import time,
    main()

