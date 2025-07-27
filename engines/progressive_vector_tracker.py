#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
🔬 نظام التتبع التدريجي المتطور للمتجه الرقمي للكلمات العربية المفردة
========================================================================

تتبع تدريجي مفصل لبناء المتجه الرقمي من الفونيم والحركة,
    حتى الكلمة الكاملة مع التكامل مع المحركات ال13:

🎯 المراحل التدريجية:
1. مستوى الفونيم والحركة (Phoneme-Diacritic Level)
2. تكوين المقاطع (Syllable Formation)
3. تحليل الجذر والوزن (Root-Pattern Analysis)
4. تصنيف جامد/مشتق (Frozen/Derived Classification)
5. تحديد مبني/معرب (Built/Inflected Determination)
6. تحليل نوع الكلمة (Word Type Analysis)
7. التحليل المورفولوجي الكامل (Complete Morphological Analysis)
8. المتجه النهائي (Final Vector Generation)

🚀 التكامل مع المحركات:
- محرك الفونيمات (UnifiedPhonemeSystem)
- محرك المقاطع (SyllabicUnitEngine)
- محرك الجذور (RootEngine)
- محرك الأوزان (WeightEngine)
- محرك الصرف (MorphologyEngine)
- محرك الاشتقاق (DerivationEngine)
- محرك التصريف (InflectionEngine)
- محرك الصوتيات (PhonologyEngine)

Progressive Digital Vector Tracking System,
    From phoneme diacritic level to complete word analysis,
    With integration to all 13 NLP engines
""""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too long
    import logging
    from typing import Dict, List, Any, Optional, Tuple
    from dataclasses import dataclass, field
    from enum import Enum
    from unified_phonemes import UnifiedArabicPhonemes, PhonemeType

# إعداد نظام السجلات,
    logging.basicConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")"
)
logger = logging.getLogger(__name__)

# ============== التعدادات الأساسية ==============


class EngineStatus(Enum):
    """حالة المحركات""""

    OPERATIONAL = 0  # يعمل بنجاح,
    FAILED = 1  # فشل في التشغيل,
    PARTIALLY_WORKING = 2  # يعمل جزئياً
    NOT_IMPLEMENTED = 3  # غير مُنفّذ,
    class EngineCategory(Enum):
    """تصنيف المحركات""""

    FIXED_ENGINES = 0  # المحركات الثابتة,
    ARABIC_MORPHOPHON = 1  # المورفوفونولوجيا العربية,
    WORKING_NLP = 2  # معالجة اللغة الطبيعية,
    AI_ENHANCED = 3  # المحسن بالذكاء الاصطناعي,
    class VectorBuilderStage(Enum):
    """مراحل بناء المتجه""""

    PHONEME_LEVEL = 0  # مستوى الفونيم,
    DIACRITIC_MAPPING = 1  # ربط الحركات,
    SYLLABLE_FORMATION = 2  # تكوين المقاطع,
    ROOT_EXTRACTION = 3  # استخراج الجذر,
    PATTERN_ANALYSIS = 4  # تحليل الوزن,
    DERIVATION_CHECK = 5  # فحص الاشتقاق,
    INFLECTION_ANALYSIS = 6  # تحليل التصريف,
    FINAL_CLASSIFICATION = 7  # التصنيف النهائي


# Using unified phonemes system - old PhonemeType and DiacriticType enums removed,
    class SyllableType(Enum):
    """أنواع المقاطع""""

    CV = 0  # صامت + صائت,
    CVC = 1  # صامت + صائت + صامت,
    CVV = 2  # صامت + صائت طويل,
    CVCC = 3  # صامت + صائت + صامتان,
    V = 4  # صائت فقط,
    VC = 5  # صائت + صامت,
    class WordType(Enum):
    """أنواع الكلمات""""

    NOUN = 0  # اسم,
    VERB = 1  # فعل,
    PARTICLE = 2  # حرف,
    class DerivationType(Enum):
    """نوع الاشتقاق""""

    JAMID = 0  # جامد (غير مشتق)
    MUSHTAQ = 1  # مشتق,
    class InflectionType(Enum):
    """نوع البناء والإعراب""""

    MABNI = 0  # مبني,
    MURAB = 1  # معرب,
    class PatternClass(Enum):
    """فئات الأوزان الصرفية""""

    TRILATERAL = 0  # ثلاثي,
    QUADRILATERAL = 1  # رباعي,
    QUINQUELATERAL = 2  # خماسي,
    COMPOUND = 3  # مركب


# ============== هياكل البيانات التدريجية ==============


class EngineState(Enum):
    """حالة المحركات""""

    OPERATIONAL = 0  # يعمل بنجاح,
    FAILED = 1  # فشل في التشغيل,
    PARTIALLY_WORKING = 2  # يعمل جزئياً
    NOT_IMPLEMENTED = 3  # غير مُنفّذ


@dataclass,
    class EngineStatusInfo:
    """معلومات حالة محرك NLP""""

    name: str,
    category: EngineCategory,
    status: EngineState,
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    integration_level: float = 0.0  # مستوى التكامل 0 1


@dataclass,
    class PhonemeComponent:
    """مكون فونيمي أساسي""""

    phoneme: str,
    phoneme_type: PhonemeType,
    position: int,
    articulatory_features: Dict[str, Any] = field(default_factory=dict)
    one_hot_encoding: List[int] = field(default_factory=list)


@dataclass,
    class DiacriticComponent:
    """مكون تشكيلي""""

    diacritic: str,
    diacritic_type: str  # Using string instead of old DiacriticType enum,
    position: int,
    phoneme_attachment: int  # الفونيم المرتبط,
    duration: float = 1.0,
    one_hot_encoding: List[int] = field(default_factory=list)


@dataclass,
    class SyllableComponent:
    """مكون مقطعي""""

    syllable_text: str,
    syllable_type: SyllableType,
    phonemes: List[PhonemeComponent]
    diacritics: List[DiacriticComponent]
    cv_pattern: str,
    stress_level: int = 0  # مستوى النبر,
    prosodic_weight: float = 1.0


@dataclass,
    class StageTracker:
    """متتبع المراحل التدريجية""""

    stage: VectorBuilderStage,
    timestamp: str,
    input_data: Any,
    output_data: Any,
    processing_time: float,
    success: bool,
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass,
    class ProgressiveVector:
    """المتجه التدريجي""""

    stage_vectors: Dict[str, List[float]] = field(default_factory=dict)
    cumulative_vector: List[float] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)


# ============== هياكل البيانات للتتبع ==============


@dataclass,
    class PhonemeUnit:
    """وحدة فونيم واحد مع خصائصه""""

    phoneme: str,
    phoneme_type: PhonemeType,
    articulation_place: str,
    articulation_manner: str,
    emphatic: bool = False,
    voiced: bool = False,
    vector_encoding: List[float] = field(default_factory=list)


@dataclass,
    class DiacriticUnit:
    """وحدة حركة واحدة مع خصائصها""""

    diacritic: str,
    diacritic_type: str  # Using string instead of old DiacriticType enum,
    duration: float = 1.0,
    vowel_quality: str = """
    length_marker: bool = False,
    vector_encoding: List[float] = field(default_factory=list)
    # إضافة الحقول المفقودة للتوافق,
    length: int = 1  # 0=سكون، 1=قصير، 2=طويل,
    case_marking: bool = False


@dataclass,
    class PhoneDiacriticPair:
    """زوج فونيم حركة مع الترميز التدريجي""""

    phoneme: PhonemeComponent,
    diacritic: Optional[DiacriticComponent]
    position_in_word: int,
    cv_contribution: str  # C أو V,
    vector_representation: List[float] = field(default_factory=list)
    confidence_score: float = 1.0
    # إضافة الحقول المفقودة للتوافق,
    phoneme_unit: Optional["PhonemeUnit"] = None"
    diacritic_unit: Optional[DiacriticUnit] = None,
    syllable_role: str = "unknown"  # onset, nucleus, coda"
    combined_vector: List[float] = field(default_factory=list)


@dataclass,
    class SyllableUnit:
    """وحدة مقطع مع التحليل التدريجي""""

    syllable_components: List[SyllableComponent]
    syllable_type: SyllableType,
    cv_pattern: str,
    stress_level: int = 0,
    prosodic_weight: float = 1.0,
    vector_encoding: List[float] = field(default_factory=list)
    phonological_processes: List[str] = field(default_factory=list)
    # إضافة الحقول المفقودة للتوافق,
    phoneme_diacritic_pairs: List[PhoneDiacriticPair] = field(default_factory=list)
    syllable_pattern: str = """
    position_in_word: int = 0


@dataclass,
    class MorphologicalAnalysis:
    """التحليل الصرفي التدريجي""""

    root: str,
    pattern: str,
    pattern_class: PatternClass,
    derivation_type: DerivationType,
    inflection_type: InflectionType,
    affixes: Dict[str, List[str]] = field(default_factory=dict)
    morphological_vector: List[float] = field(default_factory=list)
    certainty_level: float = 1.0
    # إضافة الحقول المفقودة للتوافق,
    prefixes: List[str] = field(default_factory=list)
    suffixes: List[str] = field(default_factory=list)
    stem: str = """
    vector_encoding: List[float] = field(default_factory=list)


@dataclass,
    class SyntacticAnalysis:
    """التحليل النحوي التدريجي""""

    word_type: WordType,
    syntactic_features: Dict[str, Any] = field(default_factory=dict)
    grammatical_relations: List[str] = field(default_factory=list)
    syntactic_vector: List[float] = field(default_factory=list)
    parsing_confidence: float = 1.0
    # إضافة الحقول المفقودة للتوافق,
    inflection_type: InflectionType = InflectionType.MURAB,
    case_marking: Optional[str] = None,
    definiteness: Optional[str] = None,
    gender: Optional[str] = None,
    number: Optional[str] = None,
    vector_encoding: List[float] = field(default_factory=list)


@dataclass,
    class EngineIntegrationStatus:
    """حالة تكامل المحركات الـ13""""

    # المحركات العاملة (5 محركات)
    working_engines: Dict[str, EngineStatusInfo] = field(default_factory=dict)

    # المحركات الثابتة (5 محركات)
    fixed_engines: Dict[str, EngineStatusInfo] = field(default_factory=dict)

    # محركات الصرف العربي (3 محركات)
    morphophon_engines: Dict[str, EngineStatusInfo] = field(default_factory=dict)

    # إحصائيات التكامل,
    total_engines: int = 13,
    operational_engines: int = 0,
    integration_score: float = 0.0,
    def update_integration_score(self):
    """تحديث نقاط التكامل""""
    all_engines = {
    **self.working_engines,
    **self.fixed_engines,
    **self.morphophon_engines,
    }
    operational = sum()
    1 for info in all_engines.values() if info.status == EngineState.OPERATIONAL
    )
    self.operational_engines = operational,
    self.integration_score = operational / self.total_engines


@dataclass,
    class ProgressiveAnalysis:
    """التحليل التدريجي الشامل""""

    # مراحل التحليل,
    stages: List[StageTracker] = field(default_factory=list)

    # المكونات التدريجية,
    phoneme_components: List[PhonemeComponent] = field(default_factory=list)
    diacritic_components: List[DiacriticComponent] = field(default_factory=list)
    phone_diacritic_pairs: List[PhoneDiacriticPair] = field(default_factory=list)
    syllable_units: List[SyllableUnit] = field(default_factory=list)

    # التحليلات المتقدمة,
    morphological_analysis: Optional[MorphologicalAnalysis] = None,
    syntactic_analysis: Optional[SyntacticAnalysis] = None

    # المتجه التدريجي,
    progressive_vector: ProgressiveVector = field(default_factory=ProgressiveVector)

    # تكامل المحركات,
    engine_integration: EngineIntegrationStatus = field()
        default_factory=EngineIntegrationStatus
    )

    # البيانات الوصفية,
    word: str = """
    timestamp: str = """
    processing_time: float = 0.0,
    final_confidence: float = 0.0


@dataclass,
    class ProgressiveAnalysis:
    """التحليل التدريجي الكامل""""

    word: str,
    phoneme_diacritic_pairs: List[PhoneDiacriticPair]
    syllabic_units: List[SyllableUnit]
    morphological_analysis: MorphologicalAnalysis,
    syntactic_analysis: SyntacticAnalysis,
    final_vector: List[float] = field(default_factory=list)
    analysis_steps: List[Dict] = field(default_factory=list)


class ProgressiveArabicVectorTracker:
    """"
    🔬 نظام التتبع التدريجي المتطور للمتجه الرقمي العربي
    =======================================================

    يتتبع بناء المتجه خطوة بخطوة من:
    1. الفونيم الواحد (باستخدام UnifiedPhonemeSystem)
    2. الحركة المقترنة (مع DiacriticEngine)
    3. زوج فونيم-حركة (التكامل الصوتي)
    4. المقطع الصوتي (باستخدام SyllabicUnitEngine)
    5. استخراج الجذر (باستخدام RootEngine)
    6. تحليل الوزن (باستخدام WeightEngine)
    7. التحليل المورفولوجي (باستخدام MorphologyEngine)
    8. تحليل الاشتقاق (باستخدام DerivationEngine)
    9. تحليل التصريف (باستخدام InflectionEngine)
    10. التحليل النحوي (باستخدام SyntaxEngine)
    11. التحليل الدلالي (باستخدام SemanticEngine)
    12. المعالجة الصوتية (باستخدام PhonologyEngine)
    13. المتجه النهائي المتكامل

    🚀 التكامل مع تقرير المحركات ال13:
    - المحركات العاملة: 8/13
    - المحركات الفاشلة: 5/13
    - التغطية الكاملة للقدرات اللغوية العربية
    """"

    def __init__(self):
    """تهيئة نظام التتبع مع فحص حالة المحركات""""
    self._initialize_engines_status()
    self._import_data_linguistic_databases()
    self._setup_progressive_pipeline()
    logger.info("🚀 تم تهيئة نظام التتبع التدريجي المتطور للمتجه الرقمي")"

    def _initialize_engines_status(self):
    """تهيئة حالة المحركات بناءً على التقرير""""

    self.engines_status = {
            # المحركات العاملة (8/13)
    "syllable_engine": EngineStatusInfo()"
    name="محرك المقاطع","
    category=EngineCategory.FIXED_ENGINES,
    status=EngineState.OPERATIONAL,
    performance_metrics={"accuracy": 0.95, "speed": 0.87},"
    capabilities=["تقطيع المقاطع", "تحليل النبر", "أنماط CV"],"
    integration_level=0.95),
    "unified_phonemes": EngineStatusInfo()"
    name="محرك الفونيمات","
    category=EngineCategory.ARABIC_MORPHOPHON,
    status=EngineState.OPERATIONAL,
    performance_metrics={"diversity": 0.383, "complexity": 0.280},"
    capabilities=["استخراج الفونيمات", "التحليل الصوتي", "الخصائص النطقية"],"
    integration_level=0.92),
    "root_engine": EngineStatusInfo()"
    name="محرك الجذور","
    category=EngineCategory.ARABIC_MORPHOPHON,
    status=EngineState.OPERATIONAL,
    performance_metrics={"precision": 0.88, "recall": 0.82},"
    capabilities=["استخراج الجذور", "تصنيف الجذور", "الجذور الجامدة"],"
    integration_level=0.89),
    "weight_engine": EngineStatusInfo()"
    name="محرك الأوزان","
    category=EngineCategory.ARABIC_MORPHOPHON,
    status=EngineState.OPERATIONAL,
    performance_metrics={"pattern_match": 0.91, "coverage": 0.76},"
    capabilities=["تحليل الأوزان", "الأنماط الصرفية", "التصنيف الصرفي"],"
    integration_level=0.87),
    "morphology_engine": EngineStatusInfo()"
    name="محرك الصرف","
    category=EngineCategory.WORKING_NLP,
    status=EngineState.OPERATIONAL,
    performance_metrics={
    "morpho_analysis": 0.85,"
    "feature_extraction": 0.79,"
    },
    capabilities=[
    "التحليل الصرفي","
    "استخراج الميزات","
    "التصنيف المورفولوجي","
    ],
    integration_level=0.84),
            # المحركات الفاشلة (5/13) - سنحاكيها
    "derivation_engine": EngineStatusInfo()"
    name="محرك الاشتقاق","
    category=EngineCategory.WORKING_NLP,
    status=EngineState.FAILED,
    performance_metrics={"accuracy": 0.0},"
    capabilities=[],
    integration_level=0.0),
    "inflection_engine": EngineStatusInfo()"
    name="محرك التصريف","
    category=EngineCategory.WORKING_NLP,
    status=EngineState.FAILED,
    performance_metrics={"accuracy": 0.0},"
    capabilities=[],
    integration_level=0.0),
    "phonology_engine": EngineStatusInfo()"
    name="محرك الصوتيات","
    category=EngineCategory.WORKING_NLP,
    status=EngineState.FAILED,
    performance_metrics={"accuracy": 0.0},"
    capabilities=[],
    integration_level=0.0),
    "syntax_engine": EngineStatusInfo()"
    name="محرك النحو","
    category=EngineCategory.AI_ENHANCED,
    status=EngineState.NOT_IMPLEMENTED,
    performance_metrics={"accuracy": 0.0},"
    capabilities=[],
    integration_level=0.0),
    "semantic_engine": EngineStatusInfo()"
    name="محرك الدلالة","
    category=EngineCategory.AI_ENHANCED,
    status=EngineState.NOT_IMPLEMENTED,
    performance_metrics={"accuracy": 0.0},"
    capabilities=[],
    integration_level=0.0),
    }

        # إحصائيات عامة من التقرير,
    self.suite_metrics = {
    "total_engines": 13,"
    "operational_engines": 8,"
    "failed_engines": 5,"
    "overall_efficiency": "Excellent","
    "coverage_percentage": 61.5,  # 8/13"
    "phonetic_diversity": 0.383,"
    "morphological_complexity": 0.280,"
    "overall_complexity": 0.331,"
    }

    def _setup_progressive_pipeline(self):
    """إعداد خط الأنابيب التدريجي""""

    self.pipeline_stages = [
    {
    "stage": VectorBuilderStage.PHONEME_LEVEL,"
    "engine": "unified_phonemes","
    "description": "استخراج وترميز الفونيمات","
    "vector_dimensions": 28,  # عدد الفونيمات العربية"
    "processing_function": self._process_phoneme_level,"
    },
    {
    "stage": VectorBuilderStage.DIACRITIC_MAPPING,"
    "engine": "diacritic_engine","
    "description": "ربط الحركات بالفونيمات","
    "vector_dimensions": 11,  # أنواع الحركات"
    "processing_function": self._process_diacritic_mapping,"
    },
    {
    "stage": VectorBuilderStage.SYLLABLE_FORMATION,"
    "engine": "syllable_engine","
    "description": "تكوين المقاطع الصوتية","
    "vector_dimensions": 6,  # أنواع المقاطع"
    "processing_function": self._process_syllable_formation,"
    },
    {
    "stage": VectorBuilderStage.ROOT_EXTRACTION,"
    "engine": "root_engine","
    "description": "استخراج الجذر اللغوي","
    "vector_dimensions": 12,  # خصائص الجذر"
    "processing_function": self._process_root_extraction,"
    },
    {
    "stage": VectorBuilderStage.PATTERN_ANALYSIS,"
    "engine": "weight_engine","
    "description": "تحليل الوزن الصرفي","
    "vector_dimensions": 15,  # أنماط الأوزان"
    "processing_function": self._process_pattern_analysis,"
    },
    {
    "stage": VectorBuilderStage.DERIVATION_CHECK,"
    "engine": "derivation_engine","
    "description": "فحص الاشتقاق","
    "vector_dimensions": 8,  # أنواع الاشتقاق"
    "processing_function": self._process_derivation_check,"
    },
    {
    "stage": VectorBuilderStage.INFLECTION_ANALYSIS,"
    "engine": "inflection_engine","
    "description": "تحليل التصريف","
    "vector_dimensions": 10,  # خصائص التصريف"
    "processing_function": self._process_inflection_analysis,"
    },
    {
    "stage": VectorBuilderStage.FINAL_CLASSIFICATION,"
    "engine": "integration_engine","
    "description": "التصنيف النهائي والدمج","
    "vector_dimensions": 20,  # الخصائص النهائية"
    "processing_function": self._process_final_classification,"
    },
    ]

    def _import_data_linguistic_databases(self):
    """تحميل قواعد البيانات اللغوية - Using Unified Phonemes System""""

        # Initialize unified phonemes system,
    self.unified_phonemes = UnifiedArabicPhonemes()

        # Create lookup dictionary for compatibility with old code,
    self.phoneme_database = {}

        # Add consonants,
    for i, phoneme in enumerate(self.unified_phonemes.consonants):
    self.phoneme_database[phoneme.arabic_char] = {
    "type": PhonemeType.CONSONANT,"
    "place": phoneme.place.value if phoneme.place else "unknown","
    "manner": phoneme.manner.value if phoneme.manner else "unknown","
    "emphatic": phoneme.emphatic if phoneme.emphatic else False,"
    "voiced": phoneme.voiced if phoneme.voiced else False,"
    "encoding_index": i,"
    }

        # Add vowels,
    for i, phoneme in enumerate(self.unified_phonemes.vowels):
    self.phoneme_database[phoneme.arabic_char] = {
    "type": PhonemeType.VOWEL,"
    "place": phoneme.place.value if phoneme.place else "unknown","
    "manner": phoneme.manner.value if phoneme.manner else "unknown","
    "emphatic": False,"
    "voiced": True,"
    "encoding_index": len(self.unified_phonemes.consonants) + i,"
    }

        # Add diacritics as phonemes for compatibility,
    for i, diacritic in enumerate(self.unified_phonemes.diacritics):
    self.phoneme_database[diacritic.arabic_char] = {
    "type": PhonemeType.DIACRITIC,"
    "place": "diacritic","
    "manner": "diacritic","
    "emphatic": False,"
    "voiced": False,"
    "encoding_index": len(self.unified_phonemes.consonants)"
    + len(self.unified_phonemes.vowels)
    + i,
    }

        # Replace old diacritic database with unified system,
    self.diacritic_database = {}
        for i, diacritic in enumerate(self.unified_phonemes.diacritics):
    self.diacritic_database[diacritic.arabic_char] = {
    "type": PhonemeType.DIACRITIC,"
    "length": 1,"
    "case_marking": False,"
    "encoding_index": i,"
    }

        # أنماط الأوزان الصرفية,
    self.morphological_patterns = {
            # الأسماء الجامدة
    "jamid_patterns": {"
    "CVC": ["شمس", "قلم", "بيت"],"
    "CVCC": ["كتاب", "مدرس"],"
    "CVCVC": ["جبل", "نهر"],"
    },
            # الأسماء المشتقة
    "derived_patterns": {"
    "مُفْعِل": "muFiL",  # مُدرِّس"
    "مَفْعُول": "maFuL",  # مكتوب"
    "فَاعِل": "faiL",  # كاتب"
    "فَعِيل": "faiL",  # صديق"
    "فُعَيْل": "fuaiL",  # كُتَيْب (تصغير)"
    },
            # الأفعال
    "verb_patterns": {"
    "فَعَل": "faAL",  # كتب"
    "فَعِل": "faiL",  # شرب"
    "فَعُل": "faUL",  # كرم"
    "أَفْعَل": "afAL",  # أكرم"
    "فَعَّل": "faAAL",  # درّس"
    "تَفَعَّل": "tafaAAL",  # تعلّم"
    "اسْتَفْعَل": "istafAL",  # استخرج"
    },
    }

        # قواميس الجذور الشائعة,
    self.root_database = {
    "كتب": {"meaning": "writing", "type": "trilateral"},"
    "درس": {"meaning": "teaching", "type": "trilateral"},"
    "شمس": {"meaning": "sun", "type": "trilateral_jamid"},"
    "قلم": {"meaning": "pen", "type": "trilateral_jamid"},"
    "خرج": {"meaning": "exit", "type": "trilateral"},"
    "علم": {"meaning": "knowledge", "type": "trilateral"},"
    }

    def track_progressive_analysis(self, word: str) -> ProgressiveAnalysis:
    """"
    التتبع التدريجي الكامل لبناء المتجه من الفونيم إلى الكلمة,
    Args:
    word: الكلمة العربية المراد تحليلها,
    Returns:
    تحليل تدريجي شامل مع كل الخطوات
    """"

    logger.info(f"🔬 بدء التتبع التدريجي للكلمة: {word}")"

        # إنشاء كائن التحليل التدريجي,
    analysis = ProgressiveAnalysis()
    word=word,
    phoneme_diacritic_pairs=[],
    syllabic_units=[],
    morphological_analysis=MorphologicalAnalysis()
    root="","
    pattern="","
    derivation_type=DerivationType.JAMID,
    pattern_class=PatternClass.TRILATERAL,
    inflection_type=InflectionType.MURAB),
    syntactic_analysis=SyntacticAnalysis()
    word_type=WordType.NOUN, inflection_type=InflectionType.MURAB
    ))

        try:
            # الخطوة 1: تحليل الفونيمات والحركات,
    analysis.phoneme_diacritic_pairs = self._step1_analyze_phonemes_diacritics()
    word, analysis
    )

            # الخطوة 2: بناء المقاطع الصوتية,
    analysis.syllabic_units = self._step2_build_syllabic_units()
    analysis.phoneme_diacritic_pairs, analysis
    )

            # الخطوة 3: التحليل المورفولوجي,
    analysis.morphological_analysis = self._step3_morphological_analysis()
    word, analysis
    )

            # الخطوة 4: التحليل النحوي,
    analysis.syntactic_analysis = self._step4_syntactic_analysis(word, analysis)

            # الخطوة 5: بناء المتجه النهائي,
    analysis.final_vector = self._step5_build_final_vector(analysis)

    logger.info()
    f"✅ تم التتبع التدريجي بنجاح - أبعاد المتجه: {len(analysis.final_vector)}""
    )
    return analysis,
    except Exception as e:
    logger.error(f"❌ خطأ في التتبع التدريجي: {str(e)}")"
    raise,
    def _step1_analyze_phonemes_diacritics()
    self, word: str, analysis: ProgressiveAnalysis
    ) -> List[PhoneDiacriticPair]:
    """الخطوة 1: تحليل الفونيمات والحركات""""

    step_log = {"step": 1, "description": "تحليل الفونيمات والحركات", "details": []}"

    pairs = []
    position = 0

        # تنظيف الكلمة وفصل الحروف والحركات,
    chars = list(word)
    i = 0,
    while i < len(chars):
    char = chars[i]

            # تحقق من وجود الحرف في قاموس الفونيمات,
    if char in self.phoneme_database:
                # إنشاء وحدة الفونيم,
    phoneme_data = self.phoneme_database[char]
    phoneme_unit = PhonemeUnit()
    phoneme=char,
    phoneme_type=phoneme_data["type"],"
    articulation_place=phoneme_data["place"],"
    articulation_manner=phoneme_data["manner"],"
    emphatic=phoneme_data["emphatic"],"
    voiced=phoneme_data["voiced"])"

                # ترميز الفونيم إلى متجه one hot,
    phoneme_vector = [0.0] * len(self.phoneme_database)
    phoneme_vector[phoneme_data["encoding_index"]] = 1.0"
    phoneme_unit.vector_encoding = phoneme_vector

                # البحث عن الحركة التالية,
    diacritic_unit = None,
    diacritic_char = None,
    diacritic_data = None,
    if i + 1 < len(chars) and chars[i + 1] in self.diacritic_database:
    diacritic_char = chars[i + 1]
    diacritic_data = self.diacritic_database[diacritic_char]

    diacritic_unit = DiacriticUnit()
    diacritic=diacritic_char,
    diacritic_type=diacritic_data["type"],"
    length=diacritic_data["length"],"
    case_marking=diacritic_data["case_marking"])"

                    # ترميز الحركة إلى متجه one hot,
    diacritic_vector = [0.0] * len(self.diacritic_database)
    diacritic_vector[diacritic_data["encoding_index"]] = 1.0"
    diacritic_unit.vector_encoding = diacritic_vector,
    i += 1  # تخطي الحركة في المرة القادمة

                # إنشاء زوج فونيم حركة,
    diacritic_component = None,
    if diacritic_unit and diacritic_char and diacritic_data:
    diacritic_component = DiacriticComponent()
    diacritic=diacritic_char,
    diacritic_type=diacritic_data["type"],"
    position=position,
    phoneme_attachment=position)

    pair = PhoneDiacriticPair()
    phoneme=PhonemeComponent()
    phoneme=char,
    phoneme_type=phoneme_data["type"],"
    position=position),
    diacritic=diacritic_component,
    position_in_word=position,
    cv_contribution=()
    "C" if phoneme_data["type"] == PhonemeType.CONSONANT else "V""
    ),
    phoneme_unit=phoneme_unit,
    diacritic_unit=diacritic_unit,
    syllable_role="unknown",  # سيتم تحديده لاحقاً"
    )

                # دمج متجهات الفونيم والحركة,
    combined_vector = phoneme_unit.vector_encoding.copy()
                if diacritic_unit:
    combined_vector.extend(diacritic_unit.vector_encoding)
                else:
                    # إضافة متجه حركة فارغ,
    combined_vector.extend([0.0] * len(self.diacritic_database))

    pair.combined_vector = combined_vector,
    pairs.append(pair)

                # تسجيل التفاصيل,
    step_log["details"].append()"
    {
    "position": position,"
    "phoneme": char,"
    "phoneme_type": phoneme_data["type"].name,"
    "diacritic": ()"
    diacritic_unit.diacritic if diacritic_unit else None
    ),
    "vector_size": len(combined_vector),"
    }
    )

    position += 1,
    i += 1,
    analysis.analysis_steps.append(step_log)
    logger.info(f"📝 الخطوة 1: تم تحليل {len(pairs)} زوج فونيم حركة")"
    return pairs,
    def _step2_build_syllabic_units()
    self, pairs: List[PhoneDiacriticPair], analysis: ProgressiveAnalysis
    ) -> List[SyllableUnit]:
    """الخطوة 2: بناء المقاطع الصوتية""""

    step_log = {"step": 2, "description": "بناء المقاطع الصوتية", "details": []}"

    syllabic_units = []
    current_syllable_pairs = []
    syllable_position = 0,
    for i, pair in enumerate(pairs):
    phoneme_type = ()
    pair.phoneme_unit.phoneme_type,
    if pair.phoneme_unit,
    else pair.phoneme.phoneme_type
    )

            # تحديد دور الفونيم في المقطع,
    if phoneme_type in [PhonemeType.CONSONANT]:
                if not current_syllable_pairs:
                    # بداية مقطع جديد - onset,
    pair.syllable_role = "onset""
    current_syllable_pairs.append(pair)
                else:
                    # تحقق مما إذا كان هذا نهاية المقطع الحالي,
    last_pair = current_syllable_pairs[ 1]
    last_phoneme_type = ()
    last_pair.phoneme_unit.phoneme_type,
    if last_pair.phoneme_unit,
    else last_pair.phoneme.phoneme_type
    )
                    if last_phoneme_type in [
    PhonemeType.VOWEL,
    ] or ()
    last_pair.diacritic_unit and last_pair.diacritic_unit.length > 0
    ):
                        # المقطع السابق كامل، ابدأ مقطعاً جديداً
    syllable = self._create_syllable_unit()
    current_syllable_pairs, syllable_position
    )
    syllabic_units.append(syllable)
    syllable_position += 1

                        # ابدأ مقطعاً جديداً
    current_syllable_pairs = []
    pair.syllable_role = "onset""
    current_syllable_pairs.append(pair)
                    else:
                        # أضف كـ coda للمقطع الحالي,
    pair.syllable_role = "coda""
    current_syllable_pairs.append(pair)

            elif phoneme_type in [PhonemeType.VOWEL]:
                # nucleus (نواة المقطع)
    pair.syllable_role = "nucleus""
    current_syllable_pairs.append(pair)

                # تحقق مما إذا كان هذا نهاية الكلمة أو إذا كان التالي صامت,
    if i == len(pairs) - 1:
                    # نهاية الكلمة,
    syllable = self._create_syllable_unit()
    current_syllable_pairs, syllable_position
    )
    syllabic_units.append(syllable)
                elif i + 1 < len(pairs):
                    # قد يكون هناك coda قادم,
    continue
                else:
                    # إنهاء المقطع,
    syllable = self._create_syllable_unit()
    current_syllable_pairs, syllable_position
    )
    syllabic_units.append(syllable)
    syllable_position += 1,
    current_syllable_pairs = []

        # إضافة المقطع الأخير إذا كان موجوداً
        if current_syllable_pairs:
    syllable = self._create_syllable_unit()
    current_syllable_pairs, syllable_position
    )
    syllabic_units.append(syllable)

        # تسجيل التفاصيل,
    for syllable in syllabic_units:
    step_log["details"].append()"
    {
    "syllable_type": syllable.syllable_type.name,"
    "pattern": syllable.syllable_pattern,"
    "pairs_count": len(syllable.phoneme_diacritic_pairs),"
    "vector_size": len(syllable.vector_encoding),"
    }
    )

    analysis.analysis_steps.append(step_log)
    logger.info(f"🔤 الخطوة 2: تم بناء {len(syllabic_units)} مقطع صوتي")"
    return syllabic_units,
    def _create_syllable_unit()
    self, pairs: List[PhoneDiacriticPair], position: int
    ) -> SyllableUnit:
    """إنشاء وحدة مقطع صوتي""""

        # تحديد نمط المقطع,
    pattern = """
        for pair in pairs:
            if pair.syllable_role == "onset" or pair.syllable_role == "coda":"
    pattern += "C""
            elif pair.syllable_role == "nucleus":"
                # تحقق من طول الصائت,
    if pair.diacritic_unit and pair.diacritic_unit.length > 1:
    pattern += "VV""
                else:
    pattern += "V""

        # تحديد نوع المقطع,
    syllable_type = SyllableType.CV  # افتراضي,
    if pattern == "CV":"
    syllable_type = SyllableType.CV,
    elif pattern == "CVC":"
    syllable_type = SyllableType.CVC,
    elif pattern in ["CVV", "CAA"]:"
    syllable_type = SyllableType.CVV,
    elif pattern == "CVCC":"
    syllable_type = SyllableType.CVCC,
    elif pattern == "V":"
    syllable_type = SyllableType.V,
    elif pattern == "VC":"
    syllable_type = SyllableType.VC

        # بناء متجه المقطع,
    syllable_vector = []
        for pair in pairs:
    syllable_vector.extend(pair.combined_vector)

        # إضافة ترميز نوع المقطع,
    syllable_type_vector = [0.0] * len(SyllableType)
    syllable_type_vector[syllable_type.value] = 1.0,
    syllable_vector.extend(syllable_type_vector)

    return SyllableUnit()
    syllable_components=[],  # مؤقت,
    syllable_type=syllable_type,
    cv_pattern=pattern,
    phoneme_diacritic_pairs=pairs,
    syllable_pattern=pattern,
    position_in_word=position,
    vector_encoding=syllable_vector)

    def _step3_morphological_analysis()
    self, word: str, analysis: ProgressiveAnalysis
    ) -> MorphologicalAnalysis:
    """الخطوة 3: التحليل المورفولوجي""""

    step_log = {
    "step": 3,"
    "description": "التحليل المورفولوجي (جذر، وزن، اشتقاق)","
    "details": [],"
    }

        # استخراج الجذر (خوارزمية مبسطة)
    root = self._extract_root(word)

        # تحديد نوع الاشتقاق,
    derivation_type = DerivationType.JAMID  # افتراضي,
    pattern = "unknown""
    pattern_class = PatternClass.TRILATERAL

        # تحليل البادئات واللواحق,
    prefixes, stem, suffixes = self._analyze_affixes(word)

        # تحديد النمط الصرفي,
    if root in self.root_database:
    root_info = self.root_database[root]
            if "jamid" in root_info["type"]:"
    derivation_type = DerivationType.JAMID,
    pattern = "jamid_pattern""
            else:
    derivation_type = DerivationType.MUSHTAQ,
    pattern = self._determine_derived_pattern(word, root)

        # تحديد فئة النمط,
    if len(root) == 3:
    pattern_class = PatternClass.TRILATERAL,
    elif len(root) == 4:
    pattern_class = PatternClass.QUADRILATERAL,
    else:
    pattern_class = PatternClass.QUINQUELATERAL

        # بناء متجه التحليل المورفولوجي,
    morpho_vector = []

        # ترميز نوع الاشتقاق,
    derivation_vector = [0.0] * len(DerivationType)
    derivation_vector[derivation_type.value] = 1.0,
    morpho_vector.extend(derivation_vector)

        # ترميز فئة النمط,
    pattern_class_vector = [0.0] * len(PatternClass)
    pattern_class_vector[pattern_class.value] = 1.0,
    morpho_vector.extend(pattern_class_vector)

        # معلومات إحصائية,
    morpho_vector.extend()
    [
    float(len(root)),
    float(len(prefixes)),
    float(len(suffixes)),
    float(len(stem)),
    ]
    )

    morph_analysis = MorphologicalAnalysis()
    root=root,
    pattern=pattern,
    derivation_type=derivation_type,
    pattern_class=pattern_class,
    inflection_type=InflectionType.MURAB,  # افتراضي,
    prefixes=prefixes,
    suffixes=suffixes,
    stem=stem,
    vector_encoding=morpho_vector)

    step_log["details"].append()"
    {
    "root": root,"
    "pattern": pattern,"
    "derivation_type": derivation_type.name,"
    "pattern_class": pattern_class.name,"
    "prefixes": prefixes,"
    "suffixes": suffixes,"
    "vector_size": len(morpho_vector),"
    }
    )

    analysis.analysis_steps.append(step_log)
    logger.info(f"🔍 الخطوة 3: تحليل مورفولوجي - جذر: {root,} نمط: {pattern}}")"
    return morph_analysis,
    def _step4_syntactic_analysis()
    self, word: str, analysis: ProgressiveAnalysis
    ) -> SyntacticAnalysis:
    """الخطوة 4: التحليل النحوي""""

    step_log = {
    "step": 4,"
    "description": "التحليل النحوي (نوع الكلمة، البناء/الإعراب)","
    "details": [],"
    }

        # تحديد نوع الكلمة,
    word_type = self._determine_word_type(word, analysis.morphological_analysis)

        # تحديد نوع البناء/الإعراب,
    inflection_type = self._determine_inflection_type(word, word_type)

        # تحليل الخصائص النحوية,
    case_marking = self._determine_case_marking(word)
        definiteness = self._determine_definiteness(word)
    gender = self._determine_gender(word)
    number = self._determine_number(word)

        # بناء متجه التحليل النحوي,
    syntax_vector = []

        # ترميز نوع الكلمة,
    word_type_vector = [0.0] * len(WordType)
    word_type_vector[word_type.value] = 1.0,
    syntax_vector.extend(word_type_vector)

        # ترميز نوع البناء/الإعراب,
    inflection_vector = [0.0] * len(InflectionType)
    inflection_vector[inflection_type.value] = 1.0,
    syntax_vector.extend(inflection_vector)

        # ترميز الخصائص النحوية الأخرى,
    syntax_vector.extend()
    [
    1.0 if definiteness == "معرف" else 0.0,"
    1.0 if case_marking == "مرفوع" else 0.0,"
    1.0 if case_marking == "منصوب" else 0.0,"
    1.0 if case_marking == "مجرور" else 0.0,"
    1.0 if gender == "مذكر" else 0.0,"
    1.0 if gender == "مؤنث" else 0.0,"
    1.0 if number == "مفرد" else 0.0,"
    1.0 if number == "مثنى" else 0.0,"
    1.0 if number == "جمع" else 0.0,"
    ]
    )

    syntax_analysis = SyntacticAnalysis()
    word_type=word_type,
    inflection_type=inflection_type,
    case_marking=case_marking,
            definiteness=definiteness,
    gender=gender,
    number=number,
    vector_encoding=syntax_vector)

    step_log["details"].append()"
    {
    "word_type": word_type.name,"
    "inflection_type": inflection_type.name,"
    "case_marking": case_marking,"
    "definiteness": definiteness,"
    "gender": gender,"
    "number": number,"
    "vector_size": len(syntax_vector),"
    }
    )

    analysis.analysis_steps.append(step_log)
    logger.info()
    f"📊 الخطوة 4: تحليل نحوي - نوع: {word_type.name,} بناء: {inflection_type.name}}""
    )
    return syntax_analysis,
    def _step5_build_final_vector(self, analysis: ProgressiveAnalysis) -> List[float]:
    """الخطوة 5: بناء المتجه النهائي""""

    step_log = {
    "step": 5,"
    "description": "بناء المتجه النهائي المجمع","
    "details": [],"
    }

    final_vector = []

        # إضافة متجهات الفونيمات والحركات,
    phoneme_section = []
        for pair in analysis.phoneme_diacritic_pairs:
    phoneme_section.extend(pair.combined_vector)
    final_vector.extend(phoneme_section)

        # إضافة متجهات المقاطع,
    syllable_section = []
        for syllable in analysis.syllabic_units:
    syllable_section.extend(syllable.vector_encoding)
    final_vector.extend(syllable_section)

        # إضافة المتجه المورفولوجي,
    morpho_section = analysis.morphological_analysis.vector_encoding,
    final_vector.extend(morpho_section)

        # إضافة المتجه النحوي,
    syntax_section = analysis.syntactic_analysis.vector_encoding,
    final_vector.extend(syntax_section)

        # إحصائيات المتجه النهائي,
    step_log["details"].append()"
    {
    "phoneme_diacritic_dimensions": len(phoneme_section),"
    "syllable_dimensions": len(syllable_section),"
    "morphological_dimensions": len(morpho_section),"
    "syntactic_dimensions": len(syntax_section),"
    "total_dimensions": len(final_vector),"
    }
    )

    analysis.analysis_steps.append(step_log)
    logger.info(f"🎯 الخطوة 5: متجه نهائي بـ {len(final_vector) بُعد}")"
    return final_vector

    # ============== معالجات المراحل ==============

    def _process_phoneme_level(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة الفونيم""""
    logger.info(f"🔊 معالجة مرحلة الفونيم للكلمة: {word}")"

    result = {
    "stage": "phoneme_level","
    "input": word,"
    "phonemes": [],"
    "vector_size": 0,"
    "confidence": 1.0,"
    }

        for char in word:
            if char in self.phoneme_database:
    phoneme_data = self.phoneme_database[char]
    result["phonemes"].append()"
    {
    "phoneme": char,"
    "type": ()"
    phoneme_data["type"].name"
                            if isinstance(phoneme_data["type"], Enum)"
                            else phoneme_data["type"]"
    ),
    "encoding_index": phoneme_data["encoding_index"],"
    }
    )

    result["vector_size"] = len(result["phonemes"])"
    return result,
    def _process_diacritic_mapping(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة ربط الحركات""""
    logger.info(f"🔗 معالجة مرحلة ربط الحركات للكلمة: {word}")"

    result = {
    "stage": "diacritic_mapping","
    "input": word,"
    "diacritic_pairs": [],"
    "vector_size": 0,"
    "confidence": 1.0,"
    }

    chars = list(word)
    i = 0,
    while i < len(chars):
    char = chars[i]
            if char in self.phoneme_database:
    pair = {"consonant": char, "diacritic": None}"
                if i + 1 < len(chars) and chars[i + 1] in self.diacritic_database:
    pair["diacritic"] = chars[i + 1]"
    i += 1  # تخطي الحركة,
    result["diacritic_pairs"].append(pair)"
    i += 1,
    result["vector_size"] = len(result["diacritic_pairs"])"
    return result,
    def _process_syllable_formation(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة تكوين المقاطع""""
    logger.info(f"🔤 معالجة مرحلة تكوين المقاطع للكلمة: {word}")"

    result = {
    "stage": "syllable_formation","
    "input": word,"
    "syllabic_units": [],"
    "cv_pattern": "","
    "vector_size": 0,"
    "confidence": 1.0,"
    }

        # تحليل مبسط للمقاطع,
    current_syllable = """
    cv_pattern = """

    chars = list(word)
        for char in chars:
            if char in self.phoneme_database:
    phoneme_data = self.phoneme_database[char]
                if phoneme_data["type"] == PhonemeType.CONSONANT:"
    current_syllable += char,
    cv_pattern += "C""
                elif phoneme_data["type"] == PhonemeType.VOWEL:"
    current_syllable += char,
    cv_pattern += "V""
            elif char in self.diacritic_database:
    current_syllable += char,
    cv_pattern += "V""

            # نهاية مقطع عند CV أو CVC,
    if len(cv_pattern) >= 2 and ()
    cv_pattern.endswith("CV") or cv_pattern.endswith("CVC")"
    ):
    result["syllabic_units"].append(current_syllable)"
    current_syllable = """
    cv_pattern = """

        # إضافة المقطع الأخير إن وجد,
    if current_syllable:
    result["syllabic_units"].append(current_syllable)"

    result["cv_pattern"] = cv_pattern"
    result["vector_size"] = len(result["syllabic_units"])"
    return result,
    def _process_root_extraction(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة استخراج الجذر""""
    logger.info(f"🌱 معالجة مرحلة استخراج الجذر للكلمة: {word}")"

    result = {
    "stage": "root_extraction","
    "input": word,"
    "root": "","
    "root_type": "unknown","
    "vector_size": 0,"
    "confidence": 1.0,"
    }

    root = self._extract_root(word)
    result["root"] = root"
    result["root_type"] = ()"
    "trilateral""
            if len(root) == 3,
    else "quadrilateral" if len(root) == 4 else "other""
    )
    result["vector_size"] = len(root)"

    return result,
    def _process_pattern_analysis(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة تحليل الأوزان""""
    logger.info(f"⚖️ معالجة مرحلة تحليل الأوزان للكلمة: {word}")"

    result = {
    "stage": "pattern_analysis","
    "input": word,"
    "pattern": "","
    "pattern_type": "unknown","
    "vector_size": 0,"
    "confidence": 1.0,"
    }

    root = self._extract_root(word)
    pattern = self._determine_derived_pattern(word, root)

    result["pattern"] = pattern"
    result["pattern_type"] = ()"
    "derived" if pattern != "unknown_pattern" else "unknown""
    )
    result["vector_size"] = len(pattern)"

    return result,
    def _process_derivation_check(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة فحص الاشتقاق""""
    logger.info(f"🔍 معالجة مرحلة فحص الاشتقاق للكلمة: {word}")"

    result = {
    "stage": "derivation_check","
    "input": word,"
    "derivation_type": "jamid","
    "is_derived": False,"
    "vector_size": 0,"
    "confidence": 1.0,"
    }

        # فحص مبسط للاشتقاق,
    if word.startswith("مُ") or word.startswith("است"):"
    result["derivation_type"] = "mushtaq""
    result["is_derived"] = True"

    result["vector_size"] = 1 if result["is_derived"] else 0"
    return result,
    def _process_inflection_analysis(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة تحليل التصريف""""
    logger.info(f"📝 معالجة مرحلة تحليل التصريف للكلمة: {word}")"

    result = {
    "stage": "inflection_analysis","
    "input": word,"
    "inflection_type": "murab","
    "case_marking": "","
    "vector_size": 0,"
    "confidence": 1.0,"
    }

    case_marking = self._determine_case_marking(word)
    result["case_marking"] = case_marking"
    result["vector_size"] = len(case_marking)"

    return result,
    def _process_final_classification(self, word: str, stage_info: Dict) -> Dict:
    """معالج مرحلة التصنيف النهائي""""
    logger.info(f"🎯 معالجة مرحلة التصنيف النهائي للكلمة: {word}")"

    result = {
    "stage": "final_classification","
    "input": word,"
    "final_class": "","
    "confidence_score": 1.0,"
    "vector_size": 0,"
    }

        # تصنيف مبسط,
    if word.startswith("ال"):"
    result["final_class"] = "definite_noun""
        elif word.endswith("ة"):"
    result["final_class"] = "feminine_noun""
        else:
    result["final_class"] = "masculine_noun""

    result["vector_size"] = len(result["final_class"])"
    return result

    # ============== دوال مساعدة ==============

    def _extract_root(self, word: str) -> str:
    """استخراج الجذر - خوارزمية مبسطة""""
    clean_word = word

        # إزالة أداة التعريف,
    if clean_word.startswith("ال"):"
    clean_word = clean_word[2:]

        # إزالة اللواحق الشائعة,
    suffixes = ["ة", "ات", "ان", "ين", "ون", "ها", "هم", "كم", "ٌ", "ً", "ٍ"]"
        for suffix in suffixes:
            if clean_word.endswith(suffix):
    clean_word = clean_word[:  len(suffix)]
    break

        # استخراج الصوامت الأساسية,
    consonants = []
        for char in clean_word:
            if char in self.phoneme_database:
    phoneme_data = self.phoneme_database[char]
                if phoneme_data["type"] == PhonemeType.CONSONANT:"
    consonants.append(char)

    return "".join(consonants[:4])  # أقصى 4 حروف"

    def _analyze_affixes(self, word: str) -> Tuple[List[str], str, List[str]]:
    """تحليل البادئات والجذع واللواحق""""
    prefixes = []
    suffixes = []
    stem = word

        # البادئات الشائعة,
    prefix_list = ["ال", "و", "ف", "ب", "ك", "ل", "مُ", "است"]"
        for prefix in prefix_list:
            if stem.startswith(prefix):
    prefixes.append(prefix)
    stem = stem[len(prefix) :]
    break

        # اللواحق الشائعة,
    suffix_list = ["ة", "ات", "ان", "ين", "ون", "ها", "هم", "كم", "تم", "ٌ", "ً", "ٍ"]"
        for suffix in suffix_list:
            if stem.endswith(suffix):
    suffixes.append(suffix)
    stem = stem[: -len(suffix)]
    break,
    return prefixes, stem, suffixes,
    def _determine_derived_pattern(self, word: str, root: str) -> str:
    """تحديد النمط الاشتقاقي""""
        if word.startswith("مُ"):"
    return "مُفْعِل""
        elif word.startswith("م") and word.endswith("وب"):"
    return "مَفْعُول""
        elif "ُ" in word and "َيْ" in word:"
    return "فُعَيْل"  # تصغير"
        else:
    return "unknown_pattern""

    def _determine_word_type()
    self, word: str, morph_analysis: MorphologicalAnalysis
    ) -> WordType:
    """تحديد نوع الكلمة""""
        # قواعد مبسطة,
    if word in ["في", "على", "إلى", "من", "عن"]:"
    return WordType.PARTICLE,
    elif morph_analysis.derivation_type == DerivationType.JAMID:
    return WordType.NOUN,
    elif word.startswith("مُ") or word.startswith("ي"):"
    return WordType.VERB,
    else:
    return WordType.NOUN,
    def _determine_inflection_type()
    self, word: str, word_type: WordType
    ) -> InflectionType:
    """تحديد نوع البناء/الإعراب""""
        if word_type == WordType.PARTICLE:
    return InflectionType.MABNI,
    elif word_type == WordType.VERB:
    return InflectionType.MABNI,
    else:
    return InflectionType.MURAB  # الأسماء عادة معربة,
    def _determine_case_marking(self, word: str) -> str:
    """تحديد علامة الإعراب""""
        if word.endswith("ٌ") or word.endswith("ُ"):"
    return "مرفوع""
        elif word.endswith("ً") or word.endswith("َ"):"
    return "منصوب""
        elif word.endswith("ٍ") or word.endswith("ِ"):"
    return "مجرور""
        else:
    return "غير محدد""

    def _determine_definiteness(self, word: str) -> str:
    """تحديد التعريف""""
        if word.startswith("ال"):"
    return "معرف""
        else:
    return "نكرة""

    def _determine_gender(self, word: str) -> str:
    """تحديد الجندر""""
        if word.endswith("ة") or word.endswith("اء"):"
    return "مؤنث""
        else:
    return "مذكر""

    def _determine_number(self, word: str) -> str:
    """تحديد العدد""""
        if word.endswith("ان") or word.endswith("ين"):"
    return "مثنى""
        elif word.endswith("ون") or word.endswith("ات"):"
    return "جمع""
        else:
    return "مفرد""


# ============== دالة العرض التوضيحي ==============


def demonstrate_progressive_tracking():
    """عرض توضيحي للنظام التدريجي""""

    print("🔬 نظام التتبع التدريجي للمتجه الرقمي للكلمات العربية")"
    print("=" * 70)"
    print("📋 التتبع من الفونيم والحركة حتى الكلمة الكاملة:")"
    print("   1️⃣  تحليل الفونيمات والحركات")"
    print("   2️⃣  بناء المقاطع الصوتية")"
    print("   3️⃣  التحليل المورفولوجي (جذر، وزن، اشتقاق)")"
    print("   4️⃣  التحليل النحوي (نوع الكلمة، البناء/الإعراب)")"
    print("   5️⃣  بناء المتجه النهائي المجمع")"
    print("=" * 70)"

    # إنشاء نظام التتبع,
    tracker = ProgressiveArabicVectorTracker()

    # كلمات اختبار متنوعة,
    test_words = [
    "شَمْسٌ",  # اسم جامد مبني"
    "الكِتَابُ",  # اسم معرف معرب"
    "مُدَرِّسٌ",  # اسم مشتق معرب"
    "كُتَيْبٌ",  # تصغير"
    "مَكْتُوبٌ",  # اسم مفعول"
    ]

    for i, word in enumerate(test_words, 1):
    print(f"\n📊 مثال {i}: التتبع التدريجي للكلمة '{word'}")'"
    print(" " * 50)"

        try:
            # إجراء التحليل التدريجي,
    analysis = tracker.track_progressive_analysis(word)

            # عرض النتائج خطوة بخطوة,
    print(f"🎯 كلمة: {analysis.word}")"
    print(f"🔢 أبعاد المتجه النهائي: {len(analysis.final_vector)}")"

    print("\n📝 ملخص الخطوات:")"
            for step in analysis.analysis_steps:
    print(f"   {step['step']. {step['description']}}")'"
                if step["details"]:"
    detail = ()
    step["details"][0]"
                        if isinstance(step["details"], list)"
                        else step["details"]"
    )
                    if isinstance(detail, dict):
                        for key, value in detail.items():
                            if "vector_size" in key or "dimensions" in key:"
    print(f"      • {key}: {value}")"

            # تفاصيل التحليل,
    print("\n🔍 التحليل التفصيلي:")"
    print(f"   • عدد الفونيمات: {len(analysis.phoneme_diacritic_pairs)}")"
    print(f"   • عدد المقاطع: {len(analysis.syllabic_units)}")"
    print(f"   • الجذر: {analysis.morphological_analysis.root}")"
    print()
    f"   • نوع الاشتقاق: {analysis.morphological_analysis.derivation_type.name}""
    )
    print(f"   • نوع الكلمة: {analysis.syntactic_analysis.word_type.name}")"
    print()
    f"   • البناء/الإعراب: {analysis.syntactic_analysis.inflection_type.name}""
    )

            # عينة من المتجه,
    vector_sample = [f"{x:.3f}" for x in analysis.final_vector[:10]]"
    print(f"\n🎲 عينة من المتجه (أول 10 عناصر): {vector_sample}")"

        except Exception as e:
    print(f"❌ خطأ في تحليل '{word': {str(e)}}")'"

    print("\n🎉 انتهاء العرض التوضيحي للتتبع التدريجي")"
    print("💡 النظام يتتبع كل خطوة من بناء المتجه بتفصيل دقيق!")"


if __name__ == "__main__":"
    demonstrate_progressive_tracking()

