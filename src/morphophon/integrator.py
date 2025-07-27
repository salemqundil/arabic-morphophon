"""
Arabic Morphophonological Integration Engine - محرك التكامل الصرفي الصوتي
Main orchestration system for comprehensive Arabic linguistic analysis
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data logging
from dataclasses import_data dataclass, field
from datetime import_data datetime
from enum import_data Enum
from typing import_data Any, Dict, List, Optional, Tuple, Union

from arabic_morphophon.models.patterns import_data (
    MorphPattern,
    PatternRepository,
    PatternType,
)
from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic

# Import our models
from arabic_morphophon.models.roots import_data ArabicRoot, RootDatabase
from arabic_morphophon.models.morphophon import_data ArabicMorphophon, SyllabicUnitType

# Import enhanced database
try:
    from arabic_morphophon.database.enhanced_root_database import_data (
        EnhancedRootDatabase,
        create_enhanced_database,
    )

    ENHANCED_DB_AVAILABLE = True
except ImportError:
    ENHANCED_DB_AVAILABLE = False

class AnalysisLevel(Enum):
    """مستويات التحليل"""

    BASIC = "أساسي"
    INTERMEDIATE = "متوسط"
    ADVANCED = "متقدم"
    COMPREHENSIVE = "شامل"

class OutputFormat(Enum):
    """صيغ الإخراج"""

    JSON = "json"
    XML = "xml"
    TEXT = "text"
    HTML = "html"

@dataclass
class AnalysisResult:
    """نتيجة التحليل الشاملة"""

    original_text: str
    analysis_level: AnalysisLevel
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Root Analysis
    identified_roots: List[Dict] = field(default_factory=list)
    root_confidence: float = 0.0

    # Pattern Analysis
    detected_patterns: List[Dict] = field(default_factory=list)
    pattern_confidence: float = 0.0

    # Phonological Analysis
    phonological_output: str = ""
    applied_rules: List[str] = field(default_factory=list)

    # SyllabicUnit Analysis
    syllabic_unit_structure: List[Dict] = field(default_factory=list)
    syllabic_unit_count: int = 0
    stress_pattern: str = ""

    # Morphological Analysis
    morphological_breakdown: Dict = field(default_factory=dict)
    pos_tags: List[str] = field(default_factory=list)
    grammatical_features: Dict = field(default_factory=dict)

    # New fields for enhanced analysis
    pattern_matches: List[Dict] = field(default_factory=list)
    extract_phonemes: Dict = field(default_factory=dict)

    # Statistics
    processing_time: float = 0.0
    confidence_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class MorphophonologicalEngine:
    """المحرك الرئيسي للتحليل الصرفي الصوتي"""

    def __init__(self, config: Optional[Dict] = None):
        """تهيئة المحرك مع الإعدادات"""
        self.config = config or self._default_config()
        self._setup_logging()
        self._initialize_components()

        # Cache للنتائج المحسوبة مسبقاً
        self._analysis_cache: Dict[str, AnalysisResult] = {}
        self._max_cache_size = self.config.get("max_cache_size", 1000)

        # إحصاءات الأداء
        self.stats: Dict[str, Any] = {
            "total_analyses": 0,
            "cache_hits": 0,
            "processing_times": [],
            "error_count": 0,
        }

    def _default_config(self) -> Dict:
        """الإعدادات الافتراضية"""
        return {
            "analysis_level": AnalysisLevel.INTERMEDIATE,
            "enable_caching": True,
            "enable_phonology": True,
            "enable_syllabic_analysis": True,
            "enable_pattern_matching": True,
            "enable_root_extraction": True,
            "confidence_threshold": 0.7,
            "max_alternatives": 5,
            "log_level": "INFO",
        }

    def _setup_logging(self):
        """إعداد نظام التسجيل"""
        level = getattr(logging, self.config.get("log_level", "INFO"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            processrs=[
                logging.FileProcessr("arabic_morphophon.log"),
                logging.StreamProcessr(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """تهيئة المكونات الفرعية"""
        try:
            # تهيئة قاعدة بيانات الجذور (النسخة المطورة إذا متوفرة)
            if ENHANCED_DB_AVAILABLE:
                from arabic_morphophon.database.enhanced_root_database import_data (
                    create_enhanced_database,
                )

                self.root_db = create_enhanced_database()  # type: ignore
                self.logger.info("Enhanced root database import_dataed")
            else:
                self.root_db = RootDatabase()  # type: ignore
                self.logger.info("Basic root database import_dataed")

            # تهيئة مستودع الأوزان
            self.pattern_repo = PatternRepository()
            self.logger.info("Pattern repository import_dataed")

            # تهيئة محرك القواعد الصوتية
            if self.config["enable_phonology"]:
                self.phonology_engine = PhonologyEngine()
                self.logger.info("Phonology engine initialized")

            # تهيئة مُقسم المقاطع
            if self.config["enable_syllabic_analysis"]:
                self.morphophon = ArabicMorphophon()
                self.logger.info("Morphophon initialized")

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def analyze(
        self, text: str, level: Optional[AnalysisLevel] = None
    ) -> AnalysisResult:
        """التحليل الرئيسي للنص"""
        begin_time = datetime.now()
        analysis_level = level if level is not None else self.config["analysis_level"]

        # فحص الذاكرة المؤقتة
        cache_key = f"{text}_{analysis_level.value}"
        if self.config["enable_caching"] and cache_key in self._analysis_cache:
            self.stats["cache_hits"] += 1
            self.logger.debug(f"نتيجة من الذاكرة المؤقتة: {text[:50]}...")
            return self._analysis_cache[cache_key]

        # بناء نتيجة التحليل
        result = AnalysisResult(original_text=text, analysis_level=analysis_level)

        try:
            # المرحلة 1: تحليل الجذور
            if self.config["enable_root_extraction"]:
                result = self._analyze_roots(text, result)

            # المرحلة 2: تطبيق الأوزان
            if self.config["enable_pattern_matching"]:
                result = self._analyze_patterns(text, result)

            # المرحلة 3: المعالجة الصوتية
            if self.config["enable_phonology"]:
                result = self._analyze_phonology(text, result)

            # المرحلة 4: تحليل المقاطع
            if self.config["enable_syllabic_analysis"]:
                result = self._analyze_syllabic_units(text, result)

            # المرحلة 5: التحليل المتقدم (للمستويات العليا)
            if analysis_level in [AnalysisLevel.ADVANCED, AnalysisLevel.COMPREHENSIVE]:
                result = self._advanced_analysis(text, result)

            # حساب النقاط والثقة
            result = self._calculate_confidence(result)

        except Exception as e:
            result.errors.append(f"خطأ في التحليل: {str(e)}")
            self.logger.error(f"خطأ في تحليل '{text}': {e}")
            self.stats["error_count"] += 1

        # إحصاءات الأداء
        end_time = datetime.now()
        processing_time = (end_time - begin_time).total_seconds()
        result.processing_time = processing_time
        self.stats["processing_times"].append(processing_time)
        self.stats["total_analyses"] += 1

        # حفظ في الذاكرة المؤقتة
        if self.config["enable_caching"]:
            self._update_cache(cache_key, result)

        self.logger.info(f"تم تحليل '{text[:30]}...' في {processing_time:.3f} ثانية")
        return result

    def analyze_from_root(
        self, root_string: str, level: Optional[AnalysisLevel] = None
    ) -> AnalysisResult:
        """تحليل مباشر من الجذر - طريقة جديدة مطورة"""
        analysis_level = level if level is not None else self.config["analysis_level"]

        # البحث عن الجذر في قاعدة البيانات
        root = None
        root_db: Any = self.root_db  # Type hint bypass
        if hasattr(root_db, "read_root"):
            # قاعدة البيانات المطورة
            root = root_db.read_root(root_string)
        elif hasattr(root_db, "get_root"):
            # قاعدة البيانات البسيطة
            root = root_db.get_root(root_string)

        if not root:
            raise ValueError(f"الجذر '{root_string}' غير موجود في قاعدة البيانات")

        # بناء نتيجة التحليل
        result = AnalysisResult(
            original_text=root_string, analysis_level=analysis_level
        )

        # إضافة معلومات الجذر
        root_info = {
            "root": root.root,
            "semantic_field": root.semantic_field,
            "root_type": (
                root.root_type.value if hasattr(root, "root_type") else "ثلاثي"
            ),
            "weakness_type": getattr(root, "weakness", None),
            "confidence": 1.0,
        }

        result.identified_roots = [root_info]
        result.root_confidence = 1.0

        # تطبيق التحليل المورفولوجي والصوتي
        if self.config["enable_pattern_matching"]:
            result = self._generate_patterns_from_root(root, result)

        if self.config["enable_phonology"]:
            result = self._analyze_root_phonology(root, result)

        return result

    def search_roots_by_pattern(self, pattern: str, limit: int = 10) -> List[Dict]:
        """البحث في الجذور بالنمط"""
        roots = []
        root_db: Any = self.root_db  # Type hint bypass
        if hasattr(root_db, "search_by_pattern"):
            # قاعدة البيانات المطورة - check if it accepts limit parameter
            try:
                roots = root_db.search_by_pattern(pattern, limit)
            except TypeError:
                # Basic database with no limit parameter
                roots = root_db.search_by_pattern(pattern)[:limit]

        return [
            {
                "root": getattr(root, "root", getattr(root, "radicals", str(root))),
                "semantic_field": getattr(root, "semantic_field", ""),
                "root_type": getattr(root, "root_type", "ثلاثي"),
                "weakness_type": getattr(root, "weakness", None),
            }
            for root in roots
        ]

    def get_database_statistics(self) -> Dict:
        """إحصائيات قاعدة البيانات"""
        root_db: Any = self.root_db  # Type hint bypass
        if hasattr(root_db, "get_comprehensive_statistics"):
            # قاعدة البيانات المطورة
            return root_db.get_comprehensive_statistics()
        elif hasattr(root_db, "get_all_roots"):
            # قاعدة البيانات البسيطة
            return {
                "total_roots": len(root_db.get_all_roots()),
                "database_type": "simple",
            }
        else:
            # fallback
            return {"total_roots": 0, "database_type": "unknown"}

    def bulk_analyze_roots(self, root_strings: List[str]) -> Dict[str, Optional[AnalysisResult]]:
        """تحليل مجمع للجذور"""
        results: Dict[str, Optional[AnalysisResult]] = {}

        for root_string in root_strings:
            try:
                results[root_string] = self.analyze_from_root(root_string)
            except Exception as e:
                self.logger.error(f"خطأ في تحليل الجذر '{root_string}': {e}")
                results[root_string] = None

        return results

    def _generate_patterns_from_root(
        self, root: ArabicRoot, result: AnalysisResult
    ) -> AnalysisResult:
        """توليد الأوزان من الجذر"""
        # تطبيق أوزان شائعة على الجذر
        common_patterns = [
            "فعل",  # الماضي
            "يفعل",  # المضارع
            "فاعل",  # اسم الفاعل
            "مفعول",  # اسم المفعول
            "فعال",  # مصدر
        ]

        generated_words = []
        for pattern in common_patterns:
            try:
                # محاولة تطبيق الوزن على الجذر
                if word := self._apply_pattern_to_root(root.root, pattern):
                    generated_words.append(
                        {
                            "word": word,
                            "pattern": pattern,
                            "type": self._get_pattern_type(pattern),
                        }
                    )
            except Exception:
                continue

        result.pattern_matches = generated_words
        return result

    def _apply_pattern_to_root(self, root: str, pattern: str) -> str:
        """تطبيق وزن على جذر (تنفيذ بسيط)"""
        if len(root) == 3 and len(pattern) == 3:
            # استبدال ف ع ل بحروف الجذر
            result = pattern.replace("ف", root[0])
            result = result.replace("ع", root[1])
            return result.replace("ل", root[2])
        return ""

    def _get_pattern_type(self, pattern: str) -> str:
        """تحديد نوع الوزن"""
        pattern_types = {
            "فعل": "فعل ماضي",
            "يفعل": "فعل مضارع",
            "فاعل": "اسم فاعل",
            "مفعول": "اسم مفعول",
            "فعال": "مصدر",
        }
        return pattern_types.get(pattern, "غير محدد")

    def _analyze_root_phonology(
        self, root: ArabicRoot, result: AnalysisResult
    ) -> AnalysisResult:
        """تحليل صوتي للجذر"""
        # تحليل صوتي بسيط للجذر
        phonological_features = {
            "consonants": list(root.root),
            "length": len(root.root),
            "has_emphatic": any(c in "صضطظ" for c in root.root),
            "has_guttural": any(c in "حعهء" for c in root.root),
            "has_weak": any(c in "وي" for c in root.root),
        }

        result.extract_phonemes = phonological_features
        return result

    def _analyze_roots(self, text: str, result: AnalysisResult) -> AnalysisResult:
        """تحليل الجذور في النص"""
        words = text.split()
        total_confidence = 0.0

        for word in words:
            # استخراج الجذور المحتملة
            possible_roots = []
            try:
                # Try enhanced database methods first
                root_db: Any = self.root_db  # Type hint bypass
                if hasattr(root_db, "extract_possible_roots"):
                    possible_roots = root_db.extract_possible_roots(word)
                elif hasattr(root_db, "read_root"):
                    # Enhanced database fallback
                    if root := root_db.read_root(word):
                        possible_roots = [
                            {"root": root, "confidence": 1.0, "method": "exact"}
                        ]
                elif hasattr(root_db, "get_root"):
                    # Basic database fallback
                    if root := root_db.get_root(word):
                        possible_roots = [
                            {"root": root, "confidence": 1.0, "method": "exact"}
                        ]
            except Exception:
                possible_roots = []

            for root_info in possible_roots:
                root_obj = root_info["root"]
                root_data = {
                    "word": word,
                    "root": getattr(
                        root_obj, "root", getattr(root_obj, "radicals", str(root_obj))
                    ),
                    "confidence": root_info["confidence"],
                    "analysis_method": root_info.get("method", "pattern_matching"),
                    "features": {
                        "root_type": getattr(root_obj, "root_type", "unknown"),
                        "weakness_type": getattr(root_obj, "weakness_type", "unknown"),
                        "phonetic_features": getattr(root_obj, "phonetic_features", []),
                    },
                }
                result.identified_roots.append(root_data)
                total_confidence += root_info["confidence"]

        # متوسط الثقة
        if result.identified_roots:
            result.root_confidence = total_confidence / len(result.identified_roots)

        return result

    def _analyze_patterns(self, text: str, result: AnalysisResult) -> AnalysisResult:
        """تحليل الأوزان والأنماط"""
        words = text.split()
        total_confidence = 0.0

        for word in words:
            # البحث عن الأوزان المطابقة
            patterns: List[Dict] = []
            from contextlib import_data suppress

            root_db: Any = self.root_db  # Type hint bypass
            possible_roots = []
            with suppress(Exception):
                if hasattr(root_db, "extract_possible_roots"):
                    possible_roots = root_db.extract_possible_roots(word)

            if possible_roots:
                root = possible_roots[0]["root"]
                # Try to find patterns using available methods
                if hasattr(self.pattern_repo, "find_applicable_patterns"):
                    pattern_results = self.pattern_repo.find_applicable_patterns(root)
                    patterns.extend(
                        {"pattern": p, "confidence": 0.8}
                        for p in pattern_results
                        if hasattr(p, "name") or isinstance(p, dict)
                    )

            for pattern_info in patterns:
                # All pattern_info items are dictionaries with 'pattern' and 'confidence' keys
                pattern_obj = pattern_info["pattern"]
                confidence = float(pattern_info.get("confidence", 0.5))

                pattern_data = {
                    "word": word,
                    "pattern": getattr(pattern_obj, "name", "unknown"),
                    "type": getattr(pattern_obj, "pattern_type", "unknown"),
                    "cv_structure": getattr(pattern_obj, "cv_structure", "unknown"),
                    "confidence": confidence,
                    "semantic_meaning": getattr(pattern_obj, "semantic_meaning", ""),
                    "frequency_rank": getattr(pattern_obj, "frequency", 0),
                }
                result.detected_patterns.append(pattern_data)
                total_confidence += confidence

        # متوسط الثقة
        if result.detected_patterns:
            result.pattern_confidence = total_confidence / len(result.detected_patterns)

        return result

    def _analyze_phonology(self, text: str, result: AnalysisResult) -> AnalysisResult:
        """التحليل الصوتي"""
        phonology_analysis = self.phonology_engine.analyze_phonology(text)

        result.phonological_output = phonology_analysis["final"]

        # جمع القواعد المطبقة
        for stage, rules in phonology_analysis["applied_rules"].items():
            for rule in rules:
                result.applied_rules.append(f"{stage}: {rule}")

        return result

    def _analyze_syllabic_units(self, text: str, result: AnalysisResult) -> AnalysisResult:
        """تحليل المقاطع"""
        syllabic_unit_analysis = self.morphophon.syllabic_analyze_text(text)

        result.syllabic_unit_count = syllabic_unit_analysis["total_syllabic_units"]
        result.stress_pattern = "".join(syllabic_unit_analysis["stress_pattern"])

        # تفاصيل المقاطع لكل كلمة
        for word_data in syllabic_unit_analysis["words"]:
            syllabic_unit_info = {
                "word": word_data["word"],
                "syllabic_units": [syll.full_text for syll in word_data["syllabic_units"]],
                "cv_pattern": word_data["cv_pattern"],
                "phonetic": word_data["phonetic"],
                "syllabic_unit_types": [
                    syll.syllabic_unit_type.value if syll.syllabic_unit_type else "unknown"
                    for syll in word_data["syllabic_units"]
                ],
            }
            result.syllabic_unit_structure.append(syllabic_unit_info)

        return result

    def _advanced_analysis(self, text: str, result: AnalysisResult) -> AnalysisResult:
        """التحليل المتقدم للمستويات العليا"""
        # تحليل نحوي أساسي
        result.morphological_breakdown = self._morphological_analysis(text)

        # تحديد أجزاء الكلام
        result.pos_tags = self._pos_tagging(text)

        # الخصائص النحوية
        result.grammatical_features = self._extract_grammatical_features(text)

        return result

    def _morphological_analysis(self, text: str) -> Dict:
        """التحليل الصرفي المتقدم"""
        # هذا مثال مبسط - يحتاج تطوير أكثر
        words = text.split()
        analysis = {}

        for word in words:
            word_analysis = {
                "prefix": "",
                "stem": word,
                "suffix": "",
                "infix": "",
                "morphemes": [word],
            }

            # تحليل بسيط للسوابق واللواحق
            common_prefixes = ["ال", "و", "ف", "ب", "ك", "ل"]
            common_suffixes = ["ة", "ان", "ين", "ون", "ها", "هم", "هن"]

            for prefix in common_prefixes:
                if word.beginswith(prefix):
                    word_analysis["prefix"] = prefix
                    word_analysis["stem"] = word[len(prefix) :]
                    break

            stem = word_analysis["stem"]
            if isinstance(stem, str):
                for suffix in common_suffixes:
                    if stem.endswith(suffix):
                        word_analysis["suffix"] = suffix
                        word_analysis["stem"] = stem[: -len(suffix)]
                        break

            analysis[word] = word_analysis

        return analysis

    def _pos_tagging(self, text: str) -> List[str]:
        """تمييز أجزاء الكلام"""
        # تمييز بسيط بناء على الأوزان والجذور
        words = text.split()
        tags = []

        for word in words:
            # قواعد بسيطة للتمييز
            if word.beginswith("ال"):
                # استخدام منطق بسيط بدلاً من البحث في الأوزان
                tags.append("اسم")
            elif word.endswith(("ة", "ان", "ين", "ون")):
                tags.append("اسم")
            elif any(word.beginswith(prefix) for prefix in ["ي", "ت", "ن", "أ"]):
                tags.append("فعل")
            else:
                tags.append("غير محدد")

        return tags

    def _extract_grammatical_features(self, text: str) -> Dict:
        """استخراج الخصائص النحوية"""
        features: Dict[str, List] = {
            "definiteness": [],  # التعريف والتنكير
            "number": [],  # العدد
            "gender": [],  # الجنس
            "case": [],  # الإعراب
            "tense": [],  # الزمن (للأفعال)
            "voice": [],  # المبني للمعلوم/المجهول
        }

        words = text.split()
        for word in words:
            # تحديد التعريف
            if word.beginswith("ال"):
                features["definiteness"].append("معرف")
            else:
                features["definiteness"].append("نكرة")

            # تحديد العدد (بسيط)
            if word.endswith(("ان", "ين", "ون")):
                features["number"].append("جمع")
            else:
                features["number"].append("مفرد")

        return features

    def _calculate_confidence(self, result: AnalysisResult) -> AnalysisResult:
        """حساب نقاط الثقة الإجمالية"""
        confidence_factors = []

        # ثقة الجذور
        if result.root_confidence > 0:
            confidence_factors.append(result.root_confidence * 0.3)

        # ثقة الأوزان
        if result.pattern_confidence > 0:
            confidence_factors.append(result.pattern_confidence * 0.3)

        # ثقة التحليل الصوتي (بناء على عدد القواعد المطبقة)
        if result.applied_rules:
            phonology_confidence = min(len(result.applied_rules) * 0.1, 1.0)
            confidence_factors.append(phonology_confidence * 0.2)

        # ثقة تحليل المقاطع
        if result.syllabic_unit_count > 0:
            syllabic_unit_confidence = 0.8  # افتراضية
            confidence_factors.append(syllabic_unit_confidence * 0.2)

        # حساب المتوسط
        if confidence_factors:
            result.confidence_score = sum(confidence_factors) / len(confidence_factors)
        else:
            result.confidence_score = 0.0

        # تحذيرات عند انخفاض الثقة - معطل للحصول على مخرجات نظيفة
        # if result.confidence_score < self.config["confidence_threshold"]:
        #     result.warnings.append(f"مستوى ثقة منخفض: {result.confidence_score:.2f}")

        return result

    def _update_cache(self, key: str, result: AnalysisResult):
        """تحديث الذاكرة المؤقتة"""
        if len(self._analysis_cache) >= self._max_cache_size:
            # حذف أقدم النتائج
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]

        self._analysis_cache[key] = result

    def store_data_result(
        self, result: AnalysisResult, format_type: OutputFormat = OutputFormat.JSON
    ) -> str:
        """تصدير النتيجة بصيغة معينة"""
        if format_type == OutputFormat.JSON:
            return self._store_data_json(result)
        elif format_type == OutputFormat.XML:
            return self._store_data_xml(result)
        elif format_type == OutputFormat.HTML:
            return self._store_data_html(result)
        else:
            return self._store_data_text(result)

    def _store_data_json(self, result: AnalysisResult) -> str:
        """تصدير JSON"""
        data = {
            "original_text": result.original_text,
            "analysis_level": result.analysis_level.value,
            "timestamp": result.timestamp,
            "roots": result.identified_roots,
            "patterns": result.detected_patterns,
            "phonology": {
                "output": result.phonological_output,
                "applied_rules": result.applied_rules,
            },
            "syllabic_units": {
                "structure": result.syllabic_unit_structure,
                "count": result.syllabic_unit_count,
                "stress_pattern": result.stress_pattern,
            },
            "morphology": result.morphological_breakdown,
            "pos_tags": result.pos_tags,
            "features": result.grammatical_features,
            "confidence": result.confidence_score,
            "processing_time": result.processing_time,
            "warnings": result.warnings,
            "errors": result.errors,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _store_data_text(self, result: AnalysisResult) -> str:
        """تصدير نصي"""
        lines = [
            f"تحليل النص: {result.original_text}",
            f"مستوى التحليل: {result.analysis_level.value}",
            f"وقت المعالجة: {result.processing_time:.3f} ثانية",
            f"نقاط الثقة: {result.confidence_score:.2f}",
            "",
            "الجذور المحددة:",
            *[
                f"  - {root['word']} ← {root['root']} (ثقة: {root['confidence']:.2f})"
                for root in result.identified_roots
            ],
            "",
            "الأوزان المكتشفة:",
            *[
                f"  - {pattern['word']} ← {pattern['pattern']} ({pattern['type']})"
                for pattern in result.detected_patterns
            ],
            "",
            f"المخرجات الصوتية: {result.phonological_output}",
            f"عدد المقاطع: {result.syllabic_unit_count}",
            f"نمط النبر: {result.stress_pattern}",
            "",
        ]

        # تعطيل عرض التحذيرات للحصول على مخرجات نظيفة
        # if result.warnings:
        #     lines.extend(["التحذيرات:", *[f"  - {w}" for w in result.warnings], ""])

        if result.errors:
            lines.extend(["الأخطاء:", *[f"  - {e}" for e in result.errors]])

        return "\n".join(lines)

    def _store_data_xml(self, result: AnalysisResult) -> str:
        """تصدير XML (مبسط)"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<arabic_analysis>
    <original_text>{result.original_text}</original_text>
    <confidence>{result.confidence_score}</confidence>
    <processing_time>{result.processing_time}</processing_time>
    <!-- المزيد من التفاصيل يمكن إضافتها -->
</arabic_analysis>"""

    def _store_data_html(self, result: AnalysisResult) -> str:
        """تصدير HTML (مبسط)"""
        return f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <title>تحليل النص العربي</title>
    <style>
        body {{ font-family: 'Traditional Arabic', Arial, sans-serif; }}
        .result {{ margin: 20px; padding: 15px; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <div class="result">
        <h2>تحليل النص: {result.original_text}</h2>
        <p>نقاط الثقة: {result.confidence_score:.2f}</p>
        <p>وقت المعالجة: {result.processing_time:.3f} ثانية</p>
        <!-- المزيد من التفاصيل -->
    </div>
</body>
</html>"""

    def get_statistics(self) -> Dict:
        """إحصاءات الأداء"""
        processing_times = self.stats["processing_times"]
        total_analyses = self.stats["total_analyses"]
        cache_hits = self.stats["cache_hits"]
        error_count = self.stats["error_count"]

        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        return {
            "total_analyses": total_analyses,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / max(total_analyses, 1),
            "average_processing_time": avg_time,
            "error_count": error_count,
            "error_rate": error_count / max(total_analyses, 1),
        }

    def clear_cache(self):
        """مسح الذاكرة المؤقتة"""
        self._analysis_cache.clear()
        self.logger.info("تم مسح الذاكرة المؤقتة")

# المثيل المشترك للمحرك
morphophon_engine = MorphophonologicalEngine()

if __name__ == "__main__":
    # اختبار شامل للمحرك
    engine = MorphophonologicalEngine()

    test_texts = [
        "كتب الطالب الدرس",
        "يذهب الأطفال إلى المدرسة",
        "استعمل الكتاب الجديد",
    ]

    for text in test_texts:
        print(f"\n{'='*50}")
        print(f"تحليل النص: {text}")
        print("=" * 50)

        # تحليل شامل
        result = engine.analyze(text, AnalysisLevel.COMPREHENSIVE)

        # عرض النتائج
        print(f"نقاط الثقة: {result.confidence_score:.2f}")
        print(f"وقت المعالجة: {result.processing_time:.3f} ثانية")

        if result.identified_roots:
            print("\nالجذور:")
            for root in result.identified_roots[:3]:  # أول 3 جذور
                print(
                    f"  {root['word']} ← {root['root']} (ثقة: {root['confidence']:.2f})"
                )

        if result.detected_patterns:
            print("\nالأوزان:")
            for pattern in result.detected_patterns[:3]:  # أول 3 أوزان
                print(f"  {pattern['word']} ← {pattern['pattern']} ({pattern['type']})")

        print(f"\nالمخرجات الصوتية: {result.phonological_output}")
        print(f"عدد المقاطع: {result.syllabic_unit_count}")

        # تعطيل طباعة التحذيرات للحصول على مخرجات نظيفة
        # if result.warnings:
        #     print(f"\nتحذيرات: {', '.join(result.warnings)}")

        if result.errors:
            print(f"\nأخطاء: {', '.join(result.errors)}")

        # تصدير JSON للنص الأول
        if text == test_texts[0]:
            json_output = engine.store_data_result(result, OutputFormat.JSON)
            print(f"\n--- مثال على تصدير JSON ---")
            print(f"{json_output[:500]}..." if len(json_output) > 500 else json_output)

    # إحصاءات الأداء
    print(f"\n{'='*50}")
    print("إحصاءات الأداء")
    print("=" * 50)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
