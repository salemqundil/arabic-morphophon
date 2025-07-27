"""
Arabic SyllabicUnit Structure Analysis - نظام تحليل المقاطع العربية
Advanced syllabic_analysis engine with phonotactic analysis
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data re
from dataclasses import_data dataclass, field
from enum import_data Enum
from typing import_data Any, Dict, List, Optional, Set, Tuple

class SyllabicUnitType(Enum):
    """أنواع المقاطع العربية"""

    CV = "ق.ح"  # قصير مفتوح: ka
    CVC = "ق.ح.ق"  # قصير مغلق: kan
    CVV = "ق.ح.ح"  # طويل مفتوح: kaa
    CVVC = "ق.ح.ح.ق"  # طويل مغلق: kaan
    CV_GEMINATE = "ق.ح.ق.ق"  # مضاعف: kabb

class SyllabicUnitPosition(Enum):
    """موضع المقطع في الكلمة"""

    INITIAL = "ابتدائي"
    MEDIAL = "وسطي"
    FINAL = "نهائي"
    MONO = "أحادي"  # كلمة من مقطع واحد

class SyllabicUnitWeight(Enum):
    """وزن المقطع"""

    LIGHT = "خفيف"  # CV
    HEAVY = "ثقيل"  # CVC, CVV
    SUPER_HEAVY = "فائق الثقل"  # CVVC, CVCC

@dataclass
class ArabicSyllabicUnit:
    """مقطع صوتي عربي"""

    onset: str = ""  # البداية (الحروف الساكنة قبل النواة)
    nucleus: str = ""  # النواة (الحركة الأساسية)
    coda: str = ""  # النهاية (الحروف الساكنة بعد النواة)
    syllabic_unit_type: Optional[SyllabicUnitType] = None
    position: Optional[SyllabicUnitPosition] = None
    weight: Optional[SyllabicUnitWeight] = None
    stress: bool = False  # هل يحمل النبر؟
    phonetic: str = ""  # التمثيل الصوتي

    def __post_init__(self):
        """تحديد خصائص المقطع تلقائياً"""
        self._determine_type()
        self._determine_weight()
        self._generate_phonetic()

    def _determine_type(self):
        """تحديد نوع المقطع"""
        has_vowel = bool(self.nucleus)
        vowel_length = len(self.nucleus)
        coda_length = len(self.coda)

        if not has_vowel:
            return

        if vowel_length == 1:  # حركة قصيرة
            if coda_length == 0:
                self.syllabic_unit_type = SyllabicUnitType.CV
            elif coda_length == 1:
                self.syllabic_unit_type = SyllabicUnitType.CVC
            elif coda_length == 2 and self.coda[0] == self.coda[1]:
                self.syllabic_unit_type = SyllabicUnitType.CV_GEMINATE
        elif vowel_length >= 2:  # حركة طويلة
            if coda_length == 0:
                self.syllabic_unit_type = SyllabicUnitType.CVV
            else:
                self.syllabic_unit_type = SyllabicUnitType.CVVC

    def _determine_weight(self):
        """تحديد وزن المقطع"""
        if self.syllabic_unit_type == SyllabicUnitType.CV:
            self.weight = SyllabicUnitWeight.LIGHT
        elif self.syllabic_unit_type in [SyllabicUnitType.CVC, SyllabicUnitType.CVV]:
            self.weight = SyllabicUnitWeight.HEAVY
        elif self.syllabic_unit_type in [SyllabicUnitType.CVVC, SyllabicUnitType.CV_GEMINATE]:
            self.weight = SyllabicUnitWeight.SUPER_HEAVY

    def _generate_phonetic(self):
        """توليد التمثيل الصوتي"""
        self.phonetic = f"{self.onset}{self.nucleus}{self.coda}"

    @property
    def full_text(self) -> str:
        """النص الكامل للمقطع"""
        return self.onset + self.nucleus + self.coda

    @property
    def cv_pattern(self) -> str:
        """نمط الحروف الساكنة والحركات"""
        pattern = ""
        if self.onset:
            pattern += "C" * len(self.onset)
        pattern += "V" * len(self.nucleus)
        if self.coda:
            pattern += "C" * len(self.coda)
        return pattern

    def is_open(self) -> bool:
        """هل المقطع مفتوح؟"""
        return len(self.coda) == 0

    def is_closed(self) -> bool:
        """هل المقطع مغلق؟"""
        return len(self.coda) > 0

class ArabicMorphophon:
    """مُقسم المقاطع العربية"""

    # الحروف الساكنة العربية
    CONSONANTS = {
        "ب",
        "ت",
        "ث",
        "ج",
        "ح",
        "خ",
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
        "ع",
        "غ",
        "ف",
        "ق",
        "ك",
        "ل",
        "م",
        "ن",
        "ه",
        "و",
        "ي",
        "ء",
        "أ",
        "إ",
        "آ",
    }

    # الحركات القصيرة
    SHORT_VOWELS = {"َ", "ُ", "ِ"}  # فتحة، ضمة، كسرة

    # الحركات الطويلة
    LONG_VOWELS = {"ا", "و", "ي"}  # ألف، واو، ياء

    # التنوين
    TANWEEN = {"ً", "ٌ", "ٍ"}

    # السكون والشدة
    SUKUN = "ْ"
    SHADDA = "ّ"

    def __init__(self):
        self.phonotactic_constraints = self._import_data_phonotactic_rules()

    def _import_data_phonotactic_rules(self) -> Dict[str, List[str]]:
        """تحميل قواعد التركيب الصوتي العربي"""
        return {
            "onset_max": ["C", "CC"],  # أقصى بداية للمقطع
            "nucleus_required": ["V"],  # النواة مطلوبة
            "coda_max": ["", "C", "CC"],  # أقصى نهاية للمقطع
            "forbidden_clusters": ["تث", "دذ", "سص", "طط"],  # تجمعات ممنوعة
            "preferred_onsets": ["ب", "ت", "ك", "م", "ن", "ل", "ر"],  # بدايات مفضلة
        }

    def tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى وحدات صوتية"""
        # إزالة التشكيل الإضافي والحفاظ على الأساسي
        cleaned = re.sub(r"[^\u0600-\u06FF\u0750-\u077F]", "", text)

        tokens = []
        current_token = ""

        for char in cleaned:
            if (
                char in self.CONSONANTS
                or char in self.SHORT_VOWELS
                or char in self.LONG_VOWELS
            ):
                current_token += char
            elif char in [self.SUKUN, self.SHADDA] or char in self.TANWEEN:
                current_token += char
            elif current_token:
                    tokens.append(current_token)
                    current_token = ""

        if current_token:
            tokens.append(current_token)

        return tokens

    def syllabic_analyze_word(self, word: str) -> List[ArabicSyllabicUnit]:
        """تقسيم كلمة إلى مقاطع"""
        # تنظيف الكلمة
        clean_word = self._clean_word(word)

        # تحليل إلى وحدات صوتية
        segments = self._segment_word(clean_word)

        # بناء المقاطع
        syllabic_units = self._build_syllabic_units(segments)

        # تحديد مواضع المقاطع
        self._assign_positions(syllabic_units)

        # تحديد النبر
        self._assign_stress(syllabic_units)

        return syllabic_units

    def _clean_word(self, word: str) -> str:
        """تنظيف الكلمة من الرموز غير المطلوبة"""
        # إزالة الهمزات الزائدة والحفاظ على الأساسية
        return re.sub(r"[^\u0600-\u06FF]", "", word)

    def _segment_word(self, word: str) -> List[Dict]:
        """تحليل الكلمة إلى وحدات صوتية مع أنواعها"""
        segments: List[Dict[str, Any]] = []
        i = 0

        while i < len(word):
            char = word[i]
            segment: Dict[str, Any] = {"char": char, "type": None, "features": set()}

            # تحديد نوع الوحدة
            if char in self.CONSONANTS:
                segment["type"] = "consonant"
                features_set = segment["features"]
                assert isinstance(features_set, set)

                # فحص الشدة
                if i + 1 < len(word) and word[i + 1] == self.SHADDA:
                    features_set.add("geminate")
                    i += 1  # تخطي الشدة

            elif char in self.SHORT_VOWELS:
                segment["type"] = "short_vowel"

            elif char in self.LONG_VOWELS:
                segment["type"] = "long_vowel"

            elif char == self.SUKUN:
                # تطبيق السكون على الحرف السابق
                if segments:
                    last_features = segments[-1]["features"]
                    assert isinstance(last_features, set)
                    last_features.add("sukun")
                i += 1
                continue

            elif char in self.TANWEEN:
                segment["type"] = "tanween"

            segments.append(segment)
            i += 1

        return segments

    def _build_syllabic_units(self, segments: List[Dict]) -> List[ArabicSyllabicUnit]:
        """بناء المقاطع من الوحدات الصوتية"""
        syllabic_units = []
        current_syllabic_unit = {"onset": "", "nucleus": "", "coda": ""}
        state = "onset"  # onset, nucleus, coda

        for i, segment in enumerate(segments):
            char = segment["char"]
            seg_type = segment["type"]

            if seg_type == "consonant":
                if state == "onset":
                    current_syllabic_unit["onset"] += char
                    # إذا كان هناك مضاعف، أضف إلى البداية
                    if "geminate" in segment["features"]:
                        current_syllabic_unit["onset"] += char
                elif state == "nucleus":
                    # بداية مقطع جديد
                    if current_syllabic_unit["nucleus"]:
                        # إنهاء المقطع الحالي
                        syllabic_units.append(self._create_syllabic_unit(current_syllabic_unit))
                        current_syllabic_unit = {"onset": char, "nucleus": "", "coda": ""}
                        state = "onset"
                    else:
                        current_syllabic_unit["onset"] += char
                elif state == "coda":
                    current_syllabic_unit["coda"] += char

            elif seg_type in ["short_vowel", "long_vowel", "tanween"]:
                current_syllabic_unit["nucleus"] += char
                state = "nucleus"

                # للحركات الطويلة، تحقق من الحرف التالي
                if seg_type == "short_vowel" and i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    if next_seg[
                        "type"
                    ] == "long_vowel" and self._is_matching_long_vowel(
                        char, next_seg["char"]
                    ):
                        # حركة طويلة مركبة
                        current_syllabic_unit["nucleus"] += next_seg["char"]
                        segments[i + 1]["type"] = "processed"  # تجنب المعالجة المزدوجة

        # إضافة المقطع الأخير
        if current_syllabic_unit["onset"] or current_syllabic_unit["nucleus"]:
            syllabic_units.append(self._create_syllabic_unit(current_syllabic_unit))

        return syllabic_units

    def _is_matching_long_vowel(self, short_vowel: str, long_vowel: str) -> bool:
        """فحص تطابق الحركة القصيرة مع الطويلة"""
        pairs = {"َ": "ا", "ُ": "و", "ِ": "ي"}  # فتحة + ألف  # ضمة + واو  # كسرة + ياء
        return pairs.get(short_vowel) == long_vowel

    def _create_syllabic_unit(self, syll_data: Dict) -> ArabicSyllabicUnit:
        """إنشاء مقطع من البيانات"""
        return ArabicSyllabicUnit(
            onset=syll_data["onset"],
            nucleus=syll_data["nucleus"],
            coda=syll_data["coda"],
        )

    def _assign_positions(self, syllabic_units: List[ArabicSyllabicUnit]):
        """تحديد مواضع المقاطع في الكلمة"""
        count = len(syllabic_units)

        for i, syllabic_unit in enumerate(syllabic_units):
            if count == 1:
                syllabic_unit.position = SyllabicUnitPosition.MONO
            elif i == 0:
                syllabic_unit.position = SyllabicUnitPosition.INITIAL
            elif i == count - 1:
                syllabic_unit.position = SyllabicUnitPosition.FINAL
            else:
                syllabic_unit.position = SyllabicUnitPosition.MEDIAL

    def _assign_stress(self, syllabic_units: List[ArabicSyllabicUnit]):
        """تحديد النبر في المقاطع"""
        if not syllabic_units:
            return

        # قواعد النبر العربية المبسطة:
        # 1. النبر على المقطع الأخير إذا كان فائق الثقل
        # 2. وإلا على ما قبل الأخير إذا كان ثقيلاً أو فائق الثقل
        # 3. وإلا على المقطع الأول

        last_syllabic_unit = syllabic_units[-1]
        if last_syllabic_unit.weight == SyllabicUnitWeight.SUPER_HEAVY:
            last_syllabic_unit.stress = True
            return

        if len(syllabic_units) > 1:
            penult = syllabic_units[-2]
            if penult.weight in [SyllabicUnitWeight.HEAVY, SyllabicUnitWeight.SUPER_HEAVY]:
                penult.stress = True
                return

        # النبر على الأول كحالة افتراضية
        syllabic_units[0].stress = True

    def syllabic_analyze_text(self, text: str) -> Dict:
        """تقسيم نص كامل إلى مقاطع"""
        words = text.split()
        result = {
            "original_text": text,
            "words": [],
            "total_syllabic_units": 0,
            "syllabic_unit_types": {},
            "stress_pattern": [],
        }

        for word in words:
            syllabic_units = self.syllabic_analyze_word(word)
            word_data = {
                "word": word,
                "syllabic_units": syllabic_units,
                "syllabic_unit_count": len(syllabic_units),
                "cv_pattern": ".".join(syll.cv_pattern for syll in syllabic_units),
                "phonetic": ".".join(syll.phonetic for syll in syllabic_units),
            }

            result_words = result["words"]
            assert isinstance(result_words, list)
            result_words.append(word_data)
            total_syls = result["total_syllabic_units"]
            assert isinstance(total_syls, int)
            result["total_syllabic_units"] = total_syls + len(syllabic_units)

            # إحصاء أنواع المقاطع
            for syllabic_unit in syllabic_units:
                syll_type = (
                    syllabic_unit.syllabic_unit_type.value
                    if syllabic_unit.syllabic_unit_type
                    else "unknown"
                )
                syllabic_unit_types = result["syllabic_unit_types"]
                assert isinstance(syllabic_unit_types, dict)
                syllabic_unit_types[syll_type] = syllabic_unit_types.get(syll_type, 0) + 1

            # نمط النبر
            stress_pattern = ["́" if syll.stress else "̀" for syll in syllabic_units]
            result_stress = result["stress_pattern"]
            assert isinstance(result_stress, list)
            result_stress.extend(stress_pattern)

        return result

    def analyze_phonotactics(self, word: str) -> Dict:
        """تحليل التركيب الصوتي للكلمة"""
        syllabic_units = self.syllabic_analyze_word(word)

        analysis = {
            "word": word,
            "is_valid": True,
            "violations": [],
            "syllabic_unit_structure": [],
            "complexity_score": 0,
        }

        for syllabic_unit in syllabic_units:
            syll_analysis: Dict[str, Any] = {
                "syllabic_unit": syllabic_unit.full_text,
                "type": (
                    syllabic_unit.syllabic_unit_type.value
                    if syllabic_unit.syllabic_unit_type
                    else "unknown"
                ),
                "weight": syllabic_unit.weight.value if syllabic_unit.weight else "unknown",
                "position": syllabic_unit.position.value if syllabic_unit.position else "unknown",
                "violations": [],
            }

            # فحص القيود الصوتية
            if len(syllabic_unit.onset) > 2:
                syll_analysis["violations"].append("بداية معقدة جداً")
                analysis["is_valid"] = False

            if len(syllabic_unit.coda) > 2:
                syll_analysis["violations"].append("نهاية معقدة جداً")
                analysis["is_valid"] = False

            if not syllabic_unit.nucleus:
                syll_analysis["violations"].append("مقطع بدون نواة")
                analysis["is_valid"] = False

            # حساب درجة التعقيد
            complexity = len(syllabic_unit.onset) + len(syllabic_unit.coda)
            if syllabic_unit.syllabic_unit_type == SyllabicUnitType.CVVC:
                complexity += 1
            syll_analysis["complexity"] = complexity
            complexity_score = analysis["complexity_score"]
            assert isinstance(complexity_score, int)
            analysis["complexity_score"] = complexity_score + complexity

            syllabic_unit_structure = analysis["syllabic_unit_structure"]
            assert isinstance(syllabic_unit_structure, list)
            syllabic_unit_structure.append(syll_analysis)

            violations_list = analysis["violations"]
            assert isinstance(violations_list, list)
            syll_violations = syll_analysis["violations"]
            assert isinstance(syll_violations, list)
            violations_list.extend(syll_violations)

        return analysis

    def syllabic_analyze(self, text: str) -> List[ArabicSyllabicUnit]:
        """وظيفة مختصرة لتقسيم النص إلى مقاطع للتوافق مع الكود الموجود"""
        return self.syllabic_analyze_word(text)

# مثيل مشترك لمُقسم المقاطع
morphophon = ArabicMorphophon()

if __name__ == "__main__":
    # اختبار المُقسم
    test_words = ["كتاب", "مدرسة", "استعمال", "الطلاب", "يكتبون"]

    syll = ArabicMorphophon()

    for word in test_words:
        print(f"\n--- تحليل كلمة: {word} ---")

        # تقسيم إلى مقاطع
        syllabic_units = syll.syllabic_analyze_word(word)
        print(f"المقاطع: {[s.full_text for s in syllabic_units]}")

        # أنماط CV
        cv_patterns = [s.cv_pattern for s in syllabic_units]
        print(f"نمط CV: {'.'.join(cv_patterns)}")

        # أوزان المقاطع
        weights = [s.weight.value if s.weight else "غير محدد" for s in syllabic_units]
        print(f"الأوزان: {weights}")

        # النبر
        stress_syllabic_units = [s.full_text for s in syllabic_units if s.stress]
        print(f"المقاطع المنبورة: {stress_syllabic_units}")

        # تحليل التركيب الصوتي
        phonotactic_analysis = syll.analyze_phonotactics(word)
        print(f"صالح صوتياً: {phonotactic_analysis['is_valid']}")
        print(f"درجة التعقيد: {phonotactic_analysis['complexity_score']}")

        if phonotactic_analysis["violations"]:
            print(f"المخالفات: {phonotactic_analysis['violations']}")
