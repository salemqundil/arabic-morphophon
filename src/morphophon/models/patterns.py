"""
Arabic Morphological Patterns - نماذج الأوزان العربية
Advanced pattern system for Arabic morphology with CV structure
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from __future__ import_data annotations

import_data json
import_data re
from dataclasses import_data dataclass, field
from enum import_data Enum
from typing import_data TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from arabic_morphophon.models.roots import_data ArabicRoot

class PatternType(Enum):
    """أنواع الأوزان"""

    VERB = "فعل"
    NOUN = "اسم"
    ADJECTIVE = "صفة"
    MASDAR = "مصدر"
    PARTICIPLE = "اسم فاعل/مفعول"

class VerbForm(Enum):
    """أوزان الأفعال (الصيغ)"""

    FORM_I = ("فَعَلَ", "الماضي الثلاثي المجرد")
    FORM_II = ("فَعَّلَ", "الثلاثي المزيد بحرف - التضعيف")
    FORM_III = ("فَاعَلَ", "الثلاثي المزيد بحرف - المفاعلة")
    FORM_IV = ("أَفْعَلَ", "الثلاثي المزيد بحرف - الإفعال")
    FORM_V = ("تَفَعَّلَ", "الثلاثي المزيد بحرفين")
    FORM_VI = ("تَفَاعَلَ", "الثلاثي المزيد بحرفين")
    FORM_VII = ("انْفَعَلَ", "الثلاثي المزيد بحرفين")
    FORM_VIII = ("افْتَعَلَ", "الثلاثي المزيد بحرفين")
    FORM_IX = ("افْعَلَّ", "الثلاثي المزيد بحرفين - نادر")
    FORM_X = ("اسْتَفْعَلَ", "الثلاثي المزيد بثلاثة أحرف")
    FORM_XI = ("افْعَالَّ", "الثلاثي المزيد - الألوان")
    FORM_XII = ("افْعَوْعَلَ", "الثلاثي المزيد - نادر")
    FORM_XIII = ("افْعَوَّلَ", "الثلاثي المزيد - نادر")
    FORM_XIV = ("افْعَنْلَلَ", "الثلاثي المزيد - نادر")
    FORM_XV = ("افْعَنْلَى", "الثلاثي المزيد - نادر")

@dataclass
class CVStructure:
    """بنية الحركات والسكنات"""

    pattern: str  # مثل: CVCVCV
    template: str  # مثل: فَعَلَ
    morphemes: List[str] = field(default_factory=list)  # [ف، َ، ع، َ، ل]

    def __post_init__(self):
        """تحليل القالب إلى مورفيمات"""
        if not self.morphemes:
            self.morphemes = self._parse_template()

    def _parse_template(self) -> List[str]:
        """تقسيم القالب إلى وحدات صرفية"""
        morphemes = []
        i = 0
        while i < len(self.template):
            char = self.template[i]

            # فحص الحروف مع الحركات
            if i + 1 < len(self.template):
                next_char = self.template[i + 1]
                if next_char in "ًٌٍَُِّْ":  # حركات
                    morphemes.extend([char, next_char])
                    i += 2
                    continue

            morphemes.append(char)
            i += 1

        return morphemes

    def apply_to_root(self, root_radicals: List[str]) -> str:
        """تطبيق الوزن على جذر معين"""
        if len(root_radicals) != self.pattern.count("C"):
            raise ValueError(
                f"Root has {len(root_radicals)} radicals, pattern needs {self.pattern.count('C')}"
            )

        result = ""
        root_index = 0

        for morpheme in self.morphemes:
            if morpheme in "فعلقسطنشهدمرجزتك" and root_index < len(
                root_radicals
            ):  # حروف الميزان
                result += root_radicals[root_index]
                root_index += 1
            else:
                result += morpheme  # حركة أو حرف زائد

        return result

@dataclass
class MorphPattern:
    """نموذج وزن صرفي متكامل"""

    name: str
    pattern_type: PatternType
    cv_structure: CVStructure
    semantic_meaning: str
    examples: List[Tuple[str, str]] = field(default_factory=list)  # (جذر، تطبيق)

    # خصائص صرفية
    root_constraints: Set[str] = field(default_factory=set)  # قيود على نوع الجذر
    phonological_changes: Dict[str, str] = field(default_factory=dict)
    frequency: Optional[int] = None

    def can_apply_to_root(self, root: "ArabicRoot") -> bool:
        """فحص إمكانية تطبيق الوزن على جذر"""
        # فحص القيود الأساسية - use safe attribute access
        if "sound_only" in self.root_constraints and not getattr(
            root, "is_sound", True
        ):
            return False

        if "no_doubled" in self.root_constraints and getattr(root, "is_doubled", False):
            return False

        if "no_hamza" in self.root_constraints and getattr(root, "is_hamzated", False):
            return False

        # فحص التوافق مع بنية الوزن
        expected_radicals = self.cv_structure.pattern.count("C")
        actual_radicals = len(root.radicals)

        return expected_radicals == actual_radicals

    def apply_to_root(self, root: "ArabicRoot") -> Optional[str]:
        """تطبيق الوزن على جذر مع المعالجات الصوتية"""
        if not self.can_apply_to_root(root):
            return None

        root_letters = [r.letter for r in root.radicals]
        result = self.cv_structure.apply_to_root(root_letters)

        # تطبيق التغييرات الصوتية
        for pattern, replacement in self.phonological_changes.items():
            result = re.sub(pattern, replacement, result)

        return result

    def to_dict(self) -> Dict:
        """تحويل إلى قاموس للتصدير"""
        return {
            "name": self.name,
            "type": self.pattern_type.value,
            "cv_pattern": self.cv_structure.pattern,
            "template": self.cv_structure.template,
            "meaning": self.semantic_meaning,
            "examples": self.examples,
            "constraints": list(self.root_constraints),
            "phonological_changes": self.phonological_changes,
            "frequency": self.frequency,
        }

class PatternRepository:
    """مستودع الأوزان العربية"""

    def __init__(self):
        self.verb_patterns: Dict[str, MorphPattern] = {}
        self.noun_patterns: Dict[str, MorphPattern] = {}
        self.adjective_patterns: Dict[str, MorphPattern] = {}
        self._import_data_default_patterns()

    def _import_data_default_patterns(self):
        """تحميل الأوزان الافتراضية"""
        self._import_data_verb_patterns()
        self._import_data_noun_patterns()
        self._import_data_adjective_patterns()

    def _import_data_verb_patterns(self):
        """تحميل أوزان الأفعال"""
        verb_forms = [
            # الصيغة الأولى - فَعَلَ
            MorphPattern(
                name="فَعَلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("CVCVC", "فَعَلَ"),
                semantic_meaning="الحدث البسيط في الماضي",
                examples=[("كتب", "كَتَبَ"), ("قرأ", "قَرَأَ"), ("دخل", "دَخَلَ")],
                frequency=100,
            ),
            # الصيغة الثانية - فَعَّلَ
            MorphPattern(
                name="فَعَّلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("CVCCVC", "فَعَّلَ"),
                semantic_meaning="التكثير والتكرار",
                examples=[("درس", "دَرَّسَ"), ("كسر", "كَسَّرَ"), ("علم", "عَلَّمَ")],
                frequency=85,
            ),
            # الصيغة الثالثة - فَاعَلَ
            MorphPattern(
                name="فَاعَلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("CVCVCVC", "فَاعَلَ"),
                semantic_meaning="المشاركة والمفاعلة",
                examples=[("كتب", "كَاتَبَ"), ("جلس", "جَالَسَ"), ("شرك", "شَارَكَ")],
                frequency=70,
            ),
            # الصيغة الرابعة - أَفْعَلَ
            MorphPattern(
                name="أَفْعَلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("VCCVC", "أَفْعَلَ"),
                semantic_meaning="التعدية والسببية",
                examples=[("خرج", "أَخْرَجَ"), ("كرم", "أَكْرَمَ"), ("سلم", "أَسْلَمَ")],
                frequency=90,
            ),
            # الصيغة الخامسة - تَفَعَّلَ
            MorphPattern(
                name="تَفَعَّلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("CVCCVC", "تَفَعَّلَ"),
                semantic_meaning="التدرج والتكلف",
                examples=[("علم", "تَعَلَّمَ"), ("كلم", "تَكَلَّمَ"), ("قدم", "تَقَدَّمَ")],
                frequency=80,
            ),
            # الصيغة السادسة - تَفَاعَلَ
            MorphPattern(
                name="تَفَاعَلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("CVCVCVC", "تَفَاعَلَ"),
                semantic_meaning="التشارك والتفاعل",
                examples=[("كتب", "تَكَاتَبَ"), ("عون", "تَعَاوَنَ"), ("بدل", "تَبَادَلَ")],
                frequency=65,
            ),
            # الصيغة العاشرة - اسْتَفْعَلَ
            MorphPattern(
                name="اسْتَفْعَلَ",
                pattern_type=PatternType.VERB,
                cv_structure=CVStructure("VCCVCCVC", "اسْتَفْعَلَ"),
                semantic_meaning="الطلب والاستدعاء",
                examples=[("غفر", "اسْتَغْفَرَ"), ("خرج", "اسْتَخْرَجَ"), ("عمل", "اسْتَعْمَلَ")],
                frequency=75,
            ),
        ]

        for pattern in verb_forms:
            self.verb_patterns[pattern.name] = pattern

    def _import_data_noun_patterns(self):
        """تحميل أوزان الأسماء"""
        noun_forms = [
            # فَاعِل - اسم فاعل
            MorphPattern(
                name="فَاعِل",
                pattern_type=PatternType.PARTICIPLE,
                cv_structure=CVStructure("CVCVC", "فَاعِل"),
                semantic_meaning="من يقوم بالفعل",
                examples=[("كتب", "كَاتِب"), ("قرأ", "قَارِئ"), ("عمل", "عَامِل")],
                frequency=95,
            ),
            # مَفْعُول - اسم مفعول
            MorphPattern(
                name="مَفْعُول",
                pattern_type=PatternType.PARTICIPLE,
                cv_structure=CVStructure("CVCCVC", "مَفْعُول"),
                semantic_meaning="ما وقع عليه الفعل",
                examples=[("كتب", "مَكْتُوب"), ("قرأ", "مَقْرُوء"), ("عمل", "مَعْمُول")],
                frequency=90,
            ),
            # فَعْل - مصدر
            MorphPattern(
                name="فَعْل",
                pattern_type=PatternType.MASDAR,
                cv_structure=CVStructure("CVCC", "فَعْل"),
                semantic_meaning="المصدر البسيط",
                examples=[("ضرب", "ضَرْب"), ("قتل", "قَتْل"), ("نصر", "نَصْر")],
                frequency=85,
            ),
            # فِعَالَة - حرفة أو صناعة
            MorphPattern(
                name="فِعَالَة",
                pattern_type=PatternType.NOUN,
                cv_structure=CVStructure("CVCVCV", "فِعَالَة"),
                semantic_meaning="الحرفة والصناعة",
                examples=[("زرع", "زِرَاعَة"), ("صنع", "صِنَاعَة"), ("تجر", "تِجَارَة")],
                frequency=60,
            ),
            # مَفْعَل - مكان الفعل
            MorphPattern(
                name="مَفْعَل",
                pattern_type=PatternType.NOUN,
                cv_structure=CVStructure("CVCCVC", "مَفْعَل"),
                semantic_meaning="مكان حدوث الفعل",
                examples=[("كتب", "مَكْتَب"), ("لعب", "مَلْعَب"), ("شرب", "مَشْرَب")],
                frequency=70,
            ),
        ]

        for pattern in noun_forms:
            self.noun_patterns[pattern.name] = pattern

    def _import_data_adjective_patterns(self):
        """تحميل أوزان الصفات"""
        adj_forms = [
            # فَعِيل - صفة مشبهة
            MorphPattern(
                name="فَعِيل",
                pattern_type=PatternType.ADJECTIVE,
                cv_structure=CVStructure("CVCVC", "فَعِيل"),
                semantic_meaning="الصفة الثابتة",
                examples=[("كرم", "كَرِيم"), ("جمل", "جَمِيل"), ("عظم", "عَظِيم")],
                frequency=88,
            ),
            # فَاعِل - صفة الفاعل
            MorphPattern(
                name="فَاعِل_صفة",
                pattern_type=PatternType.ADJECTIVE,
                cv_structure=CVStructure("CVCVC", "فَاعِل"),
                semantic_meaning="صفة من يقوم بالفعل",
                examples=[("عدل", "عَادِل"), ("صبر", "صَابِر"), ("شكر", "شَاكِر")],
                frequency=75,
            ),
            # أَفْعَل - التفضيل
            MorphPattern(
                name="أَفْعَل",
                pattern_type=PatternType.ADJECTIVE,
                cv_structure=CVStructure("VCCVC", "أَفْعَل"),
                semantic_meaning="اسم التفضيل",
                examples=[("كبر", "أَكْبَر"), ("صغر", "أَصْغَر"), ("جمل", "أَجْمَل")],
                frequency=92,
            ),
        ]

        for pattern in adj_forms:
            self.adjective_patterns[pattern.name] = pattern

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[MorphPattern]:
        """جلب الأوزان حسب النوع"""
        if pattern_type == PatternType.VERB:
            return list(self.verb_patterns.values())
        elif pattern_type == PatternType.NOUN:
            return list(self.noun_patterns.values())
        elif pattern_type == PatternType.ADJECTIVE:
            return list(self.adjective_patterns.values())
        elif pattern_type == PatternType.MASDAR:
            return [p for p in self.noun_patterns.values() if p.pattern_type == PatternType.MASDAR]
        elif pattern_type == PatternType.PARTICIPLE:
            return [p for p in self.noun_patterns.values() if p.pattern_type == PatternType.PARTICIPLE]
        else:
            return []

    def get_patterns_for_root(self, root: 'ArabicRoot') -> List[MorphPattern]:
        """جلب الأوزان المناسبة للجذر"""
        all_patterns = (
            list(self.verb_patterns.values()) +
            list(self.noun_patterns.values()) +
            list(self.adjective_patterns.values())
        )
        return [p for p in all_patterns if self._is_pattern_applicable(p, root)]

    def _is_pattern_applicable(self, pattern: MorphPattern, root: 'ArabicRoot') -> bool:
        """فحص قابلية تطبيق الوزن على الجذر"""
        # فحص بسيط - يمكن تطويره لاحقاً
        if len(root.root) == 3:  # الجذور الثلاثية
            return True
        elif len(root.root) == 4 and pattern.pattern_type in [PatternType.VERB]:
            return False  # الجذور الرباعية تحتاج أوزان خاصة
        return True

    def get_pattern(self, name: str) -> Optional[MorphPattern]:
        """البحث عن وزن بالاسم"""
        return next(
            (
                patterns_dict[name]
                for patterns_dict in [
                    self.verb_patterns,
                    self.noun_patterns,
                    self.adjective_patterns,
                ]
                if name in patterns_dict
            ),
            None,
        )

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[MorphPattern]:
        """إرجاع الأوزان حسب النوع"""
        return [
            pattern
            for patterns_dict in [
                self.verb_patterns,
                self.noun_patterns,
                self.adjective_patterns,
            ]
            for pattern in patterns_dict.values()
            if pattern.pattern_type == pattern_type
        ]

    def find_applicable_patterns(self, root: "ArabicRoot") -> List[MorphPattern]:
        """البحث عن الأوزان المناسبة لجذر معين"""
        all_patterns = (
            list(self.verb_patterns.values())
            + list(self.noun_patterns.values())
            + list(self.adjective_patterns.values())
        )

        applicable = [
            pattern for pattern in all_patterns if pattern.can_apply_to_root(root)
        ]

        # ترتيب حسب التكرار
        applicable.sort(key=lambda p: p.frequency or 0, reverse=True)
        return applicable

    def store_data_to_json(self, filepath: str):
        """تصدير جميع الأوزان إلى JSON"""
        data = {
            "verb_patterns": {
                name: pattern.to_dict() for name, pattern in self.verb_patterns.items()
            },
            "noun_patterns": {
                name: pattern.to_dict() for name, pattern in self.noun_patterns.items()
            },
            "adjective_patterns": {
                name: pattern.to_dict()
                for name, pattern in self.adjective_patterns.items()
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# مثيل مشترك لمستودع الأوزان
pattern_repository = PatternRepository()

if __name__ == "__main__":
    # اختبار النظام
    from .roots import_data create_root

    # اختبار جذر "كتب"
    root = create_root("كتب", "الكتابة")
    patterns = pattern_repository.find_applicable_patterns(root)

    print(f"الأوزان المناسبة للجذر '{root.root_string}':")
    for pattern in patterns[:5]:  # أول 5 أوزان
        result = pattern.apply_to_root(root)
        print(f"- {pattern.name}: {result} ({pattern.semantic_meaning})")
