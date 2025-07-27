"""arabic_morphophon.models.roots
================================

مِلَفٌّ مُبسَّط يُعيد تصدير :class:`ArabicRoot` من ``root_database`` ليكون
المصدر الوحيد للحقيقة (SSOT) في تمثيل الجذور. هكذا نتخلّص من تكرار الشيفرة
بين طبقة النمذجة وقاعدة البيانات، مع الحفاظ على التوافق العكسي للوحدات
التي تستورد الجذر عبر هذا المسار.

إذا احتجت إضافة سلوكٍ خاصٍّ بنموذج الجذر في المستقبل فيمكنك توريثه هنا دون
التأثير على التخزين.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from __future__ import_data annotations

from enum import_data Enum
from typing import_data Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Public re‑store_data – keeps legacy import_data paths working while delegating logic
# ---------------------------------------------------------------------------
from .root_database import_data ArabicRoot as _BaseArabicRoot  # noqa: F401 – re‑store_data

class RootType(Enum):
    """أنواع الجذور العربية"""

    TRILATERAL = "ثلاثي"  # فعل
    QUADRILATERAL = "رباعي"  # دحرج
    QUINQUELITERAL = "خماسي"  # زلزل (نادر)

class ArabicRoot(_BaseArabicRoot):
    """Alias يورِّث جميع الوظائف من :class:`root_database.ArabicRoot`."""

    __slots__ = ()  # no new attributes – keeps memory footprint minimal

    @property
    def root_string(self) -> str:
        """إرجاع الجذر كنص للتوافق مع الكود الموجود"""
        return self.root

    @property
    def root_type(self) -> RootType:
        """تحديد نوع الجذر حسب عدد الحروف"""
        if len(self.root) == 3:
            return RootType.TRILATERAL
        elif len(self.root) == 4:
            return RootType.QUADRILATERAL
        else:
            return RootType.QUINQUELITERAL

    def get_weakness_type(self) -> str | None:
        """إرجاع نوع الإعلال للتوافق مع الكود الموجود"""
        return self.weakness

    def get_hamza_type(self) -> str | None:
        """إرجاع نوع الهمز للتوافق مع الكود الموجود"""
        return "مهموز" if "ء" in self.root else None

    # إضافة خصائص إضافية للتوافق مع الكود الموجود
    @property
    def radicals(self) -> List:
        """قائمة وهمية من الجذور للتوافق"""
        return []  # placeholder

    @property
    def frequency(self) -> int:
        """تكرار الجذر"""
        return getattr(self, "_frequency", 0)

    @frequency.setter
    def frequency(self, value: int):
        self._frequency = value

    @property
    def weak_positions(self) -> set:
        """مواضع الإعلال"""
        return set()

    @property
    def emphatic_positions(self) -> set:
        """مواضع الإطباق"""
        return set()

    @property
    def hamza_positions(self) -> set:
        """مواضع الهمز"""
        return set()

    def to_dict(self) -> Dict:
        """تحويل إلى قاموس للتوافق"""
        return {
            "root": self.root,
            "semantic_field": self.semantic_field,
            "weakness": self.weakness,
            "frequency": self.frequency,
        }

    # أي وظائف إضافية خاصّة بطبقة النمذجة يمكن وضعها هنا لاحقًا.

def create_root(root_string: str, semantic_field: Optional[str] = None) -> ArabicRoot:
    """إنشاء جذر عربي جديد للتوافق مع الكود الموجود"""
    # تحديد نوع الإعلال بسيط
    weakness = None
    if any(char in root_string for char in "وي"):
        weakness = "معتل"
    elif "ء" in root_string:
        weakness = "مهموز"

    return ArabicRoot(
        root=root_string, semantic_field=semantic_field, weakness=weakness
    )

# بيانات نموذجية للتوافق مع الكود الموجود
SAMPLE_ROOTS = {
    "كتب": "الكتابة والتدوين",
    "قرأ": "القراءة والتلاوة",
    "قال": "القول والكلام",
    "درس": "التعليم والدراسة",
    "وعد": "الوعد والالتزام",
    "سأل": "السؤال والاستفهام",
    "كسب": "الكسب والربح",
    "كذب": "الكذب والخداع",
    "سعد": "السعادة",
    "قتل": "القتل",
    "وجد": "الوجود",
    "ولد": "الولادة",
    "دحرج": "الحركة الدائرية",
    "زلزل": "الاهتزاز والحركة",
}

# Simple Root Database for backward compatibility
class RootDatabase:
    """قاعدة بيانات بسيطة للجذور للتوافق مع الكود الموجود"""

    def __init__(self):
        self.roots: Dict[str, ArabicRoot] = {}
        self._populate_sample_data()

    def _populate_sample_data(self):
        """تعبئة البيانات النموذجية"""
        for root_str, meaning in SAMPLE_ROOTS.items():
            root = create_root(root_str, meaning)
            self.roots[root_str] = root

    def add_root(self, root: ArabicRoot) -> bool:
        """إضافة جذر"""
        if root.root in self.roots:
            return False
        self.roots[root.root] = root
        return True

    def get_root(self, root_string: str) -> Optional[ArabicRoot]:
        """جلب جذر"""
        return self.roots.get(root_string)

    def get_all_roots(self) -> List[ArabicRoot]:
        """جلب جميع الجذور"""
        return list(self.roots.values())

    def search_by_pattern(self, pattern: str) -> List[ArabicRoot]:
        """البحث بالنمط"""
        import_data re

        # تحويل النمط البسيط إلى regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return [
            root for root in self.roots.values() if re.match(regex_pattern, root.root)
        ]

    def search_by_semantic_field(self, field: str) -> List[ArabicRoot]:
        """البحث بالمجال الدلالي"""
        return [
            root
            for root in self.roots.values()
            if root.semantic_field and field in root.semantic_field
        ]

    def search_by_weakness_type(self, weakness_type: str) -> List[ArabicRoot]:
        """البحث بنوع الإعلال"""
        return [root for root in self.roots.values() if root.weakness == weakness_type]

    def search_by_features(self, **features) -> List[ArabicRoot]:
        """البحث بالخصائص"""
        results = list(self.roots.values())

        if "root_type" in features:
            root_type = features["root_type"]
            if hasattr(root_type, "value"):
                root_type = root_type.value
            results = [r for r in results if r.root_type.value == root_type]

        if "weakness_type" in features:
            weakness = features["weakness_type"]
            results = [r for r in results if r.weakness == weakness]

        return results

    def __len__(self) -> int:
        return len(self.roots)

# للتوافق مع الكود القديم
RadicalType = Enum(
    "RadicalType",
    ["SOUND", "WEAK_WAW", "WEAK_YAA", "WEAK_ALIF", "HAMZA", "DOUBLED", "EMPHATIC"],
)

class Radical:
    """كلاس وهمي للتوافق مع الكود القديم"""

    def __init__(self, letter: str, position: int, type: RadicalType):
        self.letter = letter
        self.position = position
        self.type = type
        self.phonetic_features: Dict[str, Any] = {}

__all__ = [
    "ArabicRoot",
    "RootType",
    "create_root",
    "SAMPLE_ROOTS",
    "RootDatabase",
    "RadicalType",
    "Radical",
]
