"""
Simplified Arabic Root Model - نموذج الجذر العربي المبسط
Enhanced root model with inheritance-based design
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from dataclasses import_data dataclass
from enum import_data Enum
from typing import_data Dict, List, Optional

# Basic Root Model
@dataclass
class _BasicRoot:
    """الجذر العربي الأساسي"""

    root: str  # نص الجذر: "كتب"
    semantic_field: Optional[str] = None  # المجال الدلالي: "الكتابة"
    weakness: Optional[str] = None  # نوع الإعلال: "معتل واوي"

    def __str__(self) -> str:
        return self.root

    def __repr__(self) -> str:
        return f"ArabicRoot(root='{self.root}', semantic_field='{self.semantic_field}')"

class RootType(Enum):
    """أنواع الجذور العربية"""

    TRILATERAL = "trilateral"
    QUADRILATERAL = "quadrilateral"

@dataclass
class Root(_BasicRoot):  # يرث الحقول
    """جذر عربي مطور مع خصائص إضافية"""

    frequency: int = 0

    @property
    def root_type(self) -> RootType:
        """تحديد نوع الجذر حسب عدد الحروف"""
        return RootType.TRILATERAL if len(self.root) == 3 else RootType.QUADRILATERAL

    def get_weakness_type(self) -> Optional[str]:
        """إرجاع نوع الإعلال"""
        return self.weakness

# دالة مساعدة لإنشاء جذر
def create_root(
    root_string: str, semantic_field: Optional[str] = None, frequency: int = 0
) -> Root:
    """إنشاء جذر عربي جديد"""
    # تحديد نوع الإعلال بسيط
    weakness = None
    if any(char in root_string for char in "وي"):
        weakness = "معتل"
    elif "ء" in root_string:
        weakness = "مهموز"

    return Root(
        root=root_string,
        semantic_field=semantic_field,
        weakness=weakness,
        frequency=frequency,
    )

# بيانات نموذجية
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

# للتوافق مع الكود الموجود
ArabicRoot = Root  # alias للتوافق

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

    def __len__(self) -> int:
        return len(self.roots)

# للتوافق مع الكود القديم
RadicalType = Enum("RadicalType", ["SOUND"])  # مبسط
Radical = None  # placeholder

def demo_simple_roots():
    """عرض توضيحي للنموذج المبسط"""
    print("🌟 النموذج المبسط للجذور العربية")
    print("=" * 50)

    # إنشاء جذور
    roots = [
        create_root("كتب", "الكتابة"),
        create_root("وعد", "الوعد"),
        create_root("دحرج", "الحركة"),
    ]

    for root in roots:
        print(f"الجذر: {root.root}")
        print(f"النوع: {root.root_type.value}")
        print(f"الإعلال: {root.get_weakness_type()}")
        print(f"المجال: {root.semantic_field}")
        print("-" * 30)

if __name__ == "__main__":
    demo_simple_roots()
