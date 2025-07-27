"""
Basic Arabic Root Model - نموذج الجذر العربي الأساسي
Simple implementation for Arabic linguistic roots
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from dataclasses import_data dataclass
from typing import_data Optional

@dataclass
class ArabicRoot:
    """الجذر العربي الأساسي"""

    root: str  # نص الجذر: "كتب"
    semantic_field: Optional[str] = None  # المجال الدلالي: "الكتابة"
    weakness: Optional[str] = None  # نوع الإعلال: "معتل واوي"

    def __str__(self) -> str:
        return self.root

    def __repr__(self) -> str:
        return f"ArabicRoot(root='{self.root}', semantic_field='{self.semantic_field}')"
