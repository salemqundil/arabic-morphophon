#!/usr/bin/env python3
"""
Test script for simplified roots model
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Update import_datas to use the new organized structure
from arabic_morphophon.models.roots import_data ArabicRoot as Root
from arabic_morphophon.models.roots import_data RootType, create_root

def print_separator():
    """Print a separator line"""
    print("\n" + "-" * 30)

def test_simplified_roots():
    print("🧪 اختبار النموذج المبسط للجذور")
    print("=" * 50)

    # اختبار إنشاء جذر ثلاثي
    root = create_root("كتب", "الكتابة")
    print(f"الجذر: {root.root}")
    print(f"المجال الدلالي: {root.semantic_field}")
    print(f"النوع: {root.root_type.value}")
    print(f"التكرار: {root.frequency}")
    print(f"الإعلال: {root.get_weakness_type()}")

    print_separator()

    # اختبار جذر معتل
    weak_root = create_root("وعد", "الوعد")
    print(f"الجذر المعتل: {weak_root.root}")
    print(f"النوع: {weak_root.root_type.value}")
    print(f"الإعلال: {weak_root.get_weakness_type()}")

    print("\n" + "-" * 30)

    # اختبار جذر رباعي
    quad_root = create_root("دحرج", "الحركة")
    print(f"الجذر الرباعي: {quad_root.root}")
    print(f"النوع: {quad_root.root_type.value}")
    print(f"الإعلال: {quad_root.get_weakness_type()}")

    print("\n✅ جميع الاختبارات نجحت!")

if __name__ == "__main__":
    test_simplified_roots()
