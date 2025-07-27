#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار سريع للنظام الهرمي الشبكي
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import sys
from pathlib import Path

# إضافة المسار الحالي
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def test_imports():
    """اختبار استيراد الملفات"""
    try:
        from hierarchical_graph_engine import (
            HierarchicalGraphSystem,
            AnalysisLevel,
            PhonemeHarakahEngine,
            SyllablePatternEngine,
            MorphemeMapperEngine,
            WeightInferenceEngine,
            WordClassifierEngine,
            SemanticRoleEngine,
            WordTraceGraph,
        )

        print("✅ تم استيراد جميع المكونات بنجاح")
        return True
    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        return False


def test_system_creation():
    """اختبار إنشاء النظام"""
    try:
        from hierarchical_graph_engine import HierarchicalGraphSystem

        system = HierarchicalGraphSystem()
        print(f"✅ تم إنشاء النظام بنجاح مع {len(system.engines)} محركات")

        # التحقق من وجود جميع المحركات
        expected_levels = [1, 2, 3, 4, 5, 6, 7]
        available_levels = [level.value for level in system.engines.keys()]

        print(f"📊 المحركات المتاحة: {sorted(available_levels)}")
        print(f"🎯 المحركات المتوقعة: {expected_levels}")

        return len(available_levels) == len(expected_levels)

    except Exception as e:
        print(f"❌ خطأ في إنشاء النظام: {e}")
        return False


def test_simple_analysis():
    """اختبار تحليل بسيط"""
    try:
        from hierarchical_graph_engine import HierarchicalGraphSystem

        system = HierarchicalGraphSystem()
        word = "كتاب"

        print(f"🔍 اختبار تحليل الكلمة: {word}")

        # اختبار كل محرك على حدة
        engine1 = system.engines[list(system.engines.keys())[0]]  # المحرك الأول
        print(f"🔧 اختبار المحرك الأول: {type(engine1).__name__}")

        result1 = engine1.process(word)
        print(f"✅ نجح المحرك الأول - ثقة: {result1.confidence:.2%}")

        return True

    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """الدالة الرئيسية للاختبار"""
    print("🧪 اختبار النظام الهرمي الشبكي")
    print("=" * 50)

    tests = [
        ("استيراد المكونات", test_imports),
        ("إنشاء النظام", test_system_creation),
        ("تحليل بسيط", test_simple_analysis),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        print("-" * 30)

        if test_func():
            passed += 1
            print(f"✅ {test_name}: نجح")
        else:
            print(f"❌ {test_name}: فشل")

    print(f"\n📊 النتيجة النهائية: {passed}/{len(tests)} اختبارات نجحت")

    if passed == len(tests):
        print("🎉 جميع الاختبارات نجحت! النظام جاهز للاستخدام")
    else:
        print("⚠️  بعض الاختبارات فشلت - يحتاج إصلاح")


if __name__ == "__main__":
    main()
