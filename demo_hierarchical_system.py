#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام التحليل الهرمي الشبكي للعربية - نسخة تجريبية
=====================================

هذا برنامج تجريبي لاختبار النظام الهرمي الجديد المكون من 7 محركات:
1. PhonemeHarakahEngine - تحليل الفونيمات والحركات
2. SyllablePatternEngine - تحليل المقاطع الصوتية
3. MorphemeMapperEngine - تحليل البنية الصرفية
4. WeightInferenceEngine - استنتاج الوزن الصرفي
5. WordClassifierEngine - تصنيف الكلمات نحوياً
6. SemanticRoleEngine - تحليل الأدوار الدلالية
7. WordTraceGraph - التتبع الشامل للكلمة

المؤلف: GitHub Copilot
التاريخ: 2025
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import sys
import time
import json
from pathlib import Path

# إضافة المسار الحالي إلى sys.path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from hierarchical_graph_engine import HierarchicalGraphSystem, AnalysisLevel

    print("✅ تم تحميل النظام الهرمي بنجاح!")
except ImportError as e:
    print(f"❌ خطأ في تحميل النظام: {e}")
    sys.exit(1)


def test_word_analysis(word: str):
    """اختبار تحليل كلمة واحدة"""
    print(f"\n" + "=" * 60)
    print(f"🔍 تحليل الكلمة: '{word}'")
    print("=" * 60)

    # إنشاء النظام
    system = HierarchicalGraphSystem()

    try:
        # بداية التحليل
        start_time = time.time()
        results = system.analyze_word(word)
        total_time = time.time() - start_time

        print(f"\n⏱️  إجمالي وقت التحليل: {total_time:.4f} ثانية")
        print(
            f"🎯 عدد المستويات المكتملة: {len(results)-1}/7"
        )  # -1 لاستبعاد original_word

        # عرض نتائج كل مستوى
        level_names = {
            "phoneme_harakah": "1️⃣ الفونيمات والحركات",
            "syllable_pattern": "2️⃣ المقاطع الصوتية",
            "morpheme_mapper": "3️⃣ البنية الصرفية",
            "weight_inference": "4️⃣ استنتاج الوزن",
            "word_classifier": "5️⃣ تصنيف الكلمة",
            "semantic_role": "6️⃣ الأدوار الدلالية",
            "word_tracer": "7️⃣ التتبع الشامل",
        }

        for level_key, level_name in level_names.items():
            if level_key in results:
                level_result = results[level_key]
                print(f"\n{level_name}:")
                print(f"   📊 الثقة: {level_result['confidence']:.2%}")
                print(f"   ⚡ الوقت: {level_result['processing_time']:.4f}s")
                print(f"   🔢 حجم المتجه: {len(level_result['vector'])}")

                # عرض بعض الخصائص المميزة
                if (
                    "graph_node" in level_result
                    and "features" in level_result["graph_node"]
                ):
                    features = level_result["graph_node"]["features"]
                    if features:
                        print(f"   ✨ خصائص مميزة: {list(features.keys())[:3]}...")

        # التحليل النهائي الشامل
        if "word_tracer" in results:
            tracer = results["word_tracer"]["graph_node"]
            final_analysis = tracer.get("final_analysis", {})

            print(f"\n🏆 التحليل النهائي الشامل:")
            print(
                f"   🎯 الثقة الإجمالية: {final_analysis.get('overall_confidence', 0):.2%}"
            )
            print(
                f"   📈 نسبة الاكتمال: {final_analysis.get('analysis_completeness', 0):.2%}"
            )
            print(
                f"   ⏱️  إجمالي وقت المعالجة: {final_analysis.get('total_processing_time', 0):.4f}s"
            )

            # الملخص اللغوي
            if "linguistic_summary" in final_analysis:
                summary = final_analysis["linguistic_summary"]
                print(f"   📝 الملخص اللغوي:")
                for key, value in summary.items():
                    print(f"      • {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
        import traceback

        print("🔍 تفاصيل الخطأ:")
        traceback.print_exc()
        return False


def demo_multiple_words():
    """تجربة عدة كلمات"""
    test_words = [
        "كتاب",  # كلمة بسيطة
        "المكتبة",  # كلمة بأداة التعريف
        "يكتبون",  # فعل مضارع
        "مكتوب",  # اسم مفعول
        "كاتب",  # اسم فاعل
    ]

    print("🚀 بدء تجربة النظام الهرمي الشبكي للعربية")
    print("=" * 80)

    successful_analyses = 0

    for word in test_words:
        if test_word_analysis(word):
            successful_analyses += 1

        # توقف قصير بين التحليلات
        time.sleep(0.5)

    # إحصائيات نهائية
    print(f"\n" + "=" * 80)
    print(f"📊 إحصائيات نهائية:")
    print(f"   ✅ تحليلات ناجحة: {successful_analyses}/{len(test_words)}")
    print(f"   📈 معدل النجاح: {successful_analyses/len(test_words):.1%}")
    print("=" * 80)


def interactive_mode():
    """وضع التفاعل المباشر"""
    print("\n🎮 الوضع التفاعلي - النظام الهرمي الشبكي")
    print("اكتب كلمة عربية للتحليل، أو 'خروج' للإنهاء")
    print("-" * 60)

    while True:
        word = input("\n🔤 أدخل كلمة: ").strip()

        if word.lower() in ["خروج", "exit", "quit", "q"]:
            print("👋 شكراً لاستخدام النظام!")
            break

        if not word:
            print("⚠️  يرجى إدخال كلمة صحيحة")
            continue

        if (
            not word.replace("ّ", "")
            .replace("َ", "")
            .replace("ُ", "")
            .replace("ِ", "")
            .isalpha()
        ):
            print("⚠️  يرجى إدخال كلمة عربية فقط")
            continue

        test_word_analysis(word)


def main():
    """الدالة الرئيسية"""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     النظام الهرمي الشبكي للتحليل العربي                      ║
║                        Hierarchical Graph Engine for Arabic                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  نظام تحليل متقدم من 7 مستويات:                                               ║
║  • الفونيمات والحركات • المقاطع الصوتية • البنية الصرفية                     ║
║  • استنتاج الوزن • تصنيف الكلمة • الأدوار الدلالية • التتبع الشامل           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    if len(sys.argv) > 1:
        # تحليل كلمة محددة من command line
        word = sys.argv[1]
        test_word_analysis(word)
    else:
        # قائمة الخيارات
        print("اختر نمط التشغيل:")
        print("1. تجربة عدة كلمات (Demo)")
        print("2. وضع تفاعلي (Interactive)")
        print("3. تحليل كلمة واحدة")

        choice = input("\nاختيارك (1-3): ").strip()

        if choice == "1":
            demo_multiple_words()
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            word = input("أدخل الكلمة: ").strip()
            if word:
                test_word_analysis(word)
            else:
                print("❌ لم يتم إدخال كلمة")
        else:
            print("❌ اختيار غير صحيح")


if __name__ == "__main__":
    main()
