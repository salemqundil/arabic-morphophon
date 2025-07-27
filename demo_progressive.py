#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 مثال سريع للتحليل التدريجي
============================
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


# استيراد النظام
from comprehensive_progressive_system import ComprehensiveProgressiveVectorSystem


def demo_analysis():
    """عرض توضيحي سريع"""

    print("🔥 النظام التدريجي للمتجه الرقمي العربي")
    print("=" * 50)

    # إنشاء النظام
    system = ComprehensiveProgressiveVectorSystem()

    # كلمات للاختبار
    test_words = ["شمس", "مدرسة", "كتاب", "استخراج", "مُعلِّم"]

    for word in test_words:
        print(f"\n🔍 تحليل الكلمة: '{word}'")
        print("-" * 30)

        # تحليل الكلمة
        result = system.analyze_word_progressive(word)

        # عرض النتائج
        print(f"✅ مراحل مكتملة: {result.successful_stages}/{result.total_stages}")
        print(f"📊 أبعاد المتجه: {result.vector_dimensions}")
        print(f"🎯 الثقة: {result.overall_confidence:.1%}")
        print(f"🔗 تكامل المحركات: {result.engines_integration_score:.1%}")
        print(f"⏱️ الوقت: {result.total_processing_time:.4f}s")

        # عينة من المتجه
        if result.final_vector:
            sample = [f"{x:.2f}" for x in result.final_vector[:8]]
            print(f"🎲 عينة متجه: [{', '.join(sample)}...]")

    print(f"\n🎉 تم الانتهاء من العرض التوضيحي!")


if __name__ == "__main__":
    demo_analysis()
