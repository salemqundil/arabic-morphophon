#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار سريع لمولد الأوزان الصرفية العربية
Quick test for Arabic morphological weight generator
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


try:
    print("🔬 اختبار مولد الأوزان الصرفية...")

    # استيراد النظام
    from arabic_morphological_weight_generator import (
        ArabicMorphologicalWeightGenerator,
    )  # noqa: F401

    print("✅ تم استيراد النظام بنجاح")

    # إنشاء المولد
    generator = ArabicMorphologicalWeightGenerator()
    print("✅ تم إنشاء المولد بنجاح")

    # توليد عينة من أوزان الأفعال
    print("\n🔤 توليد عينة من أوزان الأفعال...")
    verb_weights = generator.generate_verb_weights()
    print(f"✅ تم توليد {len(verb_weights)} وزن فعل")

    # توليد عينة من أوزان الأسماء
    print("\n🔤 توليد عينة من أوزان الأسماء...")
    noun_weights = generator.generate_noun_weights()
    print(f"✅ تم توليد {len(noun_weights)} وزن اسم")

    # عرض عينات
    print("\n📝 عينة من أوزان الأفعال:")
    for i, weight in enumerate(verb_weights[:5], 1):
        print(
            f"   {i}. {weight.pattern_name} - {weight.phonetic_form} ({weight.word_type.value})"
        )  # noqa: E501

    print("\n📝 عينة من أوزان الأسماء:")
    for i, weight in enumerate(noun_weights[:5], 1):
        print(
            f"   {i}. {weight.pattern_name} - {weight.phonetic_form} ({weight.word_type.value})"
        )

    # إحصائيات سريعة
    total_weights = len(verb_weights) + len(noun_weights)
    print("\n📊 إحصائيات سريعة:")
    print(f"   • إجمالي أوزان الأفعال: {len(verb_weights)}")
    print(f"   • إجمالي أوزان الأسماء: {len(noun_weights)}")
    print(f"   • المجموع الكلي: {total_weights}")

    # اختبار وزن عينة
    if verb_weights:
        sample_weight = verb_weights[0]
        print(f"\n🔍 تحليل وزن عينة '{sample_weight.pattern_name}")
        print(f"   • النمط الصوتي: {sample_weight.syllable_pattern}")
        print(f"   • الشكل الصوتي: {sample_weight.phonetic_form}")
        print(f"   • نوع الكلمة: {sample_weight.word_type.value}")
        print(f"   • الوزن العروضي: {sample_weight.prosodic_weight}")
        print(f"   • عدد المقاطع: {len(sample_weight.syllable_sequence)}")

    print("\n🎉 اختبار مولد الأوزان اكتمل بنجاح!")
    print("🚀 النظام جاهز لتطبيقات التصريف المتقدمة!")

except Exception as e:
    print(f"❌ خطأ في الاختبار: {e}")
    import traceback  # noqa: F401

    traceback.print_exc()
