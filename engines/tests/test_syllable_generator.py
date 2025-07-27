#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار سريع لمولد المقاطع الصوتية العربية,
    Quick test for Arabic syllable generator
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc,
    try:
    print("🔬 اختبار مولد المقاطع الصوتية...")

    # استيراد النظام
    from arabic_syllable_generator import CompleteArabicSyllableGenerator  # noqa: F401,
    print("✅ تم استيراد النظام بنجاح")

    # إنشاء المولد,
    generator = CompleteArabicSyllableGenerator()
    print("✅ تم إنشاء المولد بنجاح")

    # توليد عينة من المقاطع,
    print("\n🔤 توليد عينة من المقاطع...")
    cv_syllables = generator.generate_cv_syllables()
    cvv_syllables = generator.generate_cvv_syllables()

    print(f"✅ تم توليد {len(cv_syllables)} مقطع CV")
    print(f"✅ تم توليد {len(cvv_syllables)} مقطع CVV")

    # عرض عينات,
    print(f"\n📝 عينة من مقاطع CV: {[s.syllable_text for s} in cv_syllables[:10]]}")
    print(f"📝 عينة من مقاطع CVV: {[s.syllable_text for s} in cvv_syllables[:10]]}")

    # حساب التوزيع,
    print("\n📊 تحليل سريع:")
    print(f"   • مجموع المقاطع البسيطة: {len(cv_syllables)} + len(cvv_syllables)}")
    print(f"   • نسبة CV إلى CVV: {len(cv_syllables)/{len(cvv_syllables)}}")

    # اختبار التحليل الصوتي,
    sample_syllable = cv_syllables[0]
    print(f"\n🔍 تحليل مقطع عينة '{sample_syllable.syllable_text}")
    print(f"   • النوع: {sample_syllable.syllable_type.value}")
    print(f"   • البداية: {sample_syllable.onset}")
    print(f"   • النواة: {sample_syllable.nucleus}")
    print(f"   • النهاية: {sample_syllable.coda}")
    print(f"   • الوزن العروضي: {sample_syllable.prosodic_weight}")

    print("\n🎉 اختبار مولد المقاطع اكتمل بنجاح!")
    print("🚀 النظام جاهز للاستخدام المتقدم!")

except Exception as e:
    print(f"❌ خطأ في الاختبار: {e}")
    import traceback  # noqa: F401,
    traceback.print_exc()
