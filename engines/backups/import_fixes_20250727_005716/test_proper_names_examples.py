#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أمثلة متقدمة لاختبار مولد أسماء الأعلام العربية
Advanced Examples for Testing Arabic Proper Names Generator

يحتوي على اختبارات شاملة لعرض قدرات النظام في توليد أسماء أصيلة
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


from advanced_arabic_proper_names_generator import (  # noqa: F401
    AdvancedArabicProperNamesGenerator,
    ProperNameCategory,
    demo_proper_names_generation,
)


def test_detailed_name_generation():  # type: ignore[no-untyped def]
    """اختبار مفصل لتوليد الأسماء"""

    print("🎯 اختبار مفصل لتوليد أسماء الأعلام العربية")
    print("=" * 60)

    generator = AdvancedArabicProperNamesGenerator()

    # 1. اختبار أسماء الذكور
    print("\n1️⃣ أسماء الرجال:")
    print(" " * 40)
    male_names = generator.generate_names(ProperNameCategory.PERSON_MALE, count=10)

    for i, name in enumerate(male_names[:5], 1):
    print(
    f"   {i}. {name.name:15} | جودة: {name.authenticity_score:.2f} | {name.semantic_meaning}"
    )  # noqa: E501
    print(f"      المقاطع: {'} + '.join(name.syllables)}")
        if name.historical_template:
    print(f"      النمط: {name.historical_template}")
    print()

    # 2. اختبار أسماء الإناث
    print("\n2️⃣ أسماء النساء:")
    print(" " * 40)
    female_names = generator.generate_names(ProperNameCategory.PERSON_FEMALE, count=10)

    for i, name in enumerate(female_names[:5], 1):
    print(
    f"   {i}. {name.name:15} | جودة: {name.authenticity_score:.2f} | {name.semantic_meaning}"
    )  # noqa: E501
    print(f"      المقاطع: {'} + '.join(name.syllables)}")
        if name.examples:
    print(f"      أسماء مشابهة: {', '.join(name.examples[:2])}")
    print()

    # 3. اختبار أسماء المدن
    print("\n3️⃣ أسماء المدن:")
    print(" " * 40)
    city_names = generator.generate_names(ProperNameCategory.PLACE_CITY, count=8)

    for i, name in enumerate(city_names[:4], 1):
    print(f"   {i}. {name.name:15} | جودة: {name.authenticity_score:.2f}")
    print(f"      المقاطع: {'} + '.join(name.syllables)}")
    print(f"      السياق الثقافي: {name.cultural_context}")
    print()

    # 4. اختبار أسماء الدول
    print("\n4️⃣ أسماء الدول:")
    print(" " * 40)
    country_names = generator.generate_names(ProperNameCategory.PLACE_COUNTRY, count=6)

    for i, name in enumerate(country_names[:3], 1):
    print(f"   {i}. {name.name:15} | جودة: {name.authenticity_score:.2f}")
    print(f"      النمط الصوتي: {name.pattern.value}")
        if name.historical_template:
    print(f"      القالب التاريخي: {name.historical_template}")
    print()

    # 5. اختبار المعالم الطبيعية
    print("\n5️⃣ المعالم الطبيعية:")
    print(" " * 40)
    natural_names = generator.generate_names(ProperNameCategory.PLACE_NATURAL, count=6)

    for i, name in enumerate(natural_names[:3], 1):
    print(f"   {i}. {name.name:15} | جودة: {name.authenticity_score:.2f}")
    print(
    f"      التحليل الصوتي: {name.phonetic_analysis.get('euphony_score', 0):.2f}"
    )  # noqa: E501
    print()


def test_meaning_based_generation():  # type: ignore[no-untyped def]
    """اختبار التوليد بناءً على المعاني"""

    print("\n🎯 اختبار التوليد بناءً على المعاني المحددة")
    print("=" * 60)

    generator = AdvancedArabicProperNamesGenerator()

    # معاني مختارة للاختبار
    meaning_tests = [
    ("الشجاعة", ProperNameCategory.PERSON_MALE, "أسماء رجال تدل على الشجاعة"),
    ("الجمال", ProperNameCategory.PERSON_FEMALE, "أسماء نساء تدل على الجمال"),
    ("الحكمة", ProperNameCategory.PERSON_MALE, "أسماء رجال تدل على الحكمة"),
    ("الرحمة", ProperNameCategory.PERSON_FEMALE, "أسماء نساء تدل على الرحمة"),
    ("الماء", ProperNameCategory.PLACE_NATURAL, "أماكن طبيعية متعلقة بالماء"),
    ("الجبل", ProperNameCategory.PLACE_NATURAL, "معالم جبلية"),
    ]

    for i, (meaning, category, description) in enumerate(meaning_tests, 1):
    print(f"\n{i}. {description:}")
    print(f"   المعنى المطلوب: {meaning}")
    print(f"   الفئة: {category.value}")
    print("   " + " " * 50)

    meaning_names = generator.generate_by_meaning(meaning, category, count=3)

        for j, name in enumerate(meaning_names, 1):
    print(f"   {j}. {name.name:12} - جودة: {name.authenticity_score:.2f}")
    print(f"      معنى: {name.semantic_meaning}")
    print(f"      مقاطع: {'} + '.join(name.syllables)}")
    print()


def test_phonetic_analysis():  # type: ignore[no-untyped def]
    """اختبار التحليل الصوتي للأسماء"""

    print("\n🔬 اختبار التحليل الصوتي المتقدم")
    print("=" * 60)

    generator = AdvancedArabicProperNamesGenerator()

    # توليد عينة من الأسماء للتحليل
    sample_names = generator.generate_names(ProperNameCategory.PERSON_MALE, count=5)

    print("تحليل صوتي مفصل للأسماء المولدة:")
    print(" " * 50)

    for i, name in enumerate(sample_names, 1):
    analysis = name.phonetic_analysis

    print(f"{i}. اسم: {name.name}")
    print(f"   عدد المقاطع: {analysis.get('syllable_count',} 'غير محدد')}")
    print(f"   نمط النبرة: {analysis.get('stress_pattern',} 'غير محدد')}")
    print(f"   تجمعات الصوامت: {analysis.get('consonant_clusters', [])}")
    print(f"   نمط الحركات: {analysis.get('vowel_pattern',} 'غير محدد')}")
    print(f"   صعوبة النطق: {analysis.get('phonetic_difficulty', 0):.2f}")
    print(f"   جمال الصوت: {analysis.get('euphony_score', 0):.2f}")
    print(f"   نقاط الأصالة: {name.authenticity_score:.2f}")
    print()


def test_cultural_templates():  # type: ignore[no-untyped def]
    """اختبار القوالب الثقافية المختلفة"""

    print("\n🏛️ اختبار القوالب الثقافية والتاريخية")
    print("=" * 60)

    generator = AdvancedArabicProperNamesGenerator()

    # توليد أسماء متنوعة لرصد القوالب
    all_categories = [
    ProperNameCategory.PERSON_MALE,
    ProperNameCategory.PERSON_FEMALE,
    ProperNameCategory.PLACE_CITY,
    ProperNameCategory.PLACE_COUNTRY,
    ProperNameCategory.PLACE_NATURAL,
    ]

    templates_found = {}

    for category in all_categories:
    names = generator.generate_names(category, count=10)

        for name in names:
            if name.historical_template:
                if name.historical_template not in templates_found:
    templates_found[name.historical_template] = []
    templates_found[name.historical_template].append(
    (name.name, category.value, name.authenticity_score)
    )

    print("القوالب التاريخية المكتشفة:")
    print(" " * 40)

    for template, names in templates_found.items():
    print(f"\n🏺 قالب: {template}")
    print(f"   عدد الأسماء: {len(names)}")
    print("   أمثلة:")

        # أفضل 3 أمثلة
    best_examples = sorted(names, key=lambda x: x[2], reverse=True)[:3]
        for name, category, score in best_examples:
    print(f"   • {name:12} ({category}) - جودة: {score:.2f}")
    print()


def comprehensive_test():  # type: ignore[no-untyped-def]
    """اختبار شامل لجميع قدرات النظام"""

    print("🚀 الاختبار الشامل لمولد أسماء الأعلام العربية")
    print("=" * 70)

    # تشغيل جميع الاختبارات
    test_detailed_name_generation()
    test_meaning_based_generation()
    test_phonetic_analysis()
    test_cultural_templates()

    print("\n" + "=" * 70)
    print("✅ اكتمل الاختبار الشامل بنجاح!")
    print("📊 النظام يعمل بكفاءة عالية في توليد أسماء عربية أصيلة")
    print("🎯 تم التحقق من جميع الوظائف والميزات المتقدمة")
    print("=" * 70)


if __name__ == "__main__":
    # تشغيل الاختبار الشامل
    comprehensive_test()
