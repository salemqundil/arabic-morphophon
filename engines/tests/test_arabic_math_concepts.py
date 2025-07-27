#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار مولد المفاهيم الرياضية العربية المتقدم,
    Test Suite for Advanced Arabic Mathematical Concepts Generator,
    هذا الملف يحتوي على اختبارات شاملة لنظام توليد المفاهيم الرياضية,
    العربية باستخدام قاعدة بيانات المقاطع الصوتية.

المطور: نظام الذكاء الاصطناعي العربي,
    Developer: Arabic AI System,
    التاريخ: 2025,
    Date: 2025
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc
    import sys  # noqa: F401
    import json  # noqa: F401
    import time  # noqa: F401
    from typing import Dict, List, Any
    from collections import defaultdict, Counter  # noqa: F401

# استيراد النظام الرئيسي,
    try:
    from arabic_mathematical_generator import (  # noqa: F401,
    AdvancedArabicMathGenerator,
    MathConceptCategory,
    NumberGender,
    GeneratedMathConcept,
    )
except ImportError as e:
    print(f"خطأ في استيراد النظام الرئيسي: {e}")
    sys.exit(1)


def test_detailed_number_generation():  # type: ignore[no-untyped def]
    """اختبار مفصل لتوليد الأعداد"""

    print("🔢 اختبار توليد الأعداد العربية")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    # اختبار الأعداد الأساسية بالتفصيل,
    print("\n1. الأعداد الأساسية (مذكر ومؤنث):")
    test_numbers = [1, 2, 3, 5, 8, 10, 11, 15, 20, 25, 30, 50, 100]

    results = []
    for number in test_numbers:
        for gender in [NumberGender.MASCULINE, NumberGender.FEMININE]:
            try:
    concept = generator.generate_number_concept(number, 'cardinal', gender)
    gender_symbol = "♂" if gender == NumberGender.MASCULINE else "♀"

    print(
    f"   {number:3d} {gender_symbol}: {concept.term:15s} "
    f"مقاطع: {len(concept.syllables):2d} "
    f"نمط: {concept.pattern.value:8s }"
    f"صحة: {concept.authenticity_score:.1f}"
    )

    results.append(
    {
    'number': number,
    'gender': gender.value,
    'term': concept.term,
    'syllables': concept.syllables,
    'pattern': concept.pattern.value,
    'phonetic_analysis': concept.phonetic_analysis,
    }
    )

            except Exception as e:
    print(f"   خطأ في توليد العدد {number: {e}}")

    # تحليل الأنماط,
    pattern_distribution = Counter([r['pattern'] for r in results])
    print("\n   توزيع الأنماط الصوتية:")
    for pattern, count in pattern_distribution.most_common():
    print(f"     {pattern}: {count} عدد")

    assert results is not None,
    def test_ordinal_numbers():  # type: ignore[no-untyped def]
    """اختبار الأعداد الترتيبية"""

    print("\n\n🥇 اختبار الأعداد الترتيبية")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    ordinal_results = []

    print("\nالأعداد الترتيبية (1 20):")
    for number in range(1, 21):
        for gender in [NumberGender.MASCULINE, NumberGender.FEMININE]:
            try:
    concept = generator.generate_number_concept(number, 'ordinal', gender)
    gender_symbol = "♂" if gender == NumberGender.MASCULINE else "♀"

    print(
    f"   {number:2d}. {gender_symbol}: {concept.term:15s }"
    f"مقاطع: {'} - '.join(concept.syllables)}"
    )  # noqa: E501,
    ordinal_results.append(
    {
    'number': number,
    'gender': gender.value,
    'term': concept.term,
    'examples': concept.examples,
    }
    )

            except Exception as e:
    print(f"   خطأ في توليد الترتيبي {number: {e}}")

    # إحصائيات,
    print(f"\n   تم توليد {len(ordinal_results)} عدد ترتيبي")

    assert ordinal_results is not None,
    def test_comprehensive_fractions():  # type: ignore[no-untyped def]
    """اختبار شامل للكسور"""

    print("\n\n🍰 اختبار شامل للكسور العربية")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    fraction_results = []

    # الكسور البسيطة,
    print("\n1. الكسور البسيطة (البسط = 1):")
    for denominator in range(2, 11):
        try:
    concept = generator.generate_fraction_concept(1, denominator)
    decimal_value = float(concept.mathematical_value)

    print(
    f"   1/{denominator:2d}: {concept.term:10s} "
    f"= {decimal_value:.3f }"
    f"مقاطع: {'} - '.join(concept.syllables)}"
    )

    fraction_results.append(
    {
    'numerator': 1,
    'denominator': denominator,
    'term': concept.term,
    'value': decimal_value,
    'category': concept.category.value,
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد الكسر 1/{denominator: {e}}")

    # الكسور المركبة,
    print("\n2. الكسور المركبة:")
    compound_fractions = [
    (2, 3),
    (3, 4),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 6),
    (7, 8),
    (5, 12),
    (7, 10),
    ]

    for numerator, denominator in compound_fractions:
        try:
    concept = generator.generate_fraction_concept(numerator, denominator)
    decimal_value = float(concept.mathematical_value)

    print(
    f"   {numerator}/{denominator}: {concept.term:15s} "
    f"= {decimal_value:.3f }"
    f"فئة: {concept.category.value}"
    )

    fraction_results.append(
    {
    'numerator': numerator,
    'denominator': denominator,
    'term': concept.term,
    'value': decimal_value,
    'category': concept.category.value,
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد الكسر {numerator}/{denominator}: {e}")

    # تحليل النتائج,
    simple_fractions = [
    f for f in fraction_results if f['category'] == 'fraction_simple'
    ]
    compound_fractions = [
    f for f in fraction_results if f['category'] == 'fraction_compound'
    ]

    print(f"\n   إجمالي الكسور: {len(fraction_results)}")
    print(f"   الكسور البسيطة: {len(simple_fractions)}")
    print(f"   الكسور المركبة: {len(compound_fractions)}")

    assert fraction_results is not None,
    def test_mathematical_operations():  # type: ignore[no-untyped def]
    """اختبار العمليات الرياضية"""

    print("\n\n⚙️ اختبار العمليات الرياضية")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    operation_results = []

    # العمليات الأساسية,
    print("\n1. العمليات الأساسية:")
    basic_operations = ['addition', 'subtraction', 'multiplication', 'division']

    for operation in basic_operations:
        try:
    concept = generator.generate_operation_concept(operation)

    print(
    f"   {operation:15s}: {concept.term:12s }"
    f"فئة: {concept.category.value}"
    )  # noqa: E501,
    print(
    f"      المشتقات: {', '.join(concept.linguistic_features['derivatives'])}"
    )  # noqa: E501,
    print(f"      الجذر: {concept.linguistic_features['root']}")
    print(f"      أمثلة: {concept.examples[0]}")
    print()

    operation_results.append(
    {
    'operation': operation,
    'term': concept.term,
    'category': concept.category.value,
    'derivatives': concept.linguistic_features['derivatives'],
    'examples': concept.examples,
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد العملية {operation: {e}}")

    # العمليات المتقدمة,
    print("\n2. العمليات المتقدمة:")
    advanced_operations = [
    'power',
    'root',
    'logarithm',
    'factorial',
    'ratio',
    'proportion',
    ]

    for operation in advanced_operations:
        try:
    concept = generator.generate_operation_concept(operation)

    print(
    f"   {operation:15s}: {concept.term:15s }"
    f"تحليل صوتي: {concept.phonetic_analysis.get('euphony_score', 0):.2f}"
    )

    operation_results.append(
    {
    'operation': operation,
    'term': concept.term,
    'category': concept.category.value,
    'phonetic_score': concept.phonetic_analysis.get('euphony_score', 0),
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد العملية {operation: {e}}")

    assert operation_results is not None,
    def test_mathematical_concepts():  # type: ignore[no-untyped def]
    """اختبار المفاهيم الرياضية المتقدمة"""

    print("\n\n🧮 اختبار المفاهيم الرياضية المتقدمة")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    concept_results = []

    # مفاهيم الحساب,
    print("\n1. المفاهيم الحسابية:")
    arithmetic_concepts = [
    ('numbers', 'arithmetic'),
    ('operations', 'arithmetic'),
    ('properties', 'arithmetic'),
    ]

    for concept_type, domain in arithmetic_concepts:
        try:
    concept = generator.generate_concept_term(concept_type, domain)

    print(
    f"   {concept_type:12s}: {concept.term:15s }"
    f"معنى: {concept.semantic_meaning}"
    )  # noqa: E501,
    print(
    f"      مصطلحات مرتبطة: {',} '.join(concept.linguistic_features.get('related_terms', [])[:3])}"
    )  # noqa: E501,
    concept_results.append(
    {
    'type': concept_type,
    'domain': domain,
    'term': concept.term,
    'meaning': concept.semantic_meaning,
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد مفهوم {concept_type: {e}}")

    # مفاهيم الجبر,
    print("\n2. المفاهيم الجبرية:")
    algebra_concepts = [
    ('variables', 'algebra'),
    ('equations', 'algebra'),
    ('functions', 'algebra'),
    ('polynomials', 'algebra'),
    ]

    for concept_type, domain in algebra_concepts:
        try:
    concept = generator.generate_concept_term(concept_type, domain)

    print(f"   {concept_type:12s}: {concept.term:15s}")
    print(
    f"      أمثلة: {concept.examples[0] if concept.examples else} 'لا توجد'}"
    )  # noqa: E501,
    concept_results.append(
    {
    'type': concept_type,
    'domain': domain,
    'term': concept.term,
    'examples': concept.examples,
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد مفهوم {concept_type: {e}}")

    # مفاهيم الهندسة,
    print("\n3. المفاهيم الهندسية:")
    geometry_concepts = [
    ('shapes', 'geometry'),
    ('measurements', 'geometry'),
    ('angles', 'geometry'),
    ('lines', 'geometry'),
    ]

    for concept_type, domain in geometry_concepts:
        try:
    concept = generator.generate_concept_term(concept_type, domain)

    print(
    f"   {concept_type:12s}: {concept.term:15s }"
    f"تحليل: {concept.phonetic_analysis.get('mathematical_appropriateness', 0):.2f}"
    )

    concept_results.append(
    {
    'type': concept_type,
    'domain': domain,
    'term': concept.term,
    'appropriateness': concept.phonetic_analysis.get(
    'mathematical_appropriateness', 0
    ),
    }
    )

        except Exception as e:
    print(f"   خطأ في توليد مفهوم {concept_type: {e}}")

    # مفاهيم الإحصاء,
    print("\n4. المفاهيم الإحصائية:")
    statistics_concepts = [
    ('measures', 'statistics'),
    ('probability', 'statistics'),
    ('distributions', 'statistics'),
    ]

    for concept_type, domain in statistics_concepts:
        try:
    concept = generator.generate_concept_term(concept_type, domain)

    print(f"   {concept_type:12s}: {concept.term:15s}")

    concept_results.append(
    {'type': concept_type, 'domain': domain, 'term': concept.term}
    )

        except Exception as e:
    print(f"   خطأ في توليد مفهوم {concept_type}: {e}")

    print(f"\n   تم توليد {len(concept_results)} مفهوم رياضي متقدم")

    assert concept_results is not None,
    def test_comprehensive_generation():  # type: ignore[no-untyped def]
    """اختبار التوليد الشامل"""

    print("\n\n🎯 اختبار التوليد الشامل")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    print("توليد مجموعة شاملة من المفاهيم الرياضية...")
    start_time = time.time()

    try:
    comprehensive_concepts = generator.generate_comprehensive_math_concepts(100)
    generation_time = time.time() - start_time,
    print(
    f"✅ تم توليد {len(comprehensive_concepts)} مفهوم في {generation_time:.2f} ثانية"
    )  # noqa: E501

        # تحليل إحصائي شامل,
    category_stats = defaultdict(int)
    pattern_stats = defaultdict(int)
    authenticity_scores = []

        for concept in comprehensive_concepts:
    category_stats[concept.category.value] += 1,
    pattern_stats[concept.pattern.value] += 1,
    authenticity_scores.append(concept.authenticity_score)

    print("\n📊 الإحصائيات التفصيلية:")

    print("\n   توزيع الفئات:")
        for category, count in sorted(category_stats.items()):
    percentage = (count / len(comprehensive_concepts)) * 100,
    print(f"     {category:20s}: {count:3d} ({percentage:5.1f}%)")

    print("\n   توزيع الأنماط الصوتية:")
        for pattern, count in sorted(pattern_stats.items()):
    percentage = (count / len(comprehensive_concepts)) * 100,
    print(f"     {pattern:10s}: {count:3d} ({percentage:5.1f%)}")

    print("\n   نقاط الأصالة:")
    avg_authenticity = sum(authenticity_scores) / len(authenticity_scores)
    print(f"     المتوسط: {avg_authenticity:.3f}")
    print(f"     الحد الأدنى: {min(authenticity_scores):.3f}")
    print(f"     الحد الأقصى: {max(authenticity_scores):.3f}")

        # عرض عينات مميزة,
    print("\n🌟 عينات مميزة من المفاهيم المولدة:")

        # أفضل المفاهيم من ناحية الأصالة,
    best_concepts = sorted(
    comprehensive_concepts, key=lambda x: x.authenticity_score, reverse=True
    )[:10]

        for i, concept in enumerate(best_concepts, 1):
    print(
    f"   {i:2d}. {concept.term:20s} "
    f"({concept.category.value:15s) }"
    f"أصالة: {concept.authenticity_score:.3f}"
    )

        # تحليل التنوع,
    unique_terms = len(set(c.term for c in comprehensive_concepts))
    diversity_ratio = unique_terms / len(comprehensive_concepts)

    print(
    f"\n   معدل التنوع: {diversity_ratio:.3f} ({unique_terms}/{len(comprehensive_concepts)})"
    )  # noqa: E501,
    assert comprehensive_concepts is not None,
    except Exception as e:
    print(f"❌ خطأ في التوليد الشامل: {e}")
    return []


def test_phonetic_analysis():  # type: ignore[no-untyped def]
    """اختبار التحليل الصوتي المتقدم"""

    print("\n\n🔊 اختبار التحليل الصوتي المتقدم")
    print(" " * 50)

    generator = AdvancedArabicMathGenerator()

    # اختبار مصطلحات مختارة,
    test_terms = [
    generator.generate_number_concept(5, 'cardinal', NumberGender.MASCULINE),
    generator.generate_fraction_concept(1, 3),
    generator.generate_operation_concept('multiplication'),
    generator.generate_concept_term('equations', 'algebra'),
    ]

    print("تحليل صوتي مفصل للمصطلحات:")

    for i, concept in enumerate(test_terms, 1):
    print(f"\n{i}. المصطلح: {concept.term}")
    print(f"   الفئة: {concept.category.value}")
    print(f"   المقاطع: {'} - '.join(concept.syllables)}")

    analysis = concept.phonetic_analysis,
    print("   التحليل الصوتي:")
    print(f"     عدد المقاطع: {analysis.get('syllable_count', 0)}")
    print(f"     نمط النبرة: {analysis.get('stress_pattern',} 'غير محدد')}")
    print(f"     نمط الحركات: {analysis.get('vowel_pattern',} 'غير محدد')}")
    print(f"     صعوبة النطق: {analysis.get('phonetic_difficulty', 0):.3f}")
    print(f"     جمال الصوت: {analysis.get('euphony_score', 0):.3f}")
    print(
    f"     مناسبة رياضية: {analysis.get('mathematical_appropriateness', 0):.3f}"
    )  # noqa: E501,
    def generate_comprehensive_report():  # type: ignore[no-untyped def]
    """توليد تقرير شامل"""

    print("\n\n📋 تقرير شامل لنظام توليد المفاهيم الرياضية العربية")
    print("=" * 80)

    all_results = {}

    # تنفيذ جميع الاختبارات,
    print("تنفيذ الاختبارات الشاملة...")

    all_results['numbers'] = test_detailed_number_generation()
    all_results['ordinals'] = test_ordinal_numbers()
    all_results['fractions'] = test_comprehensive_fractions()
    all_results['operations'] = test_mathematical_operations()
    all_results['concepts'] = test_mathematical_concepts()
    all_results['comprehensive'] = test_comprehensive_generation()

    # حفظ النتائج,
    report_data = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_generated': sum(
    len(results) if isinstance(results, list) else 0,
    for results in all_results.values()
    ),
    'results_by_category': all_results,
    'summary': {
    'numbers_count': len(all_results['numbers']),
    'ordinals_count': len(all_results['ordinals']),
    'fractions_count': len(all_results['fractions']),
    'operations_count': len(all_results['operations']),
    'concepts_count': len(all_results['concepts']),
    'comprehensive_count': len(all_results['comprehensive']),
    },
    }

    # حفظ التقرير,
    try:
        with open('arabic_math_generator_test_report.json', 'w', encoding='utf 8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("\n💾 تم حفظ التقرير في: arabic_math_generator_test_report.json")
    except Exception as e:
    print(f"❌ خطأ في حفظ التقرير: {e}")

    # عرض الملخص النهائي,
    print("\n🎉 ملخص النتائج النهائية:")
    print(f"   إجمالي المفاهيم المولدة: {report_data['total_generated']}")
    print(f"   الأعداد الأساسية: {report_data['summary']['numbers_count']}")
    print(f"   الأعداد الترتيبية: {report_data['summary']['ordinals_count']}")
    print(f"   الكسور: {report_data['summary']['fractions_count']}")
    print(f"   العمليات: {report_data['summary']['operations_count']}")
    print(f"   المفاهيم المتقدمة: {report_data['summary']['concepts_count']}")
    print(f"   التوليد الشامل: {report_data['summary']['comprehensive_count']}")

    print("\n✅ تم إنجاز جميع الاختبارات بنجاح!")

    assert report_data is not None,
    if __name__ == "__main__":
    print("🚀 بدء اختبار نظام توليد المفاهيم الرياضية العربية المتقدم")
    print("=" * 80)

    try:
        # تنفيذ التقرير الشامل,
    final_report = generate_comprehensive_report()

        # اختبار التحليل الصوتي,
    test_phonetic_analysis()

    print("\n🎯 تم إنجاز جميع الاختبارات بنجاح!")

    except KeyboardInterrupt:
    print("\n⏹️ تم إيقاف الاختبار بواسطة المستخدم")
    except Exception as e:
    print(f"\n❌ خطأ عام في النظام: {e}")
        import traceback  # noqa: F401,
    traceback.print_exc()
