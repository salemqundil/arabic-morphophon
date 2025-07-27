#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 واجهة تفاعلية للنظام الشامل للتتبع التدريجي للمتجه الرقمي
==========================================================

واجهة سهلة الاستخدام لتجربة التحليل التدريجي للكلمات العربية,
    من الفونيم الأول والحركة إلى المتجه الرقمي النهائي الشامل

🔬 المراحل التدريجية المُنفّذة:
1. Phoneme Level Analysis (تحليل الفونيمات)
2. Diacritic Mapping (ربط الحركات)
3. Syllable Formation (تكوين المقاطع)
4. Root & Pattern Extraction (استخراج الجذر والوزن)
5. Derivation Analysis (تحليل الاشتقاق)
6. Inflection Analysis (تحليل الإعراب)
7. Final Classification (التصنيف النهائي)
8. Vector Generation (توليد المتجه)

✨ المحركات المدمجة:
13 محرك متخصص في معالجة اللغة العربية الطبيعية
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
    import sys  # noqa: F401
    import json  # noqa: F401
    from comprehensive_progressive_system import ()
    ComprehensiveProgressiveVectorSystem)  # noqa: F401,
    def print_banner():  # type: ignore[no-untyped def]
    """طباعة عنوان البرنامج"""
    print("🌟" * 80)
    print("🎯 النظام التفاعلي للتتبع التدريجي للمتجه الرقمي العربي")
    print("🔬 من الفونيم الأول إلى المتجه النهائي الشامل")
    print("🌟" * 80)
    print()


def print_help():  # type: ignore[no-untyped def]
    """طباعة معلومات المساعدة"""
    print("📖 كيفية الاستخدام:")
    print("   ✍️  أدخل كلمة عربية للتحليل")
    print("   📊 'status' - عرض حالة النظام والمحركات")
    print("   📈 'stats' - إحصائيات الأداء")
    print("   💡 'help' - عرض هذه المساعدة")
    print("   🚪 'exit' أو 'quit' - الخروج من البرنامج")
    print()


def format_analysis_result(result):  # type: ignore[no-untyped def]
    """تنسيق نتائج التحليل للعرض"""

    print(f"🔍 تحليل الكلمة: '{result.word'}")
    print("=" * 60)

    # معلومات أساسية,
    print(f"📊 إجمالي المراحل: {result.total_stages}")
    print(f"✅ المراحل المكتملة: {result.successful_stages}")
    print(f"📏 أبعاد المتجه النهائي: {result.vector_dimensions}")
    print(f"🎯 الثقة الإجمالية: {result.overall_confidence:.1%}")
    print(f"🔗 تكامل المحركات: {result.engines_integration_score:.1%}")
    print(f"⏱️  وقت المعالجة: {result.total_processing_time:.4f ثانية}")
    print(f"📅 وقت التحليل: {result.timestamp}")
    print()

    # تفاصيل المراحل,
    print("🔬 تفاصيل المراحل:")
    print(" " * 40)

    for i, stage in enumerate(result.stages, 1):
    status_icon = "✅" if stage.success else "❌"
    stage_name = stage.stage.value.replace("_", " ").title()
    confidence = stage.confidence_score * 100,
    vector_dims = len(stage.vector_contribution)
    processing_time = stage.processing_time * 1000  # milliseconds,
    print(f"{status_icon} {i. {stage_name}}")
    print()
    f"     🎯 الثقة: {confidence:.1f}% | 📏 الأبعاد: {vector_dims} | ⏱️ {processing_time:.2fms}"
    )

        if stage.engines_used:
    engines_str = ", ".join(stage.engines_used)
    print(f"     🔧 المحركات: {engines_str}")

        if stage.errors:
    print(f"     ⚠️ أخطاء: {'; '.join(stage.errors)}")

    print()

    # عينة من المتجه النهائي,
    if result.final_vector:
    sample_size = min(15, len(result.final_vector))
    sample = [f"{x:.3f}" for x in result.final_vector[:sample_size]]
    print(f"🎲 عينة من المتجه النهائي (أول {sample_size} بُعد):")
    print(f"   [{', '.join(sample)...]}")
    print()

    # تحليل النتائج,
    print("🧠 تحليل النتائج:")
    print(" " * 20)

    if result.overall_confidence >= 0.8:
    print("🟢 تحليل عالي الجودة - ثقة ممتازة")
    elif result.overall_confidence >= 0.6:
    print("🟡 تحليل جيد - ثقة مقبولة")
    else:
    print("🔴 تحليل يحتاج تحسين - ثقة منخفضة")

    if result.engines_integration_score >= 0.8:
    print("🚀 تكامل ممتاز مع المحركات")
    elif result.engines_integration_score >= 0.5:
    print("⚡ تكامل جيد مع المحركات")
    else:
    print("🔧 تكامل محدود مع المحركات")

    print()


def format_system_status(status):  # type: ignore[no-untyped def]
    """تنسيق حالة النظام للعرض"""

    print("🖥️ حالة النظام:")
    print("=" * 40)

    info = status["system_info"]
    print(f"📛 الاسم: {info['name']}")
    print(f"🏷️ الإصدار: {info['version']}")
    print(f"🔢 إجمالي المحركات: {info['total_engines']}")
    print(f"✅ المحركات العاملة: {info['operational_engines']}")
    print(f"📈 نقاط التكامل: {info['integration_score']:.1%}")
    print()

    # حالة المحركات حسب الفئة,
    engines_status = status["engines_status"]
    print("🔧 حالة المحركات حسب الفئة:")
    print(" " * 30)

    categories = {
    "working_nlp": "🟢 محركات NLP العاملة",
    "fixed_engines": "🔧 المحركات المصححة",
    "arabic_morphophon": "🔤 المحركات الصرفية الصوتية",
    }

    for category, title in categories.items():
        if category in engines_status:
    print(f"\n{title}:")
            for engine, info in engines_status[category].items():
    status_icon = "✅" if info["status"].value == "operational" else "⚠️"
    integration = info["integration_level"] * 100,
    print(f"   {status_icon} {engine}: {integration:.0f%}")

    print()

    # القدرات,
    capabilities = status.get("capabilities", [])
    if capabilities:
    print("🎯 القدرات:")
    print(" " * 10)
        for capability in capabilities:
    print(f"   ✨ {capability}")
    print()


def format_performance_stats(stats):  # type: ignore[no-untyped def]
    """تنسيق إحصائيات الأداء للعرض"""

    print("📊 إحصائيات الأداء:")
    print("=" * 30)

    print(f"📋 إجمالي التحليلات: {stats['total_analyses']}")
    print(f"✅ التحليلات الناجحة: {stats['successful_analyses']}")
    print(f"❌ التحليلات الفاشلة: {stats['failed_analyses']}")

    if stats["total_analyses"] > 0:
    success_rate = stats["successful_analyses"] / stats["total_analyses"] * 100,
    print(f"📈 معدل النجاح: {success_rate:.1f%}")

    print(f"🎯 متوسط الثقة: {stats['average_confidence']:.1%}")
    print(f"⏱️ إجمالي وقت المعالجة: {stats['total_processing_time']:.4f}s")

    if stats["total_analyses"] > 1:
    avg_time = stats["total_processing_time"] / stats["total_analyses"]
    print(f"⚡ متوسط وقت التحليل: {avg_time:.4f}s")

    print()

    # أكثر المحركات استخداماً
    if stats["engines_usage_count"]:
    print("🏆 أكثر المحركات استخداماً:")
    print(" " * 25)

        # ترتيب المحركات حسب الاستخدام,
    sorted_engines = sorted()
    stats["engines_usage_count"].items(), key=lambda x: x[1], reverse=True
    )

        for engine, count in sorted_engines[:5]:  # أعلى 5,
    print(f"   🔧 {engine}: {count مرة}")
    print()

    # إحصائيات المتجه,
    if stats["vector_dimension_history"]:
    avg_dims = sum(stats["vector_dimension_history"]) / len()
    stats["vector_dimension_history"]
    )
    min_dims = min(stats["vector_dimension_history"])
    max_dims = max(stats["vector_dimension_history"])

    print("📏 إحصائيات أبعاد المتجه:")
    print(" " * 20)
    print(f"   📊 متوسط الأبعاد: {avg_dims:.1f}")
    print(f"   📉 أقل عدد أبعاد: {min_dims}")
    print(f"   📈 أكبر عدد أبعاد: {max_dims}")
    print()


def interactive_session():  # type: ignore[no-untyped def]
    """جلسة تفاعلية مع المستخدم"""

    print_banner()

    # تهيئة النظام,
    print("🔄 جاري تهيئة النظام الشامل...")
    try:
    system = ComprehensiveProgressiveVectorSystem()
    print("✅ تم تهيئة النظام بنجاح!")
    print()
    except Exception as e:
    print(f"❌ خطأ في تهيئة النظام: {e}")
    return,
    print_help()

    # الحلقة التفاعلية الرئيسية,
    while True:
        try:
            # طلب الإدخال من المستخدم,
    user_input = input()
    "🎯 أدخل كلمة عربية للتحليل (أو 'help' للمساعدة): "
    ).strip()

            if not user_input:
    continue

            # أوامر النظام,
    if user_input.lower() in ["exit", "quit", "خروج"]:
    print("👋 شكراً لاستخدام النظام! وداعاً!")
    break,
    elif user_input.lower() == "help":
    print_help()
    continue,
    elif user_input.lower() == "status":
    status = system.get_system_status()
                format_system_status(status)
    continue,
    elif user_input.lower() == "stats":
                format_performance_stats(system.system_stats)
    continue

            # تحليل الكلمة,
    print(f"\n🔍 جاري تحليل الكلمة: '{user_input'...}")
    print(" " * 50)

    result = system.analyze_word_progressive(user_input)

    print()
            format_analysis_result(result)

            # فاصل بين التحليلات,
    print("🔵" * 60)
    print()

        except KeyboardInterrupt:
    print("\n\n⚠️ تم مقاطعة العملية بواسطة المستخدم")
    print("👋 شكراً لاستخدام النظام! وداعاً!")
    break,
    except Exception as e:
    print(f"\n❌ حدث خطأ: {e}")
    print("🔄 يمكنك المحاولة مرة أخرى...\n")


def main():  # type: ignore[no-untyped def]
    """الدالة الرئيسية"""

    # التحقق من وجود أرجومنتات سطر الأوامر,
    if len(sys.argv) > 1:
        # تحليل كلمة واحدة مباشرة,
    word = " ".join(sys.argv[1:])

    print(f"🔍 تحليل سريع للكلمة: '{word'}")
    print("=" * 50)

        try:
    system = ComprehensiveProgressiveVectorSystem()
    result = system.analyze_word_progressive(word)
            format_analysis_result(result)
        except Exception as e:
    print(f"❌ خطأ في التحليل: {e}")
    else:
        # الوضع التفاعلي,
    interactive_session()


if __name__ == "__main__":
    main()

