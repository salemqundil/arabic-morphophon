#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 أداة اختبار المتجه الرقمي للكلمات العربية المفردة
=================================================

واجهة تفاعلية لاختبار نظام توليد المتجهات الرقمية
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from arabic_vector_engine import ArabicDigitalVectorGenerator
import json


def interactive_test():
    """واجهة تفاعلية لاختبار الكلمات"""

    print("🔥 مولّد المتجه الرقمي المتقدم للكلمات العربية")
    print("=" * 60)
    print("📝 أدخل كلمة عربية مفردة لتحليلها")
    print("💡 أمثلة: الكتاب، مدرسة، كُتَيْب، مُدرِّس، الشمس")
    print("⚠️  أدخل 'خروج' للانتهاء")
    print("=" * 60)

    # إنشاء المولّد
    generator = ArabicDigitalVectorGenerator()

    while True:
        try:
            # طلب إدخال الكلمة
            word = input("\n🔤 الكلمة: ").strip()

            if word.lower() in ["خروج", "exit", "quit", "q"]:
                print("👋 وداعاً!")
                break

            if not word:
                print("⚠️  يرجى إدخال كلمة صالحة")
                continue

            # اختيار السياق (اختياري)
            print("\n🎯 السياق النحوي (اختياري - اضغط Enter للتخطي):")
            print("   1. فاعل (agent)")
            print("   2. مفعول (patient)")
            print("   3. أداة (instrument)")
            print("   4. مكان (location)")
            print("   5. زمان (time)")

            context_choice = input("الاختيار: ").strip()
            context = None

            if context_choice == "1":
                context = {"semantic_role": "agent"}
            elif context_choice == "2":
                context = {"semantic_role": "patient"}
            elif context_choice == "3":
                context = {"semantic_role": "instrument"}
            elif context_choice == "4":
                context = {"semantic_role": "location"}
            elif context_choice == "5":
                context = {"semantic_role": "time"}

            print(f"\n⚡ تحليل الكلمة '{word}'...")
            print("-" * 40)

            # توليد المتجه
            result = generator.generate_vector(word, context)

            if result["processing_status"] == "success":
                # عرض النتائج
                display_analysis_result(result)
            else:
                print(f"❌ فشل التحليل: {result['error']}")

        except KeyboardInterrupt:
            print("\n👋 تم الإنهاء بواسطة المستخدم")
            break
        except Exception as e:
            print(f"❌ خطأ غير متوقع: {str(e)}")


def display_analysis_result(result):
    """عرض نتائج التحليل بشكل منظم"""

    word = result["word"]
    summary = result["linguistic_analysis"]
    vector = result["numerical_vector"]
    components = result["vector_components"]

    print(f"🎯 نتائج تحليل الكلمة: '{word}'")
    print("=" * 50)

    # الملخص اللغوي
    print("📋 الملخص اللغوي:")
    for key, value in summary.items():
        print(f"   • {key}: {value}")

    # إحصائيات المتجه
    print(f"\n🔢 إحصائيات المتجه:")
    print(f"   • إجمالي الأبعاد: {len(vector)}")
    print(f"   • أصغر قيمة: {min(vector):.3f}")
    print(f"   • أكبر قيمة: {max(vector):.3f}")
    print(f"   • المتوسط: {sum(vector)/len(vector):.3f}")

    # عينة من المتجه
    print(f"\n🎲 عينة من المتجه (أول 15 عنصر):")
    sample = [f"{x:.3f}" for x in vector[:15]]
    print(f"   {sample}")

    # الميزات المتقدمة
    print(f"\n🔬 ميزات متقدمة:")
    print(f"   • نسبة الصوامت: {components['consonant_ratio']:.3f}")
    print(f"   • نسبة التفخيم: {components['emphatic_ratio']:.3f}")
    print(f"   • التعقد الصرفي: {components['morphological_complexity']:.3f}")
    print(f"   • الملموسية: {components['concreteness']:.3f}")
    print(f"   • عدد المقاطع: {components['syllable_count']}")
    print(f"   • طول الجذر: {components['root_length']}")

    # خيارات إضافية
    print(f"\n💾 خيارات إضافية:")
    store_data_choice = input("هل تريد حفظ النتائج في ملف JSON؟ (ن/y): ").strip().lower()

    if store_data_choice in ["ن", "نعم", "y", "yes"]:
        filename = f"analysis_{word}_{result['timestamp'][:19].replace(':', '-')}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ النتائج في: {filename}")
        except Exception as e:
            print(f"❌ فشل الحفظ: {str(e)}")


def batch_test():
    """اختبار مجموعة من الكلمات"""

    print("🔄 اختبار مجموعة من الكلمات")
    print("-" * 40)

    test_words = [
        "الكتاب",
        "مدرسة",
        "كُتَيْب",
        "مُدرِّس",
        "مكتوب",
        "الشمس",
        "القمر",
        "الطالب",
        "المعلم",
        "البيت",
        "السيارة",
        "الطعام",
        "الماء",
        "النور",
        "الحب",
    ]

    generator = ArabicDigitalVectorGenerator()
    results = []

    for word in test_words:
        print(f"⚡ تحليل: {word}")
        result = generator.generate_vector(word)
        if result["processing_status"] == "success":
            results.append(
                {
                    "word": word,
                    "dimensions": len(result["numerical_vector"]),
                    "phonemes": result["vector_components"]["phoneme_count"],
                    "syllabic_units": result["vector_components"]["syllable_count"],
                    "root_length": result["vector_components"]["root_length"],
                    "definiteness": result["linguistic_analysis"]["التعريف"],
                    "gender": result["linguistic_analysis"]["الجندر"],
                }
            )

    # عرض ملخص المجموعة
    print(f"\n📊 ملخص التحليل لـ {len(results)} كلمة:")
    print("-" * 40)

    for r in results:
        print(
            f"{r['word']:>10} | {r['definiteness']:>6} | {r['gender']:>4} | فونيمات:{r['phonemes']:>2} | مقاطع:{r['syllabic_units']:>2}"
        )

    # إحصائيات عامة
    avg_phonemes = sum(r["phonemes"] for r in results) / len(results)
    avg_syllabic_units = sum(r["syllabic_units"] for r in results) / len(results)
    avg_root_length = sum(r["root_length"] for r in results) / len(results)

    print(f"\n📈 المتوسطات:")
    print(f"   • الفونيمات: {avg_phonemes:.1f}")
    print(f"   • المقاطع: {avg_syllabic_units:.1f}")
    print(f"   • طول الجذر: {avg_root_length:.1f}")


def main():
    """القائمة الرئيسية"""

    print("🌟 أداة اختبار المتجه الرقمي للكلمات العربية")
    print("=" * 50)
    print("🎯 الخيارات المتاحة:")
    print("   1. اختبار تفاعلي لكلمة واحدة")
    print("   2. اختبار مجموعة كلمات")
    print("   3. عرض الميزات المتاحة")
    print("   4. خروج")
    print("=" * 50)

    while True:
        try:
            choice = input("\n🔢 اختر رقماً: ").strip()

            if choice == "1":
                interactive_test()
            elif choice == "2":
                batch_test()
            elif choice == "3":
                show_features()
            elif choice == "4":
                print("👋 وداعاً!")
                break
            else:
                print("⚠️  اختيار غير صالح، يرجى المحاولة مرة أخرى")

        except KeyboardInterrupt:
            print("\n👋 تم الإنهاء")
            break
        except Exception as e:
            print(f"❌ خطأ: {str(e)}")


def show_features():
    """عرض الميزات المتاحة في النظام"""

    print("🔥 الميزات المُنفّذة في مولّد المتجه الرقمي")
    print("=" * 60)

    features = [
        "🎯 التعيين المعرفي (التعريف/النكرة/العلم/الضمير)",
        "📏 حالة الاسم والإعراب (مرفوع/منصوب/مجرور)",
        "🔗 قواعد إدغام اللام مع الحروف الشمسية والقمرية",
        "🏗️  حالة الإضافة النحوية",
        "⚖️  الجندر والاتفاق الصرفي (مذكر/مؤنث)",
        "🔤 التصغير (فُعَيْل، فُعَيْلَة، فُعَيْعِل)",
        "🎵 التوزيع الصوتي واللحني (النبر والعروض)",
        "🔄 التصريف الشاذ والأنماط الاستثنائية",
        "📊 التثنية والجمع كامتداد للمفرد",
        "🧠 العلاقات الدلالية والأدوار (فاعل/مفعول/أداة/مكان/زمان)",
        "🔊 التغييرات الصوتية الاستثنائية",
        "🤖 النمذجة التنبؤية والتصنيف للتعلم الآلي",
    ]

    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")

    print("\n📐 مواصفات المتجه:")
    print("   • إجمالي الأبعاد: 47 بُعد")
    print("   • الميزات الصوتية: 18 بُعد")
    print("   • الميزات الصرفية: 7 أبعاد")
    print("   • الميزات النحوية: 8 أبعاد")
    print("   • الميزات الدلالية: 7 أبعاد")
    print("   • الميزات المتقدمة: 7 أبعاد")

    print("\n✨ قابليات خاصة:")
    print("   • دعم كامل للنص العربي مع التشكيل")
    print("   • تحليل الجذور والأوزان الصرفية")
    print("   • كشف أنماط التصغير والتصريف الشاذ")
    print("   • تحليل الخصائص الصوتية والعروضية")
    print("   • ترميز رقمي متوافق مع التعلم الآلي")


if __name__ == "__main__":
    main()
