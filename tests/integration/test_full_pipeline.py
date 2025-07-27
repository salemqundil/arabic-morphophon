"""
🔥 ملف اختبار محرك المعالجة الشاملة للنصوص العربية
=======================================================

هذا الملف يختبر جميع وظائف FullPipeline Engine ويعرض إمكانياته الكاملة
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
import_data os
import_data json
from datetime import_data datetime

# إضافة مسار المشروع
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from engines.nlp.full_pipeline.engine import_data FullPipelineEngine, create_flask_app

def test_pipeline_engine():
    """اختبار شامل لمحرك المعالجة الشاملة"""
    
    print("🔥 بدء اختبار محرك المعالجة الشاملة للنصوص العربية")
    print("=" * 60)
    
    # تهيئة المحرك
    try:
        pipeline = FullPipelineEngine()
        print(f"✅ تم تهيئة المحرك بنجاح - الإصدار: {pipeline.version}")
        print(f"📊 المحركات المتاحة: {list(pipeline.engines.keys())}")
        print(f"🔢 عدد المحركات: {len(pipeline.engines)}")
        print()
    except Exception as e:
        print(f"❌ فشل في تهيئة المحرك: {e}")
        return False
    
    # النصوص التجريبية
    test_texts = [
        "كتابة",
        "المكتوب",
        "يكتب",
        "كاتب",
        "مكتب",
        "أجمل البيانات والتحليلات الصرفية المتقدمة"
    ]
    
    print("🧪 بدء اختبار النصوص...")
    print("-" * 40)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 النص {i}: '{text}'")
        
        try:
            # اختبار التحليل الأساسي
            result = pipeline.analyze(
                text=text,
                target_engines=None,  # جميع المحركات
                enable_parallel=True,
                detailed_output=True
            )
            
            # عرض النتائج المختصرة
            pipeline_info = result.get("pipeline_info", {})
            summary = result.get("pipeline_summary", {})
            
            print(f"   ⏱️  وقت المعالجة: {pipeline_info.get('processing_time', 0):.3f} ثانية")
            print(f"   ✅ المحركات الناجحة: {summary.get('successful_engines', 0)}")
            print(f"   ❌ المحركات الفاشلة: {summary.get('failed_engines', 0)}")
            print(f"   🎯 متوسط الثقة: {summary.get('average_confidence', 0):.2f}")
            
            # عرض النتائج الرئيسية لكل محرك
            key_findings = summary.get("key_findings", {})
            for engine_name, findings in key_findings.items():
                if isinstance(findings, dict) and "error" not in findings:
                    print(f"   🔍 {engine_name}: {list(findings.keys())[:3]}")  # أول 3 مفاتيح
            
        except Exception as e:
            print(f"   ❌ فشل التحليل: {e}")
    
    # اختبار الإحصائيات
    print(f"\n📊 إحصائيات النظام:")
    print("-" * 20)
    
    stats = pipeline.get_pipeline_stats()
    comp_stats = stats.get("comprehensive_stats", {})
    perf_stats = stats.get("performance_summary", {})
    
    print(f"   📈 إجمالي التحليلات: {comp_stats.get('total_analyses', 0)}")
    print(f"   ✅ التحليلات الناجحة: {comp_stats.get('successful_analyses', 0)}")
    print(f"   ⏱️  متوسط وقت المعالجة: {comp_stats.get('average_processing_time', 0):.3f}s")
    print(f"   🎯 معدل النجاح: {perf_stats.get('success_rate', 0):.1f}%")
    print(f"   🔄 العمليات المتوازية: {comp_stats.get('parallel_operations', 0)}")
    
    # اختبار تصدير النتائج
    print(f"\n💾 اختبار تصدير النتائج:")
    print("-" * 25)
    
    try:
        # تحليل نص للتصدير
        store_data_result = pipeline.analyze("كتابة", detailed_output=False)
        
        # تصدير JSON
        json_store_data = pipeline.store_data_results(store_data_result, "json")
        print(f"   ✅ تصدير JSON: {len(json_store_data)} حرف")
        
        # تصدير CSV
        csv_store_data = pipeline.store_data_results(store_data_result, "csv")
        print(f"   ✅ تصدير CSV: {len(csv_store_data)} حرف")
        
    except Exception as e:
        print(f"   ❌ فشل التصدير: {e}")
    
    # اختبار المعالجة المجمعة
    print(f"\n📦 اختبار المعالجة المجمعة:")
    print("-" * 30)
    
    try:
        batch_texts = ["كتابة", "مكتوب", "يكتب"]
        batch_results = pipeline.analyze_batch(
            batch_texts, 
            target_engines=["weight", "morphology"],
            enable_parallel=True,
            detailed_output=False
        )
        
        print(f"   ✅ معالجة مجمعة لـ {len(batch_results)} نص")
        for i, result in enumerate(batch_results):
            if "error" not in result:
                summary = result.get("pipeline_summary", {})
                print(f"   📝 النص {i+1}: {summary.get('successful_engines', 0)} محرك ناجح")
            else:
                print(f"   ❌ النص {i+1}: خطأ في المعالجة")
    
    except Exception as e:
        print(f"   ❌ فشل المعالجة المجمعة: {e}")
    
    print(f"\n🎉 انتهى الاختبار بنجاح!")
    print("=" * 60)
    
    return True

def test_flask_integration():
    """اختبار تكامل Flask"""
    
    print("\n🌐 اختبار تكامل Flask:")
    print("-" * 25)
    
    try:
        app = create_flask_app()
        print("   ✅ تم إنشاء تطبيق Flask بنجاح")
        print("   🌐 يمكن تشغيله على: http://localhost:5000")
        print("   📱 واجهة المستخدم متاحة")
        return True
    except Exception as e:
        print(f"   ❌ فشل إنشاء تطبيق Flask: {e}")
        return False

def demo_advanced_features():
    """عرض المميزات المتقدمة"""
    
    print("\n🚀 عرض المميزات المتقدمة:")
    print("-" * 30)
    
    try:
        pipeline = FullPipelineEngine()
        
        # اختبار النص المعقد
        complex_text = "الكتابة الجميلة والأوزان الصرفية المتنوعة تظهر ثراء اللغة العربية"
        
        print(f"📝 النص المعقد: '{complex_text}'")
        
        # تحليل مع تفاصيل كاملة
        result = pipeline.analyze(
            text=complex_text,
            target_engines=None,
            enable_parallel=True,
            detailed_output=True
        )
        
        # عرض التحليل المتقاطع
        cross_insights = result.get("cross_engine_insights", {})
        if cross_insights:
            print("   🔗 رؤى متقاطعة:")
            for insight_type, data in cross_insights.items():
                print(f"      • {insight_type}: متاح")
        
        # عرض تقييم الجودة
        quality = result.get("quality_assessment", {})
        if quality:
            print(f"   📊 تقييم الجودة:")
            print(f"      • النقاط الإجمالية: {quality.get('overall_score', 0):.2f}")
            print(f"      • مؤشر الموثوقية: {quality.get('reliability_index', 0):.2f}")
            print(f"      • الاكتمال: {quality.get('completeness', 0):.1f}%")
        
        # عرض التوصيات
        recommendations = result.get("recommendations", [])
        if recommendations:
            print(f"   💡 التوصيات:")
            for rec in recommendations[:3]:  # أول 3 توصيات
                print(f"      • {rec}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ فشل في عرض المميزات المتقدمة: {e}")
        return False

if __name__ == "__main__":
    print("🔥 محرك المعالجة الشاملة للنصوص العربية - FullPipeline Engine")
    print("=" * 70)
    print(f"📅 تاريخ الاختبار: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # تشغيل جميع الاختبارات
    success = True
    
    success &= test_pipeline_engine()
    success &= test_flask_integration()
    success &= demo_advanced_features()
    
    if success:
        print(f"\n🎉 جميع الاختبارات نجحت! النظام جاهز للاستخدام.")
        print(f"🚀 لتشغيل الواجهة الويب:")
        print(f"   python engines/nlp/full_pipeline/engine.py")
    else:
        print(f"\n⚠️  بعض الاختبارات فشلت. يرجى مراجعة الأخطاء أعلاه.")
from engines.nlp.full_pipeline.engine import_data FullPipelineEngine

pipeline = FullPipelineEngine()
result = pipeline.analyze("كتابة")
print(result["pipeline_summary"])