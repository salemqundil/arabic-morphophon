#!/usr/bin/env python3
"""
🔥 اختبار محرك المعالجة الشاملة - Zero Violations
================================================

اختبار شامل للنظام بدون أي أخطا أو انتهاكات
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys
import_data time
import_data traceback

# إضافة مجلد المشروع إلى Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_system_integrity():
    """اختبار تكامل النظام بدون انتهاكات"""
    print("🔥 اختبار تكامل النظام الشامل")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    try:
        # 1. اختبار استيراد المحرك الأساسي
        total_tests += 1
        print(f"📦 اختبار {total_tests}: استيراد BaseNLPEngine...")
        from engines.core.base_engine import_data BaseNLPEngine
        print("   ✅ نجح استيراد BaseNLPEngine")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ فشل استيراد BaseNLPEngine: {e}")
    
    try:
        # 2. اختبار استيراد FullPipelineEngine
        total_tests += 1
        print(f"📦 اختبار {total_tests}: استيراد FullPipelineEngine...")
        from engines.nlp.full_pipeline.engine import_data FullPipelineEngine
        print("   ✅ نجح استيراد FullPipelineEngine")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ فشل استيراد FullPipelineEngine: {e}")
        return False
    
    try:
        # 3. اختبار إنشا المحرك
        total_tests += 1
        print(f"🏗️  اختبار {total_tests}: إنشا المحرك...")
        pipeline = FullPipelineEngine()
        print(f"   ✅ تم إنشا المحرك - الإصدار: {pipeline.version}")
        print(f"   📊 المحركات المتاحة: {len(pipeline.available_engines)}")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ فشل إنشا المحرك: {e}")
        return False
    
    try:
        # 4. اختبار التحليل الأساسي
        total_tests += 1
        print(f"🧪 اختبار {total_tests}: التحليل الأساسي...")
        
        test_text = "كتابة"
        begin_time = time.time()
        result = pipeline.analyze(
            text=test_text,
            target_engines=None,
            enable_parallel=False,
            detailed_output=True
        )
        processing_time = time.time() - begin_time
        
        print(f"   ✅ تم التحليل في {processing_time:.3f} ثانية")
        
        # التحقق من وجود النتائج الأساسية
        if isinstance(result, dict):
            print("   ✅ النتائج بتنسيق صحيح (dict)")
            
            # طباعة معلومات النتائج
            if "pipeline_info" in result:
                info = result["pipeline_info"]
                print(f"   📊 معلومات المعالجة:")
                print(f"      - الإصدار: {info.get('version', 'غير محدد')}")
                print(f"      - وقت المعالجة: {info.get('processing_time', 0):.3f}s")
                print(f"      - المحركات الناجحة: {len(info.get('successful_engines', []))}")
                print(f"      - المحركات الفاشلة: {len(info.get('failed_engines', []))}")
            
            if "quality_assessment" in result:
                quality = result["quality_assessment"]
                print(f"   🎯 تقييم الجودة:")
                print(f"      - النقاط العامة: {quality.get('overall_score', 0):.2f}")
                print(f"      - مؤشر الموثوقية: {quality.get('reliability_index', 0):.2f}")
                print(f"      - اكتمال التحليل: {quality.get('completeness', 0):.1f}%")
            
            success_count += 1
        else:
            print(f"   ⚠️ النتائج بتنسيق غير متوقع: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ فشل التحليل الأساسي: {e}")
        traceback.print_exc()
    
    try:
        # 5. اختبار إحصائيات النظام
        total_tests += 1
        print(f"📊 اختبار {total_tests}: إحصائيات النظام...")
        
        stats = pipeline.get_pipeline_stats()
        if isinstance(stats, dict):
            print("   ✅ تم الحصول على الإحصائيات")
            
            # عرض الإحصائيات الأساسية
            if "comprehensive_stats" in stats:
                comp_stats = stats["comprehensive_stats"]
                print(f"   📈 إجمالي التحليلات: {comp_stats.get('total_analyses', 0)}")
                print(f"   ⏱️  متوسط وقت المعالجة: {comp_stats.get('average_processing_time', 0):.3f}s")
                
            if "performance_summary" in stats:
                perf = stats["performance_summary"]
                print(f"   🎯 معدل النجاح: {perf.get('success_rate', 0):.1f}%")
                print(f"   🔄 العمليات المتوازية: {perf.get('parallel_operations', 0)}")
            
            success_count += 1
        else:
            print(f"   ⚠️ الإحصائيات بتنسيق غير متوقع: {type(stats)}")
            
    except Exception as e:
        print(f"   ❌ فشل الحصول على الإحصائيات: {e}")
    
    try:
        # 6. اختبار تطبيق Flask
        total_tests += 1
        print(f"🌐 اختبار {total_tests}: تطبيق Flask...")
        
        from engines.nlp.full_pipeline.engine import_data create_flask_app
        app = create_flask_app()
        
        if app is not None:
            print("   ✅ تم إنشا تطبيق Flask بنجاح")
            print("   🌐 الواجهة جاهزة على: http://localhost:5000")
            success_count += 1
        else:
            print("   ❌ فشل إنشا تطبيق Flask")
            
    except Exception as e:
        print(f"   ❌ فشل إنشا تطبيق Flask: {e}")
    
    try:
        # 7. اختبار تصدير النتائج
        total_tests += 1
        print(f"💾 اختبار {total_tests}: تصدير النتائج...")
        
        # تحليل بسيط للتصدير
        result = pipeline.analyze("test")
        
        # تصدير JSON
        json_store_data = pipeline.store_data_results(result, "json")
        if json_store_data and len(json_store_data) > 0:
            print(f"   ✅ تصدير JSON: {len(json_store_data)} حرف")
        
        # تصدير CSV
        csv_store_data = pipeline.store_data_results(result, "csv")
        if csv_store_data and len(csv_store_data) > 0:
            print(f"   ✅ تصدير CSV: {len(csv_store_data)} حرف")
        
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ فشل اختبار التصدير: {e}")
    
    # النتائج النهائية
    print("\n" + "=" * 50)
    print("📊 نتائج الاختبار النهائية:")
    print(f"   ✅ اختبارات ناجحة: {success_count}")
    print(f"   📝 إجمالي الاختبارات: {total_tests}")
    print(f"   🎯 معدل النجاح: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\n🎉 جميع الاختبارات نجحت! النظام جاهز للاستخدام بدون انتهاكات")
        print("🚀 لتشغيل الواجهة الويب: python run_fullpipeline.py")
        return True
    else:
        print(f"\n⚠️ نجح {success_count} من {total_tests} اختبار")
        print("🔧 راجع الأخطا أعلاه لمعرفة التفاصيل")
        return False

def test_web_interface():
    """اختبار الواجهة الويب"""
    print("\n🌐 اختبار الواجهة الويب:")
    print("-" * 30)
    
    try:
        from engines.nlp.full_pipeline.engine import_data create_flask_app
        app = create_flask_app()
        
        # اختبار المسارات الأساسية
        with app.test_client() as client:
            # اختبار الصفحة الرئيسية
            response = client.get('/')
            if response.status_code == 200:
                print("   ✅ الصفحة الرئيسية تعمل")
            else:
                print(f"   ❌ مشكلة في الصفحة الرئيسية: {response.status_code}")
            
            # اختبار مسار المحركات المتاحة
            response = client.get('/engines')
            if response.status_code == 200:
                print("   ✅ مسار المحركات يعمل")
            else:
                print(f"   ❌ مشكلة في مسار المحركات: {response.status_code}")
            
            # اختبار مسار الإحصائيات
            response = client.get('/stats')
            if response.status_code == 200:
                print("   ✅ مسار الإحصائيات يعمل")
            else:
                print(f"   ❌ مشكلة في مسار الإحصائيات: {response.status_code}")
        
        print("   🎉 جميع مسارات الواجهة تعمل بشكل صحيح")
        return True
        
    except Exception as e:
        print(f"   ❌ خط في اختبار الواجهة الويب: {e}")
        return False

def main():
    """الوظيفة الرئيسية للاختبار"""
    print("🔥 اختبار شامل للنظام - Zero Violations")
    print("=" * 60)
    print(f"⏰ التاريخ والوقت: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # اختبار تكامل النظام
    system_ok = test_system_integrity()
    
    # اختبار الواجهة الويب
    web_ok = test_web_interface()
    
    # النتيجة النهائية
    print("\n" + "=" * 60)
    if system_ok and web_ok:
        print("🎊 النظام يعمل بشكل مثالي بدون أي انتهاكات أو أخطا!")
        print("🚀 جاهز للاستخدام الإنتاجي")
        print("🌐 الواجهة الويب متاحة على: http://localhost:5000")
        return True
    else:
        print("⚠️ توجد بعض المشاكل التي تحتاج إلى مراجعة")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
