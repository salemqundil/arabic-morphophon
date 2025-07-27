#!/usr/bin/env python3
"""
🚀 اختبار ديناميكي شامل - Full Dynamic Functionality
================================================

اختبار النظام الكامل بشكل ديناميكي ومتطور
No Errors, No Violations - Expert Developer Level
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys
import_data time
import_data json
import_data requests
from datetime import_data datetime
from typing import_data Dict, Any, List

# إضافة مجلد المشروع إلى Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_header(title: str, char: str = "=", width: int = 70):
    """طباعة عنوان جميل"""
    print(f"\n{char * width}")
    print(f"🔥 {title}")
    print(f"{char * width}")

def print_success(message: str):
    """طباعة رسالة نجاح"""
    print(f"✅ {message}")

def print_error(message: str):
    """طباعة رسالة خطأ"""
    print(f"❌ {message}")

def print_info(message: str):
    """طباعة رسالة معلومات"""
    print(f"ℹ️  {message}")

def print_step(step: int, message: str):
    """طباعة خطوة"""
    print(f"🔸 الخطوة {step}: {message}")

class DynamicSystemTester:
    """فئة اختبار النظام الديناميكي"""
    
    def __init__(self):
        self.test_results = []
        self.begin_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
        
    def add_test_result(self, test_name: str, success: bool, details: str = ""):
        """إضافة نتيجة اختبار"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_core_import_datas(self) -> bool:
        """اختبار الاستيرادات الأساسية"""
        print_step(1, "اختبار الاستيرادات الأساسية")
        
        try:
            from engines.core.base_engine import_data BaseNLPEngine
            print_success("BaseNLPEngine تم استيراده بنجاح")
            self.add_test_result("BaseNLPEngine Import", True)
        except Exception as e:
            print_error(f"فشل استيراد BaseNLPEngine: {e}")
            self.add_test_result("BaseNLPEngine Import", False, str(e))
            return False
        
        try:
            from engines.nlp.full_pipeline.engine import_data FullPipelineEngine
            print_success("FullPipelineEngine تم استيراده بنجاح")
            self.add_test_result("FullPipelineEngine Import", True)
        except Exception as e:
            print_error(f"فشل استيراد FullPipelineEngine: {e}")
            self.add_test_result("FullPipelineEngine Import", False, str(e))
            return False
        
        return True
    
    def test_engine_creation(self) -> Dict[str, Any]:
        """اختبار إنشاء المحرك"""
        print_step(2, "اختبار إنشاء المحرك الشامل")
        
        try:
            from engines.nlp.full_pipeline.engine import_data FullPipelineEngine
            
            pipeline = FullPipelineEngine()
            
            # التحقق من الخصائص الأساسية
            engine_info = {
                "version": getattr(pipeline, 'version', 'غير محدد'),
                "engine_count": pipeline.engine_count,
                "available_engines": pipeline.available_engines,
                "created_at": datetime.now().isoformat()
            }
            
            print_success(f"تم إنشاء المحرك - الإصدار: {engine_info['version']}")
            print_info(f"عدد المحركات المتاحة: {engine_info['engine_count']}")
            print_info(f"المحركات: {', '.join(engine_info['available_engines']) if engine_info['available_engines'] else 'لا توجد محركات'}")
            
            self.add_test_result("Engine Creation", True, json.dumps(engine_info, ensure_ascii=False))
            return {"success": True, "pipeline": pipeline, "info": engine_info}
            
        except Exception as e:
            print_error(f"فشل إنشاء المحرك: {e}")
            self.add_test_result("Engine Creation", False, str(e))
            return {"success": False, "error": str(e)}
    
    def test_basic_analysis(self, pipeline) -> bool:
        """اختبار التحليل الأساسي"""
        print_step(3, "اختبار التحليل الأساسي")
        
        test_texts = [
            "كتابة",
            "المكتوب", 
            "يكتب",
            "كاتب",
            "مكتب"
        ]
        
        all_success = True
        
        for i, text in enumerate(test_texts, 1):
            try:
                begin_time = time.time()
                result = pipeline.analyze(
                    text=text,
                    target_engines=None,
                    enable_parallel=False,
                    detailed_output=True
                )
                processing_time = time.time() - begin_time
                
                # التحقق من النتائج
                if isinstance(result, dict):
                    print_success(f"النص '{text}' - تم التحليل في {processing_time:.3f}s")
                    
                    # عرض معلومات مختصرة
                    if "pipeline_info" in result:
                        info = result["pipeline_info"]
                        print_info(f"   المحركات الناجحة: {len(info.get('successful_engines', []))}")
                        print_info(f"   المحركات الفاشلة: {len(info.get('failed_engines', []))}")
                    
                    self.add_test_result(f"Analysis: {text}", True, f"Time: {processing_time:.3f}s")
                else:
                    print_error(f"النص '{text}' - نتائج غير صحيحة")
                    self.add_test_result(f"Analysis: {text}", False, "Invalid result format")
                    all_success = False
                    
            except Exception as e:
                print_error(f"النص '{text}' - خطأ في التحليل: {e}")
                self.add_test_result(f"Analysis: {text}", False, str(e))
                all_success = False
        
        return all_success
    
    def test_pipeline_stats(self, pipeline) -> bool:
        """اختبار إحصائيات المحرك"""
        print_step(4, "اختبار إحصائيات النظام")
        
        try:
            stats = pipeline.get_pipeline_stats()
            
            if isinstance(stats, dict):
                print_success("تم الحصول على الإحصائيات")
                
                # عرض الإحصائيات
                if "comprehensive_stats" in stats:
                    comp_stats = stats["comprehensive_stats"]
                    print_info(f"إجمالي التحليلات: {comp_stats.get('total_analyses', 0)}")
                    print_info(f"متوسط وقت المعالجة: {comp_stats.get('average_processing_time', 0):.3f}s")
                
                if "performance_summary" in stats:
                    perf = stats["performance_summary"]
                    print_info(f"معدل النجاح: {perf.get('success_rate', 0):.1f}%")
                
                self.add_test_result("Pipeline Stats", True, json.dumps(stats, ensure_ascii=False, default=str))
                return True
            else:
                print_error("إحصائيات بتنسيق غير صحيح")
                self.add_test_result("Pipeline Stats", False, "Invalid stats format")
                return False
                
        except Exception as e:
            print_error(f"فشل الحصول على الإحصائيات: {e}")
            self.add_test_result("Pipeline Stats", False, str(e))
            return False
    
    def test_flask_app(self) -> bool:
        """اختبار تطبيق Flask"""
        print_step(5, "اختبار تطبيق Flask")
        
        try:
            from engines.nlp.full_pipeline.engine import_data create_flask_app
            
            app = create_flask_app()
            
            if app is not None:
                print_success("تم إنشاء تطبيق Flask")
                
                # اختبار المسارات الأساسية
                with app.test_client() as client:
                    routes_tested = []
                    
                    # اختبار الصفحة الرئيسية
                    response = client.get('/')
                    if response.status_code == 200:
                        print_success("الصفحة الرئيسية تعمل")
                        routes_tested.append("home")
                    
                    # اختبار مسار المحركات
                    response = client.get('/engines')
                    if response.status_code == 200:
                        print_success("مسار المحركات يعمل")
                        routes_tested.append("engines")
                    
                    # اختبار مسار الإحصائيات
                    response = client.get('/stats')
                    if response.status_code == 200:
                        print_success("مسار الإحصائيات يعمل")
                        routes_tested.append("stats")
                
                self.add_test_result("Flask App", True, f"Routes tested: {', '.join(routes_tested)}")
                return True
            else:
                print_error("فشل إنشاء تطبيق Flask")
                self.add_test_result("Flask App", False, "App creation failed")
                return False
                
        except Exception as e:
            print_error(f"خطأ في تطبيق Flask: {e}")
            self.add_test_result("Flask App", False, str(e))
            return False
    
    def test_advanced_features(self, pipeline) -> bool:
        """اختبار المميزات المتقدمة"""
        print_step(6, "اختبار المميزات المتقدمة")
        
        try:
            # اختبار النص المعقد
            complex_text = "الكتابة الجميلة والأوزان الصرفية المتنوعة تظهر ثراء اللغة العربية"
            
            result = pipeline.analyze(
                text=complex_text,
                enable_parallel=True,
                detailed_output=True
            )
            
            if isinstance(result, dict):
                print_success("تم تحليل النص المعقد")
                
                # اختبار تصدير النتائج
                json_store_data = pipeline.store_data_results(result, "json")
                csv_store_data = pipeline.store_data_results(result, "csv")
                
                if json_store_data:
                    print_success(f"تصدير JSON: {len(json_store_data)} حرف")
                
                if csv_store_data:
                    print_success(f"تصدير CSV: {len(csv_store_data)} حرف")
                
                self.add_test_result("Advanced Features", True, "Complex text analysis and store_data")
                return True
            else:
                print_error("فشل تحليل النص المعقد")
                self.add_test_result("Advanced Features", False, "Complex text analysis failed")
                return False
                
        except Exception as e:
            print_error(f"خطأ في المميزات المتقدمة: {e}")
            self.add_test_result("Advanced Features", False, str(e))
            return False
    
    def generate_final_report(self):
        """إنشاء التقرير النهائي"""
        total_time = time.time() - self.begin_time
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print_header("التقرير النهائي", "=")
        
        print(f"⏱️  إجمالي وقت الاختبار: {total_time:.2f} ثانية")
        print(f"📊 إجمالي الاختبارات: {self.total_tests}")
        print(f"✅ الاختبارات الناجحة: {self.passed_tests}")
        print(f"❌ الاختبارات الفاشلة: {self.total_tests - self.passed_tests}")
        print(f"🎯 معدل النجاح: {success_rate:.1f}%")
        
        if success_rate == 100:
            print_header("🎊 النظام يعمل بشكل مثالي - Zero Violations!", "*")
            print("🚀 جاهز للاستخدام الإنتاجي")
            print("🌐 الواجهة الويب متاحة على: http://localhost:5000")
            print("💎 Expert Developer Level - Full Dynamic Functionality Achieved!")
        elif success_rate >= 80:
            print_header("✅ النظام يعمل بشكل جيد مع بعض التحسينات", "*")
        else:
            print_header("⚠️ النظام يحتاج إلى مراجعة", "*")
        
        return success_rate

def main():
    """الوظيفة الرئيسية"""
    print_header("🚀 اختبار ديناميكي شامل - Full Dynamic Functionality")
    print(f"⏰ بدء الاختبار: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💎 Expert Developer Level - No Errors, No Violations")
    
    tester = DynamicSystemTester()
    
    # 1. اختبار الاستيرادات
    if not tester.test_core_import_datas():
        print_error("فشل في الاستيرادات الأساسية")
        return False
    
    # 2. اختبار إنشاء المحرك
    engine_result = tester.test_engine_creation()
    if not engine_result["success"]:
        print_error("فشل في إنشاء المحرك")
        return False
    
    pipeline = engine_result["pipeline"]
    
    # 3. اختبار التحليل الأساسي
    tester.test_basic_analysis(pipeline)
    
    # 4. اختبار الإحصائيات
    tester.test_pipeline_stats(pipeline)
    
    # 5. اختبار Flask
    tester.test_flask_app()
    
    # 6. اختبار المميزات المتقدمة
    tester.test_advanced_features(pipeline)
    
    # 7. التقرير النهائي
    success_rate = tester.generate_final_report()
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 انتهى الاختبار بنتيجة: {'نجاح' if success else 'فشل'}")
    sys.exit(0 if success else 1)
