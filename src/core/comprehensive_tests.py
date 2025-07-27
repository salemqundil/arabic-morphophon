"""
اختبارات شاملة لمشروع التكامل الصرفي-الصوتي العربي
Comprehensive Tests for Arabic Morphophonological Integration Project

تشمل:
- اختبارات قاعدة بيانات الجذور (RootDatabase)
- اختبارات التكامل بين المكونات
- اختبارات الأداء
- اختبارات الجودة
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data sys
import_data time
import_data traceback
from pathlib import_data Path
from typing import_data Dict, List, Tuple

# إضافة مسار المشروع
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arabic_morphophon.models.roots import_data ArabicRoot, RootDatabase, create_root, SAMPLE_ROOTS
    from arabic_morphophon.models.patterns import_data PatternRepository, PatternType
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
    from arabic_morphophon.models.morphophon import_data ArabicMorphophon
    from arabic_morphophon.integrator import_data MorphophonologicalEngine
except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("تأكد من أن جميع الملفات موجودة ومكتوبة بشكل صحيح")
    sys.exit(1)

class TestResult:
    """نتيجة الاختبار"""
    def __init__(self, test_name: str, success: bool, message: str, duration: float = 0.0):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.duration = duration
    
    def __str__(self):
        status = "✅" if self.success else "❌"
        return f"{status} {self.test_name}: {self.message} ({self.duration:.3f}s)"

class ArabicMorphophonologyTester:
    """نظام اختبار المشروع الشامل"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.begin_time = time.time()
    
    def run_test(self, test_name: str, test_function) -> TestResult:
        """تشغيل اختبار واحد"""
        begin = time.time()
        try:
            success, message = test_function()
            duration = time.time() - begin
            result = TestResult(test_name, success, message, duration)
        except Exception as e:
            duration = time.time() - begin
            result = TestResult(test_name, False, f"استثناء: {str(e)}", duration)
            print(f"تفاصيل الخطأ في {test_name}:")
            traceback.print_exc()
        
        self.results.append(result)
        print(result)
        return result
    
    def test_root_database_basic(self) -> Tuple[bool, str]:
        """اختبار قاعدة بيانات الجذور الأساسي"""
        # إنشاء قاعدة بيانات جديدة
        db = RootDatabase()
        
        # فحص التحميل الأولي
        if len(db) == 0:
            return False, "قاعدة البيانات فارغة بعد التحميل الأولي"
        
        # اختبار إضافة جذر جديد
        test_root = create_root("سمع", "السمع والإصغاء")
        if not db.add_root(test_root):
            return False, "فشل في إضافة جذر جديد"
        
        # اختبار استخراج الجذر
        retrieved = db.get_root("سمع")
        if not retrieved or retrieved.root_string != "سمع":
            return False, "فشل في استخراج الجذر المضاف"
        
        # اختبار التحديث
        updated_root = create_root("سمع", "السمع والاستماع المحدث")
        if not db.update_root("سمع", updated_root):
            return False, "فشل في تحديث الجذر"
        
        # اختبار الحذف
        if not db.delete_root("سمع"):
            return False, "فشل في حذف الجذر"
        
        return True, f"جميع العمليات الأساسية تعمل - عدد الجذور: {len(db)}"
    
    def test_root_database_search(self) -> Tuple[bool, str]:
        """اختبار البحث في قاعدة بيانات الجذور"""
        db = RootDatabase()
        
        # البحث بالنمط
        pattern_results = db.search_by_pattern("ك*ب")
        if all(r.root_string != "كتب" for r in pattern_results):
            return False, "فشل البحث بالنمط"
        
        # البحث بالمجال الدلالي
        semantic_results = db.search_by_semantic_field("الكتابة والتدوين")
        if not semantic_results:
            return False, "فشل البحث بالمجال الدلالي"
        
        # البحث بالخصائص
        weak_roots = db.search_by_features(weakness_type="أجوف") + db.search_by_features(weakness_type="ناقص") + db.search_by_features(weakness_type="مثال")
        sound_roots = [r for r in db.get_all_roots() if not r.weakness or r.weakness == "صحيح"]
        
        total_roots = len(weak_roots) + len(sound_roots)
        if total_roots != len(db):
            return False, f"مجموع الجذور المعتلة ({len(weak_roots)}) والسالمة ({len(sound_roots)}) = {total_roots} لا يساوي العدد الكلي {len(db)}"
        
        return True, f"البحث يعمل - معتل: {len(weak_roots)}, سالم: {len(sound_roots)}"
    
    def test_root_database_statistics(self) -> Tuple[bool, str]:
        """اختبار الإحصائيات"""
        db = RootDatabase()
        stats = db.get_statistics()
        
        required_keys = ['total_roots', 'weak_roots', 'hamzated_roots', 'doubled_roots', 'coverage']
        if any(key not in stats for key in required_keys):
            return False, "إحصائيات ناقصة"
        
        if stats['total_roots'] != len(db):
            return False, "خطأ في عدد الجذور الكلي"
        
        return True, f"الإحصائيات صحيحة - {stats['total_roots']} جذر"
    
    def test_text_extraction(self) -> Tuple[bool, str]:
        """اختبار استخراج الجذور من النص"""
        db = RootDatabase()
        test_text = "كتب الطالب درساً في الكتاب المدرسي"
        
        extracted_roots = db.extract_roots_from_text(test_text)
        if not extracted_roots:
            return False, "لم يتم استخراج أي جذور من النص"
        
        # فحص وجود جذر "كتب" في النتائج
        found_ktb = any(root.root == "كتب" for root in extracted_roots)
        if not found_ktb:
            return False, "لم يتم العثور على جذر 'كتب' في النص"
        
        return True, f"تم استخراج {len(extracted_roots)} جذر محتمل"
    
    def test_file_operations(self) -> Tuple[bool, str]:
        """اختبار عمليات الملفات"""
        db = RootDatabase()
        test_file = "test_roots.json"
        
        # حفظ
        if not db.store_data_to_file(test_file):
            return False, "فشل في حفظ الملف"
        
        # تحميل
        new_db = RootDatabase()
        new_db.roots.clear()  # إفراغ البيانات الافتراضية
        
        if not new_db.import_data_from_file(test_file):
            return False, "فشل في تحميل الملف"
        
        if len(new_db) != len(db):
            return False, f"عدم تطابق عدد الجذور: الأصلي {len(db)}, المحمل {len(new_db)}"
        
        # تنظيف
        import_data os
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return True, "عمليات الملفات تعمل بشكل صحيح"
    
    def test_pattern_repository(self) -> Tuple[bool, str]:
        """اختبار مستودع الأوزان"""
        try:
            repo = PatternRepository()
            
            # اختبار جلب الأوزان حسب النوع
            verb_patterns = repo.get_patterns_by_type(PatternType.VERB)
            noun_patterns = repo.get_patterns_by_type(PatternType.NOUN)
            
            if not verb_patterns:
                return False, "لا توجد أوزان أفعال في المستودع"
            
            if not noun_patterns:
                return False, "لا توجد أوزان أسماء في المستودع"
            
            # اختبار تطبيق وزن
            test_root = create_root("كتب")
            if matched_patterns := repo.get_patterns_for_root(test_root):
                return True, f"المستودع يحتوي على {len(verb_patterns)} وزن فعل، {len(noun_patterns)} وزن اسم"
            else:
                return False, "لم يتم العثور على أوزان متوافقة مع الجذر"
        except Exception as e:
            return False, f"خطأ في مستودع الأوزان: {e}"
    
    def test_phonology_engine(self) -> Tuple[bool, str]:
        """اختبار محرك القواعد الصوتية"""
        try:
            engine = PhonologyEngine()
            test_word = "الكتاب"
            
            # تطبيق القواعد
            if processed := engine.apply_phonological_rules(test_word):
                return True, f"المعالجة الصوتية تعمل: {test_word} → {processed.get('output', 'غير محدد')}"
            else:
                return False, "لم يتم معالجة الكلمة صوتياً"
        except Exception as e:
            return False, f"خطأ في محرك الصوتيات: {e}"
    
    def test_morphophon(self) -> Tuple[bool, str]:
        """اختبار مُقسم المقاطع"""
        try:
            morphophon = ArabicMorphophon()
            test_word = "كتاب"
            
            syllabic_units = morphophon.syllabic_analyze(test_word)
            
            if not syllabic_units:
                return False, "فشل في تقسيم المقاطع"
            
            # syllabic_units is a list of syllabic_units
            syllabic_unit_count = len(syllabic_units) if isinstance(syllabic_units, list) else 1
            
            return True, f"تقسيم المقاطع يعمل: {test_word} → {syllabic_unit_count} مقطع"
        except Exception as e:
            return False, f"خطأ في مُقسم المقاطع: {e}"
    
    def test_integration_engine(self) -> Tuple[bool, str]:
        """اختبار محرك التكامل الرئيسي"""
        try:
            engine = MorphophonologicalEngine()
            test_text = "كتب"
            
            if result := engine.analyze(test_text):
                if result.original_text:
                    return True, f"محرك التكامل يعمل - تحليل: {test_text}"
                else:
                    return False, "النص الأصلي مفقود في النتيجة"
            else:
                return False, "لم يتم إرجاع نتيجة من محرك التكامل"
        except Exception as e:
            return False, f"خطأ في محرك التكامل: {e}"
    
    def test_performance(self) -> Tuple[bool, str]:
        """اختبار الأداء"""
        db = RootDatabase()
        
        # اختبار أداء البحث
        begin_time = time.time()
        for _ in range(100):
            db.get_root("كتب")
        search_time = time.time() - begin_time
        
        # اختبار أداء الإضافة
        begin_time = time.time()
        for i in range(50):
            test_root = create_root(f"جذر{i}", f"معنى {i}")
            db.add_root(test_root)
        add_time = time.time() - begin_time
        
        # معايير الأداء (مرن)
        if search_time > 1.0:  # أكثر من ثانية لـ 100 بحث
            return False, f"أداء البحث بطيء: {search_time:.3f}s لـ 100 بحث"
        
        if add_time > 2.0:  # أكثر من ثانيتين لـ 50 إضافة
            return False, f"أداء الإضافة بطيء: {add_time:.3f}s لـ 50 إضافة"
        
        return True, f"الأداء مقبول - بحث: {search_time:.3f}s, إضافة: {add_time:.3f}s"
    
    def run_all_tests(self):
        """تشغيل جميع الاختبارات"""
        print("🔄 بدء الاختبارات الشاملة للمشروع...")
        print("=" * 60)
        
        # اختبارات قاعدة بيانات الجذور
        print("\n📊 اختبارات قاعدة بيانات الجذور:")
        self.run_test("العمليات الأساسية", self.test_root_database_basic)
        self.run_test("وظائف البحث", self.test_root_database_search)
        self.run_test("الإحصائيات", self.test_root_database_statistics)
        self.run_test("استخراج من النص", self.test_text_extraction)
        self.run_test("عمليات الملفات", self.test_file_operations)
        
        # اختبارات المكونات الأخرى
        print("\n🔧 اختبارات المكونات:")
        self.run_test("مستودع الأوزان", self.test_pattern_repository)
        self.run_test("محرك الصوتيات", self.test_phonology_engine)
        self.run_test("مُقسم المقاطع", self.test_morphophon)
        self.run_test("محرك التكامل", self.test_integration_engine)
        
        # اختبارات الأداء
        print("\n⚡ اختبارات الأداء:")
        self.run_test("أداء العمليات", self.test_performance)
        
        # تقرير النتائج
        self.generate_report()
    
    def generate_report(self):
        """إنشاء تقرير النتائج"""
        total_tests = len(self.results)
        passed_tests = sum(r.success for r in self.results)
        failed_tests = total_tests - passed_tests
        total_time = time.time() - self.begin_time
        
        print("\n" + "=" * 60)
        print("📈 تقرير الاختبارات النهائي")
        print("=" * 60)
        
        print(f"المجموع: {total_tests}")
        print(f"نجح: {passed_tests} ✅")
        print(f"فشل: {failed_tests} ❌")
        print(f"معدل النجاح: {(passed_tests/total_tests)*100:.1f}%")
        print(f"الوقت الكلي: {total_time:.2f} ثانية")
        
        if failed_tests > 0:
            print(f"\n🚨 الاختبارات الفاشلة:")
            for result in self.results:
                if not result.success:
                    print(f"  • {result.test_name}: {result.message}")
        
        # حفظ التقرير
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'total_time': total_time,
            'results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'message': r.message,
                    'duration': r.duration
                } for r in self.results
            ]
        }
        
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 تم حفظ التقرير المفصل في: test_report.json")
        
        # تحديد حالة المشروع
        if passed_tests == total_tests:
            print(f"\n🎉 ممتاز! جميع الاختبارات اجتازت بنجاح")
            print("المشروع جاهز للانتقال إلى المرحلة التالية")
        elif passed_tests >= total_tests * 0.8:
            print(f"\n⚠️  المشروع في حالة جيدة ولكن يحتاج بعض الإصلاحات")
        else:
            print(f"\n🔧 المشروع يحتاج إصلاحات جوهرية قبل المتابعة")

if __name__ == "__main__":
    tester = ArabicMorphophonologyTester()
    tester.run_all_tests()
