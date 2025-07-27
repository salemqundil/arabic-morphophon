"""
🔥 Comprehensive Tests for FrozenRootsEngine
اختبارات شاملة لمحرك تصنيف الجذور الجامدة

يختبر جميع وظائف المحرك:
✅ التصنيف الأساسي (جامد/مشتق)
✅ التحليل المقطعي والفونيمي
✅ كشف أنماط الأفعال
✅ المعالجة المجمعة
✅ أداء النظام
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data unittest
import_data time
import_data sys
from pathlib import_data Path

# إضافة مسار المشروع
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from engines.nlp.frozen_root.engine import_data FrozenRootsEngine
from engines.nlp.frozen_root.models.classifier import_data AdvancedRootClassifier
from engines.nlp.frozen_root.models.syllabic_unit_check import_data SyllabicUnitAnalyzer
from engines.nlp.frozen_root.models.verb_check import_data VerbPatternRecognizer

class TestFrozenRootsEngine(unittest.TestCase):
    """اختبارات محرك تصنيف الجذور الجامدة"""
    
    @classmethod
    def setUpClass(cls):
        """إعداد الاختبارات"""
        cls.engine = FrozenRootsEngine()
        cls.test_data = {
            "frozen_words": [
                "من", "لن", "إذا", "يا", "لم", "قد", "كلا", "هل",
                "إن", "أن", "كي", "لكن", "غير", "سوف", "ليس",
                "ماذا", "متى", "أين", "كيف", "لماذا", "ذلك", "هذا"
            ],
            "derivable_words": [
                "كتب", "درس", "فعل", "علم", "شرب", "أكل",
                "كسّر", "درّس", "علّم", "قاتل", "شارك", "ساعد",
                "انكسر", "اجتمع", "استخرج", "استعمل"
            ]
        }
    
    def test_basic_classification_frozen(self):
        """اختبار التصنيف الأساسي للجذور الجامدة"""
        print("\n🧪 اختبار التصنيف الأساسي - الجذور الجامدة")
        
        for word in self.test_data["frozen_words"]:
            with self.subTest(word=word):
                result = self.engine.classify(word)
                
                self.assertEqual(result["classification"], "frozen",
                               f"الكلمة '{word}' يجب أن تصنف كجامدة")
                self.assertTrue(result["is_frozen"],
                              f"الكلمة '{word}' يجب أن تكون is_frozen=True")
                self.assertFalse(result["is_derivable"],
                               f"الكلمة '{word}' يجب أن تكون is_derivable=False")
                self.assertGreater(result["confidence"], 0.5,
                                 f"درجة الثقة للكلمة '{word}' منخفضة: {result['confidence']}")
                
                print(f"   ✅ {word}: {result['classification']} ({result['confidence']:.1%})")
    
    def test_basic_classification_derivable(self):
        """اختبار التصنيف الأساسي للجذور المشتقة"""
        print("\n🧪 اختبار التصنيف الأساسي - الجذور المشتقة")
        
        for word in self.test_data["derivable_words"]:
            with self.subTest(word=word):
                result = self.engine.classify(word)
                
                self.assertEqual(result["classification"], "derivable",
                               f"الكلمة '{word}' يجب أن تصنف كمشتقة")
                self.assertFalse(result["is_frozen"],
                               f"الكلمة '{word}' يجب أن تكون is_frozen=False")
                self.assertTrue(result["is_derivable"],
                              f"الكلمة '{word}' يجب أن تكون is_derivable=True")
                self.assertGreater(result["confidence"], 0.5,
                                 f"درجة الثقة للكلمة '{word}' منخفضة: {result['confidence']}")
                
                print(f"   ✅ {word}: {result['classification']} ({result['confidence']:.1%})")
    
    def test_detailed_analysis(self):
        """اختبار التحليل المفصل"""
        print("\n🧪 اختبار التحليل المفصل")
        
        test_cases = [
            ("من", "frozen"),
            ("كتب", "derivable"),
            ("استخرج", "derivable"),
            ("إذا", "frozen")
        ]
        
        for word, expected_type in test_cases:
            with self.subTest(word=word):
                result = self.engine.analyze(word, detailed=True)
                
                # التحقق من الأساسيات
                self.assertEqual(result["type"], expected_type)
                self.assertIn("analysis", result)
                self.assertIn("cv_pattern", result["analysis"])
                self.assertIn("phonemes", result["analysis"])
                
                # التحقق من التحليل المقطعي
                if "syllabic_unit_analysis" in result:
                    syllabic_unit_data = result["syllabic_unit_analysis"]
                    self.assertIn("syllabic_units", syllabic_unit_data)
                    self.assertIn("syllabic_unit_count", syllabic_unit_data)
                    self.assertGreater(syllabic_unit_data["syllabic_unit_count"], 0)
                
                # التحقق من تحليل الأفعال
                if "verb_pattern_analysis" in result:
                    verb_data = result["verb_pattern_analysis"]
                    self.assertIn("is_verb_pattern", verb_data)
                
                print(f"   ✅ {word}: {result['type']} | CV: {result['analysis']['cv_pattern']} | المقاطع: {result.get('syllabic_unit_analysis', {}).get('syllabic_unit_count', 'N/A')}")
    
    def test_cv_patterns(self):
        """اختبار استخراج أنماط CV"""
        print("\n🧪 اختبار أنماط CV")
        
        expected_patterns = {
            "من": "CVC",
            "إذا": "VCV", 
            "كتب": "CVCVC",
            "استخرج": "CCVCVC",
            "قاتل": "CVVCVC",
            "كسّر": "CVCCVC"
        }
        
        for word, expected_pattern in expected_patterns.items():
            with self.subTest(word=word):
                result = self.engine.analyze(word)
                actual_pattern = result["analysis"]["cv_pattern"]
                
                self.assertEqual(actual_pattern, expected_pattern,
                               f"نمط CV للكلمة '{word}' خاطئ")
                
                print(f"   ✅ {word}: {actual_pattern}")
    
    def test_phoneme_analysis(self):
        """اختبار التحليل الفونيمي"""
        print("\n🧪 اختبار التحليل الفونيمي")
        
        test_words = ["من", "كتب", "إذا", "قاتل"]
        
        for word in test_words:
            with self.subTest(word=word):
                result = self.engine.analyze(word, include_phonemes=True)
                phonemes = result["analysis"].get("phonemes", [])
                
                self.assertIsInstance(phonemes, list)
                self.assertGreater(len(phonemes), 0,
                                 f"لا توجد فونيمات للكلمة '{word}'")
                
                # التحقق من أن الفونيمات تحتوي على رموز IPA صحيحة
                for phoneme in phonemes:
                    self.assertIsInstance(phoneme, str)
                    self.assertGreater(len(phoneme), 0)
                
                print(f"   ✅ {word}: {phonemes}")
    
    def test_syllabic_unit_analysis(self):
        """اختبار تحليل المقاطع"""
        print("\n🧪 اختبار تحليل المقاطع")
        
        test_words = ["كتب", "استخرج", "قاتل", "من"]
        
        for word in test_words:
            with self.subTest(word=word):
                result = self.engine.analyze(word, include_syllabic_units=True)
                
                if "syllabic_unit_analysis" in result:
                    syllabic_unit_data = result["syllabic_unit_analysis"]
                    
                    # التحقق من وجود البيانات الأساسية
                    self.assertIn("syllabic_units", syllabic_unit_data)
                    self.assertIn("syllabic_unit_count", syllabic_unit_data)
                    self.assertIn("complexity_score", syllabic_unit_data)
                    
                    syllabic_units = syllabic_unit_data["syllabic_units"]
                    self.assertIsInstance(syllabic_units, list)
                    self.assertGreater(len(syllabic_units), 0)
                    
                    # التحقق من بنية كل مقطع
                    for syllabic_unit in syllabic_units:
                        self.assertIn("pattern", syllabic_unit)
                        self.assertIn("phonemes", syllabic_unit)
                        self.assertIn("type", syllabic_unit)
                    
                    print(f"   ✅ {word}: {syllabic_unit_data['syllabic_unit_count']} مقاطع، تعقد: {syllabic_unit_data['complexity_score']:.2f}")
    
    def test_batch_processing(self):
        """اختبار المعالجة المجمعة"""
        print("\n🧪 اختبار المعالجة المجمعة")
        
        batch_words = ["من", "كتب", "إذا", "درس", "هل", "قاتل"]
        
        result = self.engine.batch_analyze(batch_words, detailed=False)
        
        # التحقق من البنية الأساسية
        self.assertIn("batch_summary", result)
        self.assertIn("results", result)
        
        batch_summary = result["batch_summary"]
        self.assertEqual(batch_summary["total_words"], len(batch_words))
        self.assertGreaterEqual(batch_summary["successful_analyses"], 0)
        self.assertGreaterEqual(batch_summary["batch_processing_time_ms"], 0)
        
        # التحقق من النتائج الفردية
        results = result["results"]
        for word in batch_words:
            self.assertIn(word, results)
            word_result = results[word]
            self.assertIn("type", word_result)
            self.assertIn("confidence", word_result)
        
        print(f"   ✅ معالجة {batch_summary['total_words']} كلمات في {batch_summary['batch_processing_time_ms']:.1f}ms")
        print(f"   ✅ نجح: {batch_summary['successful_analyses']}, فشل: {batch_summary['failed_analyses']}")
    
    def test_performance_metrics(self):
        """اختبار مقاييس الأداء"""
        print("\n🧪 اختبار مقاييس الأداء")
        
        # تنفيذ عدة تصنيفات لتوليد إحصائيات
        test_words = ["من", "كتب", "إذا", "درس", "هل"]
        for word in test_words:
            self.engine.classify(word)
        
        stats = self.engine.get_performance_stats()
        
        # التحقق من البنية
        self.assertIn("engine_info", stats)
        self.assertIn("performance_metrics", stats)
        self.assertIn("configuration", stats)
        self.assertIn("component_stats", stats)
        
        # التحقق من المعلومات الأساسية
        engine_info = stats["engine_info"]
        self.assertEqual(engine_info["name"], "frozen_root")
        self.assertEqual(engine_info["version"], "1.0.0")
        
        # التحقق من المقاييس
        metrics = stats["performance_metrics"]
        self.assertGreaterEqual(metrics["total_classifications"], len(test_words))
        self.assertGreaterEqual(metrics["average_processing_time"], 0.0)
        
        print(f"   ✅ إجمالي التصنيفات: {metrics['total_classifications']}")
        print(f"   ✅ متوسط وقت المعالجة: {metrics['average_processing_time']:.4f}s")
        print(f"   ✅ الجذور الجامدة: {metrics['frozen_count']}")
        print(f"   ✅ الجذور المشتقة: {metrics['derivable_count']}")
    
    def test_error_handling(self):
        """اختبار معالجة الأخطاء"""
        print("\n🧪 اختبار معالجة الأخطاء")
        
        error_cases = ["", "   ", "123", "!@#", None]
        
        for case in error_cases:
            if case is None:
                continue
                
            with self.subTest(case=case):
                result = self.engine.analyze(case)
                
                # يجب أن تكون النتيجة خطأ أو تعامل صحيح
                if result["type"] == "error":
                    self.assertIn("reason", result)
                    self.assertEqual(result["confidence"], 0.0)
                    print(f"   ✅ حالة خطأ تم التعامل معها: '{case}'")
                else:
                    # في حالة التعامل الصحيح مع المدخل
                    print(f"   ✅ تم التعامل مع المدخل: '{case}' -> {result['type']}")
    
    def test_word_details(self):
        """اختبار تفاصيل الكلمة الشاملة"""
        print("\n🧪 اختبار تفاصيل الكلمة الشاملة")
        
        test_word = "استخرج"
        details = self.engine.get_word_details(test_word)
        
        # التحقق من البنية الأساسية
        self.assertIn("word", details)
        self.assertIn("comprehensive_analysis", details)
        self.assertIn("classification_explanation", details)
        self.assertIn("linguistic_features", details)
        
        # التحقق من المعالم اللغوية
        features = details["linguistic_features"]
        self.assertIn("cv_pattern", features)
        self.assertIn("phoneme_count", features)
        self.assertIn("syllabic_unit_count", features)
        self.assertIn("word_length", features)
        
        self.assertEqual(features["word_length"], len(test_word))
        
        print(f"   ✅ الكلمة: {details['word']}")
        print(f"   ✅ النمط CV: {features['cv_pattern']}")
        print(f"   ✅ عدد الفونيمات: {features['phoneme_count']}")
        print(f"   ✅ عدد المقاطع: {features['syllabic_unit_count']}")
        print(f"   ✅ درجة الثقة: {details['classification_confidence']:.1%}")
    
    def test_speed_performance(self):
        """اختبار أداء السرعة"""
        print("\n🧪 اختبار أداء السرعة")
        
        test_words = ["من", "كتب", "إذا", "استخرج", "قاتل"] * 20  # 100 كلمة
        
        # اختبار التصنيف السريع
        begin_time = time.time()
        for word in test_words:
            self.engine.classify(word)
        classify_time = time.time() - begin_time
        
        # اختبار التحليل المفصل
        begin_time = time.time()
        for word in test_words[:10]:  # 10 كلمات فقط للتحليل المفصل
            self.engine.analyze(word, detailed=True)
        detailed_time = time.time() - begin_time
        
        # التحقق من الأداء
        avg_classify_time = (classify_time / len(test_words)) * 1000  # بالميلي ثانية
        avg_detailed_time = (detailed_time / 10) * 1000
        
        self.assertLess(avg_classify_time, 10,  # أقل من 10ms لكل تصنيف
                       f"التصنيف السريع بطيء جداً: {avg_classify_time:.2f}ms")
        self.assertLess(avg_detailed_time, 100,  # أقل من 100ms للتحليل المفصل
                       f"التحليل المفصل بطيء جداً: {avg_detailed_time:.2f}ms")
        
        print(f"   ✅ متوسط وقت التصنيف: {avg_classify_time:.2f}ms")
        print(f"   ✅ متوسط وقت التحليل المفصل: {avg_detailed_time:.2f}ms")
        print(f"   ✅ معدل المعالجة: {len(test_words)/classify_time:.0f} كلمة/ثانية")

class TestSyllabicUnitAnalyzer(unittest.TestCase):
    """اختبارات محلل المقاطع منفصلة"""
    
    def setUp(self):
        self.analyzer = SyllabicUnitAnalyzer()
    
    def test_cv_pattern_extraction(self):
        """اختبار استخراج أنماط CV"""
        test_cases = {
            "كتب": "CVCVC",
            "من": "CVC", 
            "إذا": "VCV",
            "استخرج": "CCVCVC"
        }
        
        for word, expected in test_cases.items():
            actual = self.analyzer.get_cv_pattern(word)
            self.assertEqual(actual, expected,
                           f"نمط CV خاطئ للكلمة '{word}': توقع {expected}, حصل على {actual}")
    
    def test_phoneme_conversion(self):
        """اختبار تحويل الفونيمات"""
        word = "كتب"
        phonemes = self.analyzer.get_phonemes(word)
        
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)
        
        # التحقق من أن جميع الفونيمات نصوص
        for phoneme in phonemes:
            self.assertIsInstance(phoneme, str)

class TestVerbPatternRecognizer(unittest.TestCase):
    """اختبارات كاشف أنماط الأفعال منفصلة"""
    
    def setUp(self):
        self.recognizer = VerbPatternRecognizer()
    
    def test_verb_pattern_recognition(self):
        """اختبار التعرف على أنماط الأفعال"""
        verb_patterns = ["CVCVC", "CVCCVC", "CVVCVC", "CCVCVC"]
        non_verb_patterns = ["CV", "VC", "CVC"]
        
        for pattern in verb_patterns:
            self.assertTrue(self.recognizer.is_verb_form(pattern),
                          f"النمط {pattern} يجب أن يكون نمط فعل")
        
        for pattern in non_verb_patterns:
            self.assertFalse(self.recognizer.is_verb_form(pattern),
                           f"النمط {pattern} لا يجب أن يكون نمط فعل")
    
    def test_derivation_potential(self):
        """اختبار إمكانية الاشتقاق"""
        high_derivability = "CVCVC"  # فَعَلَ
        result = self.recognizer.get_derivation_potential(high_derivability)
        
        self.assertEqual(result["potential"], "high")
        self.assertGreater(result["score"], 0.8)
        self.assertIsInstance(result["possible_derivations"], list)

def run_comprehensive_tests():
    """تشغيل جميع الاختبارات الشاملة"""
    print("🔥 بدء الاختبارات الشاملة لمحرك تصنيف الجذور الجامدة")
    print("=" * 80)
    
    # تجميع جميع الاختبارات
    test_suite = unittest.TestSuite()
    
    # إضافة اختبارات المحرك الرئيسي
    test_suite.addTest(unittest.TestImporter().import_dataTestsFromTestCase(TestFrozenRootsEngine))
    
    # إضافة اختبارات المكونات الفرعية
    test_suite.addTest(unittest.TestImporter().import_dataTestsFromTestCase(TestSyllabicUnitAnalyzer))
    test_suite.addTest(unittest.TestImporter().import_dataTestsFromTestCase(TestVerbPatternRecognizer))
    
    # تشغيل الاختبارات
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # ملخص النتائج
    print("\n" + "=" * 80)
    print("📊 ملخص نتائج الاختبارات:")
    print(f"   ✅ اختبارات نجحت: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ اختبارات فشلت: {len(result.failures)}")
    print(f"   🚨 أخطاء: {len(result.errors)}")
    print(f"   📈 معدل النجاح: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ الاختبارات الفاشلة:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n🚨 الأخطاء:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 جميع الاختبارات نجحت! محرك FrozenRootsEngine جاهز للإنتاج.")
    else:
        print("\n⚠️  بعض الاختبارات فشلت. يرجى مراجعة الأخطاء أعلاه.")
        sys.exit(1)
