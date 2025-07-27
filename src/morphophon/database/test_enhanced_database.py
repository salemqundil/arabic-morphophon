"""
🧪 Enhanced Root Database Test Suite - مجموعة اختبارات قاعدة البيانات المطورة
===============================================================================

اختبارات شاملة لقاعدة بيانات الجذور العربية المطورة لضمان:
- صحة عمليات CRUD
- أداء البحث المتقدم
- دقة الإحصائيات
- استقرار النظام تحت الضغط

US-01 Testing: Complete validation of database functionality
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data tempfile
import_data time
from pathlib import_data Path
from typing import_data List

from arabic_morphophon.database.enhanced_root_database import_data (
    DatabaseConfig,
    EnhancedRootDatabase,
    create_enhanced_database,
    create_memory_database,
)
from arabic_morphophon.models.roots import_data RootType, create_root

class DatabaseTestSuite:
    """🧪 مجموعة اختبارات شاملة لقاعدة البيانات"""

    def __init__(self):
        self.results = []
        self.temp_dir = Path(tempfile.mkdtemp())

    def run_all_tests(self) -> dict:
        """تشغيل جميع الاختبارات"""
        print("🧪 بدء اختبار قاعدة البيانات المطورة...")

        tests = [
            self.test_basic_crud,
            self.test_advanced_search,
            self.test_bulk_operations,
            self.test_performance,
            self.test_statistics,
            self.test_backup_restore,
            self.test_concurrent_access,
            self.test_error_handling,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                    print(f"✅ {test.__name__}: نجح")
                else:
                    failed += 1
                    print(f"❌ {test.__name__}: فشل")
            except Exception as e:
                failed += 1
                print(f"💥 {test.__name__}: خطأ - {e}")

        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0

        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "details": self.results,
        }

        print(f"\n📊 ملخص الاختبارات:")
        print(f"  إجمالي الاختبارات: {total}")
        print(f"  نجح: {passed}")
        print(f"  فشل: {failed}")
        print(f"  معدل النجاح: {success_rate:.1f}%")

        return summary

    def test_basic_crud(self) -> bool:
        """اختبار العمليات الأساسية CRUD"""
        try:
            with create_memory_database() as db:
                # اختبار الإنشاء
                root = create_root("كتب", "الكتابة والتدوين")
                assert db.create_root(root), "فشل في إنشاء الجذر"

                # اختبار القراءة
                retrieved = db.read_root("كتب")
                assert retrieved is not None, "فشل في قراءة الجذر"
                assert retrieved.root_string == "كتب", "بيانات خاطئة"

                # اختبار التحديث
                root.frequency = 100
                assert db.update_root("كتب", root), "فشل في تحديث الجذر"

                updated = db.read_root("كتب")
                assert updated is not None, "فشل في قراءة الجذر المحدث"
                assert updated.frequency == 100, "لم يتم التحديث"

                # اختبار الحذف
                assert db.delete_root("كتب"), "فشل في حذف الجذر"
                assert db.read_root("كتب") is None, "لم يتم الحذف"

                return True

        except Exception as e:
            print(f"خطأ في اختبار CRUD: {e}")
            return False

    def test_advanced_search(self) -> bool:
        """اختبار البحث المتقدم"""
        try:
            with create_memory_database() as db:
                # إضافة بيانات اختبار
                test_roots = [
                    ("كتب", "الكتابة"),
                    ("قرأ", "القراءة"),
                    ("وعد", "الوعد"),
                    ("قال", "القول"),
                ]

                for root_str, meaning in test_roots:
                    root = create_root(root_str, meaning)
                    db.create_root(root)

                # اختبار البحث بالنمط
                pattern_results = db.search_by_pattern("ق*")
                assert len(pattern_results) == 2, f"متوقع 2، وجد {len(pattern_results)}"

                # اختبار البحث بالمجال الدلالي
                semantic_results = db.search_by_semantic_field("القراءة")
                assert len(semantic_results) == 1, "متوقع جذر واحد للقراءة"

                # اختبار البحث بالخصائص
                weak_results = db.search_by_properties(weakness_type="مثال")
                assert len(weak_results) >= 1, "يجب وجود جذور معتلة"

                return True

        except Exception as e:
            print(f"خطأ في اختبار البحث المتقدم: {e}")
            return False

    def test_bulk_operations(self) -> bool:
        """اختبار العمليات المجمعة"""
        try:
            # إنشاء ملف JSON مؤقت
            test_data = {
                "roots": [
                    {"root": "فعل", "semantic_field": "الفعل"},
                    {"root": "اسم", "semantic_field": "الاسم"},
                    {"root": "حرف", "semantic_field": "الحرف"},
                ]
            }

            import_data json

            temp_file = self.temp_dir / "test_roots.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False)

            with create_memory_database() as db:
                # اختبار الاستيراد المجمع
                stats = db.bulk_import_data_json(temp_file)
                assert (
                    stats["import_dataed"] == 3
                ), f"متوقع 3، تم استيراد {stats['import_dataed']}"
                assert stats["errors"] == 0, f"أخطاء غير متوقعة: {stats['errors']}"

                # اختبار التصدير
                store_data_file = self.temp_dir / "store_dataed_roots.json"
                assert db.store_data_to_json(store_data_file), "فشل التصدير"
                assert store_data_file.exists(), "ملف التصدير غير موجود"

                return True

        except Exception as e:
            print(f"خطأ في اختبار العمليات المجمعة: {e}")
            return False

    def test_performance(self) -> bool:
        """اختبار الأداء"""
        try:
            with create_memory_database() as db:
                # إضافة عدد كبير من الجذور
                begin_time = time.time()

                for i in range(100):
                    root_str = f"ج{i:02d}ر"
                    root = create_root(root_str, f"مجال {i}")
                    db.create_root(root)

                creation_time = time.time() - begin_time
                assert creation_time < 5.0, f"إنشاء بطيء جداً: {creation_time:.2f}s"

                # اختبار سرعة البحث
                begin_time = time.time()
                results = db.search_by_pattern("ج*")
                search_time = time.time() - begin_time

                assert search_time < 1.0, f"بحث بطيء: {search_time:.2f}s"
                assert len(results) == 100, f"نتائج ناقصة: {len(results)}"

                return True

        except Exception as e:
            print(f"خطأ في اختبار الأداء: {e}")
            return False

    def test_statistics(self) -> bool:
        """اختبار الإحصائيات"""
        try:
            with create_memory_database() as db:
                # إضافة بيانات متنوعة
                test_data = [
                    ("كتب", "الكتابة", None),  # سالم
                    ("وعد", "الوعد", "مثال"),  # معتل
                    ("قرأ", "القراءة", "مهموز"),  # مهموز
                ]

                for root_str, meaning, weakness in test_data:
                    root = create_root(root_str, meaning)
                    db.create_root(root)

                # جلب الإحصائيات
                stats = db.get_comprehensive_statistics()

                assert "basic_statistics" in stats, "إحصائيات أساسية مفقودة"
                assert "type_distribution" in stats, "توزيع الأنواع مفقود"
                assert "performance" in stats, "إحصائيات الأداء مفقودة"

                basic = stats["basic_statistics"]
                assert basic["total_roots"] >= 3, "عدد جذور خاطئ"

                return True

        except Exception as e:
            print(f"خطأ في اختبار الإحصائيات: {e}")
            return False

    def test_backup_restore(self) -> bool:
        """اختبار النسخ الاحتياطي والاستعادة"""
        try:
            backup_file = self.temp_dir / "backup.json"

            # إنشاء بيانات وحفظها
            with create_memory_database() as db1:
                root = create_root("اختبار", "اختبار النسخ الاحتياطي")
                db1.create_root(root)

                assert db1.backup_database(backup_file), "فشل النسخ الاحتياطي"
                assert backup_file.exists(), "ملف النسخ الاحتياطي غير موجود"

            # استعادة في قاعدة بيانات جديدة
            with create_memory_database() as db2:
                stats = db2.bulk_import_data_json(backup_file)
                assert stats["import_dataed"] >= 1, "فشل استعادة البيانات"

                restored = db2.read_root("اختبار")
                assert restored is not None, "الجذر غير موجود بعد الاستعادة"

                return True

        except Exception as e:
            print(f"خطأ في اختبار النسخ الاحتياطي: {e}")
            return False

    def test_concurrent_access(self) -> bool:
        """اختبار الوصول المتزامن (محاكاة)"""
        try:
            db_file = self.temp_dir / "concurrent_test.db"
            config = DatabaseConfig(db_path=db_file)

            # محاكاة عمليات متزامنة
            operations_success = 0

            for i in range(10):
                try:
                    with EnhancedRootDatabase(config) as db:
                        root = create_root(f"ت{i}", f"اختبار {i}")
                        if db.create_root(root):
                            operations_success += 1
                except Exception:
                    pass

            # التحقق من النتائج
            with EnhancedRootDatabase(config) as db:
                total_roots = len(db)
                assert total_roots >= operations_success, "فقدان بيانات"

            return True

        except Exception as e:
            print(f"خطأ في اختبار الوصول المتزامن: {e}")
            return False

    def test_error_handling(self) -> bool:
        """اختبار معالجة الأخطاء"""
        try:
            with create_memory_database() as db:
                # اختبار إدراج مكرر
                root = create_root("خطأ", "اختبار الأخطاء")
                assert db.create_root(root), "فشل الإدراج الأول"
                assert not db.create_root(root), "يجب فشل الإدراج المكرر"

                # اختبار تحديث غير موجود
                fake_root = create_root("وهمي", "غير موجود")
                assert not db.update_root(
                    "غير_موجود", fake_root
                ), "يجب فشل تحديث غير موجود"

                # اختبار حذف غير موجود
                assert not db.delete_root("غير_موجود"), "يجب فشل حذف غير موجود"

                # اختبار بحث بنمط خاطئ
                results = db.search_by_pattern("")
                assert isinstance(results, list), "يجب إرجاع قائمة فارغة"

                return True

        except Exception as e:
            print(f"خطأ في اختبار معالجة الأخطاء: {e}")
            return False

    def cleanup(self):
        """تنظيف ملفات الاختبار"""
        import_data shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

def run_comprehensive_database_test():
    """🧪 تشغيل الاختبار الشامل لقاعدة البيانات"""
    print("=" * 80)
    print("🧪 اختبار قاعدة البيانات المطورة للجذور العربية")
    print("=" * 80)

    test_suite = DatabaseTestSuite()

    try:
        results = test_suite.run_all_tests()

        print("\n" + "=" * 80)
        if results["success_rate"] >= 90:
            print("🎉 اختبارات ممتازة! قاعدة البيانات جاهزة للإنتاج")
        elif results["success_rate"] >= 70:
            print("⚠️ اختبارات جيدة، لكن تحتاج بعض التحسين")
        else:
            print("❌ اختبارات ضعيفة، تحتاج مراجعة جذرية")

        return results

    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    # تشغيل الاختبارات
    results = run_comprehensive_database_test()

    # عرض النتائج التفصيلية
    if results["failed"] > 0:
        print(f"\n⚠️ فشل {results['failed']} اختبار. يرجى مراجعة التفاصيل أعلاه.")
    else:
        print("\n✅ جميع الاختبارات نجحت بامتياز!")
