"""
🗄️ Enhanced Arabic Root Database - قاعدة بيانات الجذور العربية المطورة
==============================================================================

تصميم متقدم لقاعدة بيانات الجذور العربية مع:
- واجهة برمجية شاملة (CRUD + Advanced Search)
- دعم SQLite مع فهرسة محسنة
- نظام تخزين هجين (SQLite + JSON + Cache)
- إحصائيات وتحليلات متقدمة
- تكامل مع نظام التحليل الصرفي

US-01 Implementation: Complete CRUD operations with advanced features
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from __future__ import_data annotations

import_data json
import_data sqlite3
import_data time
from collections import_data defaultdict
from dataclasses import_data asdict, dataclass, field
from datetime import_data datetime
from pathlib import_data Path
from typing import_data Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from ..models.roots import_data ArabicRoot, RootType, create_root

# =============================================================================
# Database Schema and Configuration
# =============================================================================

ENHANCED_DB_SCHEMA = """
-- Main roots table with full Arabic linguistic properties
CREATE TABLE IF NOT EXISTS roots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    root              TEXT    UNIQUE NOT NULL,
    radicals          TEXT    NOT NULL,          -- JSON: detailed radical info
    root_type         TEXT    NOT NULL,          -- ثلاثي، رباعي، خماسي
    weakness_type     TEXT,                      -- صحيح، معتل، مهموز
    hamza_type        TEXT,                      -- مهموز الفاء/العين/اللام
    semantic_field    TEXT,                      -- المجال الدلالي
    frequency         INTEGER DEFAULT 0,         -- تكرار الاستخدام
    difficulty_score  REAL    DEFAULT 0.5,       -- صعوبة التحليل (0-1)
    phonetic_features TEXT,                      -- JSON: خصائص صوتية
    morpho_patterns   TEXT,                      -- JSON: أوزان مرتبطة
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_count       INTEGER DEFAULT 0,         -- عدد مرات الاستخدام
    confidence_score  REAL    DEFAULT 1.0        -- ثقة التحليل (0-1)
);

-- Semantic fields for categorization
CREATE TABLE IF NOT EXISTS semantic_fields (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    field_name  TEXT    UNIQUE NOT NULL,
    description TEXT,
    parent_id   INTEGER REFERENCES semantic_fields(id),
    root_count  INTEGER DEFAULT 0
);

-- Root patterns relationships
CREATE TABLE IF NOT EXISTS root_patterns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id    INTEGER REFERENCES roots(id),
    pattern    TEXT    NOT NULL,
    weight     REAL    DEFAULT 1.0,               -- قوة الارتباط
    frequency  INTEGER DEFAULT 0
);

-- Analysis cache for performance
CREATE TABLE IF NOT EXISTS analysis_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text    TEXT    NOT NULL,
    root_results  TEXT    NOT NULL,               -- JSON results
    analysis_time REAL    NOT NULL,               -- وقت التحليل بالثانية
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count     INTEGER DEFAULT 1
);

-- Performance and usage statistics
CREATE TABLE IF NOT EXISTS database_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    total_roots      INTEGER NOT NULL,
    total_queries    INTEGER DEFAULT 0,
    avg_query_time   REAL    DEFAULT 0.0,
    cache_hit_ratio  REAL    DEFAULT 0.0,
    last_updated     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for optimized queries
CREATE INDEX IF NOT EXISTS idx_roots_type ON roots(root_type);
CREATE INDEX IF NOT EXISTS idx_roots_weakness ON roots(weakness_type);
CREATE INDEX IF NOT EXISTS idx_roots_semantic ON roots(semantic_field);
CREATE INDEX IF NOT EXISTS idx_roots_frequency ON roots(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_cache_input ON analysis_cache(input_text);
CREATE INDEX IF NOT EXISTS idx_patterns_root ON root_patterns(root_id);

-- Full-text search support
CREATE VIRTUAL TABLE IF NOT EXISTS roots_fts USING fts5(
    root, radicals, semantic_field, content='roots', content_rowid='id'
);

-- Triggers for maintaining data consistency
CREATE TRIGGER IF NOT EXISTS update_root_timestamp
    AFTER UPDATE ON roots
    BEGIN
        UPDATE roots SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_semantic_count
    AFTER INSERT ON roots
    BEGIN
        UPDATE semantic_fields 
        SET root_count = root_count + 1 
        WHERE field_name = NEW.semantic_field;
    END;
"""

@dataclass
class DatabaseConfig:
    """إعدادات قاعدة البيانات"""

    db_path: Path = Path("data/enhanced_roots.db")
    json_backup_path: Path = Path("data/roots_backup.json")
    cache_size: int = 1000
    enable_fts: bool = True
    auto_backup: bool = True
    performance_logging: bool = True

@dataclass
class QueryStats:
    """إحصائيات الاستعلام"""

    query_time: float
    results_count: int
    cache_hit: bool = False
    complexity_score: float = 0.0

@dataclass
class DatabaseMetrics:
    """مقاييس أداء قاعدة البيانات"""

    total_roots: int = 0
    total_queries: int = 0
    avg_query_time: float = 0.0
    cache_hit_ratio: float = 0.0
    storage_size_mb: float = 0.0
    last_backup: Optional[datetime] = None

# =============================================================================
# Enhanced Root Database Class
# =============================================================================

class EnhancedRootDatabase:
    """
    🗄️ قاعدة بيانات الجذور العربية المطورة

    الميزات:
    - CRUD operations متقدمة
    - بحث متطور ومفهرس
    - تخزين هجين (SQLite + Cache)
    - إحصائيات وتحليل الأداء
    - نسخ احتياطي تلقائي
    - دعم FTS للبحث النصي
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """تهيئة قاعدة البيانات المطورة"""
        self.config = config or DatabaseConfig()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Database connection
        self._conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.run_commandscript(ENHANCED_DB_SCHEMA)

        # In-memory cache for performance
        self._cache: Dict[str, ArabicRoot] = {}
        self._query_cache: Dict[str, Any] = {}
        self._stats = DatabaseMetrics()

        # Performance tracking
        self._query_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

        self._initialize_database()

    def _initialize_database(self):
        """تهيئة البيانات الأولية"""
        if self._get_root_count() == 0:
            self._populate_sample_data()

        self._update_stats()

    # =========================================================================
    # Core CRUD Operations - العمليات الأساسية
    # =========================================================================

    def _validate_root_existence(
        self, root_string: str, should_exist: bool = False
    ) -> None:
        """التحقق من وجود الجذر في قاعدة البيانات"""
        exists = bool(self.read_root(root_string))
        if should_exist and not exists:
            raise ValueError(f"Root {root_string} does not exist")
        elif not should_exist and exists:
            raise ValueError(f"Root {root_string} already exists")

    def _prepare_root_data_for_db(self, root: ArabicRoot) -> tuple:
        """تحضير بيانات الجذر للإدراج في قاعدة البيانات"""
        root_data = self._serialize_root(root)
        return (
            json.dumps(root_data["radicals"], ensure_ascii=False),
            root.root_type.value,
            root.get_weakness_type(),
            root.get_hamza_type(),
            root.semantic_field,
            root.frequency or 0,
            json.dumps(root_data["phonetic_features"], ensure_ascii=False),
            json.dumps(root_data["morpho_patterns"], ensure_ascii=False),
            1.0,  # confidence_score default
        )

    def _prepare_root_data_for_update(
        self, root: ArabicRoot, root_string: str
    ) -> tuple:
        """تحضير بيانات الجذر للتحديث في قاعدة البيانات"""
        root_data = self._serialize_root(root)
        return (
            json.dumps(root_data["radicals"], ensure_ascii=False),
            root.root_type.value,
            root.get_weakness_type(),
            root.get_hamza_type(),
            root.semantic_field,
            root.frequency or 0,
            json.dumps(root_data["phonetic_features"], ensure_ascii=False),
            json.dumps(root_data["morpho_patterns"], ensure_ascii=False),
            root_string,
        )

    def _run_command_root_insert(self, root: ArabicRoot, db_data: tuple) -> None:
        """تنفيذ عملية إدراج الجذر في قاعدة البيانات"""
        query = """
            INSERT OR REPLACE INTO roots (
                root, radicals, root_type, weakness_type, hamza_type,
                semantic_field, frequency, phonetic_features, morpho_patterns,
                confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self._conn.run_command(query, (root.root_string,) + db_data)
        self._conn.commit()

        # تحديث FTS
        if self.config.enable_fts:
            self._update_fts_index(root)

        # تحديث الكاش
        self._cache[root.root_string] = root

    def _run_command_root_update(
        self, root_string: str, updated_root: ArabicRoot, db_data: tuple
    ) -> None:
        """تنفيذ عملية تحديث الجذر في قاعدة البيانات"""
        query = """
            UPDATE roots SET 
                radicals = ?, root_type = ?, weakness_type = ?, 
                hamza_type = ?, semantic_field = ?, frequency = ?,
                phonetic_features = ?, morpho_patterns = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE root = ?
        """

        self._conn.run_command(query, db_data)
        self._conn.commit()

        # تحديث الكاش
        self._cache[root_string] = updated_root

    def create_root(self, root: ArabicRoot, overwrite: bool = False) -> bool:
        """
        إنشاء جذر جديد

        Args:
            root: الجذر المراد إضافته
            overwrite: الكتابة فوق الموجود

        Returns:
            bool: نجح الإنشاء أم لا
        """
        begin_time = time.time()

        try:
            if not overwrite:
                self._validate_root_existence(root.root_string, should_exist=False)

            db_data = self._prepare_root_data_for_db(root)
            self._run_command_root_insert(root, db_data)

            self._record_operation_time(begin_time)
            return True

        except Exception as e:
            print(f"خطأ في إنشاء الجذر {root.root_string}: {e}")
            return False

    def read_root(self, root_string: str) -> Optional[ArabicRoot]:
        """
        قراءة جذر محدد

        Args:
            root_string: نص الجذر

        Returns:
            ArabicRoot أو None
        """
        begin_time = time.time()

        # فحص الكاش أولاً
        if root_string in self._cache:
            self._cache_hits += 1
            return self._cache[root_string]

        self._cache_misses += 1

        try:
            query = """
                SELECT * FROM roots WHERE root = ?
            """

            cursor = self._conn.run_command(query, (root_string,))

            if row := cursor.fetchone():
                root = self._deserialize_root(dict(row))
                # إضافة للكاش
                self._cache[root_string] = root

                # تحديث استخدام
                self._update_usage_count(root_string)

                self._record_operation_time(begin_time)

                return root

            return None

        except Exception as e:
            print(f"خطأ في قراءة الجذر {root_string}: {e}")
            return None

    def update_root(self, root_string: str, updated_root: ArabicRoot) -> bool:
        """
        تحديث جذر موجود

        Args:
            root_string: الجذر المراد تحديثه
            updated_root: البيانات الجديدة

        Returns:
            bool: نجح التحديث أم لا
        """
        if not self.read_root(root_string):
            return False

        try:
            db_data = self._prepare_root_data_for_update(updated_root, root_string)
            self._run_command_root_update(root_string, updated_root, db_data)

            return True

        except Exception as e:
            print(f"خطأ في تحديث الجذر {root_string}: {e}")
            return False

    def delete_root(self, root_string: str) -> bool:
        """
        حذف جذر

        Args:
            root_string: الجذر المراد حذفه

        Returns:
            bool: نجح الحذف أم لا
        """
        try:
            # حذف من قاعدة البيانات
            cursor = self._conn.run_command(
                "DELETE FROM roots WHERE root = ?", (root_string,)
            )
            self._conn.commit()

            # فحص إن كان تم حذف صف واحد على الأقل
            deleted = cursor.rowcount > 0

            if deleted:
                # حذف من الكاش
                self._cache.pop(root_string, None)

            return deleted

        except Exception as e:
            print(f"خطأ في حذف الجذر {root_string}: {e}")
            return False

    # =========================================================================
    # Advanced Search Operations - عمليات البحث المتقدمة
    # =========================================================================

    def search_by_pattern(self, pattern: str, limit: int = 100) -> List[ArabicRoot]:
        """
        البحث بالنمط مع دعم wildcards

        Args:
            pattern: النمط (مثل: ك*ب، *عل، ق??)
            limit: عدد النتائج القصوى

        Returns:
            قائمة الجذور المطابقة
        """
        begin_time = time.time()

        # تحويل النمط إلى SQL LIKE
        sql_pattern = pattern.replace("*", "%").replace("?", "_")

        try:
            query = """
                SELECT * FROM roots 
                WHERE root LIKE ? 
                ORDER BY frequency DESC, usage_count DESC 
                LIMIT ?
            """

            cursor = self._conn.run_command(query, (sql_pattern, limit))
            results = [self._deserialize_root(dict(row)) for row in cursor.fetchall()]

            self._record_operation_time(begin_time)

            return results

        except Exception as e:
            print(f"خطأ في البحث بالنمط {pattern}: {e}")
            return []

    def search_by_semantic_field(
        self, field: str, fuzzy: bool = False
    ) -> List[ArabicRoot]:
        """
        البحث بالمجال الدلالي

        Args:
            field: المجال الدلالي
            fuzzy: بحث ضبابي (يشمل الحقول المشابهة)

        Returns:
            قائمة الجذور في هذا المجال
        """
        try:
            if fuzzy:
                query = """
                    SELECT * FROM roots 
                    WHERE semantic_field LIKE ? 
                    ORDER BY frequency DESC
                """
                cursor = self._conn.run_command(query, (f"%{field}%",))
            else:
                query = """
                    SELECT * FROM roots 
                    WHERE semantic_field = ? 
                    ORDER BY frequency DESC
                """
                cursor = self._conn.run_command(query, (field,))

            return [self._deserialize_root(dict(row)) for row in cursor.fetchall()]

        except Exception as e:
            print(f"خطأ في البحث بالمجال الدلالي {field}: {e}")
            return []

    def search_by_properties(self, **filters) -> List[ArabicRoot]:
        """
        البحث المتقدم بالخصائص

        Args:
            **filters: مرشحات البحث
                - root_type: نوع الجذر
                - weakness_type: نوع الإعلال
                - hamza_type: نوع الهمز
                - min_frequency: أقل تكرار
                - has_semantic_field: له مجال دلالي

        Returns:
            قائمة الجذور المطابقة
        """
        conditions = []
        params = []

        for key, value in filters.items():
            if key == "root_type" and value:
                conditions.append("root_type = ?")
                params.append(value.value if hasattr(value, "value") else value)
            elif key == "weakness_type" and value:
                conditions.append("weakness_type = ?")
                params.append(value)
            elif key == "hamza_type" and value:
                conditions.append("hamza_type = ?")
                params.append(value)
            elif key == "min_frequency" and value is not None:
                conditions.append("frequency >= ?")
                params.append(value)
            elif key == "has_semantic_field" and value:
                conditions.append("semantic_field IS NOT NULL")

        if not conditions:
            return self.list_all_roots()

        query = f"""
            SELECT * FROM roots 
            WHERE {' AND '.join(conditions)}
            ORDER BY frequency DESC, usage_count DESC
        """

        try:
            cursor = self._conn.run_command(query, params)
            return [self._deserialize_root(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"خطأ في البحث بالخصائص: {e}")
            return []

    def fulltext_search(self, query: str, limit: int = 50) -> List[ArabicRoot]:
        """
        البحث النصي الكامل (FTS)

        Args:
            query: نص البحث
            limit: عدد النتائج القصوى

        Returns:
            قائمة الجذور مرتبة حسب الصلة
        """
        if not self.config.enable_fts:
            # بحث بديل بسيط
            return self.search_by_pattern(f"*{query}*", limit)

        try:
            fts_query = """
                SELECT r.* FROM roots r
                JOIN roots_fts fts ON r.id = fts.rowid
                WHERE roots_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """

            cursor = self._conn.run_command(fts_query, (query, limit))
            return [self._deserialize_root(dict(row)) for row in cursor.fetchall()]

        except Exception as e:
            print(f"خطأ في البحث النصي الكامل: {e}")
            return []

    # =========================================================================
    # Bulk Operations - العمليات المجمعة
    # =========================================================================

    def bulk_import_data_json(
        self, json_file: Union[str, Path], overwrite: bool = False
    ) -> Dict[str, int]:
        """
        استيراد مجمع من ملف JSON

        Args:
            json_file: مسار ملف JSON
            overwrite: الكتابة فوق الموجود

        Returns:
            إحصائيات الاستيراد {import_dataed, skipped, errors}
        """
        stats = {"import_dataed": 0, "skipped": 0, "errors": 0}

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.import_data(f)

            roots_data = data.get("roots", data) if isinstance(data, dict) else data

            for root_info in roots_data:
                try:
                    if isinstance(root_info, dict):
                        root_str = root_info.get("root", root_info.get("root_string"))
                        semantic_field = root_info.get("semantic_field")
                    else:
                        root_str = str(root_info)
                        semantic_field = None

                    if not root_str:
                        stats["errors"] += 1
                        continue

                    # فحص الوجود
                    if not overwrite and self.read_root(root_str):
                        stats["skipped"] += 1
                        continue

                    # إنشاء الجذر
                    root = create_root(root_str, semantic_field)
                    if self.create_root(root, overwrite=overwrite):
                        stats["import_dataed"] += 1
                    else:
                        stats["errors"] += 1

                except Exception as e:
                    print(f"خطأ في استيراد جذر: {e}")
                    stats["errors"] += 1

            return stats

        except Exception as e:
            print(f"خطأ في الاستيراد المجمع: {e}")
            return stats

    def store_data_to_json(
        self, output_file: Union[str, Path], include_metadata: bool = True
    ) -> bool:
        """
        تصدير إلى ملف JSON

        Args:
            output_file: مسار ملف الإخراج
            include_metadata: تضمين البيانات الوصفية

        Returns:
            bool: نجح التصدير أم لا
        """
        try:
            roots = self.list_all_roots()

            store_data_data: Dict[str, Any] = {"roots": [root.to_dict() for root in roots]}

            if include_metadata:
                store_data_data["metadata"] = {
                    "store_dataed_at": datetime.now().isoformat(),
                    "total_roots": len(roots),
                    "database_version": "1.0",
                    "statistics": self.get_comprehensive_statistics(),
                }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(store_data_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"خطأ في التصدير: {e}")
            return False

    # =========================================================================
    # Statistics and Analytics - الإحصائيات والتحليلات
    # =========================================================================

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """إحصائيات شاملة لقاعدة البيانات"""
        try:
            # إحصائيات أساسية
            basic_stats = self._conn.run_command(
                """
                SELECT 
                    COUNT(*) as total_roots,
                    COUNT(CASE WHEN weakness_type IS NULL THEN 1 END) as sound_roots,
                    COUNT(CASE WHEN weakness_type IS NOT NULL THEN 1 END) as weak_roots,
                    COUNT(CASE WHEN hamza_type IS NOT NULL THEN 1 END) as hamzated_roots,
                    AVG(frequency) as avg_frequency,
                    AVG(confidence_score) as avg_confidence
                FROM roots
            """
            ).fetchone()

            # توزيع أنواع الجذور
            type_distribution = {
                row[0]: row[1]
                for row in self._conn.run_command(
                    "SELECT root_type, COUNT(*) FROM roots GROUP BY root_type"
                )
            }

            # توزيع المجالات الدلالية
            semantic_distribution = {
                row[0]: row[1]
                for row in self._conn.run_command(
                    """
                    SELECT semantic_field, COUNT(*) 
                    FROM roots 
                    WHERE semantic_field IS NOT NULL 
                    GROUP BY semantic_field 
                    ORDER BY COUNT(*) DESC
                """
                )
            }

            # إحصائيات الأداء
            cache_hit_ratio = (
                (self._cache_hits / (self._cache_hits + self._cache_misses))
                if (self._cache_hits + self._cache_misses) > 0
                else 0
            )

            return {
                "basic_statistics": dict(basic_stats),
                "type_distribution": type_distribution,
                "semantic_distribution": semantic_distribution,
                "performance": {
                    "cache_hit_ratio": cache_hit_ratio,
                    "cache_size": len(self._cache),
                    "avg_query_time": (
                        sum(self._query_times) / len(self._query_times)
                        if self._query_times
                        else 0
                    ),
                    "total_queries": len(self._query_times),
                },
                "storage": {
                    "database_size_mb": (
                        self.config.db_path.stat().st_size / (1024 * 1024)
                        if self.config.db_path.exists()
                        else 0
                    ),
                    "cache_memory_usage": len(self._cache) * 1024,  # تقدير تقريبي
                },
            }

        except Exception as e:
            print(f"خطأ في حساب الإحصائيات: {e}")
            return {}

    # =========================================================================
    # Utility Methods - الطرق المساعدة
    # =========================================================================

    def list_all_roots(self, limit: Optional[int] = None) -> List[ArabicRoot]:
        """إرجاع جميع الجذور"""
        query = "SELECT * FROM roots ORDER BY frequency DESC, usage_count DESC"
        if limit:
            query += f" LIMIT {limit}"

        try:
            cursor = self._conn.run_command(query)
            return [self._deserialize_root(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"خطأ في جلب جميع الجذور: {e}")
            return []

    def clear_cache(self):
        """مسح الكاش"""
        self._cache.clear()
        self._query_cache.clear()

    def optimize_database(self):
        """تحسين قاعدة البيانات"""
        try:
            self._conn.run_command("VACUUM")
            self._conn.run_command("ANALYZE")
            self._conn.commit()
            print("✅ تم تحسين قاعدة البيانات")
        except Exception as e:
            print(f"خطأ في تحسين قاعدة البيانات: {e}")

    def backup_database(self, backup_path: Optional[Path] = None) -> bool:
        """إنشاء نسخة احتياطية"""
        backup_path = backup_path or self.config.json_backup_path
        return self.store_data_to_json(backup_path, include_metadata=True)

    def _serialize_root(self, root: ArabicRoot) -> Dict:
        """تحويل الجذر إلى قاموس قابل للتخزين"""
        serialized_radicals = []
        for radical in root.radicals:
            radical_dict = asdict(radical)
            # تحويل RadicalType إلى string للتسلسل في JSON
            if "type" in radical_dict and hasattr(radical_dict["type"], "value"):
                radical_dict["type"] = radical_dict["type"].value
            serialized_radicals.append(radical_dict)

        return {
            "radicals": serialized_radicals,
            "phonetic_features": getattr(root, "phonetic_features", {}),
            "morpho_patterns": getattr(root, "morpho_patterns", []),
        }

    def _deserialize_root(self, row_data: Dict) -> ArabicRoot:
        """إعادة بناء الجذر من بيانات قاعدة البيانات"""
        root_str = row_data["root"]
        semantic_field = row_data.get("semantic_field")
        frequency = row_data.get("frequency", 0)

        # إنشاء جذر أساسي
        root = create_root(root_str, semantic_field)
        root.frequency = frequency

        return root

    def _update_fts_index(self, root: ArabicRoot):
        """تحديث فهرس البحث النصي الكامل"""
        if self.config.enable_fts:
            try:
                self._conn.run_command(
                    """
                    INSERT OR REPLACE INTO roots_fts(rowid, root, radicals, semantic_field)
                    SELECT id, root, radicals, semantic_field FROM roots WHERE root = ?
                """,
                    (root.root_string,),
                )
                self._conn.commit()
            except Exception as e:
                print(f"خطأ في تحديث FTS: {e}")

    def _update_usage_count(self, root_string: str):
        """تحديث عداد الاستخدام"""
        try:
            self._conn.run_command(
                """
                UPDATE roots SET usage_count = usage_count + 1 
                WHERE root = ?
            """,
                (root_string,),
            )
            self._conn.commit()
        except Exception as e:
            print(f"خطأ في تحديث الاستخدام: {e}")

    def extract_possible_roots(self, word: str) -> List[Dict]:
        """
        Extract possible roots from a word using pattern matching
        مطابق لما كان في الإصدار القديم لضمان التوافق
        """
        possible_roots = []

        # Simple pattern-based extraction
        if len(word) >= 3:
            # Try exact match first
            exact_root = self.read_root(word)
            if exact_root:
                possible_roots.append(
                    {"root": exact_root, "confidence": 1.0, "method": "exact_match"}
                )

            # Try pattern matching
            pattern_results = self.search_by_pattern(f"{word[0]}*{word[-1]}")
            for root in pattern_results[:5]:  # Limit results
                if hasattr(root, "radicals"):
                    root_radicals = getattr(root, "radicals", [])
                    # Convert to list for comparison if needed
                    if isinstance(root_radicals, str):
                        root_radicals = list(root_radicals)
                    if isinstance(word, str):
                        word_list = list(word)
                    else:
                        word_list = word
                    if root_radicals != word_list:  # Avoid duplicates
                        confidence = 0.7 if len(root_radicals) == len(word) else 0.5
                        possible_roots.append(
                            {
                                "root": root,
                                "confidence": confidence,
                                "method": "pattern_matching",
                            }
                        )

            # Try fuzzy matching for short words
            if len(word) == 3:
                fts_results = self.fulltext_search(word, limit=3)
                for root in fts_results:
                    if all(
                        getattr(pr["root"], "radicals", [])
                        != getattr(root, "radicals", [])
                        for pr in possible_roots
                    ):
                        possible_roots.append(
                            {"root": root, "confidence": 0.4, "method": "fuzzy_search"}
                        )

        return possible_roots

    def _record_operation_time(self, begin_time: float):
        """Helper method to record operation time"""
        query_time = time.time() - begin_time
        self._record_query_time(query_time)

    def _record_query_time(self, query_time: float):
        """تسجيل وقت الاستعلام للإحصائيات"""
        self._query_times.append(query_time)
        # الاحتفاظ بآخر 1000 استعلام فقط
        if len(self._query_times) > 1000:
            self._query_times.pop(0)

    def _get_root_count(self) -> int:
        """عدد الجذور في قاعدة البيانات"""
        return self._conn.run_command("SELECT COUNT(*) FROM roots").fetchone()[0]

    def _populate_sample_data(self):
        """تعبئة بيانات نموذجية"""
        from ..models.roots import_data SAMPLE_ROOTS

        for root_str, meaning in SAMPLE_ROOTS.items():
            try:
                root = create_root(root_str, meaning)
                self.create_root(root)
            except Exception as e:
                print(f"خطأ في إضافة الجذر النموذجي {root_str}: {e}")

    def _update_stats(self):
        """تحديث الإحصائيات"""
        self._stats.total_roots = self._get_root_count()
        self._stats.storage_size_mb = (
            self.config.db_path.stat().st_size / (1024 * 1024)
            if self.config.db_path.exists()
            else 0
        )

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def close(self):
        """إغلاق قاعدة البيانات"""
        if self.config.auto_backup:
            self.backup_database()

        self._conn.close()

    def __enter__(self) -> "EnhancedRootDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        return self._get_root_count()

    def __contains__(self, root_string: str) -> bool:
        return self.read_root(root_string) is not None

# =============================================================================
# Factory Functions - دوال الإنشاء
# =============================================================================

def create_enhanced_database(
    db_path: Optional[str] = None, config: Optional[DatabaseConfig] = None
) -> EnhancedRootDatabase:
    """إنشاء قاعدة بيانات مطورة مع إعدادات مخصصة"""
    if config is None:
        config = DatabaseConfig()

    if db_path:
        config.db_path = Path(db_path)

    return EnhancedRootDatabase(config)

def create_memory_database() -> EnhancedRootDatabase:
    """إنشاء قاعدة بيانات في الذاكرة للاختبار"""
    config = DatabaseConfig(db_path=Path(":memory:"))
    return EnhancedRootDatabase(config)

# =============================================================================
# Example Usage - أمثلة الاستخدام
# =============================================================================

if __name__ == "__main__":
    # مثال على الاستخدام الأساسي
    print("🗄️ اختبار قاعدة البيانات المطورة...")

    with create_enhanced_database() as db:
        # إحصائيات أولية
        print(f"📊 عدد الجذور: {len(db)}")

        # اختبار البحث
        results = db.search_by_pattern("ك*ب")
        print(f"🔍 نتائج البحث بنمط 'ك*ب': {len(results)}")

        # اختبار البحث بالخصائص
        weak_roots = db.search_by_properties(weakness_type="معتل")
        print(f"📋 الجذور المعتلة: {len(weak_roots)}")

        # إحصائيات شاملة
        stats = db.get_comprehensive_statistics()
        print(f"📈 إجمالي الإحصائيات: {stats}")

        print("✅ قاعدة البيانات المطورة تعمل بنجاح!")
