"""Pytest suite for RootDatabase CRUD operations."""

from __future__ import_data annotations

import_data json
import_data sys
from pathlib import_data Path
from textwrap import_data dedent

import_data pytest

# Add current directory to path for import_datas
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

# Import from the enhanced database module
from arabic_morphophon.database.enhanced_root_database import_data (
    DatabaseConfig,
    EnhancedRootDatabase,
    create_memory_database,
)
from arabic_morphophon.models.roots import_data ArabicRoot, RootType, create_root

def verify_root_exists(db: EnhancedRootDatabase, root_string: str) -> ArabicRoot:
    """Helper function to verify root exists and return it."""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

    updated = db.read_root(root_string)
    assert updated is not None
    return updated

@pytest.fixture()
def sample_roots() -> list[ArabicRoot]:
    """Sample Arabic roots for testing"""
    return [
        create_root("فعل", "الفعل والعمل"),
        create_root("قول", "القول والكلام"),
        create_root("كتب", "الكتابة والتدوين"),
        create_root("وعد", "الوعد والالتزام"),  # معتل مثال
        create_root("قرأ", "القراءة والتلاوة"),  # مهموز اللام
    ]

@pytest.fixture()
def db(tmp_path: Path) -> EnhancedRootDatabase:
    """Create a temporary database for testing"""
    config = DatabaseConfig(db_path=tmp_path / "test_roots.db")
    return EnhancedRootDatabase(config)

@pytest.fixture()
def memory_db() -> EnhancedRootDatabase:
    """Create an in-memory database for fast testing"""
    return create_memory_database()

class TestCRUDOperations:
    """Test basic CRUD operations"""

    def test_create_and_read(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test creating and reading roots"""
        first = sample_roots[0]

        # Test create
        assert memory_db.create_root(first) is True

        # Test read
        fetched = memory_db.read_root(first.root_string)
        assert fetched is not None
        assert fetched.root_string == first.root_string
        assert fetched.semantic_field == first.semantic_field

    def test_update(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test updating existing roots"""
        root = sample_roots[0]

        # Create initial root
        assert memory_db.create_root(root) is True

        # Modify and update
        root.semantic_field = "الفعل والعمل المحدث"
        root.frequency = 100

        assert memory_db.update_root(root.root_string, root) is True

        # Verify update
        updated = verify_root_exists(memory_db, root.root_string)
        assert updated.semantic_field == "الفعل والعمل المحدث"
        assert updated.frequency == 100

    def test_delete(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test deleting roots"""
        root = sample_roots[0]

        # Create root
        assert memory_db.create_root(root) is True
        assert memory_db.read_root(root.root_string) is not None

        # Delete root
        assert memory_db.delete_root(root.root_string) is True

        # Verify deletion
        assert memory_db.read_root(root.root_string) is None

    def test_duplicate_insert(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test handling duplicate inserts"""
        root = sample_roots[0]

        # First insert should succeed
        assert memory_db.create_root(root) is True

        # Second insert should fail without overwrite
        assert memory_db.create_root(root) is False

        # With overwrite should succeed
        root.semantic_field = "معنى محدث"
        assert memory_db.create_root(root, overwrite=True) is True

        # Verify update
        updated = verify_root_exists(memory_db, root.root_string)
        assert updated.semantic_field == "معنى محدث"

class TestSearchOperations:
    """Test advanced search operations"""

    def test_search_by_pattern(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test pattern-based search"""
        # Add all sample roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Test wildcard patterns
        results = memory_db.search_by_pattern("ق*")
        root_strings = [r.root_string for r in results]
        assert "قول" in root_strings
        assert "قرأ" in root_strings

        # Test single character wildcard
        results = memory_db.search_by_pattern("?عل")
        root_strings = [r.root_string for r in results]
        assert "فعل" in root_strings

    def test_search_by_semantic_field(
        self, memory_db: EnhancedRootDatabase, sample_roots
    ):
        """Test semantic field search"""
        # Add roots with specific semantic fields
        for root in sample_roots:
            memory_db.create_root(root)

        # Exact search
        results = memory_db.search_by_semantic_field("القول والكلام")
        assert len(results) == 1
        assert results[0].root_string == "قول"

        # Fuzzy search
        results = memory_db.search_by_semantic_field("قول", fuzzy=True)
        assert len(results) >= 1
        assert any(r.root_string == "قول" for r in results)

    def test_search_by_properties(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test property-based search"""
        # Add roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Search by root type
        trilateral = memory_db.search_by_properties(root_type=RootType.TRILATERAL)
        assert len(trilateral) >= 3  # Most roots should be trilateral

        # Search by weakness type
        weak_roots = memory_db.search_by_properties(weakness_type="مثال")
        weak_root_strings = [r.root_string for r in weak_roots]
        assert "وعد" in weak_root_strings  # وعد is مثال (weak initial)

    def test_fulltext_search(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test full-text search"""
        # Add roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Search for text in semantic fields
        results = memory_db.fulltext_search("كتاب")
        root_strings = [r.root_string for r in results]
        assert "كتب" in root_strings

    def test_list_all_roots(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test listing all roots"""
        # Add roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Test list all
        all_roots = memory_db.list_all_roots()
        assert len(all_roots) >= len(sample_roots)

        # Test with limit
        limited = memory_db.list_all_roots(limit=2)
        assert len(limited) == 2

class TestBulkOperations:
    """Test bulk import_data/store_data operations"""

    def test_bulk_import_data_store_data_json(
        self, memory_db: EnhancedRootDatabase, tmp_path: Path
    ):
        """Test bulk JSON import_data and store_data"""
        # Create test data
        test_data = {
            "roots": [
                {"root": "شكر", "semantic_field": "الشكر والامتنان"},
                {"root": "أكل", "semantic_field": "الأكل والطعام"},
                {"root": "نوم", "semantic_field": "النوم والراحة"},
            ]
        }

        # Write to JSON file
        json_file = tmp_path / "test_roots.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # Test import_data
        import_data_stats = memory_db.bulk_import_data_json(json_file)
        assert import_data_stats["import_dataed"] == 3
        assert import_data_stats["errors"] == 0

        # Verify import_dataed data
        assert memory_db.read_root("شكر") is not None
        assert memory_db.read_root("أكل") is not None
        assert memory_db.read_root("نوم") is not None

        # Test store_data
        store_data_file = tmp_path / "store_dataed_roots.json"
        store_data_success = memory_db.store_data_to_json(store_data_file, include_metadata=True)
        assert store_data_success is True
        assert store_data_file.exists()

        # Verify store_dataed data
        with open(store_data_file, "r", encoding="utf-8") as f:
            store_dataed_data = json.import_data(f)

        assert "roots" in store_dataed_data
        assert "metadata" in store_dataed_data
        assert len(store_dataed_data["roots"]) >= 3

        # Check specific roots
        store_dataed_root_strings = [root["root"] for root in store_dataed_data["roots"]]
        assert "شكر" in store_dataed_root_strings
        assert "أكل" in store_dataed_root_strings
        assert "نوم" in store_dataed_root_strings

    def test_backup_and_restore(
        self, memory_db: EnhancedRootDatabase, sample_roots, tmp_path: Path
    ):
        """Test database backup and restore functionality"""
        # Add sample data
        for root in sample_roots:
            memory_db.create_root(root)

        # Create backup
        backup_file = tmp_path / "backup.json"
        backup_success = memory_db.backup_database(backup_file)
        assert backup_success is True
        assert backup_file.exists()

        # Create new database and restore
        new_db = create_memory_database()
        restore_stats = new_db.bulk_import_data_json(backup_file)
        assert restore_stats["import_dataed"] >= len(sample_roots)

        # Verify restored data
        for original_root in sample_roots:
            restored_root = new_db.read_root(original_root.root_string)
            assert restored_root is not None
            assert restored_root.semantic_field == original_root.semantic_field

class TestStatisticsAndAnalytics:
    """Test statistics and analytics functionality"""

    def test_comprehensive_statistics(
        self, memory_db: EnhancedRootDatabase, sample_roots
    ):
        """Test comprehensive statistics generation"""
        # Add diverse roots
        for root in sample_roots:
            root.frequency = len(root.root_string) * 10  # Vary frequency
            memory_db.create_root(root)

        # Get statistics
        stats = memory_db.get_comprehensive_statistics()

        # Verify structure
        assert "basic_statistics" in stats
        assert "type_distribution" in stats
        assert "semantic_distribution" in stats
        assert "performance" in stats
        assert "storage" in stats

        # Verify basic statistics
        basic = stats["basic_statistics"]
        assert basic["total_roots"] >= len(sample_roots)
        assert basic["sound_roots"] >= 0
        assert basic["weak_roots"] >= 0
        assert basic["hamzated_roots"] >= 0

    def test_database_metrics(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test database metrics and performance tracking"""
        # Add roots and perform operations to generate metrics
        for root in sample_roots:
            memory_db.create_root(root)
            memory_db.read_root(root.root_string)  # Generate read operations

        # Perform searches to generate query metrics
        memory_db.search_by_pattern("*")
        memory_db.search_by_properties(root_type=RootType.TRILATERAL)

        # Get statistics
        stats = memory_db.get_comprehensive_statistics()

        # Verify performance metrics exist
        assert "performance" in stats
        perf = stats["performance"]
        assert "cache_hit_ratio" in perf
        assert "total_queries" in perf
        assert perf["total_queries"] >= 0

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_operations(self, memory_db: EnhancedRootDatabase):
        """Test handling of invalid operations"""
        # Test reading non-existent root
        result = memory_db.read_root("غير_موجود")
        assert result is None

        # Test updating non-existent root
        fake_root = create_root("وهمي", "غير موجود")
        update_result = memory_db.update_root("غير_موجود", fake_root)
        assert update_result is False

        # Test deleting non-existent root
        delete_result = memory_db.delete_root("غير_موجود")
        assert delete_result is False

    def test_empty_search_results(self, memory_db: EnhancedRootDatabase):
        """Test handling of empty search results"""
        # Search in empty database
        results = memory_db.search_by_pattern("*")
        assert isinstance(results, list)

        # Search for non-matching pattern
        results = memory_db.search_by_pattern("xyz")
        assert len(results) == 0

        # Search for non-existent semantic field
        results = memory_db.search_by_semantic_field("غير_موجود")
        assert len(results) == 0

    def test_malformed_data_handling(
        self, memory_db: EnhancedRootDatabase, tmp_path: Path
    ):
        """Test handling of malformed import_data data"""
        # Create malformed JSON
        malformed_data = {
            "roots": [
                {"root": "صحيح", "semantic_field": "بيانات صحيحة"},
                {"invalid": "بيانات خاطئة"},  # Missing required fields
                {"root": "", "semantic_field": "جذر فارغ"},  # Empty root
            ]
        }

        json_file = tmp_path / "malformed.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(malformed_data, f, ensure_ascii=False)

        # Import should process errors gracefully
        stats = memory_db.bulk_import_data_json(json_file)
        assert stats["import_dataed"] >= 1  # At least the valid one
        assert stats["errors"] >= 1  # At least the malformed ones

class TestPerformance:
    """Test performance characteristics"""

    def test_large_dataset_performance(self, memory_db: EnhancedRootDatabase):
        """Test performance with larger datasets"""
        import_data time

        # Generate test roots
        test_roots = []
        arabic_letters = ["ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز"]

        for i in range(100):  # 100 roots for performance test
            root_str = f"{arabic_letters[i % len(arabic_letters)]}{arabic_letters[(i+1) % len(arabic_letters)]}{arabic_letters[(i+2) % len(arabic_letters)]}"
            root = create_root(root_str, f"اختبار أداء {i}")
            test_roots.append(root)

        # Test bulk insertion performance
        begin_time = time.time()
        successful_inserts = 0
        for root in test_roots:
            try:
                if memory_db.create_root(root):
                    successful_inserts += 1
            except:
                pass  # Ignore duplicate errors for performance test

        insert_time = time.time() - begin_time

        # Performance assertions
        assert successful_inserts > 0
        assert insert_time < 10.0  # Should complete in under 10 seconds

        # Test search performance
        begin_time = time.time()
        results = memory_db.search_by_pattern("*")
        search_time = time.time() - begin_time

        assert len(results) >= successful_inserts
        assert search_time < 1.0  # Search should be fast

    def test_cache_performance(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test cache performance improvements"""
        # Add roots
        for root in sample_roots:
            memory_db.create_root(root)

        # First read (cold cache)
        import_data time

        begin_time = time.time()
        for root in sample_roots[:3]:
            memory_db.read_root(root.root_string)
        cold_time = time.time() - begin_time

        # Second read (warm cache)
        begin_time = time.time()
        for root in sample_roots[:3]:
            memory_db.read_root(root.root_string)
        warm_time = time.time() - begin_time

        # Cache should improve performance
        assert warm_time <= cold_time  # Warm cache should be faster or equal

class TestDatabaseIntegration:
    """Test database integration features"""

    def test_context_manager(self, tmp_path: Path):
        """Test database context manager functionality"""
        db_path = tmp_path / "context_test.db"

        # Use database in context manager
        with EnhancedRootDatabase(DatabaseConfig(db_path=db_path)) as db:
            root = create_root("اختبار", "اختبار السياق")
            assert db.create_root(root) is True
            assert db.read_root("اختبار") is not None

        # Database should be properly closed after context
        # Create new instance to verify data persistence
        with EnhancedRootDatabase(DatabaseConfig(db_path=db_path)) as db2:
            retrieved = db2.read_root("اختبار")
            assert retrieved is not None
            assert retrieved.semantic_field == "اختبار السياق"

    def test_database_optimization(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test database optimization features"""
        # Add data
        for root in sample_roots:
            memory_db.create_root(root)

        # Test optimization (should not raise errors)
        try:
            memory_db.optimize_database()
            optimization_success = True
        except Exception:
            optimization_success = False

        assert optimization_success

        # Test cache clearing
        memory_db.clear_cache()

        # Verify data still accessible after optimization
        for root in sample_roots:
            retrieved = memory_db.read_root(root.root_string)
            assert retrieved is not None

# Integration test for the complete workflow
def test_complete_workflow(tmp_path: Path):
    """Test complete database workflow from creation to store_data"""
    db_path = tmp_path / "workflow_test.db"

    with EnhancedRootDatabase(DatabaseConfig(db_path=db_path)) as db:
        # 1. Create diverse roots
        roots = [
            create_root("كتب", "الكتابة"),
            create_root("قرأ", "القراءة"),
            create_root("وعد", "الوعد"),
            create_root("سأل", "السؤال"),
        ]

        for root in roots:
            assert db.create_root(root) is True

        # 2. Perform searches
        pattern_results = db.search_by_pattern("*")
        assert len(pattern_results) >= 4

        weak_results = db.search_by_properties(weakness_type="مثال")
        assert len(weak_results) >= 1

        # 3. Update a root
        updated_root = db.read_root("كتب")
        assert updated_root is not None
        updated_root.frequency = 150
        assert db.update_root("كتب", updated_root) is True

        # 4. Store data
        store_data_file = tmp_path / "final_store_data.json"
        assert db.store_data_to_json(store_data_file, include_metadata=True) is True

        # 5. Get comprehensive statistics
        stats = db.get_comprehensive_statistics()
        assert stats["basic_statistics"]["total_roots"] >= 4

        # 6. Verify data integrity
        for root in roots:
            retrieved = db.read_root(root.root_string)
            assert retrieved is not None
            assert retrieved.root_string == root.root_string

if __name__ == "__main__":
    # Run tests if run_commandd directly
    pytest.main([__file__, "-v"])
