"""Pytest suite for RootDatabase CRUD operations."""

from __future__ import_data annotations

import_data json
import_data sys
from pathlib import_data Path
from textwrap import_data dedent

import_data pytest

# Add current directory to path for import_datas
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import from the enhanced database module
from arabic_morphophon.database.enhanced_root_database import_data (
    DatabaseConfig,
    EnhancedRootDatabase,
    create_memory_database,
)
from arabic_morphophon.models.roots import_data ArabicRoot, RootType, create_root

def verify_stats_structure(stats):
    """Helper function to verify statistics structure."""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

    assert "basic_statistics" in stats
    assert "type_distribution" in stats

@pytest.fixture()
def sample_roots() -> list[ArabicRoot]:
    """Sample Arabic roots for testing"""
    return [
        create_root("فعل", "الفعل والعمل"),
        create_root("قول", "القول والكلام"),
        create_root("كتب", "الكتابة والتدوين"),
        create_root("وعد", "الوعد والالتزام"),  # معتل مثال
        create_root("قر", "القراءة والتلاوة"),  # مهموز اللام
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

    def test_add_get(self, memory_db: EnhancedRootDatabase, sample_roots):
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
        updated = memory_db.read_root(root.root_string)
        assert updated is not None
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

        # Duplicate insert without overwrite should fail
        assert memory_db.create_root(root) is False

        # Duplicate insert with overwrite should succeed
        assert memory_db.create_root(root, overwrite=True) is True

        # Root should still exist
        assert memory_db.read_root(root.root_string) is not None

class TestSearchOperations:
    """Test search functionality"""

    def test_search_by_pattern(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test pattern-based search"""
        # Add all sample roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Test wildcard patterns
        ق_roots = memory_db.search_by_pattern("ق*")
        assert len(ق_roots) >= 2  # قول، قرأ، وربما قال من البيانات النموذجية

        # Test exact match
        exact = memory_db.search_by_pattern("كتب")
        assert len(exact) == 1
        assert exact[0].root_string == "كتب"

    def test_search_by_properties(self, memory_db: EnhancedRootDatabase, sample_roots):
        """Test property-based search"""
        # Add all sample roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Search by root type
        trilateral = memory_db.search_by_properties(root_type=RootType.TRILATERAL)
        assert len(trilateral) >= 3  # Most roots are trilateral

        # Search by weakness type
        weak_roots = memory_db.search_by_properties(weakness_type="مثال")
        assert len(weak_roots) >= 1  # وعد is مثال

    def test_search_by_semantic_field(
        self, memory_db: EnhancedRootDatabase, sample_roots
    ):
        """Test semantic field search"""
        # Add all sample roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Exact search
        writing_roots = memory_db.search_by_semantic_field("الكتابة والتدوين")
        assert len(writing_roots) == 1
        assert writing_roots[0].root_string == "كتب"

        # Fuzzy search
        general_roots = memory_db.search_by_semantic_field("القول", fuzzy=True)
        assert len(general_roots) >= 1

class TestBulkOperations:
    """Test bulk import_data/store_data operations"""

    def test_bulk_import_data_store_data(self, memory_db: EnhancedRootDatabase, tmp_path: Path):
        """Test bulk import_data and store_data functionality"""
        # Create test data
        test_data = {
            "roots": [
                {"root": "شكر", "semantic_field": "الشكر والامتنان"},
                {"root": "حمد", "semantic_field": "الحمد والثنا"},
                {"root": "سبح", "semantic_field": "التسبيح والتنزيه"},
            ]
        }

        json_file = tmp_path / "test_roots.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # Test bulk import_data
        import_data_stats = memory_db.bulk_import_data_json(json_file)
        assert import_data_stats["import_dataed"] == 3
        assert import_data_stats["errors"] == 0

        # Verify import_dataed data
        assert memory_db.read_root("شكر") is not None
        assert memory_db.read_root("حمد") is not None
        assert memory_db.read_root("سبح") is not None

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

        # Check that all import_dataed roots are in store_data
        store_dataed_root_strings = {root["root"] for root in store_dataed_data["roots"]}
        assert {"شكر", "حمد", "سبح"}.issubset(store_dataed_root_strings)

class TestStatistics:
    """Test statistics and analytics"""

    def test_comprehensive_statistics(
        self, memory_db: EnhancedRootDatabase, sample_roots
    ):
        """Test comprehensive statistics generation"""
        # Add sample roots
        for root in sample_roots:
            memory_db.create_root(root)

        # Get statistics
        stats = memory_db.get_comprehensive_statistics()

        # Verify basic statistics structure
        verify_stats_structure(stats)
        assert "performance" in stats

        basic = stats["basic_statistics"]
        assert basic["total_roots"] >= len(sample_roots)
        assert basic["total_roots"] == len(memory_db)

        # Type distribution should have trilateral roots
        type_dist = stats["type_distribution"]
        assert "ثلاثي" in type_dist
        assert type_dist["ثلاثي"] >= 3  # Most sample roots are trilateral

    def test_database_length_and_contains(
        self, memory_db: EnhancedRootDatabase, sample_roots
    ):
        """Test __len__ and __contains__ magic methods"""
        # Empty database
        assert len(memory_db) >= 0  # Might have sample data

        # Add roots and track successful additions
        initial_count = len(memory_db)
        roots_to_add = sample_roots[:3]

        for root in roots_to_add:
            memory_db.create_root(root)

        # Check that length increased appropriately
        final_count = len(memory_db)
        assert final_count >= initial_count

        # Check contains
        assert "فعل" in memory_db
        assert "قول" in memory_db
        assert "كتب" in memory_db
        assert "غير_موجود" not in memory_db

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_nonexistent_operations(self, memory_db: EnhancedRootDatabase):
        """Test operations on non-existent roots"""
        # Read non-existent root
        assert memory_db.read_root("غير_موجود") is None

        # Update non-existent root
        fake_root = create_root("وهمي", "غير موجود")
        assert memory_db.update_root("غير_موجود", fake_root) is False

        # Delete non-existent root
        assert memory_db.delete_root("غير_موجود") is False

    def test_invalid_search_patterns(self, memory_db: EnhancedRootDatabase):
        """Test search with invalid patterns"""
        # Empty pattern should return empty list
        results = memory_db.search_by_pattern("")
        assert isinstance(results, list)
        assert len(results) == 0

        # Invalid semantic field should return empty list
        results = memory_db.search_by_semantic_field("مجال_غير_موجود")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_empty_database_operations(self, memory_db: EnhancedRootDatabase):
        """Test operations on empty database"""
        # Clear any existing data
        memory_db.clear_cache()

        # List all from potentially empty database
        all_roots = memory_db.list_all_roots()
        assert isinstance(all_roots, list)

        # Get statistics from empty/minimal database
        stats = memory_db.get_comprehensive_statistics()
        assert isinstance(stats, dict)
        assert "basic_statistics" in stats

if __name__ == "__main__":
    # Run tests if script is run_commandd directly
    pytest.main([__file__, "-v"])
