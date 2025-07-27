"""
Pytest configuration for Arabic Morphophonological Engine tests
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data sys
from pathlib import_data Path

import_data pytest

# Add the project root to Python path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

@pytest.fixture(scope="session")
def project_root():
    """Provide project root path for tests"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def sample_arabic_texts():
    """Provide sample Arabic texts for testing"""
    return [
        "أَكَلَ الوَلَدُ التُّفاحَ",
        "السَّلامُ عَلَيْكُم",
        "مَرْحَباً بِالعالَم",
        "نَحْنُ نَتَعَلَّمُ العَرَبِيَّة",
        "كتب",
        "قرأ",
        "درس"
    ]

@pytest.fixture(scope="session")
def sample_roots():
    """Provide sample Arabic roots for testing"""
    return ["كتب", "قرأ", "درس", "علم", "فهم"]

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
