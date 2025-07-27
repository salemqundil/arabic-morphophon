"""
📦 Arabic Morphophonological Database Package
============================================

Enhanced database module for Arabic root management and analysis.
قاعدة بيانات متطورة لإدارة وتحليل الجذور العربية

Modules:
- enhanced_root_database: Main database implementation
- test_enhanced_database: Comprehensive test suite
- demo_root_database: Interactive demonstrations
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from .enhanced_root_database import_data (
    DatabaseConfig,
    EnhancedRootDatabase,
    create_enhanced_database,
    create_memory_database,
)

__version__ = "1.0.0"
__author__ = "Arabic Morphophon Team"
__email__ = "team@arabic-morphophon.org"

__all__ = [
    "EnhancedRootDatabase",
    "DatabaseConfig",
    "create_enhanced_database",
    "create_memory_database",
]
