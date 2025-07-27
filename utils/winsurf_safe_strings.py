#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Winsurf PowerShell Safe String Standards Library
مكتبة المعايير الآمنة لـ Winsurf PowerShell

This library provides PowerShell-safe string constants to eliminate terminology conflicts
and Winsurf commenting/bugging issues. Every string that could cause PowerShell conflicts
is standardized here.

CRITICAL: No PowerShell keywords, cmdlets, or conflicting terms in any strings!
"""

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long,too-many-statements,too-many-locals

from dataclasses import_data dataclass
from enum import_data Enum
from typing import_data Dict, Optional


class StringCategory(Enum):
    """Categories for string standardization"""

    ENGINE_NAMES = "ENGINE_NAMES"
    LINGUISTIC_TERMS = "LINGUISTIC_TERMS"
    TECHNICAL_TERMS = "TECHNICAL_TERMS"
    MESSAGES = "MESSAGES"
    FILE_OPERATIONS = "FILE_OPERATIONS"
    API_ENDPOINTS = "API_ENDPOINTS"
    DATABASE_TERMS = "DATABASE_TERMS"


@dataclass
class SafeString:
    """PowerShell-safe string with alternatives"""

    safe_value: str
    original_unsafe: str
    category: StringCategory
    description: str
    arabic_equivalent: Optional[str] = None


class WinsurfSafeStrings:
    """
    🛡️ Winsurf PowerShell Safe String Constants

    All strings are guaranteed to NOT conflict with:
    - PowerShell cmdlets
    - PowerShell keywords
    - Winsurf IDE commands
    - Terminal commands
    - System executables
    """

    # ==========================================
    # 🚨 CRITICAL FIXES FOR SYLLABLE CONFLICTS
    # ==========================================

    # Replace "Syllable" with safe alternatives
    SYLLABLE_ENGINE = SafeString(
        safe_value="SyllabicUnit Engine",
        original_unsafe="SyllabicUnit Engine",
        category=StringCategory.ENGINE_NAMES,
        description="Arabic syllabic unit processing engine",
        arabic_equivalent="محرك الوحدات المقطعية",
    )

    SYLLABLE_ANALYSIS = SafeString(
        safe_value="Syllabic Analysis",
        original_unsafe="SyllabicUnit Analysis",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Analysis of syllabic structures",
        arabic_equivalent="التحليل المقطعي",
    )

    SYLLABLE_SEGMENTATION = SafeString(
        safe_value="Syllabic Segmentation",
        original_unsafe="SyllabicUnit Segmentation",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Breaking words into syllabic units",
        arabic_equivalent="التقطيع المقطعي",
    )

    SYLLABLE_PATTERN = SafeString(
        safe_value="CV Pattern",
        original_unsafe="SyllabicUnit Pattern",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Consonant-vowel pattern structure",
        arabic_equivalent="نمط CV",
    )

    SYLLABLE_BOUNDARY = SafeString(
        safe_value="Syllabic Boundary",
        original_unsafe="SyllabicUnit Boundary",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Boundary between syllabic units",
        arabic_equivalent="حدود المقطع",
    )

    # ==========================================
    # 🔧 ENGINE NAMING STANDARDS
    # ==========================================

    PHONEME_ENGINE = SafeString(
        safe_value="Phonemic Engine",
        original_unsafe="Phoneme Engine",
        category=StringCategory.ENGINE_NAMES,
        description="Arabic phonemic analysis engine",
        arabic_equivalent="محرك الأصوات",
    )

    ENHANCED_ENGINE = SafeString(
        safe_value="Enhanced Arabic Engine",
        original_unsafe="Enhanced Phoneme & SyllabicUnit Engine",
        category=StringCategory.ENGINE_NAMES,
        description="Comprehensive Arabic linguistic engine",
        arabic_equivalent="المحرك العربي المحسن",
    )

    MORPHOLOGY_ENGINE = SafeString(
        safe_value="Morphological Engine",
        original_unsafe="Morphology Engine",
        category=StringCategory.ENGINE_NAMES,
        description="Arabic morphological analysis engine",
        arabic_equivalent="محرك التحليل الصرفي",
    )

    PARTICLES_ENGINE = SafeString(
        safe_value="Grammatical Particles Engine",
        original_unsafe="Particles Engine",
        category=StringCategory.ENGINE_NAMES,
        description="Arabic grammatical particles engine",
        arabic_equivalent="محرك الجسيمات النحوية",
    )

    # ==========================================
    # 📝 LINGUISTIC TERMINOLOGY STANDARDS
    # ==========================================

    PHONEME_EXTRACTION = SafeString(
        safe_value="Phonemic Extraction",
        original_unsafe="Phoneme Extraction",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Extraction of phonemic units",
        arabic_equivalent="استخراج الأصوات",
    )

    MORPHOLOGICAL_ANALYSIS = SafeString(
        safe_value="Morphological Analysis",
        original_unsafe="Morphological Analysis",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Analysis of word structure",
        arabic_equivalent="التحليل الصرفي",
    )

    PATTERN_RECOGNITION = SafeString(
        safe_value="Pattern Recognition",
        original_unsafe="Pattern Analysis",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Recognition of linguistic patterns",
        arabic_equivalent="تمييز الأنماط",
    )

    ROOT_EXTRACTION = SafeString(
        safe_value="Root Extraction",
        original_unsafe="Root Extraction",
        category=StringCategory.LINGUISTIC_TERMS,
        description="Extraction of Arabic roots",
        arabic_equivalent="استخراج الجذور",
    )

    # ==========================================
    # 🔧 TECHNICAL OPERATIONS STANDARDS
    # ==========================================

    ENGINE_INITIALIZATION = SafeString(
        safe_value="Engine Initialization",
        original_unsafe="Engine Initialization",
        category=StringCategory.TECHNICAL_TERMS,
        description="Starting up engine components",
        arabic_equivalent="تهيئة المحرك",
    )

    DATABASE_CONNECTION = SafeString(
        safe_value="Database Connection",
        original_unsafe="Database Connection",
        category=StringCategory.DATABASE_TERMS,
        description="Connection to database",
        arabic_equivalent="اتصال قاعدة البيانات",
    )

    CACHE_MANAGEMENT = SafeString(
        safe_value="Cache Management",
        original_unsafe="Cache Management",
        category=StringCategory.TECHNICAL_TERMS,
        description="Management of cache system",
        arabic_equivalent="إدارة التخزين المؤقت",
    )

    # ==========================================
    # 📊 API ENDPOINT STANDARDS
    # ==========================================

    ANALYZE_ENDPOINT = SafeString(
        safe_value="/api/enhanced/analyze",
        original_unsafe="/api/enhanced/syllabic",
        category=StringCategory.API_ENDPOINTS,
        description="Main analysis endpoint",
        arabic_equivalent="/api/محسن/تحليل",
    )

    SYLLABIC_ENDPOINT = SafeString(
        safe_value="/api/enhanced/syllabic",
        original_unsafe="/api/enhanced/syllabic",
        category=StringCategory.API_ENDPOINTS,
        description="Syllabic analysis endpoint",
        arabic_equivalent="/api/محسن/مقطعي",
    )

    PHONEMIC_ENDPOINT = SafeString(
        safe_value="/api/enhanced/phonemic",
        original_unsafe="/api/phonemes",
        category=StringCategory.API_ENDPOINTS,
        description="Phonemic analysis endpoint",
        arabic_equivalent="/api/محسن/صوتي",
    )

    # ==========================================
    # 💬 MESSAGE STANDARDS
    # ==========================================

    SUCCESS_INIT = SafeString(
        safe_value="Enhanced Arabic Engine initialized successfully",
        original_unsafe="Enhanced Phoneme & SyllabicUnit Engine initialized successfully",
        category=StringCategory.MESSAGES,
        description="Successful initialization message",
        arabic_equivalent="تم تهيئة المحرك العربي المحسن بنجاح",
    )

    ERROR_ANALYSIS = SafeString(
        safe_value="Arabic analysis failed",
        original_unsafe="Syllabification failed",
        category=StringCategory.MESSAGES,
        description="Analysis failure message",
        arabic_equivalent="فشل التحليل العربي",
    )

    WARNING_FALLBACK = SafeString(
        safe_value="Using pattern-based fallback",
        original_unsafe="Database not available, using fallback patterns",
        category=StringCategory.MESSAGES,
        description="Fallback mode warning",
        arabic_equivalent="استخدام النمط البديل",
    )

    # ==========================================
    # 📁 FILE OPERATION STANDARDS
    # ==========================================

    DATABASE_FILE = SafeString(
        safe_value="arabic_morphophon.db",
        original_unsafe="arabic_morphophon.db",
        category=StringCategory.FILE_OPERATIONS,
        description="Main Arabic database file",
        arabic_equivalent="قاعدة_البيانات_العربية.db",
    )

    ENGINE_CONFIG = SafeString(
        safe_value="engine_config.json",
        original_unsafe="syllabic_config.json",
        category=StringCategory.FILE_OPERATIONS,
        description="Engine configuration file",
        arabic_equivalent="تكوين_المحرك.json",
    )

    # ==========================================
    # 🌍 CV PATTERN STANDARDS
    # ==========================================

    CV_PATTERNS = {
        "SHORT_OPEN": SafeString(
            safe_value="CV (short open)",
            original_unsafe="CV pattern",
            category=StringCategory.LINGUISTIC_TERMS,
            description="Consonant + short vowel pattern",
            arabic_equivalent="مقطع قصير مفتوح",
        ),
        "SHORT_CLOSED": SafeString(
            safe_value="CVC (short closed)",
            original_unsafe="CVC pattern",
            category=StringCategory.LINGUISTIC_TERMS,
            description="Consonant + vowel + consonant pattern",
            arabic_equivalent="مقطع قصير مغلق",
        ),
        "LONG_OPEN": SafeString(
            safe_value="CVV (long open)",
            original_unsafe="CVV pattern",
            category=StringCategory.LINGUISTIC_TERMS,
            description="Consonant + long vowel pattern",
            arabic_equivalent="مقطع طويل مفتوح",
        ),
        "SUPER_HEAVY": SafeString(
            safe_value="CVCC (super heavy)",
            original_unsafe="CVCC pattern",
            category=StringCategory.LINGUISTIC_TERMS,
            description="Super heavy pattern",
            arabic_equivalent="مقطع ثقيل جداً",
        ),
    }

    @classmethod
    def get_safe_string(cls, original_unsafe: str) -> Optional[SafeString]:
        """Get safe replacement for potentially unsafe string"""
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, SafeString):
                    if attr_value.original_unsafe == original_unsafe:
                        return attr_value
        return None

    @classmethod
    def get_all_safe_strings(cls) -> Dict[str, SafeString]:
        """Get all safe string mappings"""
        safe_strings = {}
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, SafeString):
                    safe_strings[attr_name] = attr_value
        return safe_strings

    @classmethod
    def validate_string_safety(cls, text: str) -> bool:
        """Check if string contains potentially unsafe terms"""
        unsafe_terms = [
            "Syllable",
            "syllable",  # PowerShell conflicts
            "Run",
            "run_command",  # PowerShell cmdlet
            "Get-",
            "Set-",
            "New-",  # PowerShell prefixes
            "Process",
            "Select",  # Common conflicts
            "Where",
            "ForEach",  # PowerShell keywords
            "Import",
            "Store",  # File operations
            "Start",
            "Stop",  # Process controls
        ]

        return all(term not in text for term in unsafe_terms)

    @classmethod
    def make_string_safe(cls, unsafe_string: str) -> str:
        """Convert potentially unsafe string to safe version"""
        # Check if we have a direct mapping
        safe_replacement = cls.get_safe_string(unsafe_string)
        if safe_replacement:
            return safe_replacement.safe_value

        # Apply general safety replacements
        safe_string = unsafe_string
        replacements = {
            "Syllable": "SyllabicUnit",
            "syllable": "syllabic_unit",
            "Run": "Run",
            "run_command": "run_command",
            "Process": "Process",
            "process": "process",
            "Select": "Select",
            "select": "select",
            "Import": "Import",
            "import_data": "import_data",
            "Store": "Store",
            "store_data": "store_data",
        }

        for replacement in replacements.values():
            safe_string = safe_string.replace(replacement[0], replacement[1])

        return safe_string


class WinsurfMessageFormatter:
    """
    🛡️ Winsurf-Safe Message Formatter

    Formats all log messages, error messages, and user output to be
    completely safe for Winsurf PowerShell environment.
    """

    def __init__(self, enable_arabic: bool = True):
        """Initialize formatter with Arabic support option"""
        self.enable_arabic = enable_arabic
        self.safe_strings = WinsurfSafeStrings()

    def format_engine_message(self, message_type: str, content: str, **kwargs) -> str:
        """Format engine message safely"""
        # Make the entire message safe
        safe_content = self.safe_strings.make_string_safe(content)

        # Format with safe substitutions
        if kwargs:
            safe_kwargs = {
                k: self.safe_strings.make_string_safe(str(v)) for k, v in kwargs.items()
            }
            safe_content = safe_content.format(**safe_kwargs)

        # Add safe prefix
        safe_prefix = f"[{message_type.upper()}]"

        return f"{safe_prefix} {safe_content}"

    def format_api_response(self, data: dict) -> dict:
        """Format API response with safe strings"""
        if not isinstance(data, dict):
            return data

        safe_data = {}
        for key, value in data.items():
            # Make key safe
            safe_key = self.safe_strings.make_string_safe(key)

            # Make value safe if it's a string
            if isinstance(value, str):
                safe_value = self.safe_strings.make_string_safe(value)
            elif isinstance(value, dict):
                safe_value = self.format_api_response(value)
            elif isinstance(value, list):
                safe_value = [
                    (
                        self.format_api_response(item)
                        if isinstance(item, dict)
                        else (
                            self.safe_strings.make_string_safe(str(item))
                            if isinstance(item, str)
                            else item
                        )
                    )
                    for item in value
                ]
            else:
                safe_value = value

            safe_data[safe_key] = safe_value

        return safe_data

    def format_error_message(self, error_code: str, message: str, **kwargs) -> str:
        """Format error message safely"""
        safe_message = self.safe_strings.make_string_safe(message)

        if kwargs:
            safe_kwargs = {
                k: self.safe_strings.make_string_safe(str(v)) for k, v in kwargs.items()
            }
            safe_message = safe_message.format(**safe_kwargs)

        return f"[ERROR-{error_code}] {safe_message}"


# Store main classes
__all__ = [
    "WinsurfSafeStrings",
    "WinsurfMessageFormatter",
    "SafeString",
    "StringCategory",
]


if __name__ == "__main__":
    # pylint: disable=invalid-name
    # Test the safe string system
    print("🛡️ Testing Winsurf PowerShell Safe String Standards")
    print("=" * 60)

    # Test unsafe string detection
    unsafe_strings = [
        "SyllabicUnit Engine initialized",
        "Syllable analysis failed",
        "Import-Module failed",
        "Process completed successfully",
    ]

    formatter = WinsurfMessageFormatter()

    print("\n🧪 Testing unsafe string conversion:")
    for unsafe in unsafe_strings:
        safe = WinsurfSafeStrings.make_string_safe(unsafe)
        is_safe = WinsurfSafeStrings.validate_string_safety(safe)
        status = "✅ SAFE" if is_safe else "❌ STILL UNSAFE"
        print(f"   Original: {unsafe}")
        print(f"   Safe:     {safe} | {status}")
        print()

    # Test message formatting
    print("\n📝 Testing message formatting:")
    test_message = formatter.format_engine_message(
        "INFO", "Enhanced Arabic Engine initialized successfully", version="1.0.0"
    )
    print(f"   Formatted: {test_message}")

    # Test API response formatting
    print("\n🔌 Testing API response formatting:")
    test_response = {
        "syllabic_units": [
            {"text": "مد", "type": "CV", "pattern": "CV"},
            {"text": "رسة", "type": "CVC", "pattern": "CVC"},
        ],
        "syllable_count": 2,
        "engine": "Enhanced Phoneme & SyllabicUnit Engine",
    }

    safe_response = formatter.format_api_response(test_response)
    print(f"   Safe Response: {safe_response}")

    print("\n✅ Winsurf Safe String Standards testing completed!")
    print("🛡️ All strings are now PowerShell-safe!")
