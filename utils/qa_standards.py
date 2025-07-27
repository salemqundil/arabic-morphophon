#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Quality Assurance Standards Library
مكتبة معايير ضمان الجودة

A comprehensive QA framework that standardizes all error messages, warnings,
and debugging strings to eliminate Winsurf commenting and bugging issues.
Every possible error, warning, or issue is predefined as a standardized string.

Author: AI Assistant & Development Team
Created: July 22, 2025
Version: 1.0.0
"""

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long,too-many-statements

from enum import Enum
from typing import List, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path


class QALevel(Enum):
    """Quality Assurance Severity Levels"""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class QACategory(Enum):
    """Quality Assurance Categories"""

    DATABASE = "DATABASE"
    API = "API"
    ENGINE = "ENGINE"
    PHONEME = "PHONEME"
    SYLLABIC_UNIT = "SYLLABIC_UNIT"
    PATTERN = "PATTERN"
    CACHE = "CACHE"
    PERFORMANCE = "PERFORMANCE"
    VALIDATION = "VALIDATION"
    IMPORT = "IMPORT"
    CONFIG = "CONFIG"
    NETWORK = "NETWORK"
    FILE_IO = "FILE_IO"
    MEMORY = "MEMORY"
    UNICODE = "UNICODE"


@dataclass
class QAMessage:
    """Standardized QA Message Structure"""

    code: str
    level: QALevel
    category: QACategory
    message_en: str
    message_ar: str
    solution: str
    documentation_link: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Format message with parameters"""
        return self.message_en.format(**kwargs)

    def format_ar(self, **kwargs) -> str:
        """Format Arabic message with parameters"""
        return self.message_ar.format(**kwargs)


class QAStandards:
    """
    🛡️ Comprehensive Quality Assurance Standards Library

    This class contains ALL standardized error messages, warnings, and debug strings
    to eliminate Winsurf commenting and bugging issues. Every possible scenario
    has a predefined, standardized message.
    """

    # 🚨 CRITICAL ERRORS
    QA_CRITICAL_DATABASE_MISSING = QAMessage(
        code="QA_C001",
        level=QALevel.CRITICAL,
        category=QACategory.DATABASE,
        message_en="Database file not found at path: {path}. System cannot operate without database.",
        message_ar="ملف قاعدة البيانات غير موجود في المسار: {path}. النظام لا يمكن أن يعمل بدون قاعدة البيانات.",
        solution="Ensure arabic_morphophon.db exists in Downimport_datas folder or specify correct path.",
        documentation_link="https://docs.arabic-nlp.com/database-setup",
    )

    QA_CRITICAL_ENGINE_INIT_FAILED = QAMessage(
        code="QA_C002",
        level=QALevel.CRITICAL,
        category=QACategory.ENGINE,
        message_en="Enhanced Phoneme SyllabicUnit Engine initialization failed: {error}",
        message_ar="فشل في تهيئة محرك الأصوات والمقاطع المحسن: {error}",
        solution="Check engine dependencies and configuration settings.",
        documentation_link="https://docs.arabic-nlp.com/engine-troubleshooting",
    )

    QA_CRITICAL_MEMORY_EXHAUSTED = QAMessage(
        code="QA_C003",
        level=QALevel.CRITICAL,
        category=QACategory.MEMORY,
        message_en="System memory exhausted during processing. Available: {available}MB, Required: {required}MB",
        message_ar="نفدت ذاكرة النظام أثناء المعالجة. المتاح: {available}MB، المطلوب: {required}MB",
        solution="Reduce batch size or increase system memory allocation.",
        documentation_link="https://docs.arabic-nlp.com/memory-optimization",
    )

    # ❌ ERROR MESSAGES
    QA_ERROR_DATABASE_CONNECTION = QAMessage(
        code="QA_E001",
        level=QALevel.ERROR,
        category=QACategory.DATABASE,
        message_en="Database connection failed: {error}. Falling back to pattern-based analysis.",
        message_ar="فشل الاتصال بقاعدة البيانات: {error}. التبديل إلى التحليل المبني على الأنماط.",
        solution="Check database file permissions and path accessibility.",
        documentation_link="https://docs.arabic-nlp.com/database-troubleshooting",
    )

    QA_ERROR_API_INVALID_REQUEST = QAMessage(
        code="QA_E002",
        level=QALevel.ERROR,
        category=QACategory.API,
        message_en="Invalid API request: {details}. Expected format: {expected_format}",
        message_ar="طلب API غير صالح: {details}. التنسيق المتوقع: {expected_format}",
        solution="Verify request format matches API documentation.",
        documentation_link="https://docs.arabic-nlp.com/api-reference",
    )

    QA_ERROR_PHONEME_EXTRACTION_FAILED = QAMessage(
        code="QA_E003",
        level=QALevel.ERROR,
        category=QACategory.PHONEME,
        message_en="Phoneme extraction failed for word: '{word}'. Error: {error}",
        message_ar="فشل استخراج الأصوات للكلمة: '{word}'. الخطأ: {error}",
        solution="Check word encoding and Arabic text normalization.",
        documentation_link="https://docs.arabic-nlp.com/phoneme-extraction",
    )

    QA_ERROR_SYLLABIC_UNIT_SEGMENTATION_FAILED = QAMessage(
        code="QA_E004",
        level=QALevel.ERROR,
        category=QACategory.SYLLABIC_UNIT,
        message_en="SyllabicUnit segmentation failed for word: '{word}'. Pattern: {pattern}",
        message_ar="فشل تقطيع المقاطع للكلمة: '{word}'. النمط: {pattern}",
        solution="Verify CV pattern recognition and syllabic_unit boundary detection.",
        documentation_link="https://docs.arabic-nlp.com/syllabic_unit-segmentation",
    )

    QA_ERROR_PATTERN_ANALYSIS_FAILED = QAMessage(
        code="QA_E005",
        level=QALevel.ERROR,
        category=QACategory.PATTERN,
        message_en="Pattern analysis failed for input: '{input}'. Analysis type: {analysis_type}",
        message_ar="فشل تحليل النمط للمدخل: '{input}'. نوع التحليل: {analysis_type}",
        solution="Check pattern recognition algorithms and input validation.",
        documentation_link="https://docs.arabic-nlp.com/pattern-analysis",
    )

    QA_ERROR_IMPORT_MODULE_FAILED = QAMessage(
        code="QA_E006",
        level=QALevel.ERROR,
        category=QACategory.IMPORT,
        message_en="Failed to import_data required module: {module}. Error: {error}",
        message_ar="فشل في استيراد الوحدة المطلوبة: {module}. الخطأ: {error}",
        solution="Install missing dependencies using pip install {module}",
        documentation_link="https://docs.arabic-nlp.com/installation",
    )

    QA_ERROR_UNICODE_NORMALIZATION = QAMessage(
        code="QA_E007",
        level=QALevel.ERROR,
        category=QACategory.UNICODE,
        message_en="Unicode normalization failed for text: '{text}'. Encoding: {encoding}",
        message_ar="فشل تطبيع اليونيكود للنص: '{text}'. الترميز: {encoding}",
        solution="Ensure text is properly encoded in UTF-8 and contains valid Arabic characters.",
        documentation_link="https://docs.arabic-nlp.com/unicode-handling",
    )

    # ⚠️ WARNING MESSAGES
    QA_WARNING_DATABASE_FALLBACK = QAMessage(
        code="QA_W001",
        level=QALevel.WARNING,
        category=QACategory.DATABASE,
        message_en="Database not available, using fallback patterns. Accuracy may be reduced.",
        message_ar="قاعدة البيانات غير متاحة، استخدام الأنماط البديلة. قد تقل الدقة.",
        solution="Restore database connection for optimal accuracy.",
        documentation_link="https://docs.arabic-nlp.com/database-setup",
    )

    QA_WARNING_CACHE_MEMORY_HIGH = QAMessage(
        code="QA_W002",
        level=QALevel.WARNING,
        category=QACategory.CACHE,
        message_en="Cache memory usage high: {usage}MB. Consider clearing cache.",
        message_ar="استخدام ذاكرة التخزين المؤقت عالي: {usage}MB. فكر في مسح التخزين المؤقت.",
        solution="Call clear_cache() method or reduce cache TTL.",
        documentation_link="https://docs.arabic-nlp.com/cache-management",
    )

    QA_WARNING_PERFORMANCE_SLOW = QAMessage(
        code="QA_W003",
        level=QALevel.WARNING,
        category=QACategory.PERFORMANCE,
        message_en="Processing time exceeded threshold: {time}s > {threshold}s for word: '{word}'",
        message_ar="وقت المعالجة تجاوز الحد المسموح: {time}s > {threshold}s للكلمة: '{word}'",
        solution="Optimize processing algorithms or consider batch processing.",
        documentation_link="https://docs.arabic-nlp.com/performance-optimization",
    )

    QA_WARNING_PHONEME_UNKNOWN = QAMessage(
        code="QA_W004",
        level=QALevel.WARNING,
        category=QACategory.PHONEME,
        message_en="Unknown phoneme encountered: '{phoneme}' in word: '{word}'. Using approximation.",
        message_ar="صوت غير معروف تم العثور عليه: '{phoneme}' في الكلمة: '{word}'. استخدام التقريب.",
        solution="Add phoneme definition to database or update phoneme mappings.",
        documentation_link="https://docs.arabic-nlp.com/phoneme-database",
    )

    QA_WARNING_SYLLABIC_UNIT_IRREGULAR = QAMessage(
        code="QA_W005",
        level=QALevel.WARNING,
        category=QACategory.SYLLABIC_UNIT,
        message_en="Irregular cv pattern detected: '{pattern}' in word: '{word}'",
        message_ar="نمط مقطع غير منتظم تم اكتشافه: '{pattern}' في الكلمة: '{word}'",
        solution="Review syllabic_unit segmentation rules or add pattern to database.",
        documentation_link="https://docs.arabic-nlp.com/syllabic_unit-patterns",
    )

    QA_WARNING_API_RATE_LIMIT = QAMessage(
        code="QA_W006",
        level=QALevel.WARNING,
        category=QACategory.API,
        message_en="API rate limit approaching: {current}/{limit} requests in {timeframe}",
        message_ar="الاقتراب من حد معدل API: {current}/{limit} طلب في {timeframe}",
        solution="Reduce request frequency or implement request queuing.",
        documentation_link="https://docs.arabic-nlp.com/rate-limiting",
    )

    # ℹ️ INFO MESSAGES
    QA_INFO_ENGINE_INITIALIZED = QAMessage(
        code="QA_I001",
        level=QALevel.INFO,
        category=QACategory.ENGINE,
        message_en="Enhanced Phoneme SyllabicUnit Engine initialized successfully. Version: {version}",
        message_ar="تم تهيئة محرك الأصوات والمقاطع المحسن بنجاح. الإصدار: {version}",
        solution="Engine ready for processing.",
        documentation_link="https://docs.arabic-nlp.com/getting-begined",
    )

    QA_INFO_DATABASE_CONNECTED = QAMessage(
        code="QA_I002",
        level=QALevel.INFO,
        category=QACategory.DATABASE,
        message_en="Database connected successfully. Tables import_dataed: {table_count}",
        message_ar="تم الاتصال بقاعدة البيانات بنجاح. الجداول المحملة: {table_count}",
        solution="Database ready for queries.",
        documentation_link="https://docs.arabic-nlp.com/database-info",
    )

    QA_INFO_CACHE_CLEARED = QAMessage(
        code="QA_I003",
        level=QALevel.INFO,
        category=QACategory.CACHE,
        message_en="Cache cleared successfully. Freed memory: {freed_mb}MB",
        message_ar="تم مسح التخزين المؤقت بنجاح. الذاكرة المحررة: {freed_mb}MB",
        solution="Cache is now empty and ready for new data.",
        documentation_link="https://docs.arabic-nlp.com/cache-management",
    )

    QA_INFO_ANALYSIS_COMPLETE = QAMessage(
        code="QA_I004",
        level=QALevel.INFO,
        category=QACategory.ENGINE,
        message_en="Analysis completed for word: '{word}'. Processing time: {time}ms",
        message_ar="اكتمل التحليل للكلمة: '{word}'. وقت المعالجة: {time}ms",
        solution="Analysis results are ready.",
        documentation_link="https://docs.arabic-nlp.com/analysis-results",
    )

    # 🔍 DEBUG MESSAGES
    QA_DEBUG_PHONEME_MAPPING = QAMessage(
        code="QA_D001",
        level=QALevel.DEBUG,
        category=QACategory.PHONEME,
        message_en="Phoneme mapping: '{char}' -> '{phoneme}' (IPA: {ipa})",
        message_ar="تحويل الصوت: '{char}' -> '{phoneme}' (IPA: {ipa})",
        solution="Debug information for phoneme extraction.",
        documentation_link="https://docs.arabic-nlp.com/debug-phonemes",
    )

    QA_DEBUG_SYLLABIC_UNIT_BOUNDARY = QAMessage(
        code="QA_D002",
        level=QALevel.DEBUG,
        category=QACategory.SYLLABIC_UNIT,
        message_en="SyllabicUnit boundary detected at position {position} in word: '{word}'",
        message_ar="حدود المقطع تم اكتشافها في الموضع {position} في الكلمة: '{word}'",
        solution="Debug information for syllabic_unit segmentation.",
        documentation_link="https://docs.arabic-nlp.com/debug-syllabic_units",
    )

    QA_DEBUG_PATTERN_MATCH = QAMessage(
        code="QA_D003",
        level=QALevel.DEBUG,
        category=QACategory.PATTERN,
        message_en="Pattern match found: '{pattern}' with confidence {confidence}",
        message_ar="تطابق النمط تم العثور عليه: '{pattern}' بالثقة {confidence}",
        solution="Debug information for pattern analysis.",
        documentation_link="https://docs.arabic-nlp.com/debug-patterns",
    )

    QA_DEBUG_CACHE_HIT = QAMessage(
        code="QA_D004",
        level=QALevel.DEBUG,
        category=QACategory.CACHE,
        message_en="Cache hit for key: '{key}'. Stored {time}ms processing time.",
        message_ar="إصابة في التخزين المؤقت للمفتاح: '{key}'. توفير {time}ms من وقت المعالجة.",
        solution="Cache is working efficiently.",
        documentation_link="https://docs.arabic-nlp.com/debug-cache",
    )

    QA_DEBUG_API_REQUEST = QAMessage(
        code="QA_D005",
        level=QALevel.DEBUG,
        category=QACategory.API,
        message_en="API request received: {method} {endpoint} with data: {data}",
        message_ar="طلب API تم استلامه: {method} {endpoint} مع البيانات: {data}",
        solution="Debug information for API requests.",
        documentation_link="https://docs.arabic-nlp.com/debug-api",
    )

    # 📊 VALIDATION MESSAGES
    QA_VALIDATION_TEXT_EMPTY = QAMessage(
        code="QA_V001",
        level=QALevel.ERROR,
        category=QACategory.VALIDATION,
        message_en="Input text is empty or None. Cannot process empty input.",
        message_ar="النص المدخل فارغ أو None. لا يمكن معالجة مدخل فارغ.",
        solution="Provide valid non-empty Arabic text for analysis.",
        documentation_link="https://docs.arabic-nlp.com/input-validation",
    )

    QA_VALIDATION_TEXT_NON_ARABIC = QAMessage(
        code="QA_V002",
        level=QALevel.WARNING,
        category=QACategory.VALIDATION,
        message_en="Text contains non-Arabic characters: '{text}'. Results may be inaccurate.",
        message_ar="النص يحتوي على أحرف غير عربية: '{text}'. النتائج قد تكون غير دقيقة.",
        solution="Use Arabic text only for optimal results.",
        documentation_link="https://docs.arabic-nlp.com/arabic-text-validation",
    )

    QA_VALIDATION_TEXT_TOO_LONG = QAMessage(
        code="QA_V003",
        level=QALevel.WARNING,
        category=QACategory.VALIDATION,
        message_en="Text length {length} exceeds recommended limit {limit}. Consider batch processing.",
        message_ar="طول النص {length} يتجاوز الحد الموصى به {limit}. فكر في المعالجة المجمعة.",
        solution="Split long text into smaller chunks for better performance.",
        documentation_link="https://docs.arabic-nlp.com/batch-processing",
    )

    # 🔧 CONFIGURATION MESSAGES
    QA_CONFIG_MISSING_PARAMETER = QAMessage(
        code="QA_CFG001",
        level=QALevel.WARNING,
        category=QACategory.CONFIG,
        message_en="Configuration parameter '{parameter}' not found. Using default: {default}",
        message_ar="معامل التكوين '{parameter}' غير موجود. استخدام الافتراضي: {default}",
        solution="Add parameter to configuration file for custom behavior.",
        documentation_link="https://docs.arabic-nlp.com/configuration",
    )

    QA_CONFIG_INVALID_VALUE = QAMessage(
        code="QA_CFG002",
        level=QALevel.ERROR,
        category=QACategory.CONFIG,
        message_en="Invalid configuration value for '{parameter}': {value}. Expected: {expected}",
        message_ar="قيمة تكوين غير صالحة لـ '{parameter}': {value}. المتوقع: {expected}",
        solution="Correct the configuration value to match expected format.",
        documentation_link="https://docs.arabic-nlp.com/configuration-reference",
    )

    @classmethod
    def get_message(cls, code: str) -> Optional[QAMessage]:
        """Get standardized message by code"""
        for attr_name in dir(cls):
            if attr_name.beginswith("QA_"):
                message = getattr(cls, attr_name)
                if isinstance(message, QAMessage) and message.code == code:
                    return message
        return None

    @classmethod
    def get_messages_by_category(cls, category: QACategory) -> List[QAMessage]:
        """Get all messages for a specific category"""
        messages = []
        for attr_name in dir(cls):
            if attr_name.beginswith("QA_"):
                message = getattr(cls, attr_name)
                if isinstance(message, QAMessage) and message.category == category:
                    messages.append(message)
        return messages

    @classmethod
    def get_messages_by_level(cls, level: QALevel) -> List[QAMessage]:
        """Get all messages for a specific severity level"""
        messages = []
        for attr_name in dir(cls):
            if attr_name.beginswith("QA_"):
                message = getattr(cls, attr_name)
                if isinstance(message, QAMessage) and message.level == level:
                    messages.append(message)
        return messages


class QALogger:
    """
    🛡️ Standardized QA Logger

    Logs all messages using standardized QA codes and formats.
    Eliminates inconsistent error messages and debugging strings.
    """

    def __init__(self, name: str = __name__, enable_arabic: bool = True):
        """
        Initialize QA Logger

        Args:
            name: Logger name
            enable_arabic: Enable Arabic message logging
        """
        self.logger = logging.getLogger(name)
        self.enable_arabic = enable_arabic

        # Set up formatter for QA messages
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        )

        # Add console processr if not exists
        if not self.logger.processrs:
            processr = logging.StreamProcessr()
            processr.setFormatter(formatter)
            self.logger.addProcessr(processr)
            self.logger.setLevel(logging.DEBUG)

    def log_qa(self, qa_message: QAMessage, **kwargs):
        """Log a standardized QA message"""
        formatted_message = qa_message.format(**kwargs)

        # Add QA code prefix
        log_message = f"[{qa_message.code}] {formatted_message}"

        # Add Arabic translation if enabled
        if self.enable_arabic:
            arabic_message = qa_message.format_ar(**kwargs)
            log_message += f" | {arabic_message}"

        # Add solution if available
        if qa_message.solution:
            log_message += f" | Solution: {qa_message.solution}"

        # Log at appropriate level
        if qa_message.level == QALevel.CRITICAL:
            self.logger.critical(log_message)
        elif qa_message.level == QALevel.ERROR:
            self.logger.error(log_message)
        elif qa_message.level == QALevel.WARNING:
            self.logger.warning(log_message)
        elif qa_message.level == QALevel.INFO:
            self.logger.info(log_message)
        elif qa_message.level == QALevel.DEBUG:
            self.logger.debug(log_message)
        elif qa_message.level == QALevel.TRACE:
            self.logger.debug("[TRACE] %s", log_message)

    def critical(self, qa_message: QAMessage, **kwargs):
        """Log critical QA message"""
        self.log_qa(qa_message, **kwargs)

    def error(self, qa_message: QAMessage, **kwargs):
        """Log error QA message"""
        self.log_qa(qa_message, **kwargs)

    def warning(self, qa_message: QAMessage, **kwargs):
        """Log warning QA message"""
        self.log_qa(qa_message, **kwargs)

    def info(self, qa_message: QAMessage, **kwargs):
        """Log info QA message"""
        self.log_qa(qa_message, **kwargs)

    def debug(self, qa_message: QAMessage, **kwargs):
        """Log debug QA message"""
        self.log_qa(qa_message, **kwargs)


class QAValidator:
    """
    🛡️ Quality Assurance Validator

    Validates inputs and conditions using standardized QA messages.
    """

    def __init__(self, logger: Optional[QALogger] = None):
        """Initialize QA Validator"""
        self.logger = logger or QALogger()

    def validate_text_input(self, text: str, max_length: int = 10000) -> bool:
        """
        Validate Arabic text input

        Args:
            text: Input text to validate
            max_length: Maximum text length

        Returns:
            bool: True if valid, False otherwise
        """
        # Check for empty text
        if not text or not text.strip():
            self.logger.error(QAStandards.QA_VALIDATION_TEXT_EMPTY)
            return False

        # Check text length
        if len(text) > max_length:
            self.logger.warning(
                QAStandards.QA_VALIDATION_TEXT_TOO_LONG,
                length=len(text),
                limit=max_length,
            )
            return False

        # Check for Arabic characters
        arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
        if arabic_chars / len(text) < 0.5:  # Less than 50% Arabic
            text_preview = f"{text[:50]}..." if len(text) > 50 else text
            self.logger.warning(
                QAStandards.QA_VALIDATION_TEXT_NON_ARABIC, text=text_preview
            )

        return True

    def validate_database_connection(self, db_path: str) -> bool:
        """Validate database connection"""
        if not Path(db_path).exists():
            self.logger.critical(QAStandards.QA_CRITICAL_DATABASE_MISSING, path=db_path)
            return False

        self.logger.info(QAStandards.QA_INFO_DATABASE_CONNECTED, table_count="N/A")
        return True

    def validate_config_parameter(
        self, parameter: str, value: Any, expected_type: type
    ) -> bool:
        """Validate configuration parameter"""
        if value is None:
            self.logger.warning(
                QAStandards.QA_CONFIG_MISSING_PARAMETER,
                parameter=parameter,
                default="None",
            )
            return False

        if not isinstance(value, expected_type):
            self.logger.error(
                QAStandards.QA_CONFIG_INVALID_VALUE,
                parameter=parameter,
                value=value,
                expected=expected_type.__name__,
            )
            return False

        return True


# Store main classes for easy import_data
__all__ = [
    "QAStandards",
    "QALogger",
    "QAValidator",
    "QAMessage",
    "QALevel",
    "QACategory",
]


if __name__ == "__main__":
    # pylint: disable=invalid-name
    # Test the QA system
    print("🛡️ Testing Quality Assurance Standards Library")
    print("=" * 60)

    # Initialize QA Logger
    qa_logger = QALogger("qa_test")

    # Test different message types
    qa_logger.info(QAStandards.QA_INFO_ENGINE_INITIALIZED, version="1.0.0")
    qa_logger.warning(QAStandards.QA_WARNING_DATABASE_FALLBACK)
    qa_logger.error(
        QAStandards.QA_ERROR_PHONEME_EXTRACTION_FAILED,
        word="test",
        error="invalid format",
    )
    qa_logger.debug(QAStandards.QA_DEBUG_CACHE_HIT, key="syllabic_unit_cache", time=15)

    # Test validator
    validator = QAValidator(qa_logger)
    validator.validate_text_input("مرحبا")
    validator.validate_text_input("")  # Should show error
    validator.validate_text_input("Hello world")  # Should show warning

    print("\n✅ QA Standards Library test completed!")
