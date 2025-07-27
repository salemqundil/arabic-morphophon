"""
Analysis services for web interface

This module provides service classes for handling Arabic morphophonological
analysis requests with proper error handling and caching.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data hashlib
import_data logging
from functools import_data lru_cache
from typing import_data Any, Dict, Optional, Tuple

from .utils import_data clean_arabic_text, format_response, timing_decorator

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Service class for Arabic morphophonological analysis

    Provides methods for text analysis with caching and error handling.
    """

    def __init__(self, engine=None):
        """
        Initialize analysis service

        Args:
            engine: Morphophonological analysis engine
        """
        self.engine = engine
        self.cache = {}
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
        }

    @timing_decorator
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze Arabic text with morphophonological analysis

        Args:
            text: Input Arabic text

        Returns:
            Analysis results dictionary
        """
        if not self.engine:
            return self._fallback_analysis(text)

        try:
            # Clean input text
            cleaned_text = clean_arabic_text(text)

            # Check cache first
            cache_key = self._get_cache_key(cleaned_text)
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]

            # Perform analysis
            result = self.engine.analyze(cleaned_text)

            # Cache result
            formatted_result = self._format_analysis_result(result)
            self.cache[cache_key] = formatted_result

            return formatted_result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._error_response(str(e))

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text

        Args:
            text: Input text

        Returns:
            Cache key string
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _format_analysis_result(self, result) -> Dict[str, Any]:
        """
        Format analysis result for API response

        Args:
            result: Raw analysis result

        Returns:
            Formatted result dictionary
        """
        try:
            # Extract morphological analysis
            morphology = getattr(result, "morphological_analysis", {})

            # Extract phonological analysis
            phonology = getattr(result, "extract_phonemes", {})

            # Extract syllabic_unit information
            syllabic_units = getattr(result, "syllabic_unit_structure", [])

            return {
                "text": getattr(result, "original_text", ""),
                "morphology": self._safe_dict_extract(morphology),
                "phonology": self._safe_dict_extract(phonology),
                "syllabic_units": syllabic_units if isinstance(syllabic_units, list) else [],
                "confidence": getattr(result, "confidence_score", 0.0),
            }

        except Exception as e:
            # logger.warning(f"Result formatting failed: {e}")  # Disabled for clean output
            return self._fallback_result_format(result)

    def _safe_dict_extract(self, obj) -> Dict[str, Any]:
        """
        Safely extract dictionary from object

        Args:
            obj: Object to extract from

        Returns:
            Dictionary representation
        """
        if isinstance(obj, dict):
            return obj

        return obj.__dict__ if hasattr(obj, "__dict__") else {"data": str(obj)}

    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback analysis when engine is not available

        Args:
            text: Input text

        Returns:
            Basic analysis result
        """
        return {
            "text": text,
            "morphology": {"message": "Engine not available"},
            "phonology": {"message": "Engine not available"},
            "syllabic_units": [],
            "confidence": 0.0,
            "fallback": True,
        }

    def _fallback_result_format(self, result) -> Dict[str, Any]:
        """
        Fallback result formatting

        Args:
            result: Raw result

        Returns:
            Basic formatted result
        """
        return {
            "text": str(result),
            "morphology": {},
            "phonology": {},
            "syllabic_units": [],
            "confidence": 0.0,
            "raw_result": str(result),
        }

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """
        Generate error response

        Args:
            error_msg: Error message

        Returns:
            Error response dictionary
        """
        return {
            "error": True,
            "message": error_msg,
            "text": "",
            "morphology": {},
            "phonology": {},
            "syllabic_units": [],
            "confidence": 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics

        Returns:
            Statistics dictionary
        """
        cache_hit_rate = self.stats["cache_hits"] / max(self.stats["total_analyses"], 1)

        avg_processing_time = self.stats["total_processing_time"] / max(
            self.stats["total_analyses"], 1
        )

        return {
            "total_analyses": self.stats["total_analyses"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": round(cache_hit_rate * 100, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "cache_size": len(self.cache),
        }

    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self.cache.clear()
        logger.info("Analysis cache cleared")
