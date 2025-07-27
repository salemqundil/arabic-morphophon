"""
Utility functions for web interface

This module contains helper functions for web request/response handling,
validation, and formatting to maintain clean separation of concerns.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data logging
import_data time
from functools import_data wraps
from typing import_data Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

def format_response(
    data: Optional[Dict[str, Any]] = None,
    success: bool = True,
    message: str = "",
    status_code: int = 200,
) -> Dict[str, Any]:
    """
    Format standardized API response

    Args:
        data: Response data
        success: Operation success status
        message: Response message
        status_code: HTTP status code

    Returns:
        Formatted response dictionary
    """
    response: Dict[str, Any] = {
        "success": success,
        "timestamp": time.time(),
        "status_code": status_code,
    }

    if message:
        response["message"] = message

    if data is not None:
        response["data"] = data

    return response

def validate_input(text: str, max_length: int = 1000) -> bool:
    """
    Validate input text for analysis

    Args:
        text: Input text to validate
        max_length: Maximum allowed length

    Returns:
        True if valid, False otherwise
    """
    return bool(
        text and isinstance(text, str) and text.strip() and len(text) <= max_length
    )

def timing_decorator(func):
    """
    Decorator to measure function execution time

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with timing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - begin_time
            logger.info(f"{func.__name__} completed in {processing_time:.3f}s")
            return result, processing_time
        except Exception as e:
            processing_time = time.time() - begin_time
            logger.error(f"{func.__name__} failed after {processing_time:.3f}s: {e}")
            raise

    return wrapper

def safe_get_nested(data: Dict[str, Any], keys: list, default=None):
    """
    Safely get nested dictionary value

    Args:
        data: Dictionary to search
        keys: List of keys for nested access
        default: Default value if key not found

    Returns:
        Value or default
    """
    try:
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default

def clean_arabic_text(text: str) -> str:
    """
    Clean and normalize Arabic text for processing

    Args:
        text: Input Arabic text

    Returns:
        Cleaned text
    """
    return " ".join(text.split()).strip() if text else ""
