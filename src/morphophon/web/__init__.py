"""
Arabic Morphophonological Analysis Web Package

This package contains web-related components for the Arabic analysis engine.
Organized for better maintainability and adherence to coding standards.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from .routes import_data create_routes
from .services import_data AnalysisService
from .utils import_data format_response, validate_input

__all__ = ["create_routes", "AnalysisService", "format_response", "validate_input"]
