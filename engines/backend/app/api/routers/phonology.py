"""
Phonology API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from app.services.phonology import PhonologyService
from app.core.models import TextRequest

router = APIRouter()
phonology_service = PhonologyService()


class PhonologicalAnalysisRequest(TextRequest):
    apply_recursively: bool = Field(
        default=False, description="Whether to apply rules recursively"
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations for recursive rule application",
    )


class PhonologicalRuleRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    rule_name: str = Field(..., description="Name of the rule to apply")


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration to update")


@router.post("/process", summary="Process text through phonological rules")
async def process_text(request: PhonologicalAnalysisRequest):
    """
    Process Arabic text through phonological rules

    This endpoint applies the phonological rules to the provided text and returns
    the processed result along with detailed analysis.
    """
    try:
        return phonology_service.process_text(
            request.text,
            apply_recursively=request.apply_recursively,
            max_iterations=request.max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@router.get("/rules", summary="Get available phonological rules")
async def get_rules():
    """
    Get a list of available phonological rules

    Returns information about all available phonological rules in the system,
    including their names, descriptions, and examples.
    """
    return phonology_service.get_rules()


@router.post("/apply-rule", summary="Apply a specific phonological rule")
async def apply_rule(request: PhonologicalRuleRequest):
    """
    Apply a specific phonological rule to the provided text

    This allows testing individual rules without going through the entire rule chain.
    """
    try:
        return phonology_service.apply_rule(request.text, request.rule_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply rule: {str(e)}")


@router.get("/status", summary="Get phonological engine status")
async def get_status():
    """
    Get current status of the phonological engine

    Returns information about enabled rules, statistics, and configuration.
    """
    return phonology_service.get_status()


@router.post("/update-config", summary="Update phonological engine configuration")
async def update_config(request: ConfigUpdateRequest):
    """
    Update the configuration of the phonological engine

    This allows dynamic reconfiguration of the engine without restarting the service.
    """
    try:
        return phonology_service.update_config(request.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/reset-statistics", summary="Reset phonological engine statistics")
async def reset_statistics():
    """Reset all processing statistics in the phonological engine"""
    return phonology_service.reset_statistics()
