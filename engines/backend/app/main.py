"""
Main FastAPI application for Arabic Morphophonology System
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import routers
from app.api.routers import phonology, derivation, syllable, morphology
from app.core.config import settings
from app.services.auth import get_current_user

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app.main")

# Initialize the FastAPI app
app = FastAPI(
    title="Arabic Morphophonology API",
    description="Enterprise-grade Arabic NLP processing API",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    phonology.router,
    prefix="/api/v1/phonology",
    tags=["phonology"],
    dependencies=[Depends(get_current_user)] if settings.REQUIRE_AUTH else [],
)
app.include_router(
    derivation.router,
    prefix="/api/v1/derivation",
    tags=["derivation"],
    dependencies=[Depends(get_current_user)] if settings.REQUIRE_AUTH else [],
)
app.include_router(
    syllable.router,
    prefix="/api/v1/syllable",
    tags=["syllable"],
    dependencies=[Depends(get_current_user)] if settings.REQUIRE_AUTH else [],
)
app.include_router(
    morphology.router,
    prefix="/api/v1/morphology",
    tags=["morphology"],
    dependencies=[Depends(get_current_user)] if settings.REQUIRE_AUTH else [],
)


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Welcome to Arabic Morphophonology API",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "status": "error",
        "message": "An unexpected error occurred",
        "detail": str(exc),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
