#!/usr/bin/env python3
"""
Setup Script for Arabic Morphophonology System

This script creates the necessary directory structure and starter files
for the backend application.
"""
import os
import sys
from pathlib import Path
import shutil

# Define the base directory
BASE_DIR = Path(__file__).parent

# Define the backend directory
BACKEND_DIR = BASE_DIR / "backend"

# Define the structure we want to create
STRUCTURE = {
    "api": {
        "endpoints": {
            "phonology.py": """from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ...models.phonology import (
    PhonologyRequest,
    PhonologyResponse,
    PhonemeRequest,
    PhonemeResponse
)
from ...services.phonology import PhonologyService
from ...core.dependencies import get_phonology_service

router = APIRouter(prefix="/phonology", tags=["phonology"])

@router.post("/process", response_model=PhonologyResponse)
async def process_text(
    request: PhonologyRequest,
    phonology_service: PhonologyService = Depends(get_phonology_service)
):
    """Process text using the phonology engine"""
    try:
        result = phonology_service.process_text(request.text, request.options)
        return PhonologyResponse(
            success=True,
            result=result,
            message="Text processed successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )

@router.post("/phonemes", response_model=PhonemeResponse)
async def extract_phonemes(
    request: PhonemeRequest,
    phonology_service: PhonologyService = Depends(get_phonology_service)
):
    """Extract phonemes from text"""
    try:
        phonemes = phonology_service.extract_phonemes(request.text)
        return PhonemeResponse(
            success=True,
            phonemes=phonemes,
            message="Phonemes extracted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting phonemes: {str(e)}"
        )
"""
        },
        "router.py": """from fastapi import APIRouter
from .endpoints import phonology

# Create the main API router
api_router = APIRouter()

# Include the endpoint routers
api_router.include_router(phonology.router)

# Add more routers here as needed
# api_router.include_router(syllable.router)
# api_router.include_router(derivation.router)
"""
    },
    "core": {
        "config.py": """from pydantic import BaseSettings, PostgresDsn, RedisDsn
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    # General settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Arabic Morphophonology System"
    VERSION: str = "1.0.0"

    # CORS settings
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Database settings
    DATABASE_URL: Optional[PostgresDsn] = None

    # Redis settings
    REDIS_URL: Optional[RedisDsn] = None

    # Security settings
    JWT_SECRET: str = "your_very_secure_jwt_secret_key_change_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 86400  # 24 hours in seconds

    # NLP Engine settings
    DEFAULT_MODEL: str = "advanced"
    CACHE_RESULTS: bool = True
    CACHE_EXPIRATION: int = 3600  # 1 hour in seconds

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
""",
        "dependencies.py": """from fastapi import Depends

from ..services.phonology import PhonologyService
from ..repositories.phonology import PhonologyRepository

def get_phonology_repository():
    """Get phonology repository instance"""
    return PhonologyRepository()

def get_phonology_service(
    repository: PhonologyRepository = Depends(get_phonology_repository)
):
    """Get phonology service instance"""
    return PhonologyService(repository)
""",
        "security.py": """from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError

from .config import settings
from ..models.auth import TokenPayload, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")

def create_access_token(*, sub: str) -> str:
    """Create a JWT access token"""
    payload = {
        "sub": sub,
        "exp": datetime.utcnow() + timedelta(seconds=settings.JWT_EXPIRATION),
        "iat": datetime.utcnow(),
    }
    encoded_jwt = jwt.encode(
        payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise credentials_exception

    # In a real application, you would look up the user in the database here
    user = User(username=token_data.sub)

    if not user:
        raise credentials_exception

    return user
"""
    },
    "db": {
        "migrations.py": """#!/usr/bin/env python3
\"\"\"
Database migration script for Arabic Morphophonology System
\"\"\"
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import settings

def run_migrations():
    \"\"\"Run database migrations\"\"\"
    print("Running database migrations...")

    if not settings.DATABASE_URL:
        print("No DATABASE_URL configured, skipping migrations.")
        return

    # In a real application, you would use alembic or another migration tool
    # For this example, we'll just simulate the migration
    print("Migration completed successfully.")

if __name__ == "__main__":
    run_migrations()
""",
        "seed.py": """#!/usr/bin/env python3
\"\"\"
Database seeding script for Arabic Morphophonology System
\"\"\"
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import settings

def seed_database():
    \"\"\"Seed the database with initial data\"\"\"
    print("Seeding database with initial data...")

    if not settings.DATABASE_URL:
        print("No DATABASE_URL configured, skipping database seeding.")
        return

    # In a real application, you would seed your database with initial data
    # For this example, we'll just simulate the seeding
    print("Database seeding completed successfully.")

if __name__ == "__main__":
    seed_database()
"""
    },
    "models": {
        "auth.py": """from pydantic import BaseModel, Field

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str = None
    exp: int = None

class User(BaseModel):
    username: str
    email: str = None
    full_name: str = None
    disabled: bool = False

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str""",
        "phonology.py": """from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union

class PhonologyOptions(BaseModel):
    """Options for phonological processing"""
    include_vowels: bool = True
    include_consonants: bool = True
    include_syllables: bool = False
    normalize_text: bool = True
    advanced_analysis: bool = False

class PhonologyRequest(BaseModel):
    """Request model for phonological processing"""
    text: str
    options: Optional[PhonologyOptions] = Field(default_factory=PhonologyOptions)

class PhonologyResponse(BaseModel):
    """Response model for phonological processing"""
    success: bool
    result: Dict[str, Any]
    message: str

class PhonemeRequest(BaseModel):
    """Request model for phoneme extraction"""
    text: str
    include_details: bool = False

class PhonemeResponse(BaseModel):
    """Response model for phoneme extraction"""
    success: bool
    phonemes: List[Dict[str, Any]]
    message: str
"""
    },
    "repositories": {
        "phonology.py": """from typing import Dict, List, Any, Optional

class PhonologyRepository:
    """Repository for phonology-related operations"""

    def get_phonology_engine(self, engine_type: str = "advanced"):
        """
        Get phonology engine instance based on type

        Args:
            engine_type: The type of phonology engine (advanced, basic, etc.)

        Returns:
            A phonology engine instance
        """
        # In a real application, this would return an actual engine instance
        # For this example, we'll just return a mock object
        if engine_type == "advanced":
            from nlp.phonological.engine import PhonoEngine
            return PhonoEngine()
        else:
            from nlp.base_engine import BaseEngine
            return BaseEngine()

    def save_processing_result(self, text: str, result: Dict[str, Any]) -> bool:
        """
        Save processing result to cache or database

        Args:
            text: The processed text
            result: The processing result

        Returns:
            True if successful, False otherwise
        """
        # In a real application, this would save to a database or cache
        # For this example, we'll just return True
        return True

    def get_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached processing result if available

        Args:
            text: The text to look up

        Returns:
            Cached result if available, None otherwise
        """
        # In a real application, this would check a cache or database
        # For this example, we'll just return None
        return None
"""
    },
    "services": {
        "phonology.py": """from typing import Dict, List, Any, Optional
from ..repositories.phonology import PhonologyRepository

class PhonologyService:
    """Service for phonology-related operations"""

    def __init__(self, repository: PhonologyRepository):
        """
        Initialize the service with a repository

        Args:
            repository: The phonology repository instance
        """
        self.repository = repository

    def process_text(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process text using the phonology engine

        Args:
            text: The text to process
            options: Processing options

        Returns:
            Processing result
        """
        # Check cache first
        cached_result = self.repository.get_cached_result(text)
        if cached_result:
            return cached_result

        # Get the engine and process
        engine = self.repository.get_phonology_engine(
            engine_type=options.get("engine_type", "advanced") if options else "advanced"
        )

        # Process the text
        result = engine.process_text(text, options)

        # Cache the result
        self.repository.save_processing_result(text, result)

        return result

    def extract_phonemes(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract phonemes from text

        Args:
            text: The text to process

        Returns:
            List of phonemes with their properties
        """
        engine = self.repository.get_phonology_engine()
        return engine.extract_phonemes(text)
"""
    },
    "main.py": """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict

from .core.config import settings
from .api.router import api_router

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Arabic Morphophonology System",
)

# Add CORS middleware if enabled
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/api/health", tags=["health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.VERSION}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
""",
    "requirements.txt": """fastapi>=0.101.0
uvicorn>=0.23.0
pydantic>=2.2.0
python-jose>=3.3.0
passlib>=1.7.4
python-multipart>=0.0.6
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.6
redis>=4.6.0
python-dotenv>=1.0.0
pyyaml>=6.0
httpx>=0.24.1
"""
}

def create_directory_structure(base_dir: Path, structure: dict, current_dir: Path = None):
    """Create the directory structure recursively"""
    if current_dir is None:
        current_dir = base_dir

    for name, content in structure.items():
        path = current_dir / name

        if isinstance(content, dict):
            # If the content is a dictionary, it's a directory
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
            create_directory_structure(base_dir, content, path)
        else:
            # If the content is not a dictionary, it's a file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created file: {path}")

def main():
    """Main entry point"""
    try:
        print(f"Setting up backend directory structure at {BACKEND_DIR}")
        create_directory_structure(BACKEND_DIR, STRUCTURE)
        print("Setup completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
