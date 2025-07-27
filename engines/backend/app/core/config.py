"""
Configuration settings for the application
"""

from pydantic import BaseSettings, Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""

    # Basic settings
    APP_NAME: str = "Arabic Morphophonology API"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # API settings
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],  # Default to allow all origins in dev, restrict in prod
        env="ALLOWED_ORIGINS",
    )

    # Security
    SECRET_KEY: str = Field(
        default="change_this_in_production_environment", env="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    REQUIRE_AUTH: bool = Field(default=False, env="REQUIRE_AUTH")

    # Database
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Paths
    DATA_DIR: str = Field(
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        ),
        env="DATA_DIR",
    )

    # Engine configuration
    PHONOLOGICAL_CONFIG_PATH: Optional[str] = Field(
        default=None, env="PHONOLOGICAL_CONFIG_PATH"
    )
    PHONOLOGICAL_RULES_PATH: Optional[str] = Field(
        default=None, env="PHONOLOGICAL_RULES_PATH"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
