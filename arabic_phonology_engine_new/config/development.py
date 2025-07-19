"""
Configuration module for different deployment environments
Supports development, production, and distributed deployments
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API configuration for different deployment modes"""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

    # Stats backend
    deployment_mode: str = "development"  # development, production, distributed
    redis_url: Optional[str] = None

    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"

    # Security
    cors_origins: str = "*"
    max_content_length: int = 16 * 1024 * 1024  # 16MB

    # Feature flags
    enable_distributed_stats: bool = False
    enable_phoneme_registry: bool = True
    enable_advanced_analysis: bool = True


class DevelopmentConfig(APIConfig):
    """Development environment configuration"""

    debug = True
    log_level = "DEBUG"
    deployment_mode = "development"


class ProductionConfig(APIConfig):
    """Production environment configuration"""

    debug = False
    log_level = "WARNING"
    deployment_mode = "production"
    enable_distributed_stats = True
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class DistributedConfig(ProductionConfig):
    """Distributed deployment configuration"""

    deployment_mode = "distributed"
    enable_distributed_stats = True
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 5000))


def get_config() -> APIConfig:
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "development")
    deployment_mode = os.getenv("DEPLOYMENT_MODE", "development")

    if deployment_mode == "distributed":
        return DistributedConfig()
    elif deployment_mode == "production" or env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()


# Global configuration instance
config = get_config()
