#!/usr/bin/env python3
"""
Arabic Morphophonology System Setup Script

This script initializes the project structure for the Arabic Morphophonology System.
It creates the necessary directories and files for both backend and frontend components.
"""
import os
import shutil
import argparse
from pathlib import Path
import json
import subprocess
import sys

# Base paths
BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
INFRASTRUCTURE_DIR = BASE_DIR / "infrastructure"
DATA_DIR = BASE_DIR / "data"

# Backend structure
BACKEND_DIRS = [
    "app/api/routers",
    "app/core",
    "app/database",
    "app/services",
    "app/tests",
]

# Frontend structure
FRONTEND_DIRS = [
    "public",
    "src/components",
    "src/pages",
    "src/services",
    "src/utils",
    "src/assets",
    "src/hooks",
    "src/context",
]

# Infrastructure structure
INFRASTRUCTURE_DIRS = [
    "docker",
    "kubernetes",
    "scripts",
]

# Data structure
DATA_DIRS = [
    "phonology",
    "morphology",
    "derivation",
    "syllable",
]


def create_directory_structure():
    """Create the basic directory structure for the project"""
    print("Creating directory structure...")

    # Create base directories
    for dir_path in [BACKEND_DIR, FRONTEND_DIR, INFRASTRUCTURE_DIR, DATA_DIR]:
        dir_path.mkdir(exist_ok=True)
        print(f"Created {dir_path}")

    # Create backend directories
    for dir_path in BACKEND_DIRS:
        (BACKEND_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if "app/" in dir_path:
            init_file = BACKEND_DIR / dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
        print(f"Created {BACKEND_DIR / dir_path}")

    # Create frontend directories
    for dir_path in FRONTEND_DIRS:
        (FRONTEND_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created {FRONTEND_DIR / dir_path}")

    # Create infrastructure directories
    for dir_path in INFRASTRUCTURE_DIRS:
        (INFRASTRUCTURE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created {INFRASTRUCTURE_DIR / dir_path}")

    # Create data directories
    for dir_path in DATA_DIRS:
        (DATA_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created {DATA_DIR / dir_path}")


def create_frontend_package_json():
    """Create package.json for the frontend"""
    package_json = {
        "name": "arabic-morphophon-ui",
        "version": "1.0.0",
        "private": True,
        "dependencies": {
            "@emotion/react": "^11.11.1",
            "@emotion/styled": "^11.11.0",
            "@mui/icons-material": "^5.14.9",
            "@mui/material": "^5.14.9",
            "@testing-library/jest-dom": "^5.17.0",
            "@testing-library/react": "^13.4.0",
            "@testing-library/user-event": "^13.5.0",
            "@types/jest": "^27.5.2",
            "@types/node": "^16.18.50",
            "@types/react": "^18.2.21",
            "@types/react-dom": "^18.2.7",
            "axios": "^1.5.0",
            "i18next": "^23.5.1",
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-i18next": "^13.2.2",
            "react-router-dom": "^6.16.0",
            "react-scripts": "5.0.1",
            "typescript": "^4.9.5",
            "web-vitals": "^2.1.4",
        },
        "scripts": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test",
            "eject": "react-scripts eject",
        },
        "eslintConfig": {"extends": ["react-app", "react-app/jest"]},
        "browserslist": {
            "production": [">0.2%", "not dead", "not op_mini all"],
            "development": [
                "last 1 chrome version",
                "last 1 firefox version",
                "last 1 safari version",
            ],
        },
    }

    package_json_path = FRONTEND_DIR / "package.json"
    with open(package_json_path, 'w') as f:
        json.dump(package_json, f, indent=2)

    print(f"Created {package_json_path}")


def create_env_file():
    """Create .env.example file for environment variables"""
    env_content = """# Application settings
DEBUG=0
LOG_LEVEL=INFO
MAX_WORKERS=4

# Security
SECRET_KEY=change_this_in_production_environment
ACCESS_TOKEN_EXPIRE_MINUTES=30
REQUIRE_AUTH=0

# Database
DB_USER=arabic_morphophon
DB_PASSWORD=change_this_in_production
DB_NAME=morphophon_db
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@db:5432/${DB_NAME}

# Redis
REDIS_URL=redis://redis:6379/0

# Paths
DATA_DIR=/app/data

# CORS
ALLOWED_ORIGINS=["http://localhost", "http://localhost:3000"]
"""

    env_path = BASE_DIR / ".env.example"
    with open(env_path, 'w') as f:
        f.write(env_content)

    print(f"Created {env_path}")


def setup_git_ignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
.hypothesis/
.pytest_cache/
venv/
.venv/

# React
node_modules/
/build
/coverage
.env.local
.env.development.local
.env.test.local
.env.production.local
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Docker
.docker/
.dockerignore

# IDEs and editors
/.idea
/.vscode
*.swp
*.swo
*~
.project
.classpath
.c9/
*.launch
.settings/
*.sublime-workspace

# Environment
.env

# OS
.DS_Store
Thumbs.db
"""

    gitignore_path = BASE_DIR / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)

    print(f"Created {gitignore_path}")


def initialize_git_repo():
    """Initialize Git repository"""
    try:
        subprocess.run(["git", "init"], cwd=BASE_DIR, check=True)
        print("Initialized Git repository")
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not initialize Git repository. Make sure Git is installed."
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup Arabic Morphophonology System")
    parser.add_argument("--no-git", action="store_true", help="Skip Git initialization")

    args = parser.parse_args()

    print("Setting up Arabic Morphophonology System...")

    create_directory_structure()
    create_frontend_package_json()
    create_env_file()
    setup_git_ignore()

    if not args.no_git:
        initialize_git_repo()

    print("\nSetup complete! Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run docker compose up -d to start the system")
    print("3. Access the web UI at http://localhost")
    print("4. Access the API documentation at http://localhost/docs")


if __name__ == "__main__":
    main()
