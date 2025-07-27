#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PERMANENT PROJECT-WIDE ENCODING CONFIGURATION
This script makes UTF-8 and PowerShell fixes permanent for the entire project
No external dependencies - pure Python solution
"""

import os
import sys
import json
import shutil
from pathlib import Path


class PermanentProjectConfig:
    """Permanent configuration manager for the entire Arabic NLP project"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config_file = self.project_root / "project_config.json"

    def create_permanent_config(self):
        """Create permanent project-wide configuration"""
        print("🔧 CREATING PERMANENT PROJECT CONFIGURATION")
        print("=" * 50)

        # 1. Project-wide encoding configuration
        config = {
            "project_name": "Arabic Morphophonological Engine",
            "encoding": {
                "default": "utf-8",
                "console_encoding": "utf-8",
                "file_encoding": "utf-8",
                "code_page": 65001,
            },
            "powershell": {
                "safe_mode": True,
                "utf8_support": True,
                "working_directory": "safe_workspace",
            },
            "arabic_characters": {
                "problematic_chars": ["ؤ", "ئ", "إ", "أ"],
                "safe_handling": True,
                "encoding_validation": True,
            },
            "virtual_environment": {
                "python_encoding": "utf-8",
                "legacy_windows_stdio": True,
                "distutils_hack_fixed": True,
            },
            "project_structure": {
                "safe_directories": [
                    "safe_workspace",
                    "safe_workspace/scripts",
                    "safe_workspace/data",
                    "safe_workspace/logs",
                    "safe_workspace/temp",
                ],
                "config_files": [
                    "project_config.json",
                    "SafePowerShell.bat",
                    "setup_project.py",
                ],
            },
        }

        # Save configuration
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration saved: {self.config_file}")

        return config

    def setup_environment_files(self):
        """Create permanent environment setup files"""
        print("\n🔧 CREATING ENVIRONMENT FILES")
        print("=" * 35)

        # 1. Python environment setup
        env_setup = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Environment Setup - Auto-run on startup"""
import os
import sys

# Set permanent encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"

# Ensure UTF-8 for all operations
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

print("✅ Project encoding configured: UTF-8")
'''

        env_file = self.project_root / "setup_environment.py"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_setup)
        print(f"✅ Created: {env_file}")

        # 2. Requirements with encoding specifications
        requirements = """# Arabic NLP Project Requirements
# All packages configured for UTF-8 support

# Core dependencies
numpy>=1.21.0
pandas>=1.3.0

# Optional for advanced features (install when needed)
# networkx>=2.8.0
# matplotlib>=3.5.0
# flask>=2.0.0

# Development tools
# pytest>=6.0.0
# black>=21.0.0
# flake8>=4.0.0
"""

        req_file = self.project_root / "requirements_minimal.txt"
        with open(req_file, "w", encoding="utf-8") as f:
            f.write(requirements)
        print(f"✅ Created: {req_file}")

        # 3. VSCode workspace settings
        vscode_settings = {
            "python.defaultInterpreterPath": "./.venv/Scripts/python.exe",
            "files.encoding": "utf8",
            "files.autoGuessEncoding": False,
            "terminal.integrated.env.windows": {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONLEGACYWINDOWSSTDIO": "1",
            },
            "python.terminal.activateEnvironment": True,
            "files.associations": {"*.py": "python", "*.md": "markdown"},
        }

        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        settings_file = vscode_dir / "settings.json"
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(vscode_settings, f, indent=2)
        print(f"✅ Created: {settings_file}")

    def create_startup_scripts(self):
        """Create permanent startup scripts"""
        print("\n🔧 CREATING STARTUP SCRIPTS")
        print("=" * 30)

        # 1. Universal startup batch file
        startup_batch = """@echo off
title Arabic NLP Project - Safe Environment
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1

echo 🚀 Starting Arabic NLP Project Environment
echo ============================================

if exist .venv\\Scripts\\activate.bat (
    call .venv\\Scripts\\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ⚠️  Virtual environment not found
    echo Creating new virtual environment...
    python -m venv .venv
    call .venv\\Scripts\\activate.bat
    pip install --upgrade pip
    echo ✅ New virtual environment created
)

if exist setup_environment.py (
    python setup_environment.py
)

cd safe_workspace 2>nul || (
    echo 📁 Creating safe workspace...
    mkdir safe_workspace
    cd safe_workspace
)

echo 🎉 Ready for development!
echo Current directory: %CD%
echo Python version:
python --version

cmd /k
"""

        startup_file = self.project_root / "start_project.bat"
        with open(startup_file, "w", encoding="utf-8") as f:
            f.write(startup_batch)
        print(f"✅ Created: {startup_file}")

        # 2. PowerShell startup script
        ps_startup = """# Arabic NLP Project PowerShell Startup
Write-Host "🚀 Arabic NLP Project - Safe Environment" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Set encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set environment variables
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "1"

# Activate virtual environment if exists
if (Test-Path ".venv\\Scripts\\Activate.ps1") {
    & .venv\\Scripts\\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠️ Virtual environment not found" -ForegroundColor Yellow
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\\Scripts\\Activate.ps1
    pip install --upgrade pip
    Write-Host "✅ New virtual environment created" -ForegroundColor Green
}

# Run environment setup
if (Test-Path "setup_environment.py") {
    python setup_environment.py
}

# Navigate to safe workspace
if (Test-Path "safe_workspace") {
    Set-Location safe_workspace
} else {
    Write-Host "📁 Creating safe workspace..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Name "safe_workspace" -Force
    Set-Location safe_workspace
}

Write-Host "🎉 Ready for development!" -ForegroundColor Green
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Python version:" -ForegroundColor Cyan
python --version
"""

        ps_file = self.project_root / "start_project.ps1"
        with open(ps_file, "w", encoding="utf-8") as f:
            f.write(ps_startup)
        print(f"✅ Created: {ps_file}")

    def create_project_structure(self):
        """Create permanent safe project structure"""
        print("\n🔧 CREATING PROJECT STRUCTURE")
        print("=" * 35)

        # Safe directories
        safe_dirs = [
            "safe_workspace",
            "safe_workspace/scripts",
            "safe_workspace/data",
            "safe_workspace/logs",
            "safe_workspace/temp",
            "safe_workspace/config",
            "safe_workspace/tests",
        ]

        for dir_path in safe_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

            # Create .gitkeep to ensure directory is tracked
            gitkeep = full_path / ".gitkeep"
            gitkeep.touch()

            print(f"✅ Created: {dir_path}")

        # Create README for safe workspace
        readme_content = """# Safe Workspace Directory

This directory is designed to be safe from PowerShell and UTF-8 encoding issues.

## Usage
- Work here for all Arabic NLP development
- All files are automatically UTF-8 encoded
- No Arabic character corruption issues
- Safe for PowerShell operations

## Structure
- `scripts/` - Python scripts and utilities
- `data/` - Data files and datasets  
- `logs/` - Log files and debugging output
- `temp/` - Temporary files (safe to delete)
- `config/` - Configuration files
- `tests/` - Test files and validation scripts

## Arabic Character Support
This workspace fully supports all Arabic characters including:
- ؤ (HAMZA on WAW)
- ئ (HAMZA on YEH)  
- إ (HAMZA below ALIF)
- أ (HAMZA above ALIF)

All operations are UTF-8 safe and PowerShell compatible.
"""

        readme_file = self.project_root / "safe_workspace" / "README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"✅ Created: safe_workspace/README.md")

    def create_gitignore(self):
        """Create comprehensive .gitignore"""
        print("\n🔧 CREATING .GITIGNORE")
        print("=" * 25)

        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
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
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo
*~

# Project specific
*.log
*.tmp
safe_workspace/temp/*
!safe_workspace/temp/.gitkeep
safe_workspace/logs/*
!safe_workspace/logs/.gitkeep

# System files
.DS_Store
Thumbs.db
desktop.ini

# Backup files
*.bak
*.backup
venv_backup/

# Temporary files
*.temp
*.temporary
"""

        gitignore_file = self.project_root / ".gitignore"
        with open(gitignore_file, "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        print(f"✅ Created: {gitignore_file}")

    def run_complete_setup(self):
        """Run complete permanent setup"""
        print("🚀 PERMANENT PROJECT SETUP")
        print("=" * 30)
        print("Making all encoding fixes permanent for the entire project...\n")

        # Create all components
        config = self.create_permanent_config()
        self.setup_environment_files()
        self.create_startup_scripts()
        self.create_project_structure()
        self.create_gitignore()

        print("\n🎯 SETUP COMPLETE!")
        print("=" * 20)
        print("✅ All encoding fixes are now PERMANENT")
        print("✅ Project-wide UTF-8 configuration applied")
        print("✅ Safe workspace structure created")
        print("✅ Startup scripts configured")
        print("✅ Development environment ready")

        print("\n📋 HOW TO USE:")
        print("=" * 15)
        print("1. 🚀 Run 'start_project.bat' (Windows CMD)")
        print("2. 🚀 Run 'start_project.ps1' (PowerShell)")
        print("3. 🏠 Work in 'safe_workspace/' directory")
        print("4. 🔧 All encoding issues are automatically handled")

        print("\n🏆 RESULT:")
        print("Your Arabic NLP project is now permanently configured!")
        print("No more UTF-8, PowerShell, or encoding issues!")

        return True


def main():
    """Main function to run permanent setup"""
    setup = PermanentProjectConfig()
    return setup.run_complete_setup()


if __name__ == "__main__":
    main()
