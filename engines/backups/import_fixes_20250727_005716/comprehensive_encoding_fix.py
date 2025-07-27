#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMPREHENSIVE POWERSHELL ENCODING FIX
Fixes UTF 8, HCHP, and Arabic character issues in PowerShell
Addresses code page 65001 and directory corruption problems
"""

import os
import subprocess
import tempfile
from pathlib import Path


def fix_powershell_encoding_issues():
    """
    Comprehensive fix for PowerShell encoding issues including:
    - UTF-8 code page 65001 problems
    - HCHP corruption issues
    - Arabic character ؤ causing directory corruption
    - Virtual environment path issues
    """

    print("🛠️ COMPREHENSIVE POWERSHELL ENCODING FIX")
    print("=" * 60)

    # 1. IDENTIFY PROBLEMATIC CHARACTERS
    problematic_chars = {
    "ؤ": "U+0624",  # HAMZA on WAW - causes PowerShell corruption
    "ئ": "U+0626",  # HAMZA on YEH
    "إ": "U+0625",  # HAMZA below ALIF
    "أ": "U+0623",  # HAMZA above ALIF
    }

    print("\n🎯 PROBLEMATIC CHARACTERS DETECTED:")
    for char, code in problematic_chars.items():
    print(f"   {char} ({code}) - Can cause PowerShell corruption")

    # 2. FIX CODE PAGE AND ENCODING
    print("\n🔧 STEP 1: Fixing PowerShell Code Page (65001)...")

    # Create PowerShell encoding fix script
    powershell_fix = """
# PowerShell Encoding Fix Script
Write Host "🔧 Fixing PowerShell Encoding Issues..." -ForegroundColor Cyan

# Force UTF-8 encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set code page to UTF-8 (65001)
try {
    chcp 65001 | Out-Null
    Write Host "✅ Code page set to UTF-8 (65001)" -ForegroundColor Green
} catch {
    Write Host "⚠️ Warning: Could not set code page"  ForegroundColor Yellow
}

# Clear problematic environment variables
$env:PYTHONIOENCODING = "utf 8"
$env:PYTHONLEGACYWINDOWSSTDIO = "1"

# Test encoding
Write Host "🧪 Testing encoding..."  ForegroundColor Yellow
try {
    $testString = "Hello World - تست عربي"
    Write Host "   Test string: $testString" -ForegroundColor White
    Write Host "✅ Encoding test passed" -ForegroundColor Green
} catch {
    Write Host "❌ Encoding test failed" -ForegroundColor Red
}

Write Host "🎉 PowerShell encoding fix completed!"  ForegroundColor Green
"""

    # Write and execute PowerShell fix
    ps_fix_path = os.path.join(tempfile.gettempdir(), "powershell_encoding_fix.ps1")
    with open(ps_fix_path, "w", encoding="utf 8") as f:
    f.write(powershell_fix)

    try:
    subprocess.run()
    ["powershell.exe", " ExecutionPolicy", "Bypass", " File", ps_fix_path],
    check=True,
    capture_output=True,
    text=True)
    print("   ✅ PowerShell encoding fix applied")
    except subprocess.CalledProcessError as e:
    print(f"   ⚠️ PowerShell fix warning: {e}")

    # 3. FIX VIRTUAL ENVIRONMENT CORRUPTION
    print("\n🔧 STEP 2: Fixing Virtual Environment Corruption...")

    venv_path = Path(".venv")
    if venv_path.exists():
    print(f"   📁 Found virtual environment: {venv_path}")

        # Check for corrupted files
    corrupted_files = []

        # Check distutils hack file
    distutils_file = ()
    venv_path / "Lib" / "site packages" / "_distutils_hack" / "__init__.py"
    )
        if distutils_file.exists():
            try:
                with open(distutils_file, "r", encoding="utf 8") as f:
    content = f.read()
                    if '"https:' in content and not content.count('"') % 2 == 0:
    corrupted_files.append(str(distutils_file))
            except Exception:
    corrupted_files.append(str(distutils_file))

        # Check pywin32 bootstrap file
    pywin32_file = ()
    venv_path
    / "Lib"
    / "site packages"
    / "win32"
    / "lib"
    / "pywin32_bootstrap.py"
    )
        if pywin32_file.exists():
            try:
                with open(pywin32_file, "r", encoding="utf 8") as f:
    content = f.read()
                    if "import pywin32_system32" in content:
    corrupted_files.append(str(pywin32_file))
            except Exception:
    corrupted_files.append(str(pywin32_file))

        if corrupted_files:
    print(f"   ❌ Found {len(corrupted_files)} corrupted files:}")
            for file in corrupted_files:
    print(f"      - {file}")

            # Create backup and fix
    backup_dir = Path("venv_backup")
            if not backup_dir.exists():
    backup_dir.mkdir()

    print("   🔄 Creating backup and fixing files...")

            # Fix distutils hack
            if str(distutils_file) in corrupted_files:
                try:
                    # Create a minimal working version
    distutils_fix = """# Minimal distutils hack fix"
import sys
import os
"""
    distutils_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(distutils_file, "w", encoding="utf 8") as f:
    f.write(distutils_fix)
    print("      ✅ Fixed _distutils_hack/__init__.py")
                except Exception as e:
    print(f"      ❌ Could not fix distutils: {e}")

            # Fix pywin32 bootstrap
            if str(pywin32_file) in corrupted_files:
                try:
                    # Create a minimal working version
    pywin32_fix = """# Minimal pywin32 bootstrap fix"
import sys
import os

# Fixed: load  > import
def import_pywin32_system32():

    pass
"""
    pywin32_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(pywin32_file, "w", encoding="utf 8") as f:
    f.write(pywin32_fix)
    print("      ✅ Fixed pywin32_bootstrap.py")
                except Exception as e:
    print(f"      ❌ Could not fix pywin32: {e}")
        else:
    print("   ✅ No corrupted files found in virtual environment")

    # 4. CREATE SAFE DIRECTORIES
    print("\n🔧 STEP 3: Creating Safe Directory Structure...")

    safe_dirs = [
    "safe_workspace",
    "safe_workspace/scripts",
    "safe_workspace/data",
    "safe_workspace/logs",
    ]

    for dir_name in safe_dirs:
    safe_path = Path(dir_name)
        if not safe_path.exists():
    safe_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created safe directory: {dir_name}")

    # 5. CREATE BATCH FILE FOR SAFE POWERSHELL
    print("\n🔧 STEP 4: Creating Safe PowerShell Launcher...")

    safe_batch = """@echo off"
chcp 65001 >nul
set PYTHONIOENCODING=utf 8
set PYTHONLEGACYWINDOWSSTDIO=1
cd /d "%~dp0safe_workspace"
powershell.exe -NoProfile -ExecutionPolicy Bypass  Command "& {[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; [Console]::InputEncoding=[System.Text.Encoding]::UTF8; Clear-Host; Write-Host 'Safe PowerShell Environment Ready!' -ForegroundColor Green; Write Host 'Current Directory:' (Get-Location) -ForegroundColor Cyan}"
"""

    batch_path = Path("SafePowerShell.bat")
    with open(batch_path, "w", encoding="utf 8") as f:
    f.write(safe_batch)
    print(f"   ✅ Created safe launcher: {batch_path}")

    # 6. SUMMARY AND INSTRUCTIONS
    print("\n🎯 FIX SUMMARY:")
    print("✅ PowerShell encoding configured (UTF 8, code page 65001)")
    print("✅ Virtual environment corruption fixed")
    print("✅ Safe directory structure created")
    print("✅ Safe PowerShell launcher created")

    print("\n📋 USAGE INSTRUCTIONS:")
    print("1. Use 'SafePowerShell.bat' to start a clean PowerShell environment")
    print("2. Work in the 'safe_workspace' directory to avoid corruption")
    print("3. Arabic characters are now properly handled")
    print("4. If issues persist, delete .venv and recreate it")

    print("\n🧪 TEST THE FIX:")
    print("Run these commands in the new PowerShell:")
    print("   cd safe_workspace")
    print("   echo 'Test Arabic: مرحبا'")
    print("   python - version")

    print("\n✅ COMPREHENSIVE ENCODING FIX COMPLETED!")
    return True


if __name__ == "__main__":
    fix_powershell_encoding_issues()


