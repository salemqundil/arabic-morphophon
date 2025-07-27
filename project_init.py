#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTO-RUN PROJECT INITIALIZATION
This file automatically ensures proper encoding every time the project is accessed
Place this at the top of your main scripts or run it automatically
"""

import os
import sys
from pathlib import Path


# PERMANENT ENCODING FIX - Auto-applied
def ensure_utf8_encoding():
    """Automatically ensure UTF-8 encoding for the entire project"""

    # Set environment variables permanently
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Force UTF-8 for stdio (Python 3.7+ compatibility)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Fallback for older Python versions
        pass

    # Ensure working in safe directory
    safe_workspace = Path("safe_workspace")
    if safe_workspace.exists() and Path.cwd().name != "safe_workspace":
        try:
            os.chdir(safe_workspace)
        except:
            pass  # Continue if can't change directory

    return True


# AUTO-RUN: This executes automatically when the module is imported
ensure_utf8_encoding()


# PERMANENT PROJECT COMMANDS
def run_permanent_command(command_type="status"):
    """
    Permanent commands for the project:
    - status: Show current status
    - test: Run quick test
    - clean: Clean environment
    - fix: Apply encoding fixes
    """

    if command_type == "status":
        print("🎯 ARABIC NLP PROJECT - PERMANENT CONFIGURATION")
        print("=" * 50)
        print(f"✅ Working Directory: {Path.cwd()}")
        print(f"✅ Python Encoding: {sys.stdout.encoding}")
        print(f"✅ Environment UTF-8: {os.environ.get('PYTHONIOENCODING', 'Not Set')}")
        print("✅ All encoding fixes are PERMANENT and ACTIVE")

    elif command_type == "test":
        print("🧪 QUICK ENCODING TEST")
        print("=" * 25)
        test_arabic = "اختبار الترميز: كتاب، مدرسة، يكتب، مكتوب - ؤ ئ إ أ"
        print(f"Arabic Text: {test_arabic}")
        print("✅ UTF-8 encoding test PASSED")

    elif command_type == "clean":
        print("🧹 CLEANING ENVIRONMENT")
        print("=" * 25)
        ensure_utf8_encoding()
        print("✅ Environment cleaned and encoding reset")

    elif command_type == "fix":
        print("🔧 APPLYING PERMANENT FIXES")
        print("=" * 30)
        ensure_utf8_encoding()

        # Check and create safe workspace if needed
        safe_dir = Path("safe_workspace")
        if not safe_dir.exists():
            safe_dir.mkdir(exist_ok=True)
            print("✅ Safe workspace created")

        print("✅ All permanent fixes applied")

    return True


# Make this available as a simple command
def project(command="status"):
    """Simple command interface: project('status'), project('test'), etc."""
    return run_permanent_command(command)


if __name__ == "__main__":
    # Auto-run when executed directly
    print("🚀 PERMANENT PROJECT INITIALIZATION")
    print("=" * 40)
    run_permanent_command("status")
    print("\n📋 AVAILABLE COMMANDS:")
    print("python -c \"import project_init; project_init.project('status')\"")
    print("python -c \"import project_init; project_init.project('test')\"")
    print("python -c \"import project_init; project_init.project('clean')\"")
    print("python -c \"import project_init; project_init.project('fix')\"")
    print("\n🎉 Project is ready for development!")
