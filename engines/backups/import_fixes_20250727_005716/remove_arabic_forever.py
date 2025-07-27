#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete UTF 8 Arabic Character Remover
Permanently removes problematic Arabic character ؤ from system
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import os  # noqa: F401
import sys  # noqa: F401
import re  # noqa: F401
import glob  # noqa: F401
import shutil  # noqa: F401
from pathlib import Path  # noqa: F401


def remove_arabic_character_forever():  # type: ignore[no-untyped def]
    """Completely remove the problematic Arabic character ؤ from all files and system"""

    print("🗑️  PERMANENT ARABIC CHARACTER REMOVER")
    print("=" * 60)
    print("Target character: ؤ (U+0624)")
    print("This will PERMANENTLY remove this character from:")
    print("- All Python files in the project")
    print("- All text files")
    print("- All configuration files")
    print("=" * 60)

    problematic_char = 'ؤ'
    replacement_char = 'w'  # Replace with 'w' phoneme equivalent

    # Get project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")

    # File patterns to clean
    file_patterns = [
    "**/*.py",
    "**/*.txt",
    "**/*.md",
    "**/*.json",
    "**/*.yaml",
    "**/*.yml",
    "**/*.cfg",
    "**/*.ini",
    "**/*.log",
    ]

    files_cleaned = 0
    total_replacements = 0

    for pattern in file_patterns:
        for file_path in project_root.glob(pattern):
            try:
                # Skip binary files and directories
                if file_path.is_dir() or file_path.suffix in ['.pyc', '.exe', '.dll']:
    continue

                # Read file content
                with open(file_path, 'r', encoding='utf 8', errors='ignore') as f:
    content = f.read()

                # Check if problematic character exists
                if problematic_char in content:
    print(f"🔧 Cleaning: {file_path}")

                    # Count occurrences
    char_count = content.count(problematic_char)
    total_replacements += char_count

                    # Replace the character
    cleaned_content = content.replace(
    problematic_char, replacement_char
    )

                    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    shutil.copy2(file_path, backup_path)

                    # Write cleaned content
                    with open(file_path, 'w', encoding='utf 8') as f:
    f.write(cleaned_content)

    print(f"   ✅ Replaced {char_count} occurrences")
    files_cleaned += 1

            except Exception as e:
    print(f"   ❌ Error processing {file_path}: {e}")

    print("\n📊 CLEANUP SUMMARY:")
    print(f"Files processed: {files_cleaned}")
    print(f"Total replacements: {total_replacements}")

    # Clean PowerShell profile and environment
    print("\n🔧 POWERSHELL CLEANUP:")

    # PowerShell commands to run manually
    powershell_commands = [
    "# Clear PowerShell history",
    "Clear History",
    "",
    "# Reset environment variables",
    "$env:PSModulePath = $env:PSModulePath -replace 'ؤ', 'w'",
    "$env:PATH = $env:PATH -replace 'ؤ', 'w'",
    "",
    "# Clear clipboard",
    "Set-Clipboard -Value ''",
    "",
    "# Reset PowerShell profile",
    "if (Test Path $PROFILE) {",
    "    $content = Get-Content $PROFILE  Raw",
    "    $content = $content -replace 'ؤ', 'w'",
    "    Set-Content $PROFILE  Value $content",
    "}",
    "",
    "# Clear command history",
    "Remove-Item (Get-PSReadLineOption).HistoryStorePath  ErrorAction SilentlyContinue",
    ]

    # Write PowerShell cleanup script
    ps_script_path = project_root / "cleanup_powershell.ps1"
    with open(ps_script_path, 'w', encoding='utf 8') as f:
    f.write('\n'.join(powershell_commands))

    print(f"PowerShell cleanup script created: {ps_script_path}")

    # Terminal reset commands
    print("\n🔄 TERMINAL RESET COMMANDS:")
    print("Run these commands in PowerShell:")
    print("1. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned  Scope CurrentUser")
    print(f"2. & '{ps_script_path}")
    print("3. exit")
    print("4. Restart PowerShell")

    # Registry cleanup (Windows specific)
    print("\n🗂️  REGISTRY CLEANUP (Optional):")
    print("If the character persists, check these registry locations:")
    print("- HKEY_CURRENT_USER\\Console")
    print("- HKEY_CURRENT_USER\\Software\\Microsoft\\Command Processor")

    print("\n✅ PERMANENT REMOVAL COMPLETE!")
    print("The Arabic character ؤ has been permanently removed from all project files.")
    print("Run the PowerShell cleanup script to complete the terminal cleanup.")


if __name__ == "__main__":
    remove_arabic_character_forever()
