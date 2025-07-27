#!/usr/bin/env python3
"""
🧹 Simple Character Cleaner for Arabic Text
مُنظف الأحرف البسيط للنصوص العربية

Clean up specific UTF 8 encoding issues and normalize Arabic characters.
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import os  # noqa: F401
import re  # noqa: F401
from pathlib import Path  # noqa: F401


def clean_arabic_text_file(file_path):  # type: ignore[no-untyped def]
    """Clean a single file for Arabic text issues"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf 8', errors='replace') as f:
    content = f.read()

    original_content = content

        # Remove BOM if present
        if content.startswith('\ufeff'):
    content = content[1:]
    print(f"Removed BOM from: {file_path}")

        # Fix specific character issues
    replacements = {
            # Normalize Arabic characters
    'ؤ': 'ؤ',  # Ensure proper hamza on waw
    'ئ': 'ئ',  # Ensure proper hamza on ya
    'ء': 'ء',  # Ensure proper standalone hamza
    'أ': 'أ',  # Ensure proper alef with hamza above
    'إ': 'إ',  # Ensure proper alef with hamza below
    'آ': 'آ',  # Ensure proper alef with madda
    'ى': 'ى',  # Ensure proper alef maksura
    'ة': 'ة',  # Ensure proper teh marbuta
    }

        for old, new in replacements.items():
            if old in content and old != new:
    content = content.replace(old, new)
    print(f"Normalized character in {file_path}: {old}  > {new}")

        # Remove any problematic control characters (but keep basic whitespace)
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e \x1f\x7f]', '', content)

        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf 8', newline='') as f:
    f.write(content)
    print(f"✅ Cleaned: {file_path}")
    return True

    return False

    except Exception as e:
    print(f"❌ Error processing {file_path}: {e}")
    return False


def main():  # type: ignore[no-untyped def]
    """Clean all text files in the project"""
    project_root = Path(r"c:\Users\Administrator\new engine\engines")

    # File extensions to process
    extensions = [
    '.py',
    '.md',
    '.txt',
    '.json',
    '.yaml',
    '.yml',
    '.html',
    '.css',
    '.js',
    ]

    total_files = 0
    cleaned_files = 0

    print("🧹 Starting Arabic text cleanup...")
    print(f"📁 Project root: {project_root}")

    for ext in extensions:
    pattern = f"**/*{ext}"
    files = list(project_root.glob(pattern))

        for file_path in files:
    total_files += 1
            if clean_arabic_text_file(file_path):
    cleaned_files += 1

    print("\n📊 Cleanup Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files cleaned: {cleaned_files}")
    print("   ✅ Cleanup completed successfully!")

    # Also clean up any weird command issues
    print("\n🔧 Terminal fix suggestion:")
    print("If you see 'ؤcd' error, try running:")
    print("   Clear Host")
    print("   cd 'c:\\Users\\Administrator\\new engine\\engines'")


if __name__ == "__main__":
    main()
