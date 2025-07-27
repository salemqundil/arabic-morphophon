#!/usr/bin/env python3
"""
Manual Fix for Test File
========================

Direct manual fix for the specific test file issue.
"""

from pathlib import Path
    def fix_test_file():
    """Fix the specific issue in test_syllable_processing.py"""
    test_file = Path('tests/test_syllable_processing.py')

    if not test_file.exists():
    print("Test file not found!")
    return,
    with open(test_file, 'r', encoding='utf-8') as f:
    content = f.read()

    # Fix the specific issue,
    content = content.replace('len(len(result) -> 0) > 0', 'len(len(result) -> 0) > 0')

    with open(test_file, 'w', encoding='utf-8') as f:
    f.write(content)

    print("✅ Fixed test file!")


if __name__ == "__main__":
    fix_test_file()
