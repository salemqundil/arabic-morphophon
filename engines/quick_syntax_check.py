#!/usr/bin/env python3
"""
Quick syntax validation check after indentation fix
"""

import ast
import os
from pathlib import Path


def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def main():
    """Main validation function"""
    print("🔍 QUICK SYNTAX VALIDATION CHECK")
    print("=" * 40)

    base_dir = Path.cwd()
    python_files = list(base_dir.rglob("*.py"))

    valid_files = 0
    invalid_files = 0
    errors = []

    for py_file in python_files:
        is_valid, error = check_syntax(py_file)
        if is_valid:
            valid_files += 1
        else:
            invalid_files += 1
            rel_path = py_file.relative_to(base_dir)
            errors.append(f"❌ {rel_path}: {error}")

    # Summary
    total_files = len(python_files)
    success_rate = (valid_files / total_files * 100) if total_files > 0 else 0

    print(f"📊 VALIDATION RESULTS:")
    print(f"   Total files: {total_files}")
    print(f"   ✅ Valid: {valid_files}")
    print(f"   ❌ Invalid: {invalid_files}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print()

    if errors:
        print("🔴 SYNTAX ERRORS (First 20):")
        for error in errors[:20]:
            print(f"   {error}")

    return valid_files, invalid_files, success_rate


if __name__ == "__main__":
    main()
