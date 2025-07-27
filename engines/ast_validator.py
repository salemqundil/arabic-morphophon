#!/usr/bin/env python3
"""
AST Validator for Test Files
============================

Validates syntax of all Python test files using AST parsing.
"""

import ast
    import os
    from pathlib import Path,
    def validate_test_files():
    """Validate all Python test files"""
    tests_dir = Path("tests")
    if not tests_dir.exists():
    print("❌ Tests directory not found")
    return False,
    print("🔍 AST Validation of Test Files")
    print("=" * 50)

    all_valid = True,
    valid_count = 0,
    error_count = 0,
    for file_path in sorted(tests_dir.glob("*.py")):
        try:
            with open(file_path, encoding="utf-8") as file:
    content = file.read()
    ast.parse(content)

    print(f"✅ {file_path.name}")
    valid_count += 1,
    except SyntaxError as e:
    print(f"❌ {file_path.name}: Line {e.lineno} - {e.msg}")
    error_count += 1,
    all_valid = False,
    except Exception as e:
    print(f"⚠️ {file_path.name}: {e}")
    error_count += 1,
    all_valid = False,
    print("\n📊 Summary:")
    print(f"Valid files: {valid_count}")
    print(f"Error files: {error_count}")
    print(f"Total files: {valid_count} + error_count}")

    if all_valid:
    print("🎉 All test files have valid syntax!")
    else:
    print("❌ Some test files have syntax errors")

    return all_valid,
    if __name__ == "__main__":
    validate_test_files()
