#!/usr/bin/env python3
"""
Quick Status Checker
"""

import os
import ast


def check_syntax():
    """Check syntax of all Python files"""
    error_files = []
    success_files = []

    for root, _, files in os.walk('.'):
        for f in files:
            if f.endswith('.py'):
    filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
    ast.parse(file.read())
    success_files.append(f)
                except Exception as e:
    error_files.append((f, str(e)))

    total = len(success_files) + len(error_files)
    success_rate = (len(success_files) / total * 100) if total > 0 else 0

    print(f"Status Report:")
    print(f"Success: {len(success_files)} files")
    print(f"Errors: {len(error_files)} files")
    print(f"Total: {total} files")
    print(f"Success rate: {success_rate:.1f}%")

    if error_files:
    print(f"\nFirst 10 error files:")
        for f, e in error_files[:10]:
    print(f"❌ {f}: {e[:50]}...")


if __name__ == "__main__":
    check_syntax()
