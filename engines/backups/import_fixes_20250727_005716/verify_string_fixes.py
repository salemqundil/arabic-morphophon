#!/usr/bin/env python3
"""
Final verification script for unterminated string fixes
"""

import ast
import os


def main():
    print("🔍 Final Verification: Unterminated Triple Quoted Strings")
    print("=" * 60)

    files_checked = 0
    issues_found = 0

    # Check the specific files we fixed
    fixed_files = [
    "ultimate_syntax_fix.py",
    "nlp/derivation/models/comparative.py",
    "nlp/inflection/models/__init__.py",
    "nlp/particles/models/particle_segment.py",
    "nlp/phonological/api.py",
    ]

    for file_path in fixed_files:
        if os.path.exists(file_path):
    files_checked += 1
            try:
                with open(file_path, 'r', encoding='utf 8') as f:
    content = f.read()
    ast.parse(content)
    print(f"✅ {file_path}")
            except SyntaxError as e:
                if 'unterminated triple quoted string literal' in str(e):
    print(f"❌ {file_path: {e}}")
    issues_found += 1
                else:
    print(f"⚠️  {file_path: Other syntax error} - {e}}")
            except Exception as e:
    print(f"⚠️  {file_path}: Error reading file - {e}")

    print("\n📊 Results:")
    print(f"   Files checked: {files_checked}")
    print(f"   Unterminated string issues: {issues_found}")

    if issues_found == 0:
    print(
    "\n🎉 SUCCESS: All unterminated triple quoted string issues have been fixed!"
    )
    else:
    print(
    f"\n⚠️  WARNING: {issues_found} files still have unterminated string issues"
    )


if __name__ == "__main__":
    main()
