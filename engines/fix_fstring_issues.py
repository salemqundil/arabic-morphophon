#!/usr/bin/env python3
"""
🔧 F-String Fixer - Fix unterminated f-strings and syntax issues,
    Fixes common f-string syntax problems.
"""

import os
    import re,
    def fix_fstring_issues():
    """Fix f-string syntax issues."""

    print("🔧 Fixing f-string syntax issues...")

    fixed_files = 0,
    total_fixes = 0,
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
    path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
    content = file.read()

    original_content = content,
    file_fixes = 0

                    # Fix 1: Unterminated f-strings - missing closing quote
                    # Pattern: f"text {variable'}" -> f"text {variable}"
    pattern1 = r'f"([^"]*\{[^}]*)\'"}\)'
                    if re.search(pattern1, content):
    content = re.sub(pattern1, r'f"\1")', content)
    file_fixes += len(re.findall(pattern1, original_content))

                    # Fix 2: Missing closing quote in f-strings
                    # Pattern: f"text {variable" ->} f"text {variable}"
    pattern2 = r'f"([^"]*\{[^}]*)\'"'
                    if re.search(pattern2, content):
    content = re.sub(pattern2, r'f"\1"', content)
    file_fixes += len(re.findall(pattern2, original_content))

                    # Fix 3: Double quote issues in f-strings
                    # Pattern: f"text {variable'}" -> f"text {variable}"
    pattern3 = r'f"([^"]*)\'"}\)'
                    if re.search(pattern3, content):
    content = re.sub(pattern3, r'f"\1")', content)
    file_fixes += len(re.findall(pattern3, original_content))

                    # Fix 4: Fix missing quote at end,
    pattern4 = r'f"([^"]*\{[^}]*}[^"]*)\'"'
                    if re.search(pattern4, content):
    content = re.sub(pattern4, r'f"\1"', content)
    file_fixes += len(re.findall(pattern4, original_content))

                    # Fix 5: Broken f-string with extra quote
                    # Pattern: f"{variable}" -> f"{variable}"
    pattern5 = r'f"\{([^}]*)\'\}"'
                    if re.search(pattern5, content):
    content = re.sub(pattern5, r'f"{\1}"', content)
    file_fixes += len(re.findall(pattern5, original_content))

                    # Fix 6: Fix missing comma in f-string contexts
                    # Pattern: f"text {fixes} safe fixes}" -> f"text {fixes} safe fixes"
    pattern6 = r'f"([^"]*\{[^}]*)\s+([^}]*)\s+([^"]*)"'
                    if re.search(pattern6, content):
    content = re.sub(pattern6, r'f"\1} \2 \3"', content)
    file_fixes += len(re.findall(pattern6, original_content))

                    if file_fixes > 0:
                        with open(path, 'w', encoding='utf-8') as file:
    file.write(content)

    print(f"✅ Fixed {file_fixes} f-string issues in {path}")
    fixed_files += 1,
    total_fixes += file_fixes,
    except Exception as e:
    print(f"⚠️ Error processing {path}: {e}")

    print(f"\n🎯 F-STRING FIXER SUMMARY:")
    print(f"   Files fixed: {fixed_files}")
    print(f"   Total fixes: {total_fixes}")


if __name__ == "__main__":
    fix_fstring_issues()
