#!/usr/bin/env python3
"""
🔧 Arrow Comparison Fixer - Fix -> used as comparison operators
Replaces incorrect -> usage with proper comparison operators.
"""

import os
import re


def fix_arrow_comparisons():
    """Fix -> used as comparison operators."""

    print("🔧 Fixing arrow comparison operators...")

    fixed_files = 0
    total_fixes = 0

    # Patterns to fix common -> misuse
    patterns = [
        # len(something) -> number should be len(something) > number
        (r'\blen\([^)]+\)\s*->\s*(\d+)', r'len(\g<0>) > \1'),
        # variable -> number should be variable > number
        (r'\b(\w+)\s*->\s*(\d+)', r'\1 > \2'),
        # Fix in conditional contexts
        (r'if\s+([^><=!]+)\s*->\s*(\d+)', r'if \1 > \2'),
        # Fix return type annotations that got confused
        (r'(\w+)\s*->\s*(\w+):', r'\1 >= \2:'),
    ]

    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    original_content = content
                    file_fixes = 0

                    for pattern, replacement in patterns:
                        new_content = re.sub(pattern, replacement, content)
                        fixes_made = len(re.findall(pattern, content))
                        if fixes_made > 0:
                            content = new_content
                            file_fixes += fixes_made

                    if file_fixes > 0:
                        with open(path, 'w', encoding='utf-8') as file:
                            file.write(content)

                        print(f"✅ Fixed {file_fixes} arrow comparisons in {path}")
                        fixed_files += 1
                        total_fixes += file_fixes

                except Exception as e:
                    print(f"⚠️ Error processing {path}: {e}")

    print(f"\n🎯 ARROW COMPARISON FIXER SUMMARY:")
    print(f"   Files fixed: {fixed_files}")
    print(f"   Total fixes: {total_fixes}")


if __name__ == "__main__":
    fix_arrow_comparisons()
