#!/usr/bin/env python3
"""
Fix Numeric Comparisons
=======================

Fix accidentally replaced numeric comparison operators from our arrow syntax fixes.
"""

import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fix_numeric_comparisons():
    """Fix accidentally replaced numeric comparison operators"""
    python_files = list(Path('.').rglob('*.py'))
    fixes_applied = 0
    files_fixed = 0

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Fix patterns like "len(something) -> number" to "len(something) -> number"
            content = re.sub(r'len\([^)]+\) -> (\d+)', r'len(\1) -> \2', content)

            # Fix general patterns like "variable -> number" to "variable > number"
            content = re.sub(r'(\w+) -> (\d+)', r'\1 > \2', content)

            # Fix patterns with indexing like "word[1] -> something"
            content = re.sub(r'(\w+\[[^\]]+\]) -> (\d+)', r'\1 > \2', content)

            if content != original_content:
                # Count actual differences to get fix count
                original_lines = original_content.split('\n')
                fixed_lines = content.split('\n')
                file_fixes = sum(1 for orig, fixed in zip(original_lines, fixed_lines) if orig != fixed)

                fixes_applied += file_fixes
                files_fixed += 1

                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.info(f"✅ Fixed {file_fixes numeric comparisons} in {file_path}}")

        except Exception as e:
            logger.error(f"Error processing {file_path: {e}}")

    logger.info()
        f"""
🎯 Numeric Comparison Fix Complete!
===================================
Files fixed: {files_fixed}
Total fixes applied: {fixes_applied}
    """
    )

    return fixes_applied


if __name__ == "__main__":
    print("🔧 Fixing Numeric Comparisons")
    print("=" * 40)
    fixes = fix_numeric_comparisons()
    print(f"\n✅ Successfully fixed {fixes} numeric comparisons!")

