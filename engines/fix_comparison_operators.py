#!/usr/bin/env python3
"""
Fix Comparison Operators
========================

Fix accidentally replaced comparison operators from our arrow syntax fixes.
"""

import logging
    from pathlib import Path,
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fix_comparison_operators():
    """Fix accidentally replaced comparison operators"""
    python_files = list(Path('.').rglob('*.py'))
    fixes_applied = 0,
    files_fixed = 0,
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

    original_content = content

            # Fix the comparison operators,
    content = content.replace('>= ', '>= ')
    content = content.replace('<= ', '<= ')
    content = content.replace('>= ', '>= ')  # Just in case,
    if content != original_content:
                # Count fixes,
    file_fixes = ()
    original_content.count('>= ') + original_content.count('<= ') + original_content.count('>= ')
    )
    fixes_applied += file_fixes,
    files_fixed += 1

                # Write fixed content,
    with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

    logger.info(f"✅ Fixed {file_fixes comparison operators} in {file_path}}")

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")

    logger.info()
    f"""
🎯 Comparison Operator Fix Complete!
===================================
Files fixed: {files_fixed}
Total fixes applied: {fixes_applied}
    """
    )

    return fixes_applied,
    if __name__ == "__main__":
    print("🔧 Fixing Comparison Operators")
    print("=" * 40)
    fixes = fix_comparison_operators()
    print(f"\n✅ Successfully fixed {fixes} comparison operators!")

