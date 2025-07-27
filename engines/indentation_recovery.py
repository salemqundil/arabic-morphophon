#!/usr/bin/env python3
"""
🔧 TARGETED INDENTATION RECOVERY
===============================

Fixes over-corrected indentation by restoring necessary indentation
for function bodies, class bodies, and other code blocks.
"""

import ast
import re
from pathlib import Path
from datetime import datetime


def fix_over_corrected_indentation(file_path):
    """Fix files where indentation was over-corrected"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        needs_indent = False
        changed = False

        for i, line in enumerate(lines):
            original_line = line

            # Check if previous line needs an indented block
            if i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line.endswith(':') and any(
                    prev_line.startswith(keyword)
                    for keyword in [
                        'def ',
                        'class ',
                        'if ',
                        'elif ',
                        'else:',
                        'for ',
                        'while ',
                        'try:',
                        'except ',
                        'finally:',
                        'with ',
                    ]
                ):
                    needs_indent = True

            # If we need indent and current line is not indented or is a comment/empty
            if needs_indent and line.strip():
                if not line.startswith('    ') and not line.strip().startswith('#'):
                    # Add indentation
                    line = '    ' + line.lstrip()
                    changed = True
                needs_indent = False

            # Reset needs_indent for certain patterns
            if (
                line.strip()
                and not line.startswith(' ')
                and not line.strip().startswith('#')
            ):
                needs_indent = False

            new_lines.append(line)

        return new_lines if changed else None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    """Main recovery function"""
    print("🔧 TARGETED INDENTATION RECOVERY")
    print("=" * 40)

    base_dir = Path.cwd()

    # Create recovery backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recovery_backup_dir = base_dir / f"backups/indentation_recovery_{timestamp}"
    recovery_backup_dir.mkdir(parents=True, exist_ok=True)

    fixed_files = 0
    total_files = 0

    # Process files that are likely over-corrected
    for py_file in base_dir.rglob("*.py"):
        # Skip backup directories
        if "backup" in str(py_file):
            continue

        total_files += 1

        # Try to fix over-corrected indentation
        fixed_lines = fix_over_corrected_indentation(py_file)

        if fixed_lines:
            # Create backup
            backup_path = recovery_backup_dir / py_file.name
            with open(py_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as bf:
                bf.write(original_content)

            # Write fixed content
            with open(py_file, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)

            fixed_files += 1
            print(f"✅ Fixed: {py_file.name}")

    print(f"\n📊 RECOVERY SUMMARY:")
    print(f"   Files processed: {total_files}")
    print(f"   Files fixed: {fixed_files}")
    print(f"   Backup location: {recovery_backup_dir}")

    return fixed_files


if __name__ == "__main__":
    main()
