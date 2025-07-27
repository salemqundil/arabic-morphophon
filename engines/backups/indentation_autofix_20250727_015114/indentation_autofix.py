#!/usr/bin/env python3
"""
🔧 AUTOMATIC INDENTATION FIXER
==============================

Automatically fixes common indentation issues in Python files:
- Converts tabs to 4 spaces
- Fixes excessive indentation (8+ spaces to 4 spaces)
- Creates timestamped backups
- Provides detailed reporting
"""

import os
import re
from pathlib import Path
from datetime import datetime


def main():
    """Main indentation fixing function"""
    print("🔧 AUTOMATIC INDENTATION FIXER")
    print("=" * 40)

    # Directory to scan
    BASE_DIR = Path.cwd()

    # Error pattern to look for
    INDENT_ERROR_PATTERN = re.compile(r"^\s+\S")

    # Collect report and fixes
    fix_log = []
    fixed_files = 0
    fixes_applied = 0

    # Timestamped backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / f"backups/indentation_autofix_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Scanning directory: {BASE_DIR}")
    print(f"💾 Backup location: {backup_dir}")
    print()

    # Process each Python file
    python_files = list(BASE_DIR.rglob("*.py"))
    print(f"🔍 Found {len(python_files)} Python files to process")
    print()

    for py_file in python_files:
        try:
            with open(py_file, encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            file_fixes = 0

            for line_num, line in enumerate(lines, 1):
                original_line = line

                # Replace tabs with 4 spaces
                fixed_line = line.replace("\t", "    ")
                if fixed_line != original_line:
                    file_fixes += 1
                    changed = True

                # Strip redundant leading whitespace for lines with unexpected indent
                if INDENT_ERROR_PATTERN.match(
                    fixed_line
                ) and not fixed_line.lstrip().startswith(
                    (
                        "#",
                        "def",
                        "class",
                        "if",
                        "for",
                        "while",
                        "with",
                        "try",
                        "except",
                        "elif",
                        "else",
                        "import",
                        "from",
                    )
                ):
                    stripped = fixed_line.lstrip()
                    leading_spaces = len(fixed_line) - len(stripped)
                    if leading_spaces >= 8:
                        fixed_line = "    " + stripped  # reduce to 4 spaces
                        file_fixes += 1
                        changed = True

                new_lines.append(fixed_line)

            if changed:
                # Save backup
                backup_path = backup_dir / py_file.name
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.writelines(lines)

                # Write fixed content
                with open(py_file, "w", encoding="utf-8") as wf:
                    wf.writelines(new_lines)

                fix_log.append(
                    f"✅ Fixed: {py_file.relative_to(BASE_DIR)} ({file_fixes} fixes)"
                )
                fixed_files += 1
                fixes_applied += file_fixes
                print(f"✅ {py_file.name}: {file_fixes} indentation fixes")

        except Exception as e:
            error_msg = f"⚠️ Error processing {py_file.name}: {str(e)}"
            fix_log.append(error_msg)
            print(error_msg)

    # Generate summary report
    print()
    print("📊 INDENTATION FIX SUMMARY")
    print("=" * 35)
    print(f"📁 Files processed: {len(python_files)}")
    print(f"🔧 Files fixed: {fixed_files}")
    print(f"⚡ Total fixes applied: {fixes_applied}")
    print(f"💾 Backup directory: {backup_dir}")
    print()

    # Save detailed log
    log_file = BASE_DIR / f"indentation_fix_log_{timestamp}.txt"
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"INDENTATION FIX REPORT - {datetime.now().isoformat()}\n")
        lf.write("=" * 60 + "\n\n")
        lf.write(f"Files processed: {len(python_files)}\n")
        lf.write(f"Files fixed: {fixed_files}\n")
        lf.write(f"Total fixes applied: {fixes_applied}\n")
        lf.write(f"Backup directory: {backup_dir}\n\n")
        lf.write("DETAILED LOG:\n")
        lf.write("-" * 20 + "\n")
        for log_entry in fix_log:
            lf.write(log_entry + "\n")

    print(f"📋 Detailed log saved: {log_file}")

    # Create summary DataFrame-style output
    if fixed_files > 0:
        print()
        print("📋 FIXED FILES SUMMARY:")
        print("-" * 50)
        for i, log_entry in enumerate(
            [line for line in fix_log if line.startswith("✅")], 1
        ):
            file_info = log_entry.replace("✅ Fixed: ", "")
            print(f"{i:3}. {file_info}")

    return fixed_files, fixes_applied, str(backup_dir)


if __name__ == "__main__":
    fixed_files, fixes_applied, backup_location = main()
    print(f"\n🎉 Indentation fixing complete!")
    print(
        f"   Files: {fixed_files} | Fixes: {fixes_applied} | Backup: {backup_location}"
    )
