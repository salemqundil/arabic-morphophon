#!/usr/bin/env python3
"""
🛠 STEP 1: EASY SYNTAX FIXES
==========================

Fixes the 2 simple syntax errors identified in the validation report:
1. manual_test_fix.py: Trailing comma in import
2. tools\validate_tools_ast.py: Trailing comma in import

✅ Low-risk, high-confidence changes
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def create_backup(file_path):
    """Create a timestamped backup of the file"""
    backup_dir = Path("backups_easy_fixes")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{Path(file_path).name}_{timestamp}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"📁 Backup created: {backup_path}")
    return backup_path


def fix_trailing_comma_import(file_path, line_content):
    """Fix trailing comma in import statement"""
    print(f"🔧 Fixing trailing comma in: {file_path}")

    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    except UnicodeDecodeError:
    print(f"⚠️ Unicode error in {file_path}, skipping...")
    return False

    # Create backup
    create_backup(file_path)

    # Fix the specific line - remove trailing comma
    for i, line in enumerate(lines):
        if line.strip() == line_content.strip():
            # Remove trailing comma and add newline if needed
    fixed_line = line.rstrip().rstrip(',') + '\n'
    lines[i] = fixed_line
    print(f"   Line {i+1}: '{line.strip()}' → '{fixed_line.strip()}'")
    break

    # Write back the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
    print(f"✅ Fixed: {file_path}")
    return True
    except Exception as e:
    print(f"❌ Error writing {file_path}: {e}")
    return False


def main():
    """Apply the 2 easy syntax fixes"""
    print("🛠 STEP 1: EASY SYNTAX FIXES")
    print("=" * 50)

    fixes_applied = 0

    # Fix 1: manual_test_fix.py - trailing comma
    file1 = "manual_test_fix.py"
    if os.path.exists(file1):
        if fix_trailing_comma_import(file1, "from pathlib import Path,"):
    fixes_applied += 1
    else:
    print(f"⚠️ File not found: {file1}")

    # Fix 2: tools\validate_tools_ast.py - trailing comma
    file2 = Path("tools") / "validate_tools_ast.py"
    if file2.exists():
        if fix_trailing_comma_import(str(file2), "import ast,"):
    fixes_applied += 1
    else:
    print(f"⚠️ File not found: {file2}")

    print(f"\n📊 RESULTS:")
    print(f"Fixes applied: {fixes_applied}/2")
    print(f"Expected improvement: +{fixes_applied} files to clean status")

    if fixes_applied > 0:
    print(f"\n🎯 Next: Run syntax validator to confirm fixes")
    print(f"💡 Then proceed to Step 2 (Indentation Fixer)")


if __name__ == "__main__":
    main()
