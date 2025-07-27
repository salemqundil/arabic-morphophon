#!/usr/bin/env python3
"""
Surgical Syntax Repair Tool
===========================

Conservative approach to fix only the most critical and safe syntax patterns.

Author: AI Assistant
Date: July 26, 2025
"""

import re
import ast
from pathlib import Path
from typing import Dict, Tuple


def is_parseable(content: str) -> bool:
    """Check if content can be parsed as Python"""
    try:
    ast.parse(content)
    return True
    except SyntaxError:
    return False


def safe_fix_import_data(content: str) -> Tuple[str, int]:
    """Safely fix import statements"""
    # Only fix clear import patterns
    pattern = r'\bimport_data\s+(\w+)'
    matches = re.findall(pattern, content)
    if matches:
    fixed = re.sub(pattern, r'import \1', content)
    return fixed, len(matches)
    return content, 0


def safe_fix_pathlib_import(content: str) -> Tuple[str, int]:
    """Fix pathlib import specifically"""
    pattern = r'from\s+pathlib\s+import\s+Path'
    if re.search(pattern, content):
    fixed = re.sub(pattern, 'from pathlib import Path', content)
    return fixed, 1
    return content, 0


def safe_fix_typing_import(content: str) -> Tuple[str, int]:
    """Fix typing import specifically"""
    pattern = r'from\s+typing\s+import\s+'
    if re.search(pattern, content):
    fixed = re.sub(pattern, 'from typing import ', content)
    return fixed, 1
    return content, 0


def safe_fix_self_comments(content: str) -> Tuple[str, int]:
    """Fix self.# comment patterns"""
    pattern = r'^\s*self\.#\s*(.*)$'
    matches = re.findall(pattern, content, re.MULTILINE)
    if matches:
    fixed = re.sub(pattern, r'        # \1', content, flags=re.MULTILINE)
    return fixed, len(matches)
    return content, 0


def safe_fix_double_commas(content: str) -> Tuple[str, int]:
    """Fix obvious double comma issues"""
    pattern = r',\s*Optional\b'
    matches = re.findall(pattern, content)
    if matches:
    fixed = re.sub(pattern, ', ', content)
    return fixed, len(matches)
    return content, 0


def surgical_fix_file(file_path: Path) -> Dict[str, int]:
    """Apply only safe, surgical fixes to a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
    original_content = f.read()
    except Exception:
    return {"read_error": 1}

    # Check if originally parseable
    originally_parseable = is_parseable(original_content)

    # Apply safe fixes one by one
    current_content = original_content
    total_fixes = 0

    # Only apply fixes that are known to be safe
    safe_fixes = [
    safe_fix_import_data,
    safe_fix_pathlib_import,
    safe_fix_typing_import,
    safe_fix_self_comments,
    safe_fix_double_commas,
    ]

    for fix_func in safe_fixes:
    new_content, fixes = fix_func(current_content)

        # Only keep the fix if it doesn't break parsing worse'
        if fixes > 0:
    new_parseable = is_parseable(new_content)

            # Keep fix if: it improves parsing OR it doesn't make it worse'
            if new_parseable or (not originally_parseable and not new_parseable):
    current_content = new_content
    total_fixes += fixes

    # Only write if we made improvements
    if total_fixes > 0:
        # Final safety check
    final_parseable = is_parseable(current_content)

        # Don't write if we made parsing worse'
        if not originally_parseable or final_parseable:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
    f.write(current_content)

    return {"fixes_applied": total_fixes, "syntax_improved": final_parseable and not originally_parseable}
            except Exception:
    return {"write_error": 1}

    return {"no_changes": 1}


def main():
    """Main execution with conservative approach"""
    workspace_path = Path(".")

    print("🔧 Surgical Syntax Repair Tool")
    print("=" * 50)
    print("Conservative approach - only safe, proven fixes")

    # Target specific problematic files
    target_files = [
    "citation_standardization_system.py",
    "precision_violation_fixer.py",
    "validate_citations.py",
    "violation_elimination_system.py",
    "ultimate_winsurf_eliminator.py",
    "core/base_engine.py",
    ]

    stats = {"files_processed": 0, "files_improved": 0, "total_fixes": 0, "syntax_improvements": 0}

    for file_name in target_files:
    file_path = workspace_path / file_name
        if not file_path.exists():
    continue

    print(f"\nProcessing: {file_name}")
    result = surgical_fix_file(file_path)

        if "fixes_applied" in result:
    fixes = result["fixes_applied"]
    improved = result.get("syntax_improved", False)

    print(f"  ✅ Applied {fixes} safe fixes}")
            if improved:
    print("  🎉 Syntax improved!")

    stats["files_improved"] += 1
    stats["total_fixes"] += fixes
            if improved:
    stats["syntax_improvements"] += 1
        else:
    print("  📝 No safe fixes available")

    stats["files_processed"] += 1

    print("\n📊 SURGICAL REPAIR RESULTS")
    print("=" * 50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files improved: {stats['files_improved']}")
    print(f"Total safe fixes: {stats['total_fixes']}")
    print(f"Syntax improvements: {stats['syntax_improvements']}")


if __name__ == "__main__":
    main()

