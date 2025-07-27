#!/usr/bin/env python3
"""
🧪 Targeted Syntax Scanner - Isolate Exact Synt    print(f"🎯 TOTAL SYNTAX ERRORS: {len(errors)} files")
    print(f"📁 TOTAL FILES SCANNED: {sum(1 for _ in Path('.').rglob('*.py'))}") Errors,
    Scans all Python files and reports specific syntax issues for targeted fixing.
"""

import ast
    import os
    from collections import defaultdict
    from pathlib import Path,
    def scan_syntax_errors():
    """Scan for unparseable files and categorize common error patterns."""

    print("🧪 Scanning for unparseable files...\n")

    errors = []
    error_patterns = defaultdict(list)

    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
    path = os.path.join(root, f)
                try:
                    with open(path, encoding="utf-8") as file:
    content = file.read()
    ast.parse(content)
                except SyntaxError as e:
    error_info = {
    'file': path,
    'line': e.lineno,
    'message': e.msg,
    'text': e.text.strip() if e.text else "",
    }
    errors.append(error_info)

                    # Categorize common patterns,
    if "'->' used as comparison" in str(e.msg) or "->" in str(
    e.text or ""
    ):
    error_patterns["arrow_comparison"].append(error_info)
                    elif "import ()" in str(e.text or ""):
    error_patterns["empty_import"].append(error_info)
                    elif "f\"" in str(e.text or "") or "f'" in str(e.text or ""):
    error_patterns["fstring_issue"].append(error_info)
                    elif "def " in str(e.text or "") and "()" in str(e.text or ""):
    error_patterns["empty_function_args"].append(error_info)
                    elif "unterminated" in str(e.msg).lower():
    error_patterns["unterminated_string"].append(error_info)
                    elif "level=logging.INFO," in str(e.text or ""):
    error_patterns["logging_config"].append(error_info)
                    else:
    error_patterns["other"].append(error_info)

    print(f"❌ {path}: Line {e.lineno} → {e.msg}")
                    if e.text:
    print(f"   Text: {e.text.strip()}")
    print()
                except Exception as e:
    print(f"⚠️  {path}: Cannot read file - {e}")

    # Print summary by error type,
    print("\n" + "=" * 80)
    print("📊 ERROR CATEGORIZATION SUMMARY")
    print("=" * 80)

    for pattern, pattern_errors in error_patterns.items():
    print(
    f"\n🔍 {pattern.upper().replace('_', ' ')} ({len(pattern_errors)} files):"
    )
        for error in pattern_errors[:5]:  # Show first 5 examples,
    print(f"   • {error['file']}:{error['line']} - {error['message']}")
        if len(pattern_errors) > 5:
    print(f"   ... and {len(pattern_errors) - 5} more")

    print(f"\n🎯 TOTAL SYNTAX ERRORS: {len(errors)} files")
    print(f"📁 TOTAL FILES SCANNED: {sum(1 for _ in Path('.').rglob('*.py'))}")

    return errors, error_patterns,
    if __name__ == "__main__":
    scan_syntax_errors()
