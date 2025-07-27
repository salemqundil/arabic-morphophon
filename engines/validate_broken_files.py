#!/usr/bin/env python3
"""
Comprehensive AST Validator with Report Generation,
    Scans all Python files and generates detailed syntax error report
"""

import ast
    import os
    from collections import defaultdict,
    def validate_all_files():
    """Validate all Python files and generate comprehensive report"""
    error_patterns = defaultdict(list)
    total_files = 0,
    broken_files = 0,
    with open("syntax_error_report.txt", "w", encoding="utf-8") as report:
    report.write("🔍 COMPREHENSIVE SYNTAX ERROR REPORT\n")
    report.write("=" * 60 + "\n\n")

        for root, _, files in os.walk("."):
            for f in files:
                if f.endswith(".py"):
    total_files += 1,
    path = os.path.join(root, f)
                    try:
                        with open(path, encoding="utf-8") as file:
    ast.parse(file.read())
                    except SyntaxError as e:
    broken_files += 1,
    error_msg = f"❌ {path}: Line {e.lineno} - {e.msg}"
    report.write(error_msg + "\n")

                        # Categorize error patterns,
    if "-> " in str(e.msg):
    error_patterns["arrow_comparison"].append(path)
                        elif "f-string" in str(e.msg).lower():
    error_patterns["f_string"].append(path)
                        elif "import" in str(e.msg).lower():
    error_patterns["import_issue"].append(path)
                        elif "logging" in str(e.msg).lower():
    error_patterns["logging_config"].append(path)
                        elif "unterminated" in str(e.msg).lower():
    error_patterns["unterminated_string"].append(path)
                        else:
    error_patterns["other"].append(path)

        # Generate summary,
    report.write(f"\n📊 SUMMARY:\n")
    report.write(f"Total Python files: {total_files}\n")
    report.write(f"Files with syntax errors: {broken_files}\n")
    report.write(
    f"Success rate: {((total_files - broken_files) / total_files * 100):.1f}%\n\n"
    )

        # Error pattern breakdown,
    report.write("🏷️ ERROR PATTERNS:\n")
        for pattern, files in error_patterns.items():
    report.write(f"  {pattern}: {len(files)} files\n")
            for file in files[:5]:  # Show first 5 examples,
    report.write(f"    - {file}\n")
            if len(files) > 5:
    report.write(f"    ... and {len(files) - 5} more\n")
    report.write("\n")

    print(f"📄 Report generated: syntax_error_report.txt")
    print(f"📊 Status: {broken_files}/{total_files} files have syntax errors")
    return error_patterns,
    if __name__ == "__main__":
    validate_all_files()
