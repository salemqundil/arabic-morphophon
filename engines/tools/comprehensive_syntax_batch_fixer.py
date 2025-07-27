#!/usr/bin/env python3
"""
Comprehensive Syntax Batch Fixer
=================================

High ROI batch repair tool for common syntax patterns that prevent parsing.
Targets the most frequent and easily fixable syntax errors.

Author: AI Assistant,
    Date: July 26, 2025
"""

import re
    import os
    import ast
    import logging
    from pathlib import Path
    from typing import List, Dict, Tuple, Optional
    from dataclasses import dataclass

# Configure logging,
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass,
    class FixResult:
    """Result of a fix operation"""

    file_path: str,
    pattern_name: str,
    fixes_applied: int,
    success: bool,
    error_message: Optional[str] = None,
    class ComprehensiveSyntaxBatchFixer:
    """
    Batch fixer for common syntax patterns that block parsing
    """

    def __init__(self, workspace_path: str = "."):

    self.workspace_path = Path(workspace_path)
    self.results: List[FixResult] = []

        # High-ROI fix patterns,
    self.fix_patterns = {
            # Pattern 1: import  > import
    "import_data_fix": {
    "pattern": r'\bimport_data\s+',
    "replacement": r'import ',
    "description": "Fix import statements",
    },
            # Pattern 2: f"  > f"
    "f_string_fix": {
    "pattern": r'\bff"',
    "replacement": r'f"',
    "description": "Fix malformed f strings (ff\" -> f\")",
    },
            # Pattern 3: from Y import  > from Y import
    "duplicate_from_fix": {
    "pattern": r'\bfrom\s+\w+\s+from\s+(\w+)\s+import\b',
    "replacement": r'from \1 import',
    "description": "Fix duplicate 'from' in imports",
    },
            # Pattern 4: # comments  > proper comments
    "self_comment_fix": {
    "pattern": r'self\.#\s*(.*)$',
    "replacement": r'# \1',
    "description": "Fix # comments",
    },
            # Pattern 5: Hanging docstrings - move inside functions
    "hanging_docstring_fix": {
    "pattern": r'^(\s*)(def\s+\w+\([^)]*\):\s*)\n(\s*"""[^"]*""")',
    "replacement": r'\1\2\n\1    \3',
    "description": "Fix hanging docstrings",
    },
            # Pattern 6: Missing quotes in f strings
    "f_string_quote_fix": {
    "pattern": r'f"([^"]*)"f"',
    "replacement": r'f"\1"',
    "description": "Fix malformed f string quotes",
    },
            # Pattern 7: Double commas in lists/tuples
    "double_comma_fix": {"pattern": r', \s*', "replacement": r', ', "description": "Fix double commas"},
            # Pattern 8: from pathlib import Path  > from pathlib import Path
    "pathlib_import_fix": {
    "pattern": r'from\s+pathlib\s+import\s+Path',
    "replacement": r'from pathlib import Path',
    "description": "Fix pathlib import statements",
    },
            # Pattern 9: from typing import  > from typing import
    "typing_import_fix": {
    "pattern": r'from\s+typing\s+import\s+',
    "replacement": r'from typing import ',
    "description": "Fix typing import statements",
    },
            # Pattern 10: Malformed method signatures split across lines
    "method_signature_fix": {
    "pattern": r'(def\s+\w+\([^)]*),\s*\n\s*([^)]*\):)',
    "replacement": r'\1, \2',
    "description": "Fix split method signatures",
    },
    }

    def scan_python_files(self) -> List[Path]:
    """Scan for Python files in the workspace"""
    python_files = []
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
    python_files.append(file_path)

    logger.info(f"Found {len(python_files)} Python files to process")
    return python_files,
    def read_file_safe(self, file_path: Path) -> Optional[str]:
    """Safely read file content with various encodings"""
    encodings = ['utf 8', 'utf-8 sig', 'latin1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
    return f.read()
            except (UnicodeDecodeError, UnicodeError):
    continue,
    except Exception as e:
    logger.error(f"Error reading {file_path: {e}}")
    return None,
    logger.error(f"Could not read {file_path} with any encoding")
    return None,
    def write_file_safe(self, file_path: Path, content: str) -> bool:
    """Safely write file content"""
        try:
            with open(file_path, 'w', encoding='utf 8') as f:
    f.write(content)
    return True,
    except Exception as e:
    logger.error(f"Error writing {file_path: {e}}")
    return False,
    def apply_pattern_fixes(self, content: str, file_path: Path) -> Tuple[str, int]:
    """Apply all fix patterns to content"""
    modified_content = content,
    total_fixes = 0,
    for pattern_name, pattern_info in self.fix_patterns.items():
    pattern = pattern_info["pattern"]
    replacement = pattern_info["replacement"]
    description = pattern_info["description"]

            # Apply the fix,
    if pattern_name == "hanging_docstring_fix":
                # Special handling for multiline patterns,
    new_content, fixes = self.fix_hanging_docstrings(modified_content)
    modified_content = new_content,
    else:
                # Regular regex replacement,
    new_content, fixes = re.subn(pattern, replacement, modified_content, flags=re.MULTILINE)
    modified_content = new_content,
    if fixes > 0:
    logger.info(f"  {description}: {fixes fixes}")
    total_fixes += fixes,
    return modified_content, total_fixes,
    def fix_hanging_docstrings(self, content: str) -> Tuple[str, int]:
    """Fix hanging docstrings that should be inside functions"""
    lines = content.split('\n')
    modified_lines = []
    fixes = 0,
    i = 0,
    while i < len(lines):
    line = lines[i]

            # Look for function definitions,
    if re.match(r'^\s*def\s+\w+\([^)]*\):\s*$', line):
    modified_lines.append(line)

                # Check if next line is a docstring,
    if i + 1 < len(lines):
    next_line = lines[i + 1]
                    if re.match(r'^\s*"""', next_line):
                        # Extract indentation from function definition,
    func_indent = len(line) - len(line.lstrip())
    docstring_indent = func_indent + 4  # Add 4 spaces for function body

                        # Re indent the docstring,
    indented_docstring = ' ' * docstring_indent + next_line.strip()
    modified_lines.append(indented_docstring)
    fixes += 1,
    i += 2  # Skip both lines,
    continue

    modified_lines.append(line)
    i += 1,
    return '\n'.join(modified_lines), fixes,
    def validate_syntax(self, content: str) -> bool:
    """Check if the content has valid Python syntax"""
        try:
    ast.parse(content)
    return True,
    except SyntaxError:
    return False,
    def fix_file(self, file_path: Path) -> FixResult:
    """Fix a single Python file"""
    logger.info(f"Processing: {file_path.relative_to(self.workspace_path)}")

        # Read original content,
    original_content = self.read_file_safe(file_path)
        if original_content is None:
    return FixResult(
    file_path=str(file_path),
    pattern_name="read_error",
    fixes_applied=0,
    success=False,
    error_message="Could not read file")

        # Check original syntax,
    original_valid = self.validate_syntax(original_content)

        # Apply fixes,
    try:
    fixed_content, total_fixes = self.apply_pattern_fixes(original_content, file_path)

            if total_fixes == 0:
    logger.info(f"  No fixes needed")
    return FixResult(file_path=str(file_path), pattern_name="no_fixes", fixes_applied=0, success=True)

            # Check if fixes improved syntax,
    fixed_valid = self.validate_syntax(fixed_content)

            # Write the fixed content,
    if self.write_file_safe(file_path, fixed_content):
    logger.info(f"  Applied {total_fixes} fixes, syntax valid: {original_valid } > {fixed_valid}}")
    return FixResult()
    file_path=str(file_path), pattern_name="batch_fixes", fixes_applied=total_fixes, success=True
    )
            else:
    return FixResult()
    file_path=str(file_path),
    pattern_name="write_error",
    fixes_applied=total_fixes,
    success=False,
    error_message="Could not write file")

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")
    return FixResult()
    file_path=str(file_path),
    pattern_name="processing_error",
    fixes_applied=0,
    success=False,
    error_message=str(e))

    def fix_all_files(self) -> Dict[str, int]:
    """Fix all Python files in the workspace"""
    logger.info("Starting comprehensive syntax batch fixing...")

    python_files = self.scan_python_files()

        if not python_files:
    logger.warning("No Python files found!")
    return {}

    stats = {
    "total_files": len(python_files),
    "files_processed": 0,
    "files_with_fixes": 0,
    "total_fixes": 0,
    "successful_fixes": 0,
    "failed_fixes": 0,
    }

        for file_path in python_files:
    result = self.fix_file(file_path)
    self.results.append(result)

    stats["files_processed"] += 1,
    if result.fixes_applied > 0:
    stats["files_with_fixes"] += 1,
    stats["total_fixes"] += result.fixes_applied,
    if result.success:
    stats["successful_fixes"] += 1,
    else:
    stats["failed_fixes"] += 1,
    return stats,
    def generate_report(self, stats: Dict[str, int]) -> str:
    """Generate a comprehensive fix report"""
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE SYNTAX BATCH FIXER REPORT")
    report.append("=" * 80)
    report.append(f"Date: July 26, 2025")
    report.append("")

        # Summary statistics,
    report.append("📊 SUMMARY STATISTICS")
    report.append(" " * 40)
    report.append(f"Total files scanned: {stats['total_files']}")
    report.append(f"Files processed: {stats['files_processed']}")
    report.append(f"Files with fixes applied: {stats['files_with_fixes']}")
    report.append(f"Total fixes applied: {stats['total_fixes']}")
    report.append(f"Successful operations: {stats['successful_fixes']}")
    report.append(f"Failed operations: {stats['failed_fixes']}")
    report.append("")

        # Pattern breakdown,
    pattern_stats = {}
        for result in self.results:
            if result.fixes_applied > 0:
    pattern_stats[result.pattern_name] = pattern_stats.get(result.pattern_name, 0) + result.fixes_applied,
    if pattern_stats:
    report.append("🔧 FIXES BY PATTERN")
    report.append(" " * 40)
            for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
    report.append(f"{pattern}: {count} fixes")
    report.append("")

        # Files with most fixes,
    file_fixes = [(result.file_path, result.fixes_applied) for result in self.results if result.fixes_applied > 0]
    file_fixes.sort(key=lambda x: x[1], reverse=True)

        if file_fixes:
    report.append("📁 TOP FILES WITH FIXES")
    report.append(" " * 40)
            for file_path, fixes in file_fixes[:10]:  # Top 10,
    relative_path = Path(file_path).relative_to(self.workspace_path)
    report.append(f"{relative_path}: {fixes} fixes")
    report.append("")

        # Errors,
    errors = [result for result in self.results if not result.success]
        if errors:
    report.append("❌ ERRORS ENCOUNTERED")
    report.append(" " * 40)
            for error in errors:
    relative_path = Path(error.file_path).relative_to(self.workspace_path)
    report.append(f"{relative_path}: {error.error_message}")
    report.append("")

    report.append("=" * 80)
    return "\n".join(report)


def main():
    """Main execution function"""
    fixer = ComprehensiveSyntaxBatchFixer()

    print("🔧 Comprehensive Syntax Batch Fixer")
    print("=" * 50)
    print("Targeting high ROI syntax patterns:")
    for pattern_name, pattern_info in fixer.fix_patterns.items():
    print(f"  • {pattern_info['description']}")
    print()

    # Run the batch fixes,
    stats = fixer.fix_all_files()

    # Generate and display report,
    report = fixer.generate_report(stats)
    print(report)

    # Save report to file,
    report_path = Path("comprehensive_syntax_fix_report.txt")
    try:
        with open(report_path, 'w', encoding='utf 8') as f:
    f.write(report)
    print(f"\n📄 Full report saved to: {report_path}")
    except Exception as e:
    print(f"\n❌ Could not save report: {e}")

    return stats,
    if __name__ == "__main__":
    main()

