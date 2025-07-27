#!/usr/bin/env python3
"""
Final Arrow Syntax Fixer
=========================

Emergency system to fix the remaining 5,978 syntax errors across the codebase.
Most errors are the same pattern: function type annotations with ") -> type:" instead of ") -> type:"

This tool will systematically fix all files with this critical syntax error pattern.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FinalArrowSyntaxFixer:
    """Emergency fixer for type annotation arrow syntax errors"""

    def __init__(self):
    self.fixes_applied = 0
    self.files_processed = 0
    self.files_fixed = 0
    self.error_patterns = [
            # Pattern 1: ") -> TypeHint:" - most common
    (r'\)\s*>\s*([A-Za-z_][A-Za-z0-9_\[\],\s]*?):', r') -> \1:'),
            # Pattern 2: ") -> TypeHint[Generic]:" - generics
    (r'\)\s*>\s*([A-Za-z_][A-Za-z0-9_]*\[[^\]]+\]):', r') -> \1:'),
            # Pattern 3: "def func() -> Type:" - standalone functions
    (
    r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*>\s*([A-Za-z_][A-Za-z0-9_\[\],\s]*?):',
    r'def \g<1>() -> \2:',
    ),
    ]

    def fix_arrow_syntax_in_content(self, content: str) -> Tuple[str, int]:
    """Fix arrow syntax errors in file content"""
    fixes_in_file = 0
    fixed_content = content

        # Apply each pattern fix
        for pattern, replacement in self.error_patterns:
    old_content = fixed_content
    fixed_content = re.sub(pattern, replacement, fixed_content)

            # Count fixes made by this pattern
    pattern_fixes = len(re.findall(pattern, old_content))
    fixes_in_file += pattern_fixes

            if pattern_fixes > 0:
    logger.debug(f"Applied {pattern_fixes} fixes for pattern: {pattern}")

    return fixed_content, fixes_in_file

    def validate_syntax(self, content: str) -> bool:
    """Check if Python content has valid syntax"""
        try:
    ast.parse(content)
    return True
        except SyntaxError:
    return False

    def fix_file(self, file_path: Path) -> Dict[str, Any]:
    """Fix arrow syntax errors in a single file"""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

            # Check if file needs fixing
    needs_fixing = any(
    re.search(pattern, original_content)
                for pattern, _ in self.error_patterns
    )
            if not needs_fixing:
    return {
    'status': 'skipped',
    'reason': 'No arrow syntax errors found',
    'fixes_applied': 0,
    }

            # Apply fixes
    fixed_content, fixes_applied = self.fix_arrow_syntax_in_content(
    original_content
    )

            if fixes_applied == 0:
    return {
    'status': 'no_changes',
    'reason': 'No fixes needed',
    'fixes_applied': 0,
    }

            # Validate syntax
            if not self.validate_syntax(fixed_content):
    return {
    'status': 'syntax_error',
    'reason': 'Fixed content has syntax errors',
    'fixes_applied': fixes_applied,
    }

            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

    self.fixes_applied += fixes_applied
    self.files_fixed += 1

    return {
    'status': 'fixed',
    'reason': f'Applied {fixes_applied} arrow syntax fixes',
    'fixes_applied': fixes_applied,
    }

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")
    return {'status': 'error', 'reason': str(e), 'fixes_applied': 0}

    def fix_all_files(self) -> Dict[str, Any]:
    """Fix arrow syntax errors in all Python files"""
    logger.info("🔧 Starting final arrow syntax repair...")

        # Find all Python files
    python_files = list(Path('.').rglob('*.py'))
    logger.info(f"Found {len(python_files)} Python files to check")

    results = {
    'files_processed': 0,
    'files_fixed': 0,
    'total_fixes': 0,
    'file_results': {},
    'errors': [],
    }

        for file_path in python_files:
    self.files_processed += 1

    logger.info(
    f"Processing {file_path} ({self.files_processed}/{len(python_files)})"
    )

    result = self.fix_file(file_path)
    results['file_results'][str(file_path)] = result

            if result['status'] == 'fixed':
    results['files_fixed'] += 1
    results['total_fixes'] += result['fixes_applied']
    logger.info(f"✅ Fixed {result['fixes_applied']} issues in {file_path}")
            elif result['status'] == 'error':
    results['errors'].append(f"{file_path: {result['reason']}}")
    logger.error(f"❌ Error in {file_path: {result['reason']}}")

    results['files_processed'] = self.files_processed
    results['files_fixed'] = self.files_fixed
    results['total_fixes'] = self.fixes_applied

        # Summary
    logger.info(
    f"""
🎯 Final Arrow Syntax Repair Complete!
==========================================
Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Total fixes applied: {results['total_fixes']}
Errors encountered: {len(results['errors'])}
    """
    )

    return results


def main():
    """Main function to run the final arrow syntax fixer"""
    print("🔧 Final Arrow Syntax Fixer")
    print("=" * 50)

    fixer = FinalArrowSyntaxFixer()
    results = fixer.fix_all_files()

    if results['total_fixes'] > 0:
    print(f"\n✅ Successfully applied {results['total_fixes']} arrow syntax fixes!")
    print("🧪 Run 'python -m pytest tests/ -v' to verify functionality")
    print("📊 Run 'ruff check . --statistics' to see error reduction")
    else:
    print("\n⚠️ No arrow syntax errors found to fix")

    return results


if __name__ == "__main__":
    main()
