#!/usr/bin/env python3
"""
Fix Return Type Arrow - Surgical Approach
==========================================

Fix ONLY the function return type arrow syntax errors:
- ") >" -> ") ->"

This is a surgical fix focusing on ONE specific pattern.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ReturnTypeArrowFixer:
    """Surgical fixer for function return type arrows"""

    def __init__(self):
    self.fixes_applied = 0
    self.files_processed = 0
    self.files_fixed = 0

    def fix_return_type_arrows(self, content: str) -> tuple[str, int]:
    """Fix return type arrow syntax: ') >' -> ') ->'"""
    fixes_in_file = 0
    lines = content.split('\n')
    fixed_lines = []

        for line in lines:
    original_line = line

            # Look specifically for function definitions with return type arrows
            # Pattern: ") >" followed by type annotation and ":"
            if ') ->' in line and ('def ' in line or line.strip().startswith(')')):
                # Replace ") >" with ") ->" only in function contexts
                if ':' in line:  # Must have colon for function definition
    line = line.replace(') >', ') ->')
                    if line != original_line:
    fixes_in_file += 1
    logger.debug(f"Fixed arrow: {original_line.strip()}")

    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_in_file

    def fix_file(self, file_path: Path) -> Dict[str, Any]:
    """Fix return type arrows in a single file"""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    original_content = f.read()

            # Check if file needs fixing
            if ') >' not in original_content:
    return {'status': 'skipped', 'reason': 'No return type arrow errors found', 'fixes_applied': 0}

            # Apply fixes
    fixed_content, fixes_applied = self.fix_return_type_arrows(original_content)

            if fixes_applied == 0:
    return {'status': 'no_changes', 'reason': 'No fixes needed', 'fixes_applied': 0}

            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

    self.fixes_applied += fixes_applied
    self.files_fixed += 1

    return {
    'status': 'fixed',
    'reason': f'Applied {fixes_applied} return type arrow fixes',
    'fixes_applied': fixes_applied,
    }

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")
    return {'status': 'error', 'reason': str(e), 'fixes_applied': 0}

    def fix_all_files(self) -> Dict[str, Any]:
    """Fix return type arrows in all Python files"""
    logger.info("🎯 Starting surgical return type arrow fix...")

        # Find all Python files
    python_files = list(Path('.').rglob('*.py'))
    logger.info(f"Found {len(python_files)} Python files to check")

    results = {'files_processed': 0, 'files_fixed': 0, 'total_fixes': 0, 'file_results': {}, 'errors': []}

        for file_path in python_files:
    self.files_processed += 1

    result = self.fix_file(file_path)
    results['file_results'][str(file_path)] = result

            if result['status'] == 'fixed':
    results['files_fixed'] += 1
    results['total_fixes'] += result['fixes_applied']
    logger.info(f"✅ Fixed {result['fixes_applied'] arrows} in {file_path}}")
            elif result['status'] == 'error':
    results['errors'].append(f"{file_path: {result['reason']}}")

    results['files_processed'] = self.files_processed
    results['files_fixed'] = self.files_fixed
    results['total_fixes'] = self.fixes_applied

        # Summary
    logger.info()
    f"""
🎯 Return Type Arrow Fix Complete!
==================================
Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Total fixes applied: {results['total_fixes']}
Errors encountered: {len(results['errors'])}
    """
    )

    return results


def main():
    """Main function"""
    print("🎯 Fix Return Type Arrow - Surgical Approach")
    print("=" * 50)

    fixer = ReturnTypeArrowFixer()
    results = fixer.fix_all_files()

    if results['total_fixes'] > 0:
    print(f"\n✅ Successfully applied {results['total_fixes']} return type arrow fixes!")
    print("🧪 Run 'python -m pytest tests/ -v' to verify tests still pass")
    else:
    print("\n⚠️ No return type arrow errors found")

    return results


if __name__ == "__main__":
    main()

