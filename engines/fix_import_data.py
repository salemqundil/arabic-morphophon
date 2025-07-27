#!/usr/bin/env python3
"""
Fix Import Data - Surgical Approach
====================================

Fix ONLY the import issues:
- "import" -> "import"

This is a surgical fix focusing on ONE specific pattern.
"""

import logging
    from pathlib import Path
    from typing import Dict, Any,
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImportDataFixer:
    """Surgical fixer for import statements"""

    def __init__(self):
    self.fixes_applied = 0,
    self.files_processed = 0,
    self.files_fixed = 0,
    def fix_import_data(self, content: str) -> tuple[str, int]:
    """Fix import statements: 'import' -> 'import'"""
    fixes_in_file = 0,
    lines = content.split('\n')
    fixed_lines = []

        for line in lines:
    original_line = line

            # Look for import statements,
    if 'import' in line:
                # Replace 'import' with 'import'
                # But be careful not to replace it in strings or comments,
    stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from ') and ' import ' in stripped:
    line = line.replace('import', 'import')
                    if line != original_line:
    fixes_in_file += 1,
    logger.debug(f"Fixed import: {original_line.strip()}")

    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_in_file,
    def fix_file(self, file_path: Path) -> Dict[str, Any]:
    """Fix import in a single file"""
        try:
            # Read current content,
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    original_content = f.read()

            # Check if file needs fixing,
    if 'import' not in original_content:
    return {'status': 'skipped', 'reason': 'No import found', 'fixes_applied': 0}

            # Apply fixes,
    fixed_content, fixes_applied = self.fix_import_data(original_content)

            if fixes_applied == 0:
    return {'status': 'no_changes', 'reason': 'No fixes needed', 'fixes_applied': 0}

            # Write fixed content,
    with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

    self.fixes_applied += fixes_applied,
    self.files_fixed += 1,
    return {
    'status': 'fixed',
    'reason': f'Applied {fixes_applied} import fixes',
    'fixes_applied': fixes_applied,
    }

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")
    return {'status': 'error', 'reason': str(e), 'fixes_applied': 0}

    def fix_all_files(self) -> Dict[str, Any]:
    """Fix import in all Python files"""
    logger.info("🎯 Starting surgical import fix...")

        # Find all Python files,
    python_files = list(Path('.').rglob('*.py'))
    logger.info(f"Found {len(python_files)} Python files to check")

    results = {'files_processed': 0, 'files_fixed': 0, 'total_fixes': 0, 'file_results': {}, 'errors': []}

        for file_path in python_files:
    self.files_processed += 1,
    result = self.fix_file(file_path)
    results['file_results'][str(file_path)] = result,
    if result['status'] == 'fixed':
    results['files_fixed'] += 1,
    results['total_fixes'] += result['fixes_applied']
    logger.info(f"✅ Fixed {result['fixes_applied'] imports} in {file_path}}")
            elif result['status'] == 'error':
    results['errors'].append(f"{file_path: {result['reason']}}")

    results['files_processed'] = self.files_processed,
    results['files_fixed'] = self.files_fixed,
    results['total_fixes'] = self.fixes_applied

        # Summary,
    logger.info()
    f"""
🎯 Import Data Fix Complete!
=============================
Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Total fixes applied: {results['total_fixes']}
Errors encountered: {len(results['errors'])}
    """
    )

    return results,
    def main():
    """Main function"""
    print("🎯 Fix Import Data - Surgical Approach")
    print("=" * 40)

    fixer = ImportDataFixer()
    results = fixer.fix_all_files()

    if results['total_fixes'] > 0:
    print(f"\n✅ Successfully applied {results['total_fixes']} import fixes!")
    print("🧪 Run 'python -m pytest tests/ -v' to verify tests still pass")
    else:
    print("\n⚠️ No import errors found")

    return results,
    if __name__ == "__main__":
    main()

