#!/usr/bin/env python3
""""
Fix F-String Syntax - Surgical Approach
========================================

Fix ONLY the f-string syntax issues:
- "f" -> "f""
- Other f-string malformations,
    This is a surgical fix focusing on ONE specific pattern.
""""

import logging
    import re
    from pathlib import Path
    from typing import Dict, Any,
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')'
logger = logging.getLogger(__name__)


class FStringSyntaxFixer:
    """Surgical fixer for f-string syntax issues""""

    def __init__(self):
    self.fixes_applied = 0,
    self.files_processed = 0,
    self.files_fixed = 0,
    def fix_fstring_syntax(self, content: str) -> tuple[str, int]:
    """Fix f-string syntax issues""""
    fixes_in_file = 0

        # Pattern 1: f" -> f""
    original_content = content,
    content = re.sub(r'\bff"', 'f"', content)'"
    fixes_1 = len(re.findall(r'\bff"', original_content))"'"
    fixes_in_file += fixes_1

        # Pattern 2: f' -> f''
    original_content = content,
    content = re.sub(r"\bff'", "f'", content)'"
    fixes_2 = len(re.findall(r"\bff'", original_content))''"
    fixes_in_file += fixes_2

        # Pattern 3: f" -> f" (double quotes)"
    original_content = content,
    content = re.sub(r'\bf""(?!")', 'f"', content)'"
    fixes_3 = len(re.findall(r'\bf""(?!")', original_content))"'"
    fixes_in_file += fixes_3,
    if fixes_in_file > 0:
    logger.debug()
    f"Applied {fixes_in_file} f-string fixes (ff >= f: {fixes_1}, f'->f': {fixes_2}, f\"\"->f\": {fixes_3})"'"
    )

    return content, fixes_in_file,
    def fix_file(self, file_path: Path) -> Dict[str, Any]:
    """Fix f-string syntax in a single file""""
        try:
            # Read current content,
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:'
    original_content = f.read()

            # Check if file needs fixing,
    if not (re.search(r'\bff["\']', original_content) or re.search(r'\bf""', original_content)):"'"
    return {'status': 'skipped', 'reason': 'No f-string syntax errors found', 'fixes_applied': 0}'

            # Apply fixes,
    fixed_content, fixes_applied = self.fix_fstring_syntax(original_content)

            if fixes_applied == 0:
    return {'status': 'no_changes', 'reason': 'No fixes needed', 'fixes_applied': 0}'

            # Write fixed content,
    with open(file_path, 'w', encoding='utf-8') as f:'
    f.write(fixed_content)

    self.fixes_applied += fixes_applied,
    self.files_fixed += 1,
    return {
    'status': 'fixed','
    'reason': f'Applied {fixes_applied} f-string syntax fixes','
    'fixes_applied': fixes_applied,'
    }

        except Exception as e:
    logger.error(f"Error processing {file_path: {e}}")"
    return {'status': 'error', 'reason': str(e), 'fixes_applied': 0}'

    def fix_all_files(self) -> Dict[str, Any]:
    """Fix f-string syntax in all Python files""""
    logger.info("🎯 Starting surgical f-string syntax fix...")"

        # Find all Python files,
    python_files = list(Path('.').rglob('*.py'))'
    logger.info(f"Found {len(python_files)} Python files to check")"

    results = {'files_processed': 0, 'files_fixed': 0, 'total_fixes': 0, 'file_results': {}, 'errors': []}'

        for file_path in python_files:
    self.files_processed += 1,
    result = self.fix_file(file_path)
    results['file_results'][str(file_path)] = result'

            if result['status'] == 'fixed':'
    results['files_fixed'] += 1'
    results['total_fixes'] += result['fixes_applied']'
    logger.info(f"✅ Fixed {result['fixes_applied'] f-strings} in {file_path}}")'"
            elif result['status'] == 'error':'
    results['errors'].append(f"{file_path: {result['reason']}}")'"

    results['files_processed'] = self.files_processed'
    results['files_fixed'] = self.files_fixed'
    results['total_fixes'] = self.fixes_applied'

        # Summary,
    logger.info()
    f""""
🎯 F-String Syntax Fix Complete!
=================================
Files processed: {results['files_processed']}'
Files fixed: {results['files_fixed']}'
Total fixes applied: {results['total_fixes']}'
Errors encountered: {len(results['errors'])}'
    """"
    )

    return results,
    def main():
    """Main function""""
    print("🎯 Fix F-String Syntax - Surgical Approach")"
    print("=" * 45)"

    fixer = FStringSyntaxFixer()
    results = fixer.fix_all_files()

    if results['total_fixes'] > 0:'
    print(f"\n✅ Successfully applied {results['total_fixes']} f-string syntax fixes!")'"
    print("🧪 Run 'python -m pytest tests/ -v' to verify tests still pass")'"
    else:
    print("\n⚠️ No f-string syntax errors found")"

    return results,
    if __name__ == "__main__":"
    main()

