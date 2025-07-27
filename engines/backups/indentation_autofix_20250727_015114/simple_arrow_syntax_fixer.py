#!/usr/bin/env python3
"""
Simple Arrow Syntax Fixer
=========================

Direct string replacement approach to fix all ") -> " type annotation errors.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import ast

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleArrowSyntaxFixer:
    """Simple string replacement for arrow syntax errors"""

    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.files_fixed = 0

    def fix_arrow_syntax_in_content(self, content: str) -> tuple[str, int]:
        """Fix arrow syntax errors using simple string replacement"""
        fixes_in_file = 0
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            original_line = line

            # Look for the pattern ") -> " followed by type annotation
            if ') ->' in line and ':' in line:
                # Replace ") -> " with ") -> "
                line = line.replace(') >', ') ->')
                if line != original_line:
                    fixes_in_file += 1
                    logger.debug(f"Fixed: {original_line.strip()} -> {line.strip()}")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixes_in_file

    def validate_syntax(self, content: str) -> bool:
        """Check if Python content has valid syntax"""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.debug(f"Syntax error: {e}")
            return False

    def fix_file(self, file_path: Path) -> Dict[str, Any]:
        """Fix arrow syntax errors in a single file"""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Check if file needs fixing
            if ') ->' not in original_content:
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
        logger.info("🔧 Starting simple arrow syntax repair...")

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

            if self.files_processed % 20 == 0:
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

        results['files_processed'] = self.files_processed
        results['files_fixed'] = self.files_fixed
        results['total_fixes'] = self.fixes_applied

        # Summary
        logger.info(
            f"""
🎯 Simple Arrow Syntax Repair Complete!
========================================
Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Total fixes applied: {results['total_fixes']}
Errors encountered: {len(results['errors'])}
        """
        )

        return results


def main():
    """Main function to run the simple arrow syntax fixer"""
    print("🔧 Simple Arrow Syntax Fixer")
    print("=" * 40)

    fixer = SimpleArrowSyntaxFixer()
    results = fixer.fix_all_files()

    if results['total_fixes'] > 0:
        print(f"\n✅ Successfully applied {results['total_fixes']} arrow syntax fixes!")
    else:
        print("\n⚠️ No arrow syntax errors found to fix")

    return results


if __name__ == "__main__":
    main()
