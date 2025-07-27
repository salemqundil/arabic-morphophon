#!/usr/bin/env python3
"""
🔧 Advanced Automated Syntax Fixer
Fixes the specific syntax patterns identified in the codebase.
"""

import ast
import re
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedSyntaxFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        self.backup_dir = Path(
            f"backups/syntax_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification."""
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        return backup_path

    def fix_indentation_issues(self, content: str) -> Tuple[str, int]:
        """Fix unexpected indentation in imports and other common patterns."""
        lines = content.split('\n')
        fixed_lines = []
        fixes = 0

        for i, line in enumerate(lines):
            # Fix indented imports that should be at top level
            if re.match(r'^\s+(import\s+|from\s+)', line):
                # Check if this is actually inside a function/class or if it's a stray indent
                prev_lines = lines[max(0, i - 5) : i]
                in_function_or_class = any(
                    re.match(
                        r'^(def\s+|class\s+|if\s+|elif\s+|else:|try:|except|with\s+|for\s+|while\s+)',
                        prev_line.strip(),
                    )
                    for prev_line in prev_lines
                )

                if not in_function_or_class:
                    # Remove leading whitespace from import
                    fixed_line = line.lstrip()
                    fixed_lines.append(fixed_line)
                    fixes += 1
                    logger.debug(
                        f"Fixed indented import: '{line.strip()}' -> '{fixed_line}'"
                    )
                else:
                    fixed_lines.append(line)

            # Fix other unexpected indents (like version strings)
            elif re.match(r'^\s+(__version__|__author__|__email__)', line):
                fixed_line = line.lstrip()
                fixed_lines.append(fixed_line)
                fixes += 1
                logger.debug(
                    f"Fixed indented module variable: '{line.strip()}' -> '{fixed_line}'"
                )

            # Fix indented docstrings at module level
            elif re.match(r'^\s+"""', line) and i < 10:  # Module docstring near top
                fixed_line = line.lstrip()
                fixed_lines.append(fixed_line)
                fixes += 1
                logger.debug(f"Fixed indented module docstring")

            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixes

    def fix_string_literal_issues(self, content: str) -> Tuple[str, int]:
        """Fix unterminated string literals and malformed docstrings."""
        fixes = 0

        # Fix the """" pattern (should be """)
        pattern1 = r'""""|""""(?:\s*\n)'
        if re.search(pattern1, content):
            content = re.sub(pattern1, '"""\n', content)
            fixes += re.search(pattern1, content, re.MULTILINE) is not None
            logger.debug(f"Fixed unterminated string literals (\"\"\"\")")

        # Fix incomplete docstrings
        pattern2 = r'"""(?:\s*\n\s*)?$'
        if re.search(pattern2, content, re.MULTILINE):
            content = re.sub(pattern2, '"""\nPass\n"""', content, flags=re.MULTILINE)
            fixes += 1
            logger.debug(f"Fixed incomplete docstring")

        return content, fixes

    def fix_trailing_comma_issues(self, content: str) -> Tuple[str, int]:
        """Fix trailing comma issues in imports."""
        fixes = 0

        # Fix trailing comma in import statements
        pattern = r'(from\s+[\w.]+\s+import\s+[\w\s,]+),\s*$'
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
            content = re.sub(pattern, r'\1', content, flags=re.MULTILINE)
            fixes += len(matches)
            logger.debug(f"Fixed {len(matches)} trailing comma issues in imports")

        return content, fixes

    def fix_invalid_syntax_issues(self, content: str) -> Tuple[str, int]:
        """Fix various invalid syntax patterns."""
        fixes = 0

        # Fix "import ast," -> "import ast"
        pattern1 = r'^import\s+([\w.]+),\s*$'
        matches = re.findall(pattern1, content, re.MULTILINE)
        if matches:
            content = re.sub(pattern1, r'import \1', content, flags=re.MULTILINE)
            fixes += len(matches)
            logger.debug(f"Fixed {len(matches)} invalid import comma issues")

        return content, fixes

    def validate_syntax(self, content: str) -> bool:
        """Validate that the content has valid Python syntax."""
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False

    def fix_file(self, file_path: Path) -> bool:
        """Fix a single Python file and return True if fixes were applied."""
        try:
            # Read original content
            original_content = file_path.read_text(encoding='utf-8')

            # Check if already valid
            if self.validate_syntax(original_content):
                logger.debug(f"✅ {file_path} - already valid syntax")
                return False

            # Apply fixes
            content = original_content
            total_fixes = 0

            # 1. Fix indentation issues
            content, fixes = self.fix_indentation_issues(content)
            total_fixes += fixes

            # 2. Fix string literal issues
            content, fixes = self.fix_string_literal_issues(content)
            total_fixes += fixes

            # 3. Fix trailing comma issues
            content, fixes = self.fix_trailing_comma_issues(content)
            total_fixes += fixes

            # 4. Fix invalid syntax issues
            content, fixes = self.fix_invalid_syntax_issues(content)
            total_fixes += fixes

            # Validate the result
            if not self.validate_syntax(content):
                logger.warning(
                    f"⚠️ {file_path} - fixes applied but syntax still invalid"
                )
                return False

            if total_fixes > 0:
                # Create backup
                self.backup_file(file_path)

                # Write fixed content
                file_path.write_text(content, encoding='utf-8')

                self.fixes_applied += total_fixes
                self.files_fixed += 1

                logger.info(f"🔧 {file_path} - {total_fixes} fixes applied")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {str(e)}")
            return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
        """Fix all Python files in directory."""
        logger.info(f"🔧 Starting automated syntax fixing in {directory}")

        files_processed = 0

        for file_path in directory.rglob('*.py'):
            # Skip backup directories, virtual environments, and hidden files
            if any(
                part.startswith('.')
                or part in ['backups', 'venv', '__pycache__', 'node_modules']
                for part in file_path.parts
            ):
                continue

            files_processed += 1
            self.fix_file(file_path)

        # Generate report
        report = {
            'files_processed': files_processed,
            'files_fixed': self.files_fixed,
            'total_fixes': self.fixes_applied,
            'success_rate': (self.files_fixed / max(files_processed, 1)) * 100,
            'backup_directory': str(self.backup_dir),
        }

        return report


def main():
    """Main entry point for automated syntax fixing."""
    logger.info("🔧 Starting Advanced Automated Syntax Fixing")

    fixer = AdvancedSyntaxFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🔧 AUTOMATED SYNTAX FIXING SUMMARY")
    print("=" * 70)
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"🔧 Files fixed: {report['files_fixed']}")
    print(f"⚡ Total fixes applied: {report['total_fixes']}")
    print(f"📊 Fix rate: {report['success_rate']:.1f}%")
    print(f"💾 Backups saved to: {report['backup_directory']}")

    if report['files_fixed'] > 0:
        print(f"\n✅ Successfully fixed {report['files_fixed']} files!")
        print("💡 Run syntax validation again to verify fixes")
    else:
        print("\n🤔 No files needed fixing or all files already had valid syntax")

    return report['files_fixed']


if __name__ == "__main__":
    main()
