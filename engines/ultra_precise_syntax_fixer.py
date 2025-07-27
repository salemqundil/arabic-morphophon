#!/usr/bin/env python3
"""
🎯 Ultra-Precise Syntax Fixer
Fixes the most stubborn syntax issues with surgical precision.
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


class UltraPreciseSyntaxFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_fixed = 0
    self.backup_dir = Path(
    f"backups/ultra_precise_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

    def fix_indentation_issues_ultra_precise(self, content: str) -> Tuple[str, int]:
    """Fix indentation issues with ultra precision."""
    lines = content.split('\n')
    fixed_lines = []
    fixes = 0

    i = 0
        while i < len(lines):
    line = lines[i]
    original_line = line

            # Ultra-precise pattern matching for problematic indentations

            # 1. Fix import statements with leading spaces at module level
            if re.match(r'^\s+(import\s+|from\s+)', line):
                # Check if this is truly at module level by looking at context
    context_lines = lines[max(0, i - 3) : i]

                # If previous lines are comments, docstrings, or other module-level items
    in_module_level = True
                for prev_line in reversed(context_lines):
    stripped = prev_line.strip()
                    if stripped and not (
    stripped.startswith('#')
    or stripped.startswith('"""')
    or stripped.endswith('"""')
    or stripped.startswith('"""')
    or 'pylint:' in stripped
    or 'flake8:' in stripped
    or 'mypy:' in stripped
    or stripped.startswith('__')
    or stripped.startswith('from __future__')
    or stripped.startswith('import')
    or stripped.startswith('from ')
    ):
                        if not (
    stripped.startswith('def ') or stripped.startswith('class ')
    ):
    in_module_level = False
    break

                if in_module_level:
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed module-level import: '{line.strip()}'")
                else:
    fixed_lines.append(line)

            # 2. Fix logging.basicConfig calls that are wrongly indented
            elif re.match(r'^\s+(logging\.basicConfig)', line):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed logging config: '{line.strip()}'")

            # 3. Fix logger assignments that are wrongly indented
            elif re.match(r'^\s+(logger\s*=)', line):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed logger assignment: '{line.strip()}'")

            # 4. Fix module-level variables
            elif re.match(r'^\s+(__version__|__author__|__email__)', line):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed module variable: '{line.strip()}'")

            else:
    fixed_lines.append(line)

    i += 1

    return '\n'.join(fixed_lines), fixes

    def fix_string_issues_ultra_precise(self, content: str) -> Tuple[str, int]:
    """Fix string issues with ultra precision."""
    fixes = 0
    original_content = content

        # Fix """" -> """
    content = re.sub(r'""""', '"""', content)
    fixes += len(re.findall(r'""""', original_content))

        # Fix incomplete docstrings at module level
    lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == '"""' and i < 15:  # Likely module docstring
                # Check if it's not properly closed
    remaining_lines = lines[i + 1 : i + 10]
                if not any(
    '"""' in remaining_line for remaining_line in remaining_lines
    ):
    lines[i] = '"""\nModule documentation.\n"""'
    fixes += 1

    content = '\n'.join(lines)
    return content, fixes

    def fix_comma_issues_ultra_precise(self, content: str) -> Tuple[str, int]:
    """Fix comma issues with ultra precision."""
    fixes = 0

        # Fix trailing commas in imports like "from collections import Counter, defaultdict  # noqa: F401,"
    pattern = r'(from\s+[\w.]+\s+import\s+[^#\n]+),(\s*#.*)?$'
    matches = list(re.finditer(pattern, content, re.MULTILINE))
        for match in reversed(matches):  # Reverse to maintain positions
    start, end = match.span()
    before_comma = match.group(1)
    comment_part = match.group(2) if match.group(2) else ''
    replacement = before_comma + comment_part
    content = content[:start] + replacement + content[end:]
    fixes += 1

        # Fix simple import trailing commas
    pattern2 = r'^(import\s+[\w.]+),\s*$'
    content = re.sub(pattern2, r'\1', content, flags=re.MULTILINE)
    fixes += len(re.findall(pattern2, content, re.MULTILINE))

    return content, fixes

    def validate_syntax(self, content: str) -> Tuple[bool, str]:
    """Validate syntax and return error message if invalid."""
        try:
    ast.parse(content)
    return True, ""
        except SyntaxError as e:
    return False, f"Line {e.lineno}: {e.msg}"

    def fix_file(self, file_path: Path) -> bool:
    """Fix a single Python file and return True if fixes were applied."""
        try:
            # Read original content
    original_content = file_path.read_text(encoding='utf-8')

            # Check if already valid
    is_valid, error_msg = self.validate_syntax(original_content)
            if is_valid:
    logger.debug(f"✅ {file_path} - already valid syntax")
    return False

    logger.debug(f"🔍 {file_path} - syntax error: {error_msg}")

            # Apply fixes step by step
    content = original_content
    total_fixes = 0

            # 1. Fix indentation issues
    content, fixes = self.fix_indentation_issues_ultra_precise(content)
    total_fixes += fixes

            # 2. Fix string issues
    content, fixes = self.fix_string_issues_ultra_precise(content)
    total_fixes += fixes

            # 3. Fix comma issues
    content, fixes = self.fix_comma_issues_ultra_precise(content)
    total_fixes += fixes

            # Validate the result
    is_valid, error_msg = self.validate_syntax(content)

            if is_valid:
                if total_fixes > 0:
                    # Create backup
    self.backup_file(file_path)

                    # Write fixed content
    file_path.write_text(content, encoding='utf-8')

    self.fixes_applied += total_fixes
    self.files_fixed += 1

    logger.info(
    f"✅ {file_path} - {total_fixes} fixes applied, now valid!"
    )
    return True
                else:
    logger.debug(f"✅ {file_path} - already valid")
    return False
            else:
                if total_fixes > 0:
    logger.warning(
    f"⚠️ {file_path} - {total_fixes} fixes applied but still: {error_msg}"
    )
                else:
    logger.debug(f"❌ {file_path} - no fixes possible for: {error_msg}")
    return False

        except Exception as e:
    logger.error(f"❌ Error fixing {file_path}: {str(e)}")
    return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
    """Fix all Python files in directory."""
    logger.info(f"🎯 Starting ultra-precise syntax fixing in {directory}")

    files_processed = 0

        for file_path in directory.rglob('*.py'):
            # Skip backup directories and system files
            if any(
    part.startswith('.') or part in ['backups', 'venv', '__pycache__']
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
    """Main entry point for ultra-precise syntax fixing."""
    logger.info("🎯 Starting Ultra-Precise Syntax Fixing")

    fixer = UltraPreciseSyntaxFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🎯 ULTRA-PRECISE SYNTAX FIXING SUMMARY")
    print("=" * 70)
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"✅ Files fixed: {report['files_fixed']}")
    print(f"⚡ Total fixes applied: {report['total_fixes']}")
    print(f"📊 Success rate: {report['success_rate']:.1f}%")
    print(f"💾 Backups saved to: {report['backup_directory']}")

    if report['files_fixed'] > 0:
    print(f"\n🎉 Successfully fixed {report['files_fixed']} files!")
    print("💡 Run syntax validation again to verify fixes")
    else:
    print("\n🤔 No files needed fixing or were already valid")

    return report['files_fixed']


if __name__ == "__main__":
    main()
