#!/usr/bin/env python3
"""
🎯 Surgical Indentation Fixer
Fixes specific indentation issues in Python files.
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


class SurgicalIndentationFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_fixed = 0
    self.backup_dir = Path(
    f"backups/indentation_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

    def fix_indentation_issues(self, content: str) -> Tuple[str, int]:
    """Fix unexpected indentation issues comprehensively."""
    lines = content.split('\n')
    fixed_lines = []
    fixes = 0
    in_function = False
    in_class = False
    brace_depth = 0

        for i, line in enumerate(lines):
    original_line = line

            # Track if we're inside a function or class
    stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('class '):
    in_function = True
    in_class = stripped.startswith('class ')
            elif stripped == '' or (
    not line.startswith(' ') and not line.startswith('\t')
    ):
                if not stripped.startswith('@'):  # decorators are ok
    in_function = False
    in_class = False

            # Count braces for context
    brace_depth += line.count('(') - line.count(')')
    brace_depth += line.count('[') - line.count(']')
    brace_depth += line.count('{') - line.count('}')

            # Fix various indentation issues
            if (
    line.startswith('    ')
    and not in_function
    and not in_class
    and brace_depth <= 0
    ):
                # Module-level statements that shouldn't be indented
                if any(
    line.strip().startswith(keyword)
                    for keyword in [
    'import ',
    'from ',
    '__version__',
    '__author__',
    '__email__',
    'logger',
    'logging.',
    'def ',
    'class ',
    'if __name__',
    ]
    ):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed indented module statement: '{line.strip()}'")
    continue

            # Fix docstrings that are indented when they shouldn't be
            if (
    line.strip().startswith('"""')
    and i < 20
    and not in_function
    and not in_class
    ):
                if line.startswith('    '):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed indented module docstring")
    continue

            # Fix comments that are wrongly indented
            if (
    line.strip().startswith('#')
    and line.startswith('    ')
    and not in_function
    and not in_class
    ):
    fixed_line = line.lstrip()
    fixed_lines.append(fixed_line)
    fixes += 1
    logger.debug(f"Fixed indented comment")
    continue

    fixed_lines.append(original_line)

    return '\n'.join(fixed_lines), fixes

    def fix_string_issues(self, content: str) -> Tuple[str, int]:
    """Fix string literal issues."""
    fixes = 0

        # Fix """" -> """
    pattern = r'""""'
    count = len(re.findall(pattern, content))
        if count > 0:
    content = re.sub(pattern, '"""', content)
    fixes += count
    logger.debug(f"Fixed {count} unterminated string literals")

    return content, fixes

    def fix_comma_issues(self, content: str) -> Tuple[str, int]:
    """Fix trailing comma issues."""
    fixes = 0

        # Fix trailing commas in imports
    pattern = r'(from\s+[\w.]+\s+import\s+[^,\n]+),\s*$'
    matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
    content = re.sub(pattern, r'\1', content, flags=re.MULTILINE)
    fixes += len(matches)
    logger.debug(f"Fixed {len(matches)} trailing commas in imports")

        # Fix "import module," -> "import module"
    pattern2 = r'^(import\s+[\w.]+),\s*$'
    matches2 = re.findall(pattern2, content, re.MULTILINE)
        if matches2:
    content = re.sub(pattern2, r'\1', content, flags=re.MULTILINE)
    fixes += len(matches2)
    logger.debug(f"Fixed {len(matches2)} trailing commas in simple imports")

    return content, fixes

    def validate_syntax(self, content: str) -> bool:
    """Validate that the content has valid Python syntax."""
        try:
    ast.parse(content)
    return True
        except SyntaxError as e:
    logger.debug(f"Syntax error: {e}")
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

            # Apply fixes step by step
    content = original_content
    total_fixes = 0

            # 1. Fix indentation issues
    content, fixes = self.fix_indentation_issues(content)
    total_fixes += fixes

            # 2. Fix string issues
    content, fixes = self.fix_string_issues(content)
    total_fixes += fixes

            # 3. Fix comma issues
    content, fixes = self.fix_comma_issues(content)
    total_fixes += fixes

            # Check if we made progress
            if self.validate_syntax(content):
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
    f"⚠️ {file_path} - {total_fixes} fixes applied but syntax still invalid"
    )
                else:
    logger.debug(f"❌ {file_path} - no fixes could be applied")
    return False

        except Exception as e:
    logger.error(f"❌ Error fixing {file_path}: {str(e)}")
    return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
    """Fix all Python files in directory."""
    logger.info(f"🎯 Starting surgical indentation fixing in {directory}")

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
    """Main entry point for surgical indentation fixing."""
    logger.info("🎯 Starting Surgical Indentation Fixing")

    fixer = SurgicalIndentationFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🎯 SURGICAL INDENTATION FIXING SUMMARY")
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
