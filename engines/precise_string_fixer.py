#!/usr/bin/env python3
"""
🎯 Precise String Fixer - Phase 1B
Fixes specific string literal patterns identified in the codebase.
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


class PreciseStringFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_fixed = 0
    self.backup_dir = Path(
    f"backups/precise_string_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

    def fix_string_issues_precise(self, content: str) -> Tuple[str, int]:
    """Fix string issues with surgical precision."""
    fixes = 0
    lines = content.split('\n')

        for i, line in enumerate(lines):
    original_line = line

            # Fix 1: """" at start of line -> """
            if line.strip().startswith('""""'):
    lines[i] = line.replace('""""', '"""', 1)
    fixes += 1
    logger.debug(f"Line {i+1}: Fixed '\"\"\"\"' -> '\"\"\"'")

            # Fix 2: Lines ending with """" -> """
            elif line.strip().endswith('""""'):
    lines[i] = line.replace('""""', '"""')
    fixes += 1
    logger.debug(f"Line {i+1}: Fixed ending '\"\"\"\"' -> '\"\"\"'")

            # Fix 3: Mixed quote issues - "" at end where it should be """
            elif line.strip().endswith("''") and i > 0 and '"""' in lines[i - 1]:
                # This looks like a broken docstring continuation
    lines[i] = line.replace("''", '')
    fixes += 1
    logger.debug(f"Line {i+1}: Fixed broken docstring continuation")

    content = '\n'.join(lines)

        # Fix 4: Handle multi-line docstring issues
        # Look for patterns like """"..."""" and fix them
    content = re.sub(r'""""([^"]*?)""""', r'"""\1"""', content, flags=re.DOTALL)
    fixes += len(re.findall(r'""""[^"]*?""""', content, re.DOTALL))

    return content, fixes

    def validate_syntax(self, content: str) -> Tuple[bool, str]:
    """Validate syntax and return error message if invalid."""
        try:
    ast.parse(content)
    return True, ""
        except SyntaxError as e:
    return False, f"Line {e.lineno}: {e.msg}"

    def has_string_error(self, file_path: Path) -> bool:
    """Check if file has string-related syntax errors."""
        try:
    content = file_path.read_text(encoding='utf-8')
    ast.parse(content)
    return False
        except SyntaxError as e:
    error_msg = e.msg or ""
    return any(
    keyword in error_msg
                for keyword in [
    "unterminated string",
    "unterminated triple",
    "EOL while scanning",
    "EOF while scanning",
    ]
    )
        except Exception:
    return False

    def fix_file(self, file_path: Path) -> bool:
    """Fix a single Python file if it has string issues."""
        try:
            # Check if this file has string-related errors
            if not self.has_string_error(file_path):
    return False

            # Read original content
    original_content = file_path.read_text(encoding='utf-8')

            # Apply string fixes
    content, fixes = self.fix_string_issues_precise(original_content)

            # Validate the result
    is_valid, error_msg = self.validate_syntax(content)

            if is_valid and fixes > 0:
                # Create backup
    self.backup_file(file_path)

                # Write fixed content
    file_path.write_text(content, encoding='utf-8')

    self.fixes_applied += fixes
    self.files_fixed += 1

    logger.info(
    f"✅ {file_path.name} - {fixes} string fixes applied, now valid!"
    )
    return True
            elif fixes > 0:
    logger.warning(
    f"⚠️ {file_path.name} - {fixes} fixes applied but still invalid: {error_msg}"
    )
    return False
            else:
    logger.debug(f"❌ {file_path.name} - no applicable string fixes")
    return False

        except Exception as e:
    logger.error(f"❌ Error fixing {file_path}: {str(e)}")
    return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
    """Fix all Python files with string issues in directory."""
    logger.info(f"🎯 Starting precise string fixing in {directory}")

    files_processed = 0
    target_files = []

        # First, identify target files
        for file_path in directory.rglob('*.py'):
            # Skip backup directories and system files
            if any(
    part.startswith('.') or part in ['backups', 'venv', '__pycache__']
                for part in file_path.parts
    ):
    continue

            if self.has_string_error(file_path):
    target_files.append(file_path)

    logger.info(f"🎯 Found {len(target_files)} files with string issues")

        # Now fix them
        for file_path in target_files:
    files_processed += 1
    self.fix_file(file_path)

        # Generate report
    report = {
    'target_files': len(target_files),
    'files_processed': files_processed,
    'files_fixed': self.files_fixed,
    'total_fixes': self.fixes_applied,
    'success_rate': (self.files_fixed / max(len(target_files), 1)) * 100,
    'backup_directory': str(self.backup_dir),
    }

    return report


def main():
    """Main entry point for precise string fixing."""
    logger.info("🎯 Starting Precise String Fixing - Phase 1B")

    fixer = PreciseStringFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🎯 PRECISE STRING FIXING SUMMARY - PHASE 1B")
    print("=" * 70)
    print(f"🎯 Target files identified: {report['target_files']}")
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"✅ Files fixed: {report['files_fixed']}")
    print(f"⚡ Total fixes applied: {report['total_fixes']}")
    print(f"📊 Success rate: {report['success_rate']:.1f}%")
    print(f"💾 Backups saved to: {report['backup_directory']}")

    if report['files_fixed'] > 0:
    print(f"\n🎉 Phase 1B SUCCESS: Fixed {report['files_fixed']} files!")
    print("📈 Improvement achieved in string literal syntax")
    print("💡 Run syntax validation to verify improvements")
    else:
    print("\n🤔 String issues require more complex fixes")
    print("💡 Consider manual review of the most problematic files")

    return report['files_fixed']


if __name__ == "__main__":
    main()
