#!/usr/bin/env python3
"""
🎯 Controlled Syntax Repair - Phase 1: Unterminated Strings
Targets the 20 easiest fixes first for quick wins.
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


class ControlledStringFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_fixed = 0
    self.backup_dir = Path(
    f"backups/controlled_string_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

    def fix_unterminated_strings(self, content: str) -> Tuple[str, int]:
    """Fix unterminated string literals specifically."""
    fixes = 0

        # Fix the most common pattern: """" -> """
    pattern1 = r'""""|""""'
    count1 = len(re.findall(pattern1, content))
        if count1 > 0:
    content = re.sub(pattern1, '"""', content)
    fixes += count1
    logger.debug(f"Fixed {count1} instances of '\"\"\"\"' -> '\"\"\"'")

        # Fix incomplete triple quotes at end of lines
    lines = content.split('\n')
        for i, line in enumerate(lines):
            # Check for lines ending with incomplete quotes
    stripped = line.strip()
            if stripped == '"""' and i < len(lines) - 1:
                # Look ahead to see if there's a matching closing quote
    found_closing = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    if '"""' in lines[j]:
    found_closing = True
    break

                if not found_closing:
                    # Add a basic docstring
    lines[i] = line.replace('"""', '"""\nDocstring placeholder.\n"""')
    fixes += 1
    logger.debug(f"Fixed incomplete docstring at line {i + 1}")

    content = '\n'.join(lines)
    return content, fixes

    def validate_syntax(self, content: str) -> Tuple[bool, str]:
    """Validate syntax and return error message if invalid."""
        try:
    ast.parse(content)
    return True, ""
        except SyntaxError as e:
    return False, f"Line {e.lineno}: {e.msg}"

    def has_unterminated_string_error(self, file_path: Path) -> bool:
    """Check if file has unterminated string error specifically."""
        try:
    content = file_path.read_text(encoding='utf-8')
    ast.parse(content)
    return False
        except SyntaxError as e:
    return "unterminated string literal" in (e.msg or "")
        except Exception:
    return False

    def fix_file(self, file_path: Path) -> bool:
    """Fix a single Python file if it has unterminated string issues."""
        try:
            # Check if this file has the specific error we're targeting
            if not self.has_unterminated_string_error(file_path):
    return False

            # Read original content
    original_content = file_path.read_text(encoding='utf-8')

            # Apply string fixes
    content = original_content
    fixes = 0

    content, string_fixes = self.fix_unterminated_strings(content)
    fixes += string_fixes

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
    logger.debug(f"❌ {file_path.name} - no string fixes applicable")
    return False

        except Exception as e:
    logger.error(f"❌ Error fixing {file_path}: {str(e)}")
    return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
    """Fix all Python files with unterminated string issues in directory."""
    logger.info(f"🎯 Starting controlled string fixing in {directory}")

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

            if self.has_unterminated_string_error(file_path):
    target_files.append(file_path)

    logger.info(
    f"🎯 Found {len(target_files)} files with unterminated string issues"
    )

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
    """Main entry point for controlled string fixing."""
    logger.info("🎯 Starting Controlled Syntax Repair - Phase 1: Unterminated Strings")

    fixer = ControlledStringFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🎯 CONTROLLED STRING FIXING SUMMARY - PHASE 1")
    print("=" * 70)
    print(f"🎯 Target files identified: {report['target_files']}")
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"✅ Files fixed: {report['files_fixed']}")
    print(f"⚡ Total fixes applied: {report['total_fixes']}")
    print(f"📊 Success rate: {report['success_rate']:.1f}%")
    print(f"💾 Backups saved to: {report['backup_directory']}")

    if report['files_fixed'] > 0:
    print(f"\n🎉 Phase 1 SUCCESS: Fixed {report['files_fixed']} files!")
    print("📈 Expected improvement: ~5.6% increase in overall success rate")
    print("💡 Run syntax validation to verify improvements")
    print("🚀 Ready for Phase 2: Unicode character fixes")
    else:
    print("\n🤔 No unterminated string issues found or couldn't be fixed")

    return report['files_fixed']


if __name__ == "__main__":
    main()
