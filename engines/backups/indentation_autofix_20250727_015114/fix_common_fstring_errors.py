#!/usr/bin/env python3
"""
🔧 F-String Error Fixer
Fixes common f-string syntax errors automatically with rollback capability.
"""

import re
import os
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FStringFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.backup_dir = (
            Path("backups")
            / f"fstring_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Common f-string error patterns and their fixes
        self.patterns = [
            # Unmatched closing braces
            (r'f"([^"]*?)}}([^"]*?)"', r'f"\1}\2"'),
            (r"f'([^']*?)}}([^']*?)'", r"f'\1}\2'"),
            # Missing closing braces
            (r'f"([^"]*?)\{([^}]*?)([^"}]*?)"', r'f"\1{\2}\3"'),
            (r"f'([^']*?)\{([^}]*?)([^'}]*?)'", r"f'\1{\2}\3'"),
            # Common get() method fixes
            (r"\.get\('([^']+)',\s*\}", r".get('\1', '')"),
            (r'\.get\("([^"]+)",\s*\}', r'.get("\1", "")'),
            # Fix unterminated f-strings with missing quotes
            (r'f"([^"]*?)\{([^}]*?)\s*$', r'f"\1{\2}"'),
            (r"f'([^']*?)\{([^}]*?)\s*$", r"f'\1{\2}'"),
            # Fix malformed variable references
            (r'f"([^"]*?)\{([^}]*?)\s+([^}]*?)\}"', r'f"\1{\2_\3}"'),
            (r"f'([^']*?)\{([^}]*?)\s+([^}]*?)\}'", r"f'\1{\2_\3}'"),
            # Fix broken format specifiers
            (r'f"([^"]*?)\{([^}:]*?):\s*\}"', r'f"\1{\2}"'),
            (r"f'([^']*?)\{([^}:]*?):\s*\}'", r"f'\1{\2}'"),
        ]

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification."""
        backup_path = self.backup_dir / file_path.name
        backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        return backup_path

    def fix_file(self, file_path: Path) -> int:
        """Fix f-string errors in a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            fixes_in_file = 0

            # Apply each pattern fix
            for pattern, replacement in self.patterns:
                old_content = content
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                if content != old_content:
                    matches = len(re.findall(pattern, old_content, flags=re.MULTILINE))
                    fixes_in_file += matches
                    logger.debug(
                        f"Applied pattern '{pattern}' {matches} times in {file_path}"
                    )

            # Special case: fix completely broken f-strings
            content = self.fix_broken_fstrings(content)

            if content != original_content:
                # Backup original file
                self.backup_file(file_path)

                # Write fixed content
                file_path.write_text(content, encoding='utf-8')
                self.fixes_applied += fixes_in_file
                logger.info(f"✅ Fixed {fixes_in_file} f-string errors in {file_path}")
                return fixes_in_file

            return 0

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return 0

    def fix_broken_fstrings(self, content: str) -> str:
        """Fix completely broken f-string patterns."""
        # Fix f-strings with missing closing quotes
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Check for unterminated f-strings
            if 'f"' in line and line.count('"') % 2 != 0:
                # Add missing closing quote
                line = line.rstrip() + '"'
            elif "f'" in line and line.count("'") % 2 != 0:
                # Add missing closing quote
                line = line.rstrip() + "'"

            # Fix print statements with broken f-strings
            if re.match(r'\s*print\(f"[^"]*$', line):
                line = line.rstrip() + '")'
            elif re.match(r"\s*print\(f'[^']*$", line):
                line = line.rstrip() + "')"

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def process_directory(self, directory: Path = Path('.')) -> dict:
        """Process all Python files in directory."""
        results = {
            'files_processed': 0,
            'files_fixed': 0,
            'total_fixes': 0,
            'errors': [],
        }

        for file_path in directory.rglob('*.py'):
            # Skip backup directories and virtual environments
            if any(
                part.startswith('.') or part in ['backups', 'venv', '__pycache__']
                for part in file_path.parts
            ):
                continue

            results['files_processed'] += 1
            fixes = self.fix_file(file_path)

            if fixes > 0:
                results['files_fixed'] += 1
                results['total_fixes'] += fixes

        return results


def main():
    """Main entry point."""
    logger.info("🔧 Starting F-String Error Fixer")

    fixer = FStringFixer()
    results = fixer.process_directory()

    # Print summary
    print("\n" + "=" * 60)
    print("🔧 F-STRING ERROR FIXER SUMMARY")
    print("=" * 60)
    print(f"📁 Files processed: {results['files_processed']}")
    print(f"🔧 Files fixed: {results['files_fixed']}")
    print(f"✅ Total fixes applied: {results['total_fixes']}")
    print(f"💾 Backups saved to: {fixer.backup_dir}")

    if results['total_fixes'] > 0:
        print("\n🎉 F-string errors have been fixed!")
        print("💡 Run AST validation to verify fixes")
    else:
        print("\n✨ No f-string errors found!")

    return results['total_fixes']


if __name__ == "__main__":
    main()
