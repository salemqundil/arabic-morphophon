#!/usr/bin/env python3
"""
🔧 Broken Comparison Fixer
Fixes -> comparison operators that should be >=, >, etc.
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


class BrokenComparisonFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_processed = 0
    self.backup_dir = (
    Path("backups")
    / f"comparison_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Patterns for broken comparisons that should be fixed
    self.patterns = [
            # Common -> that should be >=
    (r'len\([^)]+\)\s*->\s*(\d+)', r'len(\g<1>) >= \1'),
    (r'count\([^)]+\)\s*->\s*(\d+)', r'count(\g<1>) >= \1'),
    (r'(\w+)\s*->\s*(\d+)', r'\1 >= \2'),
            # Arrow in function annotations (type hints) - should be left alone
            # These patterns specifically avoid type annotations
            # But fix arrows in conditions and expressions
    (r'if\s+([^:]+?)\s*->\s*([^:]+?):', r'if \1 >= \2:'),
    (r'elif\s+([^:]+?)\s*->\s*([^:]+?):', r'elif \1 >= \2:'),
    (r'while\s+([^:]+?)\s*->\s*([^:]+?):', r'while \1 >= \2:'),
            # Fix in list comprehensions and generator expressions
    (
    r'for\s+\w+\s+in\s+[^]]+if\s+([^]]+?)\s*->\s*([^]]+?)\]',
    r'for \w+ in \g<0> if \1 >= \2]',
    ),
    ]

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
    return backup_path

    def is_type_annotation(self, line: str, arrow_pos: int) -> bool:
    """Check if the -> is part of a type annotation."""
        # Look for function definition patterns
    before_arrow = line[:arrow_pos].strip()
    after_arrow = line[arrow_pos + 2 :].strip()

        # Function definition with return type annotation
        if 'def ' in before_arrow and '(' in before_arrow and ')' in before_arrow:
    return True

        # Lambda with type annotation
        if 'lambda' in before_arrow and ':' in after_arrow:
    return True

    return False

    def fix_comparison_operators(self, content: str) -> tuple[str, int]:
    """Fix broken comparison operators while preserving type annotations."""
    lines = content.split('\n')
    fixed_lines = []
    fixes_count = 0

        for line in lines:
    original_line = line

            # Find all -> occurrences
    arrow_positions = [m.start() for m in re.finditer(r'->', line)]

            for pos in reversed(arrow_positions):  # Process from right to left
                if not self.is_type_annotation(line, pos):
                    # This -> is likely a broken comparison
    before = line[:pos].strip()
    after = line[pos + 2 :].strip()

                    # Try to determine the correct operator
                    if any(word in before.lower() for word in ['len', 'count', 'size']):
                        # Likely should be >=
    line = line[:pos] + ' >= ' + line[pos + 2 :]
    fixes_count += 1
                    elif re.search(r'\b\d+\s*$', before):
                        # Number comparison, likely >=
    line = line[:pos] + ' >= ' + line[pos + 2 :]
    fixes_count += 1
                    elif any(keyword in before for keyword in ['if', 'elif', 'while']):
                        # In conditional, likely >=
    line = line[:pos] + ' >= ' + line[pos + 2 :]
    fixes_count += 1

            if line != original_line:
    logger.debug(
    f"Fixed comparison: '{original_line.strip()}' -> '{line.strip()}'"
    )

    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_count

    def fix_file(self, file_path: Path) -> int:
    """Fix broken comparison operators in a single file."""
        try:
    content = file_path.read_text(encoding='utf-8')
    original_content = content

            # Apply pattern-based fixes
    fixes_in_file = 0
            for pattern, replacement in self.patterns:
    old_content = content
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                if content != old_content:
    matches = len(re.findall(pattern, old_content, flags=re.MULTILINE))
    fixes_in_file += matches
    logger.debug(
    f"Applied comparison pattern '{pattern}' {matches} times in {file_path}"
    )

            # Apply advanced operator fixes
    content, advanced_fixes = self.fix_comparison_operators(content)
    fixes_in_file += advanced_fixes

            if content != original_content:
                # Backup original file
    self.backup_file(file_path)

                # Write fixed content
    file_path.write_text(content, encoding='utf-8')
    self.fixes_applied += fixes_in_file
    logger.info(
    f"✅ Fixed {fixes_in_file} comparison operators in {file_path}"
    )
    return fixes_in_file

    return 0

        except Exception as e:
    logger.error(f"❌ Error processing {file_path}: {e}")
    return 0

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
    logger.info("🔧 Starting Broken Comparison Fixer")

    fixer = BrokenComparisonFixer()
    results = fixer.process_directory()

    # Print summary
    print("\n" + "=" * 60)
    print("🔧 BROKEN COMPARISON FIXER SUMMARY")
    print("=" * 60)
    print(f"📁 Files processed: {results['files_processed']}")
    print(f"🔧 Files fixed: {results['files_fixed']}")
    print(f"✅ Total fixes applied: {results['total_fixes']}")
    print(f"💾 Backups saved to: {fixer.backup_dir}")

    if results['total_fixes'] > 0:
    print("\n🎉 Broken comparison operators have been fixed!")
    print("💡 Run AST validation to verify fixes")
    else:
    print("\n✨ No broken comparison operators found!")

    return results['total_fixes']


if __name__ == "__main__":
    main()
