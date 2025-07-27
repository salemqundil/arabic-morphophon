#!/usr/bin/env python3
"""
🏁 Final Comprehensive Syntax Fixer
Fixes all remaining syntax issues in the codebase.
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


class FinalComprehensiveSyntaxFixer:
    def __init__(self):
    self.fixes_applied = 0
    self.files_fixed = 0
    self.backup_dir = Path(
    f"backups/final_comprehensive_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

    def fix_unicode_characters(self, content: str) -> Tuple[str, int]:
    """Remove problematic Unicode characters."""
    fixes = 0

        # Map of problematic Unicode characters to safe replacements
    unicode_fixes = {
    '🎯': 'TARGET',
    '🔬': 'ANALYSIS',
    '🚀': 'ROCKET',
    '🔥': 'FIRE',
    '🏗': 'CONSTRUCTION',
    '🌟': 'STAR',
    '→': '->',
    '،': ',',  # Arabic comma
    '📋': 'CLIPBOARD',
    '🔤': 'TEXT',
    '🏥': 'HOSPITAL',
    '🎨': 'ART',
    '💾': 'DISK',
    '⚡': 'LIGHTNING',
    '📊': 'CHART',
    '🎉': 'CELEBRATION',
    '💡': 'BULB',
    '🤔': 'THINKING',
    '❌': 'X',
    '✅': 'CHECK',
    '⚠️': 'WARNING',
    '🆘': 'SOS',
    '🔧': 'WRENCH',
    }

        for unicode_char, replacement in unicode_fixes.items():
            if unicode_char in content:
    content = content.replace(unicode_char, replacement)
    fixes += 1
    logger.debug(
    f"Fixed Unicode character: {unicode_char} -> {replacement}"
    )

    return content, fixes

    def fix_leading_zeros(self, content: str) -> Tuple[str, int]:
    """Fix leading zeros in decimal integers."""
    fixes = 0

        # Pattern to find decimal numbers with leading zeros
    pattern = r'\b0+([1-9]\d*)\b'
    matches = list(re.finditer(pattern, content))

        for match in reversed(matches):  # Reverse to maintain positions
    start, end = match.span()
    leading_zero_num = match.group(0)
    fixed_num = match.group(1)

            # Make sure this is actually a standalone number, not part of a string or other context
    before_char = content[start - 1] if start > 0 else ' '
    after_char = content[end] if end < len(content) else ' '

            # Check if it's in a valid context for decimal literal
            if (before_char in ' \t\n=+-(,[{') and (
    after_char in ' \t\n,)]}+\-*/%><!='
    ):
    content = content[:start] + fixed_num + content[end:]
    fixes += 1
    logger.debug(f"Fixed leading zeros: {leading_zero_num} -> {fixed_num}")

    return content, fixes

    def fix_unterminated_strings(self, content: str) -> Tuple[str, int]:
    """Fix unterminated string literals and triple quotes."""
    fixes = 0
    lines = content.split('\n')

    in_triple_quote = False
    triple_quote_start = None

        for i, line in enumerate(lines):
            # Check for unterminated triple quotes
    triple_quote_count = line.count('"""')

            if triple_quote_count % 2 == 1:  # Odd number means start or end
                if not in_triple_quote:
    in_triple_quote = True
    triple_quote_start = i
                else:
    in_triple_quote = False
    triple_quote_start = None

            # Fix unterminated strings at end of file
            if (
    i == len(lines) - 1
    and in_triple_quote
    and triple_quote_start is not None
    ):
                # Add closing triple quotes
    lines.append('"""')
    fixes += 1
    logger.debug(
    f"Fixed unterminated triple quote starting at line {triple_quote_start + 1}"
    )

        # Fix malformed docstrings
    content = '\n'.join(lines)

        # Fix """" -> """
    content = re.sub(r'""""', '"""', content)
    fixes += content.count('""""')

    return content, fixes

    def fix_invalid_syntax_patterns(self, content: str) -> Tuple[str, int]:
    """Fix various invalid syntax patterns."""
    fixes = 0

        # Fix unmatched parentheses/brackets
        # This is a simple approach - count and balance
    paren_balance = content.count('(') - content.count(')')
    bracket_balance = content.count('[') - content.count(']')
    brace_balance = content.count('{') - content.count('}')

        if paren_balance > 0:
    content += ')' * paren_balance
    fixes += paren_balance
        elif paren_balance < 0:
    content = ('(' * abs(paren_balance)) + content
    fixes += abs(paren_balance)

        if bracket_balance > 0:
    content += ']' * bracket_balance
    fixes += bracket_balance
        elif bracket_balance < 0:
    content = ('[' * abs(bracket_balance)) + content
    fixes += abs(bracket_balance)

        if brace_balance > 0:
    content += '}' * brace_balance
    fixes += brace_balance
        elif brace_balance < 0:
    content = ('{' * abs(brace_balance)) + content
    fixes += abs(brace_balance)

    return content, fixes

    def fix_extreme_indentation_issues(self, content: str) -> Tuple[str, int]:
    """Fix remaining extreme indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    fixes = 0

        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
    fixed_lines.append(line)
    continue

            # Check for extremely wrong indentation patterns
            if line.startswith('        ') and not line.strip().startswith(
    ('#', '"""', "'''")
    ):
                # Extremely deep indentation that's probably wrong
                # Try to dedent to reasonable level
    stripped = line.lstrip()
                if any(
    stripped.startswith(keyword)
                    for keyword in [
    'import ',
    'from ',
    'def ',
    'class ',
    'if ',
    'try:',
    'except',
    'finally:',
    ]
    ):
    fixed_lines.append(stripped)
    fixes += 1
    logger.debug(f"Fixed extreme indentation: {line.strip()}")
                else:
    fixed_lines.append(line)
            else:
    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes

    def validate_syntax(self, content: str) -> Tuple[bool, str]:
    """Validate syntax and return error message if invalid."""
        try:
    ast.parse(content)
    return True, ""
        except SyntaxError as e:
    return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
    return False, f"Parse error: {str(e)}"

    def fix_file(self, file_path: Path) -> bool:
    """Fix a single Python file and return True if fixes were applied."""
        try:
            # Read original content
    original_content = file_path.read_text(encoding='utf-8', errors='replace')

            # Check if already valid
    is_valid, error_msg = self.validate_syntax(original_content)
            if is_valid:
    logger.debug(f"✅ {file_path.name} - already valid syntax")
    return False

    logger.debug(f"🔍 {file_path.name} - fixing: {error_msg}")

            # Apply fixes step by step
    content = original_content
    total_fixes = 0

            # 1. Fix Unicode characters
    content, fixes = self.fix_unicode_characters(content)
    total_fixes += fixes

            # 2. Fix leading zeros
    content, fixes = self.fix_leading_zeros(content)
    total_fixes += fixes

            # 3. Fix unterminated strings
    content, fixes = self.fix_unterminated_strings(content)
    total_fixes += fixes

            # 4. Fix invalid syntax patterns
    content, fixes = self.fix_invalid_syntax_patterns(content)
    total_fixes += fixes

            # 5. Fix extreme indentation issues
    content, fixes = self.fix_extreme_indentation_issues(content)
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
    f"✅ {file_path.name} - {total_fixes} fixes applied, now valid!"
    )
    return True
                else:
    logger.debug(f"✅ {file_path.name} - already valid")
    return False
            else:
                if total_fixes > 0:
    logger.warning(
    f"⚠️ {file_path.name} - {total_fixes} fixes applied but still invalid: {error_msg}"
    )
                else:
    logger.debug(
    f"❌ {file_path.name} - no fixes possible: {error_msg}"
    )
    return False

        except Exception as e:
    logger.error(f"❌ Error fixing {file_path}: {str(e)}")
    return False

    def fix_directory(self, directory: Path = Path('.')) -> Dict:
    """Fix all Python files in directory."""
    logger.info(f"🏁 Starting final comprehensive syntax fixing in {directory}")

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
    """Main entry point for final comprehensive syntax fixing."""
    logger.info("🏁 Starting Final Comprehensive Syntax Fixing")

    fixer = FinalComprehensiveSyntaxFixer()
    report = fixer.fix_directory()

    # Print summary
    print("\n" + "=" * 70)
    print("🏁 FINAL COMPREHENSIVE SYNTAX FIXING SUMMARY")
    print("=" * 70)
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"✅ Files fixed: {report['files_fixed']}")
    print(f"⚡ Total fixes applied: {report['total_fixes']}")
    print(f"📊 Success rate: {report['success_rate']:.1f}%")
    print(f"💾 Backups saved to: {report['backup_directory']}")

    if report['files_fixed'] > 0:
    print(f"\n🎉 Successfully fixed {report['files_fixed']} files!")
    print("💡 Run syntax validation again to verify final results")
    else:
    print("\n🤔 No files needed fixing or were already valid")

    return report['files_fixed']


if __name__ == "__main__":
    main()
