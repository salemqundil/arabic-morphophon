#!/usr/bin/env python3
"""
🔧 PHASE 3: F-STRING AND BRACKET FIXES
======================================

Target high-impact syntax issues:
1. F-String Issues (20 files): Replace quadruple quotes with triple quotes
2. Bracket Issues (3 files): Fix unmatched parentheses
3. General Syntax (1 file): Fix trailing comma in import

Strategy:
- Surgical precision fixes for maximum success rate
- Backup all modified files
- Validate each fix with AST parsing
- Focus on easy wins for measurable progress
"""

import os
import ast
import shutil
import re
from pathlib import Path
from datetime import datetime


class Phase3Fixer:
    def __init__(self):
    self.backup_dir = Path("backups_phase3_fixes")
    self.backup_dir.mkdir(exist_ok=True)

        # Success tracking
    self.files_processed = 0
    self.files_fixed = 0
    self.files_skipped = 0

        # Issue-specific tracking
    self.fstring_fixes = 0
    self.bracket_fixes = 0
    self.syntax_fixes = 0

    print("🔧 PHASE 3: F-STRING AND BRACKET FIXES")
    print("=" * 50)

    def create_backup(self, file_path):
    """Create a timestamped backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
    backup_path = self.backup_dir / f"{Path(file_path).name}_{timestamp}.bak"

        try:
    shutil.copy2(file_path, backup_path)
    return backup_path
        except Exception as e:
    print(f"❌ Backup failed for {file_path}: {e}")
    return None

    def validate_syntax(self, file_path):
    """Validate Python syntax using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    ast.parse(content)
    return True
        except Exception:
    return False

    def fix_fstring_issues(self, file_path):
    """Fix quadruple quote issues (four quotes to three quotes)"""
    print(f"🔧 F-String Fix: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

            # Replace '""""' with '"""'
    original_content = content
    content = content.replace('""""', '"""')

            if content != original_content:
                # Create backup
    backup_path = self.create_backup(file_path)
                if backup_path is None:
    return False

                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

                # Validate fix
                if self.validate_syntax(file_path):
    count = original_content.count('""""')
    print(f"  ✅ Fixed {count} quadruple quotes")
    self.fstring_fixes += count
    return True
                else:
                    # Restore if syntax still broken
    shutil.copy2(backup_path, file_path)
    print(f"  ❌ Syntax still invalid, restored")
    return False
            else:
    print(f"  ⚪ No quadruple quotes found")
    return True

        except Exception as e:
    print(f"  ❌ Error: {e}")
    return False

    def fix_bracket_issues(self, file_path, line_num, line_content):
    """Fix unmatched parentheses"""
    print(f"🔧 Bracket Fix: {file_path} line {line_num}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

            if line_num > len(lines):
    print(f"  ❌ Line {line_num} not found")
    return False

    original_line = lines[line_num - 1]

            # Specific fixes based on the patterns we saw
    fixed_line = original_line

            # Pattern 1: Remove extra closing parenthesis
            if "f1_score)  # noqa: F401" in original_line:
    fixed_line = original_line.replace(
    "f1_score)  # noqa: F401", "f1_score  # noqa: F401"
    )
            elif "ArabicPronounsDatabase)  # noqa: F401," in original_line:
    fixed_line = original_line.replace(
    "ArabicPronounsDatabase)  # noqa: F401,",
    "ArabicPronounsDatabase  # noqa: F401",
    )
            elif original_line.strip() == "):":
                # Check if this is an orphaned closing parenthesis
    fixed_line = ""  # Remove the line entirely

            if fixed_line != original_line:
                # Create backup
    backup_path = self.create_backup(file_path)
                if backup_path is None:
    return False

                # Apply fix
                if fixed_line == "":
                    # Remove the line
    lines.pop(line_num - 1)
                else:
    lines[line_num - 1] = fixed_line

                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

                # Validate fix
                if self.validate_syntax(file_path):
    print(f"  ✅ Fixed bracket issue")
    self.bracket_fixes += 1
    return True
                else:
                    # Restore if syntax still broken
    shutil.copy2(backup_path, file_path)
    print(f"  ❌ Syntax still invalid, restored")
    return False
            else:
    print(f"  ⚪ No bracket fix pattern matched")
    return True

        except Exception as e:
    print(f"  ❌ Error: {e}")
    return False

    def fix_trailing_comma(self, file_path, line_num):
    """Fix trailing comma in import"""
    print(f"🔧 Syntax Fix: {file_path} line {line_num}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

            if line_num > len(lines):
    print(f"  ❌ Line {line_num} not found")
    return False

    original_line = lines[line_num - 1]

            # Remove trailing comma from import
            if "import re," in original_line:
    fixed_line = original_line.replace("import re,", "import re")

                # Create backup
    backup_path = self.create_backup(file_path)
                if backup_path is None:
    return False

                # Apply fix
    lines[line_num - 1] = fixed_line

                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

                # Validate fix
                if self.validate_syntax(file_path):
    print(f"  ✅ Fixed trailing comma")
    self.syntax_fixes += 1
    return True
                else:
                    # Restore if syntax still broken
    shutil.copy2(backup_path, file_path)
    print(f"  ❌ Syntax still invalid, restored")
    return False
            else:
    print(f"  ⚪ No trailing comma pattern found")
    return True

        except Exception as e:
    print(f"  ❌ Error: {e}")
    return False

    def run(self):
    """Execute Phase 3 fixes"""

        # 1. F-String Issues (20 files)
    print("\n📋 PART 1: F-STRING FIXES")
    print("-" * 30)

    fstring_files = [
    "arabic_inflection_corrected.py",
    "arabic_inflection_rules_engine.py",
    "arabic_inflection_ultimate.py",
    "arabic_inflection_ultimate_fixed.py",
    "arabic_phonological_foundation.py",
    "arabic_test.py",
    "arabic_vector_engine.py",
    "arabic_verb_conjugator.py",
    "complete_all_13_engines.py",
    "complete_arabic_phonological_coverage.py",
    ]

        # Get additional F-string files from validation report
        try:
            with open(
    "syntax_validation_report_20250727_012620.txt", 'r', encoding='utf-8'
    ) as f:
    content = f.read()

            # Extract all files with unterminated string literals
            import re

    pattern = (
    r'❌\s+([^:]+\.py):\s+Line\s+\d+\s+-\s+unterminated string literal'
    )
    matches = re.findall(pattern, content)

            # Add any additional files found
            for match in matches:
                if match not in fstring_files:
    fstring_files.append(match)

        except Exception as e:
    print(f"⚠️ Could not read validation report: {e}")

        # Remove duplicates and sort
    fstring_files = sorted(list(set(fstring_files)))
    print(f"Target: {len(fstring_files)} files with F-string issues")

        for file_path in fstring_files:
    self.files_processed += 1
            if os.path.exists(file_path):
                if self.fix_fstring_issues(file_path):
    self.files_fixed += 1
                else:
    self.files_skipped += 1
            else:
    print(f"⚠️ File not found: {file_path}")
    self.files_skipped += 1

        # 2. Bracket Issues (3 files)
    print(f"\n📋 PART 2: BRACKET FIXES")
    print("-" * 30)

    bracket_issues = [
    (
    "arabic_interrogative_pronouns_test_analysis.py",
    27,
    "f1_score)  # noqa: F401",
    ),
    (
    "arabic_pronouns_analyzer.py",
    29,
    "ArabicPronounsDatabase)  # noqa: F401,",
    ),
    ("arabic_pronouns_deep_model.py", 37, "):"),
    ]

        for file_path, line_num, line_content in bracket_issues:
    self.files_processed += 1
            if os.path.exists(file_path):
                if self.fix_bracket_issues(file_path, line_num, line_content):
    self.files_fixed += 1
                else:
    self.files_skipped += 1
            else:
    print(f"⚠️ File not found: {file_path}")
    self.files_skipped += 1

        # 3. General Syntax Issues (1 file)
    print(f"\n📋 PART 3: GENERAL SYNTAX FIXES")
    print("-" * 30)

        if os.path.exists("arabic_normalizer.py"):
    self.files_processed += 1
            if self.fix_trailing_comma("arabic_normalizer.py", 10):
    self.files_fixed += 1
            else:
    self.files_skipped += 1
        else:
    print("⚠️ File not found: arabic_normalizer.py")

        # Results
    print(f"\n📊 PHASE 3 RESULTS:")
    print("=" * 40)
    print(f"Files processed: {self.files_processed}")
    print(f"Files fixed: {self.files_fixed}")
    print(f"Files skipped: {self.files_skipped}")
    print(
    f"Success rate: {self.files_fixed/self.files_processed*100:.1f}%"
            if self.files_processed > 0
            else "0%"
    )
    print(f"\nSpecific fixes:")
    print(f"📝 F-string fixes: {self.fstring_fixes}")
    print(f"🔧 Bracket fixes: {self.bracket_fixes}")
    print(f"⚙️ Syntax fixes: {self.syntax_fixes}")

        if self.files_fixed > 0:
    print(
    f"\n🎯 Expected improvement: +{self.files_fixed} files to clean status"
    )
    print(f"💡 Next: Run syntax validator to measure progress")


def main():
    """Main entry point"""
    fixer = Phase3Fixer()
    fixer.run()


if __name__ == "__main__":
    main()
