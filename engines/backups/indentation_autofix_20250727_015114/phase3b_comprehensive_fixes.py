#!/usr/bin/env python3
"""
🔧 PHASE 3B: COMPREHENSIVE SYNTAX FIXES
=======================================

Multi-issue approach:
- Fix quadruple quotes AND any indentation issues in the same file
- Don't validate until ALL issues in a file are addressed
- Target files that have multiple syntax problems
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class Phase3BFixer:
    def __init__(self):
        self.backup_dir = Path("backups_phase3b_comprehensive")
        self.backup_dir.mkdir(exist_ok=True)

        self.files_processed = 0
        self.files_fixed = 0

        print("🔧 PHASE 3B: COMPREHENSIVE SYNTAX FIXES")
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

    def fix_comprehensive(self, file_path):
        """Fix multiple issues in a file comprehensively"""
        print(f"🔧 Comprehensive Fix: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            changes_made = 0

            # Process each line
            for i, line in enumerate(lines):
                original_line = line

                # Fix 1: Replace quadruple quotes with triple quotes
                if '""""' in line:
                    line = line.replace('""""', '"""')
                    if line != original_line:
                        changes_made += 1
                        print(f"  Line {i+1}: Fixed quadruple quotes")

                # Fix 2: Remove indentation from import statements
                stripped = line.strip()
                if line.startswith(' ') and (
                    stripped.startswith('import ')
                    or (stripped.startswith('from ') and ' import ' in stripped)
                ):
                    line = line.lstrip()
                    if line != original_line:
                        changes_made += 1
                        print(f"  Line {i+1}: Fixed import indentation")

                # Fix 3: Remove trailing commas from imports
                if line.strip().endswith('import re,'):
                    line = line.replace('import re,', 'import re')
                    if line != original_line:
                        changes_made += 1
                        print(f"  Line {i+1}: Fixed trailing comma")

                # Fix 4: Fix unmatched parentheses patterns
                if "f1_score)  # noqa: F401" in line:
                    line = line.replace(
                        "f1_score)  # noqa: F401", "f1_score  # noqa: F401"
                    )
                    if line != original_line:
                        changes_made += 1
                        print(f"  Line {i+1}: Fixed bracket issue")
                elif "ArabicPronounsDatabase)  # noqa: F401," in line:
                    line = line.replace(
                        "ArabicPronounsDatabase)  # noqa: F401,",
                        "ArabicPronounsDatabase  # noqa: F401",
                    )
                    if line != original_line:
                        changes_made += 1
                        print(f"  Line {i+1}: Fixed bracket issue")

                lines[i] = line

            if changes_made > 0:
                # Create backup
                backup_path = self.create_backup(file_path)
                if backup_path is None:
                    return False

                # Write all changes
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"  ✅ Applied {changes_made} comprehensive fixes")
                return True
            else:
                print(f"  ⚪ No fixes needed")
                return True

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def run(self):
        """Execute comprehensive fixes"""

        # Target the files that failed in Phase 3 due to multiple issues
        target_files = [
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
            "arabic_interrogative_pronouns_test_analysis.py",
            "arabic_pronouns_analyzer.py",
            "arabic_pronouns_deep_model.py",
            "arabic_normalizer.py",
        ]

        print(f"Target: {len(target_files)} files with multiple syntax issues")

        for file_path in target_files:
            self.files_processed += 1
            if os.path.exists(file_path):
                if self.fix_comprehensive(file_path):
                    self.files_fixed += 1
            else:
                print(f"⚠️ File not found: {file_path}")

        # Results
        print(f"\n📊 PHASE 3B RESULTS:")
        print("=" * 40)
        print(f"Files processed: {self.files_processed}")
        print(f"Files with fixes applied: {self.files_fixed}")
        print(
            f"Success rate: {self.files_fixed/self.files_processed*100:.1f}%"
            if self.files_processed > 0
            else "0%"
        )

        if self.files_fixed > 0:
            print(f"\n💡 Next: Run syntax validator to measure improvement")
            print(f"Note: Some files may still have remaining syntax issues")
            print(f"but should show progress toward clean syntax")


def main():
    """Main entry point"""
    fixer = Phase3BFixer()
    fixer.run()


if __name__ == "__main__":
    main()
