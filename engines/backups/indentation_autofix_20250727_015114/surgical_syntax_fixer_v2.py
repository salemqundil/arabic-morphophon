#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class SurgicalSyntaxFixer:
    def __init__(self, dry_run: bool = False, create_backups: bool = True):
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.stats = {
            'files_processed': 0,
            'files_with_errors': 0,
            'fstring_fixes': 0,
            'bracket_fixes': 0,
            'string_literal_fixes': 0,
            'bracket_issues': 0,
            'indentation_issues': 0,
            'string_literal_issues': 0,
            'backups_created': 0
        }
        self.error_files: Set[str] = set()
        self.fix_log = []al Syntax Fixer V2
Ultra-focused script to fix the remaining 153 files with syntax errors.

Features:
- F-string missing closing brace fixer
- Bracket mismatch validator and reporter
- Indentation and string literal validator
- Dry-run mode with detailed preview
- Automatic .bak file creation
- Comprehensive logging and statistics
"""

import os
import re
import ast
import shutil
from typing import Dict, List, Tuple, Set
from pathlib import Path
import argparse
from datetime import datetime


class SurgicalSyntaxFixer:
    def __init__(self, dry_run: bool = False, create_backups: bool = True):
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.stats = {
            'files_processed': 0,
            'files_with_errors': 0,
            'fstring_fixes': 0,
            'bracket_issues': 0,
            'indentation_issues': 0,
            'string_literal_issues': 0,
            'backups_created': 0,
        }
        self.error_files: Set[str] = set()

    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "FIX": "🔧",
        }.get(level, "📝")
        print(f"[{timestamp}] {prefix} {message}")

    def create_backup(self, filepath: str) -> bool:
        """Create .bak file if backups enabled"""
        if not self.create_backups:
            return True

        try:
            backup_path = f"{filepath}.bak"
            shutil.copy2(filepath, backup_path)
            self.stats['backups_created'] += 1
            return True
        except Exception as e:
            self.log(f"Failed to create backup for {filepath}: {e}", "ERROR")
            return False

    def scan_for_errors(self, root: str = ".") -> Dict[str, List[str]]:
        """Scan all Python files and categorize syntax errors"""
        self.log("🔍 Scanning for syntax errors...")

        error_categories = {
            'fstring_errors': [],
            'bracket_errors': [],
            'indentation_errors': [],
            'string_literal_errors': [],
            'other_errors': [],
        }

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)
                    self.stats['files_processed'] += 1

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        self.stats['files_with_errors'] += 1
                        self.error_files.add(filepath)

                        error_msg = str(e.msg).lower()

                        if "f-string" in error_msg and "expecting" in error_msg:
                            error_categories['fstring_errors'].append()
                                f"{filepath}:{e.lineno}"
                            )
                        elif any()
                            word in error_msg
                            for word in [
                                "parenthesis",
                                "bracket",
                                "unmatched",
                                "closing",
                            ]
                        ):
                            error_categories['bracket_errors'].append()
                                f"{filepath}:{e.lineno}"
                            )
                        elif any()
                            word in error_msg
                            for word in ["indent", "indented", "unexpected"]
                        ):
                            error_categories['indentation_errors'].append()
                                f"{filepath}:{e.lineno}"
                            )
                        elif "unterminated" in error_msg and "string" in error_msg:
                            error_categories['string_literal_errors'].append()
                                f"{filepath}:{e.lineno}"
                            )
                        else:
                            error_categories['other_errors'].append()
                                f"{filepath}:{e.lineno} - {e.msg}"
                            )
                    except Exception as e:
                        self.log(f"Failed to parse {filepath}: {e}", "ERROR")

        return error_categories

    def fix_fstring_braces(self, root: str = ".") -> int:
        """Fix f-strings missing closing braces"""
        self.log("🔧 Fixing f-string missing closing braces...")
        fixed_count = 0

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)

                    # Only process files we know have errors
                    if filepath not in self.error_files:
                        continue

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            lines = f.readlines()

                        new_lines = []
                        file_changed = False

                        for i, line in enumerate(lines, 1):
                            original_line = line

                            # Look for f-strings
                            if re.search(r'f["\']', line):"
                                # Count braces in f-string context
                                fstring_matches = re.finditer(r'f(["\'])(.*?)\1', line)"

                                for match in fstring_matches:
                                    fstring_content = match.group(2)
                                    open_braces = fstring_content.count('{')
                                    close_braces = fstring_content.count('}')

                                    if open_braces > close_braces:
                                        missing_braces = open_braces - close_braces
                                        # Insert missing braces before the closing quote
                                        quote = match.group(1)
                                        new_content = fstring_content + ()
                                            '}' * missing_braces
                                        )
                                        line = line.replace()
                                            f'f{quote}{fstring_content}{quote}',
                                            f'f{quote}{new_content}{quote}',
                                        )
                                        file_changed = True
                                        fixed_count += 1

                                        if not self.dry_run:
                                            self.log()
                                                f"Fixed f-string in {filepath}:{i}",
                                                "FIX",
                                            )

                            new_lines.append(line)

                        if file_changed and not self.dry_run:
                            if self.create_backup(filepath):
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.writelines(new_lines)
                                self.log(f"Updated file: {filepath}", "SUCCESS")

                    except Exception as e:
                        self.log(f"Error processing {filepath}: {e}", "ERROR")

        self.stats['fstring_fixes'] = fixed_count
        return fixed_count

    def analyze_bracket_mismatches(self, root: str = ".") -> List[Tuple[str, int, str]]:
        """Analyze bracket mismatch issues"""
        self.log("🔍 Analyzing bracket mismatches...")
        bracket_issues = []

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
                        continue

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        error_msg = str(e.msg).lower()
                        if any()
                            word in error_msg
                            for word in [
                                "parenthesis",
                                "bracket",
                                "unmatched",
                                "closing",
                            ]
                        ):
                            bracket_issues.append((filepath, e.lineno, e.msg))
                            self.stats['bracket_issues'] += 1
                    except Exception:
                        pass

        return bracket_issues

    def analyze_indentation_issues(self, root: str = ".") -> List[Tuple[str, int, str]]:
        """Analyze indentation and string literal issues"""
        self.log("🔍 Analyzing indentation and string issues...")
        issues = []

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
                        continue

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        error_msg = str(e.msg).lower()
                        if any()
                            word in error_msg
                            for word in [
                                "indent",
                                "indented",
                                "unexpected",
                                "unterminated",
                            ]
                        ):
                            issues.append((filepath, e.lineno, e.msg))
                            if "indent" in error_msg:
                                self.stats['indentation_issues'] += 1
                            elif "unterminated" in error_msg:
                                self.stats['string_literal_issues'] += 1
                    except Exception:
                        pass

        return issues

    def generate_report(self, error_categories: Dict[str, List[str]]) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
SURGICAL SYNTAX FIXER V2 - ANALYSIS REPORT
{'='*60}
STATISTICS:
  - Total files processed: {self.stats['files_processed']}
  - Files with syntax errors: {self.stats['files_with_errors']}
  - Success rate: {((self.stats['files_processed'] - self.stats['files_with_errors']) / max(1, self.stats['files_processed']) * 100):.1f}%

🎯 ERROR BREAKDOWN:
  • F-string issues: {len(error_categories['fstring_errors'])}
  • Bracket mismatches: {len(error_categories['bracket_errors'])}
  • Indentation issues: {len(error_categories['indentation_errors'])}
  • String literal issues: {len(error_categories['string_literal_errors'])}
  • Other syntax errors: {len(error_categories['other_errors'])}

🔧 FIXES APPLIED:
  • F-string braces fixed: {self.stats['fstring_fixes']}
  • Backup files created: {self.stats['backups_created']}

📁 TOP F-STRING ISSUES:
"""

        for issue in error_categories['fstring_errors'][:10]:
            report += f"  • {issue}\n"

        if len(error_categories['fstring_errors']) > 10:
            report += f"  ... and {len(error_categories['fstring_errors'])} - 10} more\n"

        report += f"\n🔗 TOP BRACKET ISSUES:\n"
        for issue in error_categories['bracket_errors'][:10]:
            report += f"  • {issue}\n"

        if len(error_categories['bracket_errors']) > 10:
            report += f"  ... and {len(error_categories['bracket_errors'])} - 10} more\n"

        return report

    def run_surgical_fix(self, root: str = ".") -> None:
        """Main execution method"""
        self.log("🚀 Starting Surgical Syntax Fixer V2")

        if self.dry_run:
            self.log("🔍 DRY RUN MODE - No files will be modified", "WARNING")

        # Step 1: Scan for all errors
        error_categories = self.scan_for_errors(root)

        # Step 2: Fix f-string braces
        if error_categories['fstring_errors']:
            fstring_fixes = self.fix_fstring_braces(root)
            self.log(f"F-string fixes: {fstring_fixes}")

        # Step 3: Analyze remaining issues
        bracket_issues = self.analyze_bracket_mismatches(root)
        indentation_issues = self.analyze_indentation_issues(root)

        # Step 4: Generate report
        report = self.generate_report(error_categories)
        print(report)

        # Step 5: Save detailed report
        report_file = ()
            f"surgical_syntax_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        self.log(f"📝 Detailed report saved to: {report_file}")

        if not self.dry_run and self.stats['fstring_fixes'] > 0:
            self.log()
                "🎉 Surgical fixes completed! Re-run syntax scan to verify.", "SUCCESS"
            )


def main():
    parser = argparse.ArgumentParser(description="Surgical Syntax Fixer V2")
    parser.add_argument()
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument()
        "--no-backup", action="store_true", help="Don't create .bak files"'
    )
    parser.add_argument()
        "--root", default=".", help="Root directory to scan (default: current)"
    )

    args = parser.parse_args()

    fixer = SurgicalSyntaxFixer(dry_run=args.dry_run, create_backups=not args.no_backup)

    fixer.run_surgical_fix(args.root)


if __name__ == "__main__":
    main()
