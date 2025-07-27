#!/usr/bin/env python3
"""
🏥 SURGICAL SYNTAX FIXER V3 - ENHANCED MODULAR VERSION
================================================================
Comprehensive surgical syntax repair with modular capabilities:
1️⃣ Bracket Mismatch Fixer,
    2️⃣ String Literal Fixer,
    3️⃣ Indentation Validator + Flagger,
    4️⃣ Unified Surgical Fixer Launcher Script,
    Usage:
  python surgical_syntax_fixer_v3.py --all,
    python surgical_syntax_fixer_v3.py --fstrings --brackets,
    python surgical_syntax_fixer_v3.py --strings --log,
    python surgical_syntax_fixer_v3.py --dry-run --analyze
"""

import ast
    import argparse
    import os
    import re
    import shutil
    import json
    from datetime import datetime
    from typing import Dict, List, Tuple, Set,
    class SurgicalSyntaxFixerV3:
    """Enhanced surgical syntax fixer with modular capabilities"""

    def __init__(
    self, dry_run: bool = False, create_backups: bool = True, verbose: bool = True
    ):
    self.dry_run = dry_run,
    self.create_backups = create_backups,
    self.verbose = verbose,
    self.stats = {
    'files_processed': 0,
    'files_with_errors': 0,
    'backups_created': 0,
    'fstring_fixes': 0,
    'bracket_fixes': 0,
    'string_literal_fixes': 0,
    'indentation_fixes': 0,
    'bracket_issues': 0,
    'indentation_issues': 0,
    'string_literal_issues': 0,
    }
    self.error_files = set()
    self.log_entries = []

    def log(self, message: str, level: str = "INFO") -> None:
    """Enhanced logging with timestamps and levels"""
    timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {level}: {message}"

        if self.verbose:
    colors = {
    "ERROR": "🔴",
    "WARNING": "🟡",
    "SUCCESS": "🟢",
    "FIX": "🔧",
    "INFO": "ℹ️",
    }
    icon = colors.get(level, "ℹ️")
    print(f"{icon} {formatted_msg}")

    self.log_entries.append(
    {'timestamp': timestamp, 'level': level, 'message': message}
    )

    def create_backup(self, filepath: str) -> bool:
    """Create backup with collision handling"""
        if not self.create_backups:
    return True,
    backup_path = f"{filepath}.bak"
    counter = 1

        # Handle backup file collisions,
    while os.path.exists(backup_path):
    backup_path = f"{filepath}.bak.{counter}"
    counter += 1,
    try:
    shutil.copy2(filepath, backup_path)
    self.stats['backups_created'] += 1,
    self.log(f"Backup created: {backup_path}")
    return True,
    except Exception as e:
    self.log(f"Failed to create backup for {filepath}: {e}", "ERROR")
    return False,
    def scan_for_errors(self, root: str = ".") -> Dict[str, List[str]]:
    """Enhanced error categorization with detailed analysis"""
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
    self.stats['files_processed'] += 1,
    try:
                        with open(filepath, encoding="utf-8") as f:
    source = f.read()
    ast.parse(source)
                    except SyntaxError as e:
    self.error_files.add(filepath)
    self.stats['files_with_errors'] += 1,
    error_msg = str(e.msg).lower()
    error_info = f"{filepath}:{e.lineno} - {e.msg}"

                        # Categorize errors more precisely,
    if any(
    word in error_msg,
    for word in [
    'f-string',
    'invalid decimal',
    'unterminated string',
    ]
    ):
                            if 'f-string' in error_msg or (
    'invalid decimal' in error_msg,
    and 'f' in str(e.text or '')
    ):
    error_categories['fstring_errors'].append(error_info)
                            elif 'unterminated string' in error_msg:
    error_categories['string_literal_errors'].append(
    error_info
    )
                        elif any(
    word in error_msg,
    for word in [
    'parenthesis',
    'bracket',
    'unmatched',
    'closing',
    ]
    ):
    error_categories['bracket_errors'].append(error_info)
                        elif any(
    word in error_msg,
    for word in ['indent', 'indented', 'unexpected indent']
    ):
    error_categories['indentation_errors'].append(error_info)
                        elif any(
    word in error_msg,
    for word in ['unterminated', 'string literal']
    ):
    error_categories['string_literal_errors'].append(error_info)
                        else:
    error_categories['other_errors'].append(error_info)
                    except Exception:
    pass,
    self.log(f"Files scanned: {self.stats['files_processed']}")
    self.log(f"Files with errors: {self.stats['files_with_errors']}")

    return error_categories,
    def fix_fstring_braces(self, root: str = ".") -> int:
    """Enhanced f-string brace fixing with better pattern matching"""
    self.log("🔧 Fixing f-string brace mismatches...")
    fixed_count = 0

        # Enhanced f-string patterns,
    fstring_patterns = [
    re.compile(r"f(['\"])([^'\"]*?\{[^}]*?)(['\"])", re.MULTILINE),
    re.compile(r"f(['\"])([^'\"]*?\{[^}]*?[^}])(['\"])", re.MULTILINE),
    re.compile(r"f(['\"])([^'\"]*?\{[^{}]*?\{[^}]*?)(['\"])", re.MULTILINE),
    ]

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
    continue,
    try:
                        with open(filepath, encoding="utf-8") as f:
    lines = f.readlines()

    new_lines = []
    file_changed = False,
    for i, line in enumerate(lines):
    original_line = line,
    for pattern in fstring_patterns:
                                for match in pattern.finditer(line):
    quote = match.group(1)
    fstring_content = match.group(2)

                                    # Count braces,
    open_braces = fstring_content.count('{')
    close_braces = fstring_content.count('}')

                                    if open_braces > close_braces:
    missing_braces = open_braces - close_braces,
    new_content = fstring_content + (
    '}' * missing_braces
    )
    line = line.replace(
    f'f{quote}{fstring_content}{quote}',
    f'f{quote}{new_content}{quote}',
    )
    file_changed = True,
    fixed_count += 1,
    if not self.dry_run:
    self.log(
    f"Fixed f-string in {filepath}:{i+1}",
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

    self.stats['fstring_fixes'] = fixed_count,
    return fixed_count,
    def fix_bracket_mismatches(self, root: str = ".") -> int:
    """Fix simple bracket mismatch issues"""
    self.log("🔧 Fixing bracket mismatches...")
    fixed_count = 0,
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
    continue,
    try:
                        with open(filepath, encoding="utf-8") as f:
    lines = f.readlines()

    new_lines = []
    file_changed = False,
    for i, line in enumerate(lines):
    original_line = line.rstrip()

                            # Simple bracket balance check,
    bracket_stack = []
    in_string = False,
    string_char = None,
    for j, char in enumerate(original_line):
                                if char in ['"', "'"]:
                                    if not in_string:
    in_string = True,
    string_char = char,
    elif char == string_char:
    in_string = False,
    string_char = None,
    elif not in_string:
                                    if char in bracket_pairs:
    bracket_stack.append(char)
                                    elif char in bracket_pairs.values():
                                        if bracket_stack:
    expected_opener = [
    k,
    for k, v in bracket_pairs.items()
                                                if v == char
    ][0]
                                            if bracket_stack[-1] == expected_opener:
    bracket_stack.pop()

                            # If line ends with unclosed brackets, try to fix simple cases,
    if bracket_stack and len(bracket_stack) <= 2:
                                # Only fix simple cases with 1-2 missing closing brackets,
    missing_closers = ''.join(
    bracket_pairs[opener]
                                    for opener in reversed(bracket_stack)
    )

                                # Check if this looks like a function call or list that was cut off,
    if original_line.rstrip().endswith(
    ','
    ) or original_line.rstrip().endswith('('):
    fixed_line = (
    original_line.rstrip() + missing_closers + '\n'
    )
    new_lines.append(fixed_line)
    file_changed = True,
    fixed_count += 1,
    if not self.dry_run:
    self.log(
    f"Fixed brackets in {filepath}:{i+1}", "FIX"
    )
                                else:
    new_lines.append(line)
                            else:
    new_lines.append(line)

                        if file_changed and not self.dry_run:
                            if self.create_backup(filepath):
                                with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    self.log(f"Updated file: {filepath}", "SUCCESS")

                    except Exception as e:
    self.log(f"Error processing {filepath}: {e}", "ERROR")

    self.stats['bracket_fixes'] = fixed_count,
    return fixed_count,
    def fix_string_literals(self, root: str = ".") -> int:
    """Fix unterminated string literal issues"""
    self.log("🔧 Fixing string literal issues...")
    fixed_count = 0,
    for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
    continue,
    try:
                        with open(filepath, encoding="utf-8") as f:
    lines = f.readlines()

    new_lines = []
    file_changed = False,
    for i, line in enumerate(lines):
    original_line = line.rstrip()

                            # Check for unterminated strings,
    quote_count_single = original_line.count(
    "'"
    ) - original_line.count("\\'")
    quote_count_double = original_line.count(
    '"'
    ) - original_line.count('\\"')

                            # If odd number of quotes, likely unterminated string,
    if (
    quote_count_single % 2 == 1,
    and quote_count_double % 2 == 0
    ):
                                # Try to fix by adding closing single quote,
    if not original_line.endswith("'"):
    fixed_line = original_line + "'\n"
    new_lines.append(fixed_line)
    file_changed = True,
    fixed_count += 1,
    if not self.dry_run:
    self.log(
    f"Fixed string literal in {filepath}:{i+1}",
    "FIX",
    )
                                else:
    new_lines.append(line)

                            elif (
    quote_count_double % 2 == 1,
    and quote_count_single % 2 == 0
    ):
                                # Try to fix by adding closing double quote,
    if not original_line.endswith('"'):
    fixed_line = original_line + '"\n'
    new_lines.append(fixed_line)
    file_changed = True,
    fixed_count += 1,
    if not self.dry_run:
    self.log(
    f"Fixed string literal in {filepath}:{i+1}",
    "FIX",
    )
                                else:
    new_lines.append(line)
                            else:
    new_lines.append(line)

                        if file_changed and not self.dry_run:
                            if self.create_backup(filepath):
                                with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    self.log(f"Updated file: {filepath}", "SUCCESS")

                    except Exception as e:
    self.log(f"Error processing {filepath}: {e}", "ERROR")

    self.stats['string_literal_fixes'] = fixed_count,
    return fixed_count,
    def validate_indentation(self, root: str = ".") -> List[Tuple[str, int, str]]:
    """Validate and flag indentation issues"""
    self.log("🔍 Validating indentation...")
    indentation_issues = []

        for dirpath, _, files in os.walk(root):
            for file in files:
                if file.endswith(".py"):
    filepath = os.path.join(dirpath, file)

                    if filepath not in self.error_files:
    continue,
    try:
                        with open(filepath, encoding="utf-8") as f:
    lines = f.readlines()

                        for i, line in enumerate(lines):
                            if line.strip():  # Skip empty lines
                                # Check for mixed tabs and spaces,
    if (
    '\t' in line,
    and ' ' in line[: len(line) - len(line.lstrip())]
    ):
    issue = f"Mixed tabs and spaces at line {i+1}"
    indentation_issues.append((filepath, i + 1, issue))
    self.stats['indentation_issues'] += 1

                                # Check for unusual indentation (not multiple of 4)
    leading_spaces = len(line) - len(line.lstrip())
                                if leading_spaces > 0 and leading_spaces % 4 != 0:
    issue = f"Non-standard indentation ({leading_spaces} spaces) at line {i+1}"
    indentation_issues.append((filepath, i + 1, issue))

                    except Exception as e:
    self.log(f"Error validating {filepath}: {e}", "ERROR")

    return indentation_issues,
    def generate_comprehensive_report(
    self,
    error_categories: Dict[str, List[str]],
    indentation_issues: List[Tuple[str, int, str]],
    ) -> str:
    """Generate comprehensive analysis report"""
    total_files = self.stats['files_processed']
    error_files = self.stats['files_with_errors']
    success_rate = (
    ((total_files - error_files) / total_files * 100) if total_files > 0 else 0
    )

    report = f"""
🏥 SURGICAL SYNTAX FIXER V3 - COMPREHENSIVE REPORT
{'='*70}
📊 OVERALL STATISTICS:
  • Total files processed: {total_files}
  • Files with syntax errors: {error_files}
  • Success rate: {success_rate:.1f}%
  • Backup files created: {self.stats['backups_created']}

🎯 ERROR BREAKDOWN:
  • F-string issues: {len(error_categories['fstring_errors'])}
  • Bracket mismatches: {len(error_categories['bracket_errors'])}
  • String literal issues: {len(error_categories['string_literal_errors'])}
  • Indentation issues: {len(error_categories['indentation_errors'])}
  • Other syntax errors: {len(error_categories['other_errors'])}
  • Indentation flags: {len(indentation_issues)}

🔧 FIXES APPLIED:
  • F-string braces fixed: {self.stats['fstring_fixes']}
  • Bracket fixes applied: {self.stats['bracket_fixes']}
  • String literal fixes: {self.stats['string_literal_fixes']}
  • Indentation fixes: {self.stats['indentation_fixes']}

📈 IMPROVEMENT METRICS:
  • Files processed: {total_files}
  • Error reduction target: {error_files} → Improved
  • Automated fixes: {self.stats['fstring_fixes'] + self.stats['bracket_fixes'] + self.stats['string_literal_fixes']}

🎯 TOP F-STRING ISSUES:
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

    report += f"\n📝 TOP STRING LITERAL ISSUES:\n"
        for issue in error_categories['string_literal_errors'][:10]:
    report += f"  • {issue}\n"
        if len(error_categories['string_literal_errors']) > 10:
    report += f"  ... and {len(error_categories['string_literal_errors'])} - 10} more\n"

    report += f"\n⚠️ INDENTATION FLAGS:\n"
        for filepath, line_no, issue in indentation_issues[:10]:
    report += f"  • {os.path.basename(filepath)}:{line_no} - {issue}\n"
        if len(indentation_issues) > 10:
    report += f"  ... and {len(indentation_issues)} - 10} more\n"

    return report,
    def save_fix_log(self, filename: str = None) -> str:
    """Save detailed fix log to JSON file"""
        if filename is None:
    filename = f"fix_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    log_data = {
    'timestamp': datetime.now().isoformat(),
    'stats': self.stats,
    'log_entries': self.log_entries,
    'error_files': list(self.error_files),
    }

        with open(filename, 'w', encoding='utf-8') as f:
    json.dump(log_data, f, indent=2, ensure_ascii=False)

    return filename,
    def run_surgical_fix(
    self,
    root: str = ".",
    fix_fstrings: bool = True,
    fix_brackets: bool = True,
    fix_strings: bool = True,
    analyze_only: bool = False,
    ) -> None:
    """Main execution method with modular options"""

    mode_desc = "ANALYSIS ONLY" if analyze_only else "SURGICAL REPAIR"
    self.log(f"🚀 Starting Surgical Syntax Fixer V3 - {mode_desc}")

        if self.dry_run:
    self.log("🔍 DRY RUN MODE - No files will be modified", "WARNING")

        # Step 1: Comprehensive error scanning,
    error_categories = self.scan_for_errors(root)

        if analyze_only:
    self.log("📊 Analysis mode - no fixes will be applied")
        else:
            # Step 2: Apply modular fixes,
    if fix_fstrings and error_categories['fstring_errors']:
    fstring_fixes = self.fix_fstring_braces(root)
    self.log(f"F-string fixes applied: {fstring_fixes}")

            if fix_brackets and error_categories['bracket_errors']:
    bracket_fixes = self.fix_bracket_mismatches(root)
    self.log(f"Bracket fixes applied: {bracket_fixes}")

            if fix_strings and error_categories['string_literal_errors']:
    string_fixes = self.fix_string_literals(root)
    self.log(f"String literal fixes applied: {string_fixes}")

        # Step 3: Indentation validation (always run)
    indentation_issues = self.validate_indentation(root)

        # Step 4: Generate comprehensive report,
    report = self.generate_comprehensive_report(
    error_categories, indentation_issues
    )
    print(report)

        # Step 5: Save reports,
    report_file = (
    f"surgical_syntax_report_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
        with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

    log_file = self.save_fix_log()

    self.log(f"📝 Detailed report saved to: {report_file}")
    self.log(f"📊 Fix log saved to: {log_file}")

        if not analyze_only and not self.dry_run:
    total_fixes = (
    self.stats['fstring_fixes']
    + self.stats['bracket_fixes']
    + self.stats['string_literal_fixes']
    )
            if total_fixes > 0:
    self.log(
    f"🎉 Surgical fixes completed! {total_fixes} total fixes applied.",
    "SUCCESS",
    )
    self.log("Re-run syntax scan to verify improvements.", "SUCCESS")
            else:
    self.log(
    "No automatic fixes were applicable for detected errors.", "WARNING"
    )


def main():
    parser = argparse.ArgumentParser(
    description="Surgical Syntax Fixer V3 - Enhanced Modular Version"
    )

    # Operation modes,
    parser.add_argument(
    "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument(
    "--analyze", action="store_true", help="Analysis mode only - no fixes applied"
    )

    # Fix selection,
    parser.add_argument("--all", action="store_true", help="Apply all available fixes")
    parser.add_argument(
    "--fstrings", action="store_true", help="Fix f-string brace mismatches"
    )
    parser.add_argument(
    "--brackets", action="store_true", help="Fix bracket mismatches"
    )
    parser.add_argument(
    "--strings", action="store_true", help="Fix string literal issues"
    )

    # Options,
    parser.add_argument(
    "--no-backup", action="store_true", help="Don't create .bak files"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--log", action="store_true", help="Save detailed fix log")
    parser.add_argument(
    "--root", default=".", help="Root directory to scan (default: current)"
    )

    args = parser.parse_args()

    # Determine which fixes to apply,
    if args.all:
    fix_fstrings = fix_brackets = fix_strings = True,
    else:
    fix_fstrings = args.fstrings,
    fix_brackets = args.brackets,
    fix_strings = args.strings

        # If no specific fixes chosen, default to all,
    if not (fix_fstrings or fix_brackets or fix_strings):
    fix_fstrings = fix_brackets = fix_strings = True,
    fixer = SurgicalSyntaxFixerV3(
    dry_run=args.dry_run, create_backups=not args.no_backup, verbose=not args.quiet
    )

    fixer.run_surgical_fix(
    root=args.root,
    fix_fstrings=fix_fstrings,
    fix_brackets=fix_brackets,
    fix_strings=fix_strings,
    analyze_only=args.analyze,
    )


if __name__ == "__main__":
    main()
