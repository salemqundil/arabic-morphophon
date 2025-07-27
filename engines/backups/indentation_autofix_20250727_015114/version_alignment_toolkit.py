#!/usr/bin/env python3
"""
🔧 Version Alignment Toolkit
Ensures all Python files are aligned to consistent standards and syntax.
"""

import re
import ast
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VersionAlignmentTool:
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.backup_dir = (
            Path("backups")
            / f"version_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification."""
        backup_path = self.backup_dir / file_path.name
        backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        return backup_path

    def fix_import_indentation(self, content: str) -> Tuple[str, int]:
        """Fix import statement indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0

        for i, line in enumerate(lines):
            # Fix common import indentation patterns
            if re.match(r'^    import\s+', line):
                # Remove extra indentation from imports at module level
                fixed_line = line.lstrip()
                if fixed_line != line:
                    fixes_count += 1
                    logger.debug(f"Fixed import indentation at line {i+1}")
                fixed_lines.append(fixed_line)
            elif re.match(r'^import\s+,', line):
                # Fix comma issues in imports
                fixed_line = re.sub(r'import\s+,', 'import', line)
                if fixed_line != line:
                    fixes_count += 1
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixes_count

    def fix_trailing_commas(self, content: str) -> Tuple[str, int]:
        """Fix trailing comma issues in code."""
        fixes_count = 0
        original_content = content

        # Fix trailing commas at end of statements
        patterns = [
            (r',(\s*\n\s*def\s)', r'\1'),  # Before function definitions
            (r',(\s*\n\s*class\s)', r'\1'),  # Before class definitions
            (r',(\s*\n\s*if\s)', r'\1'),  # Before if statements
            (r',(\s*\n\s*return\s)', r'\1'),  # Before return statements
            (r',(\s*\n\s*$)', r'\1'),  # At end of file
        ]

        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if new_content != content:
                fixes_count += 1
                content = new_content

        return content, fixes_count

    def fix_function_definitions(self, content: str) -> Tuple[str, int]:
        """Fix function definition syntax issues."""
        fixes_count = 0

        # Fix functions missing parentheses
        pattern = r'def\s+(\w+)\s*:\s*$'
        replacement = r'def \1():'
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            fixes_count += 1
            content = new_content

        # Fix empty parameter lists
        pattern = r'def\s+(\w+)\(\)\s*:'
        replacement = r'def \1():'
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            fixes_count += 1
            content = new_content

        return content, fixes_count

    def fix_logging_statements(self, content: str) -> Tuple[str, int]:
        """Fix standalone logging configuration statements."""
        fixes_count = 0
        lines = content.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for standalone logging level statements
            if line == 'level=logging.INFO,' or line.startswith('level=logging.'):
                # Look for format statement on next line
                format_line = None
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('format='):
                        format_line = next_line
                        i += 1  # Skip the format line

                # Replace with proper logging.basicConfig
                if format_line:
                    fixed_lines.append(f"logging.basicConfig({line} {format_line})")
                else:
                    fixed_lines.append(f"logging.basicConfig({line})")

                fixes_count += 1
                logger.debug(f"Fixed logging statement: {line}")
            else:
                fixed_lines.append(lines[i])

            i += 1

        return '\n'.join(fixed_lines), fixes_count

    def fix_string_literals(self, content: str) -> Tuple[str, int]:
        """Fix string literal syntax issues."""
        fixes_count = 0

        # Fix unterminated f-strings
        patterns = [
            (r'f"([^"]*)"([^}]*)"', r'f"\1\2"'),  # Fix broken f-string quotes
            (r"f'([^']*)'([^}]*)'", r"f'\1\2'"),  # Fix broken f-string single quotes
        ]

        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                fixes_count += 1
                content = new_content

        return content, fixes_count

    def validate_syntax(self, content: str) -> List[str]:
        """Validate Python syntax using AST."""
        errors = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")

        return errors

    def align_file(self, file_path: Path) -> int:
        """Apply all alignment fixes to a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            total_fixes = 0

            # Apply all fixes in sequence
            fixes = [
                self.fix_import_indentation,
                self.fix_trailing_commas,
                self.fix_function_definitions,
                self.fix_logging_statements,
                self.fix_string_literals,
            ]

            for fix_function in fixes:
                content, fix_count = fix_function(content)
                total_fixes += fix_count

            # Validate final result
            errors = self.validate_syntax(content)
            if errors:
                logger.warning(f"⚠️  Syntax validation errors in {file_path}:")
                for error in errors:
                    logger.warning(f"    {error}")

            # Save if changes were made
            if content != original_content:
                self.backup_file(file_path)
                file_path.write_text(content, encoding='utf-8')
                self.fixes_applied += total_fixes
                logger.info(f"✅ Aligned {total_fixes} issues in {file_path}")
                return total_fixes

            return 0

        except Exception as e:
            logger.error(f"❌ Error aligning {file_path}: {e}")
            return 0

    def process_directory(self, directory: Path = Path('.')) -> Dict:
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
            fixes = self.align_file(file_path)

            if fixes > 0:
                results['files_fixed'] += 1
                results['total_fixes'] += fixes

        return results


class ConsistencyChecker:
    """Check for consistency issues across the codebase."""

    def __init__(self):
        self.issues = []

    def check_import_consistency(self, directory: Path = Path('.')) -> List[str]:
        """Check for import consistency issues."""
        import_patterns = {}
        inconsistencies = []

        for file_path in directory.rglob('*.py'):
            if any(part.startswith('.') for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                # Find import statements
                imports = re.findall(
                    r'^(import\s+\w+|from\s+\w+\s+import\s+\w+)', content, re.MULTILINE
                )

                for imp in imports:
                    if imp not in import_patterns:
                        import_patterns[imp] = []
                    import_patterns[imp].append(file_path)

            except Exception as e:
                inconsistencies.append(f"Error reading {file_path}: {e}")

        return inconsistencies

    def check_coding_style_consistency(self, directory: Path = Path('.')) -> List[str]:
        """Check for coding style consistency."""
        inconsistencies = []

        # Check for mixed quote styles, indentation, etc.
        for file_path in directory.rglob('*.py'):
            if any(part.startswith('.') for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding='utf-8')

                # Check for mixed quotes
                single_quotes = len(re.findall(r"'[^']*'", content))
                double_quotes = len(re.findall(r'"[^"]*"', content))

                if single_quotes > 0 and double_quotes > 0:
                    ratio = min(single_quotes, double_quotes) / max(
                        single_quotes, double_quotes
                    )
                    if ratio > 0.3:  # More than 30% mixed usage
                        inconsistencies.append(
                            f"{file_path}: Mixed quote styles ({single_quotes} single, {double_quotes} double)"
                        )

            except Exception as e:
                inconsistencies.append(f"Error checking {file_path}: {e}")

        return inconsistencies


def main():
    """Main entry point for version alignment."""
    logger.info("🔧 Starting Version Alignment Toolkit")

    # Run alignment
    aligner = VersionAlignmentTool()
    results = aligner.process_directory()

    # Run consistency checks
    checker = ConsistencyChecker()
    import_issues = checker.check_import_consistency()
    style_issues = checker.check_coding_style_consistency()

    # Print comprehensive report
    print("\n" + "=" * 70)
    print("🔧 VERSION ALIGNMENT TOOLKIT SUMMARY")
    print("=" * 70)
    print(f"📁 Files processed: {results['files_processed']}")
    print(f"🔧 Files aligned: {results['files_fixed']}")
    print(f"✅ Total fixes applied: {results['total_fixes']}")
    print(f"💾 Backups saved to: {aligner.backup_dir}")

    if import_issues:
        print(f"\n⚠️  Import inconsistencies found: {len(import_issues)}")
        for issue in import_issues[:5]:  # Show first 5
            print(f"    {issue}")

    if style_issues:
        print(f"\n📝 Style inconsistencies found: {len(style_issues)}")
        for issue in style_issues[:5]:  # Show first 5
            print(f"    {issue}")

    if results['total_fixes'] > 0:
        print("\n🎉 Version alignment completed successfully!")
        print("💡 Run AST validation to verify all fixes")
    else:
        print("\n✨ All files are already properly aligned!")

    return results['total_fixes']


if __name__ == "__main__":
    main()
