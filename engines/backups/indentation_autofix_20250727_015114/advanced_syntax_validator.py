#!/usr/bin/env python3
"""
🧪 Advanced Syntax Validator
Comprehensive AST-based validation with detailed error reporting.
"""

import ast
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntaxError_Detail:
    def __init__(
        self, filename: str, line_no: int, message: str, line_content: str = ""
    ):
        self.filename = filename
        self.line_no = line_no
        self.message = message
        self.line_content = line_content
        self.category = self.categorize_error()

    def categorize_error(self) -> str:
        """Categorize the syntax error for better reporting."""
        msg_lower = self.message.lower()

        if 'f-string' in msg_lower or 'unterminated string' in msg_lower:
            return 'F-String Issues'
        elif 'unexpected indent' in msg_lower or 'indentation' in msg_lower:
            return 'Indentation Issues'
        elif 'unmatched' in msg_lower or 'closing parenthesis' in msg_lower:
            return 'Bracket/Parenthesis Issues'
        elif 'import' in msg_lower:
            return 'Import Issues'
        elif 'invalid syntax' in msg_lower and 'comma' in msg_lower:
            return 'Comma/Punctuation Issues'
        elif 'invalid syntax' in msg_lower:
            return 'General Syntax Issues'
        elif 'cannot parse' in msg_lower:
            return 'Parse Errors'
        else:
            return 'Other Issues'


class AdvancedSyntaxValidator:
    def __init__(self):
        self.errors: List[SyntaxError_Detail] = []
        self.files_processed = 0
        self.files_with_errors = 0

    def validate_file(self, file_path: Path) -> List[SyntaxError_Detail]:
        """Validate a single Python file and return detailed errors."""
        file_errors = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Try to parse with AST
            try:
                ast.parse(content)
                # File is syntactically correct
                return []

            except SyntaxError as e:
                error_detail = SyntaxError_Detail(
                    filename=str(file_path),
                    line_no=e.lineno or 0,
                    message=e.msg or "Unknown syntax error",
                    line_content=(
                        lines[e.lineno - 1]
                        if e.lineno and e.lineno <= len(lines)
                        else ""
                    ),
                )
                file_errors.append(error_detail)

        except UnicodeDecodeError:
            error_detail = SyntaxError_Detail(
                filename=str(file_path),
                line_no=0,
                message="File encoding error - not valid UTF-8",
                line_content="",
            )
            file_errors.append(error_detail)

        except Exception as e:
            error_detail = SyntaxError_Detail(
                filename=str(file_path),
                line_no=0,
                message=f"Unexpected error: {str(e)}",
                line_content="",
            )
            file_errors.append(error_detail)

        return file_errors

    def validate_directory(self, directory: Path = Path('.')) -> Dict:
        """Validate all Python files in directory."""
        self.errors.clear()
        self.files_processed = 0
        self.files_with_errors = 0

        for file_path in directory.rglob('*.py'):
            # Skip backup directories, virtual environments, and hidden files
            if any(
                part.startswith('.')
                or part in ['backups', 'venv', '__pycache__', 'node_modules']
                for part in file_path.parts
            ):
                continue

            self.files_processed += 1
            file_errors = self.validate_file(file_path)

            if file_errors:
                self.files_with_errors += 1
                self.errors.extend(file_errors)
                logger.debug(f"❌ {len(file_errors)} errors in {file_path}")
            else:
                logger.debug(f"✅ {file_path} - OK")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate a comprehensive validation report."""
        # Categorize errors
        error_categories = {}
        for error in self.errors:
            category = error.category
            if category not in error_categories:
                error_categories[category] = []
            error_categories[category].append(error)

        # Generate statistics
        report = {
            'files_processed': self.files_processed,
            'files_with_errors': self.files_with_errors,
            'files_clean': self.files_processed - self.files_with_errors,
            'total_errors': len(self.errors),
            'error_categories': error_categories,
            'success_rate': (self.files_processed - self.files_with_errors)
            / max(self.files_processed, 1)
            * 100,
        }

        return report

    def save_detailed_report(self, output_path: Path = None) -> Path:
        """Save a detailed report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"syntax_validation_report_{timestamp}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("🧪 ADVANCED SYNTAX VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"📊 SUMMARY\n")
            f.write(f"Files processed: {self.files_processed}\n")
            f.write(f"Files with errors: {self.files_with_errors}\n")
            f.write(f"Clean files: {self.files_processed - self.files_with_errors}\n")
            f.write(f"Total errors: {len(self.errors)}\n")
            f.write(
                f"Success rate: {(self.files_processed - self.files_with_errors) / max(self.files_processed, 1) * 100:.1f}%\n\n"
            )

            # Group errors by category
            error_categories = {}
            for error in self.errors:
                category = error.category
                if category not in error_categories:
                    error_categories[category] = []
                error_categories[category].append(error)

            f.write("📋 ERRORS BY CATEGORY\n")
            f.write("-" * 40 + "\n")
            for category, errors in sorted(
                error_categories.items(), key=lambda x: len(x[1]), reverse=True
            ):
                f.write(f"\n{category}: {len(errors)} errors\n")
                for error in errors[:10]:  # Show first 10 of each category
                    f.write(
                        f"  ❌ {error.filename}: Line {error.line_no} - {error.message}\n"
                    )
                    if error.line_content.strip():
                        f.write(f"     Code: {error.line_content.strip()}\n")
                if len(errors) > 10:
                    f.write(f"     ... and {len(errors) - 10} more\n")

            f.write(f"\n📁 ALL ERRORS BY FILE\n")
            f.write("-" * 40 + "\n")

            # Group by file
            file_errors = {}
            for error in self.errors:
                if error.filename not in file_errors:
                    file_errors[error.filename] = []
                file_errors[error.filename].append(error)

            for filename, errors in sorted(file_errors.items()):
                f.write(f"\n📄 {filename}: {len(errors)} errors\n")
                for error in errors:
                    f.write(f"  Line {error.line_no}: {error.message}\n")
                    if error.line_content.strip():
                        f.write(f"    Code: {error.line_content.strip()}\n")

        logger.info(f"📋 Detailed report saved to: {output_path}")
        return output_path


def main():
    """Main entry point for syntax validation."""
    logger.info("🧪 Starting Advanced Syntax Validation")

    validator = AdvancedSyntaxValidator()
    report = validator.validate_directory()

    # Print summary to console
    print("\n" + "=" * 70)
    print("🧪 ADVANCED SYNTAX VALIDATION SUMMARY")
    print("=" * 70)
    print(f"📁 Files processed: {report['files_processed']}")
    print(f"✅ Clean files: {report['files_clean']}")
    print(f"❌ Files with errors: {report['files_with_errors']}")
    print(f"🚨 Total errors: {report['total_errors']}")
    print(f"📊 Success rate: {report['success_rate']:.1f}%")

    if report['error_categories']:
        print(f"\n📋 ERROR BREAKDOWN:")
        for category, errors in sorted(
            report['error_categories'].items(), key=lambda x: len(x[1]), reverse=True
        ):
            print(f"  {category}: {len(errors)} errors")

    # Save detailed report
    report_path = validator.save_detailed_report()

    if report['total_errors'] == 0:
        print("\n🎉 All files have valid syntax!")
    else:
        print(f"\n📋 Detailed report saved to: {report_path}")
        print("💡 Use the version alignment toolkit to fix common issues")

    return report['total_errors']


if __name__ == "__main__":
    main()
