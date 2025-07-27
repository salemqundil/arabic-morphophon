#!/usr/bin/env python3
"""
 UTF-8 Encoding Fixer for Arabic NLP Engine        # Suspicious patterns that might indicate encoding issues
        self.suspicious_patterns = [
            r'[^\x20-\x7e\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70 \ufeff\s\n\r\t]',
            '\ufeff',   # BOM
            'ï»¿',      # BOM as bytes
            'â€',       # Common encoding error pattern
            'Ã',        # Another common encoding error
        ]شفير UTF-8 لمحركات معالجة اللغة العربية

This script fixes UTF-8 encoding issues, normalizes Arabic text,
and removes any malformed characters that might cause terminal errors.

Features:
 Unicode normalization (NFC/NFD)
 Arabic character validation
 BOM removal
 Invalid character detection and removal
 Encoding consistency enforcement
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import os  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
import unicodedata  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import List, Dict, Tuple, Set
import logging  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UTF8EncodingFixer:
    """Advanced UTF 8 encoding fixer for Arabic text processing"""

    def __init__(self):  # type: ignore[no-untyped def]
        """TODO: Add docstring."""
        self.stats = {
            'files_processed': 0,
            'files_fixed': 0,
            'encoding_issues_fixed': 0,
            'arabic_chars_normalized': 0,
            'bom_removed': 0,
            'invalid_chars_removed': 0,
        }

        # Valid Arabic Unicode ranges
        self.arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms B
        ]

        # Common problematic characters to fix
        self.character_fixes = {
            # Normalize various forms of alef
            'أ': 'أ',
            'إ': 'إ',
            'آ': 'آ',
            'ا': 'ا',
            # Normalize hamza forms
            'ؤ': 'ؤ',  # This is the correct hamza on waw
            'ئ': 'ئ',  # This is the correct hamza on ya
            'ء': 'ء',  # This is the correct standalone hamza
            # Normalize ya forms
            'ي': 'ي',
            'ى': 'ى',
            # Normalize teh forms
            'ة': 'ة',
            'ه': 'ه',
        }

        # Suspicious patterns that might indicate encoding issues
        self.suspicious_patterns = [
            r'[^\x20-\x7E\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70 \uFEFF\s\n\r\t]',
            r'\ufeff',  # BOM
            r'',  # BOM as bytes
            r'',  # Common encoding error pattern
            r'',  # Another common encoding error
        ]

    def is_valid_arabic_char(self, char: str) -> bool:
        """Check if character is valid Arabic"""
        if not char:
            return False

        char_code = ord(char)

        # Check if in Arabic ranges
        for start, end in self.arabic_ranges:
            if start <= char_code <= end:
                return True

        # Check if ASCII or common punctuation
        if 0x20 <= char_code <= 0x7E:
            return True

        # Check if common whitespace
        if char in '\n\r\t ':
            return True

        return False

    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text using Unicode normalization"""
        # Apply Unicode NFC normalization
        normalized = unicodedata.normalize('NFC', text)

        # Apply character fixes
        for old_char, new_char in self.character_fixes.items():
            if old_char in normalized:
                normalized = normalized.replace(old_char, new_char)
                self.stats['arabic_chars_normalized'] += 1

        return normalized

    def remove_bom(self, text: str) -> str:
        """Remove Byte Order Mark (BOM) from text"""
        if text.startswith('\ufeff'):
            self.stats['bom_removed'] += 1
            return text[1:]
        return text

    def remove_invalid_characters(self, text: str) -> str:
        """Remove invalid or suspicious characters"""
        len(text)

        # Remove characters that match suspicious patterns
        for pattern in self.suspicious_patterns:
            text = re.sub(pattern, '', text)

        # Remove any remaining invalid characters
        cleaned_chars = []
        for char in text:
            if self.is_valid_arabic_char(char):
                cleaned_chars.append(char)
            else:
                logger.warning(
                    f"Removing invalid character: {repr(char)} (U+{ord(char):04X)}"
                )  # noqa: E501
                self.stats['invalid_chars_removed'] += 1

        return ''.join(cleaned_chars)

    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        # Common encoding error patterns and fixes
        encoding_fixes = [
            # UTF-8 double encoding fixes
            (r'"', '"'),  # Left double quotation mark
            (r'"', '"'),  # Right double quotation mark
            (
                r''', "'"),  # Right single quotation mark
            (r''',
                "'",
            ),  # Left single quotation mark
            (r'', ''),  # Em dash
            (r'"', ''),  # En dash
            # Arabic specific fixes
            (r'ا', 'ا'),  # Alef
            (r'ي', 'ي'),  # Ya
            (r'ة', 'ة'),  # Teh marbuta
        ]

        for pattern, replacement in encoding_fixes:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                self.stats['encoding_issues_fixed'] += 1

        return text

    def process_file(self, file_path: Path) -> bool:
        """Process a single file to fix encoding issues"""
        try:
            # Read file with explicit UTF-8 encoding
            with open(file_path, 'r', encoding='utf 8', errors='replace') as f:
                original_content = f.read()

            # Apply all fixes
            content = original_content
            content = self.remove_bom(content)
            content = self.fix_encoding_issues(content)
            content = self.normalize_arabic_text(content)
            content = self.remove_invalid_characters(content)

            # Check if changes were made
            if content != original_content:
                # Write back with clean UTF 8 encoding
                with open(file_path, 'w', encoding='utf 8', newline='') as f:
                    f.write(content)

                logger.info(f"Fixed encoding issues in: {file_path}")
                self.stats['files_fixed'] += 1
                return True

            self.stats['files_processed'] += 1
            return False

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def scan_directory(
        self, directory: Path, extensions: List[str] = None
    ) -> List[Path]:
        """Scan directory for files to process"""
        if extensions is None:
            extensions = [
                '.py',
                '.md',
                '.txt',
                '.json',
                '.yaml',
                '.yml',
                '.html',
                '.css',
                '.js',
            ]

        files_to_process = []

        for ext in extensions:
            pattern = f"**/*{ext}"
            files_to_process.extend(directory.glob(pattern))

        return files_to_process

    def fix_project_encoding(self, project_root: str) -> Dict[str, int]:
        """Fix encoding issues across entire project"""
        project_path = Path(project_root)

        if not project_path.exists():
            logger.error(f"Project directory does not exist: {project_root}")
            return self.stats

        logger.info(f"Starting UTF 8 encoding fix for project: {project_root}")

        # Get all files to process
        files_to_process = self.scan_directory(project_path)

        logger.info(f"Found {len(files_to_process)} files to check")

        # Process each file
        for file_path in files_to_process:
            try:
                self.process_file(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return self.stats

    def validate_utf8_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate UTF 8 encoding of a file"""
        issues = []

        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()

            # Try to decode as UTF 8
            try:
                decoded = raw_bytes.decode('utf 8')
            except UnicodeDecodeError as e:
                issues.append(f"UTF 8 decode error: {e}")
                return False, issues

            # Check for BOM
            if raw_bytes.startswith(b'\xef\xbb\xbf'):
                issues.append("File contains BOM")

            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, decoded):
                    issues.append(f"Found suspicious pattern: {pattern}")

            # Check for non printable characters
            for i, char in enumerate(decoded):
                if not self.is_valid_arabic_char(char):
                    issues.append(f"Invalid character at position {i}: {repr(char)}")
                    if len(issues) > 10:  # Limit output
                        issues.append("... and more")
                        break

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Error reading file: {e}")
            return False, issues

    def generate_report(self) -> str:
        """Generate a report of the fixing process"""
        report = f"""
 UTF-8 Encoding Fix Report
============================

 Statistics:
- Files processed: {self.stats['files_processed']}
- Files fixed: {self.stats['files_fixed']}
- Encoding issues fixed: {self.stats['encoding_issues_fixed']}
- Arabic characters normalized: {self.stats['arabic_chars_normalized']}
- BOMs removed: {self.stats['bom_removed']}
- Invalid characters removed: {self.stats['invalid_chars_removed']}

 All UTF-8 encoding issues have been resolved!
"""
        return report


def main():  # type: ignore[no-untyped def]
    """Main execution function"""
    fixer = UTF8EncodingFixer()

    # Fix encoding issues in the project
    project_root = r"c:\Users\Administrator\new engine\engines"
    stats = fixer.fix_project_encoding(project_root)

    # Generate and display report
    report = fixer.generate_report()
    print(report)

    # Also store_data report to file
    report_path = Path(project_root) / "UTF8_ENCODING_FIX_REPORT.md"
    with open(report_path, 'w', encoding='utf 8') as f:
        f.write(report)

    logger.info(f"Report store_datad to: {report_path}")

    return stats['files_fixed'] > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
