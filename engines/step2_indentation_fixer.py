#!/usr/bin/env python3
"""
🔧 STEP 2: INDENTATION FIXER (215 FILES)
========================================

Controlled strategy to fix import statement indentation issues:
- Normalize all import and from statements to column 0
- Preserve UTF-8 BOM and Arabic characters
- Leave Arabic docstrings, f-strings, and comments intact
- Backup original files before patching
- Skip files with decoding errors

🎯 Pattern: import statements are inconsistently or incorrectly indented
🧠 Safety-first approach with comprehensive logging
"""

import os
import ast
import shutil
import re
from pathlib import Path
from datetime import datetime
import logging


class IndentationFixer:
    def __init__(self):
    self.backup_dir = Path("backups_indentation_fixes")
    self.backup_dir.mkdir(exist_ok=True)

    self.skip_log_file = "indentation_skip_log.txt"
    self.success_log_file = "indentation_success_log.txt"

        # Setup logging
    logging.basicConfig(
    level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler(
    f'indentation_fixer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    ),
    logging.StreamHandler(),
    ],
    )
    self.logger = logging.getLogger(__name__)

        # Stats
    self.files_processed = 0
    self.files_fixed = 0
    self.files_skipped = 0
    self.files_no_changes = 0

        # Import patterns to fix
    self.import_patterns = [
    re.compile(r'^(\s+)(import\s+.+)$', re.MULTILINE),
    re.compile(r'^(\s+)(from\s+.+\s+import\s+.+)$', re.MULTILINE),
    ]

    def create_backup(self, file_path):
    """Create a timestamped backup of the file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
    :17
    ]  # microseconds for uniqueness
    backup_path = self.backup_dir / f"{Path(file_path).name}_{timestamp}.bak"

        try:
    shutil.copy2(file_path, backup_path)
    return backup_path
        except Exception as e:
    self.logger.error(f"Failed to create backup for {file_path}: {e}")
    return None

    def detect_encoding(self, file_path):
    """Detect file encoding, preserving UTF-8 BOM if present"""
        try:
            # Check for BOM first
            with open(file_path, 'rb') as f:
    raw = f.read(3)
                if raw.startswith(b'\xef\xbb\xbf'):
    return 'utf-8-sig'

            # Try UTF-8
            with open(file_path, 'r', encoding='utf-8') as f:
    f.read()
    return 'utf-8'

        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['cp1256', 'iso-8859-6', 'latin1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
    f.read()
    return encoding
                except UnicodeDecodeError:
    continue

    return None

    def is_import_statement(self, line):
    """Check if line is an import statement"""
    stripped = line.strip()
    return (
    stripped.startswith('import ')
    or stripped.startswith('from ')
    and ' import ' in stripped
    )

    def is_in_multiline_context(self, lines, line_idx):
    """Check if we're inside a multiline string, comment block, or other context"""
        # Simple heuristic: check if we're inside triple quotes
    in_string = False
    quote_count = 0

        for i in range(line_idx):
    line = lines[i]
            # Count triple quotes
    triple_single = line.count("'''")
    triple_double = line.count('"""')

    quote_count += triple_single + triple_double

        # If odd number of triple quotes, we're inside a multiline string
    return quote_count % 2 == 1

    def fix_indentation(self, file_path):
    """Fix indentation issues in a single file"""
    self.files_processed += 1

        # Detect encoding
    encoding = self.detect_encoding(file_path)
        if encoding is None:
    self.logger.warning(f"Could not determine encoding for {file_path}")
    self.log_skip(file_path, "Encoding detection failed")
    self.files_skipped += 1
    return False

        try:
            # Read file content
            with open(file_path, 'r', encoding=encoding) as f:
    original_content = f.read()
    lines = original_content.splitlines(keepends=True)

    modified_lines = []
    changes_made = 0

            for i, line in enumerate(lines):
                # Skip if we're in a multiline string context
                if self.is_in_multiline_context(lines, i):
    modified_lines.append(line)
    continue

                # Check if this is an indented import statement
                if self.is_import_statement(line) and line.startswith(' '):
                    # Remove leading whitespace from import statements
    fixed_line = line.lstrip()
                    if line != fixed_line:
    changes_made += 1
    self.logger.info(
    f"  Line {i+1}: '{line.rstrip()}' → '{fixed_line.rstrip()}'"
    )
    modified_lines.append(fixed_line)
                else:
    modified_lines.append(line)

            if changes_made > 0:
                # Create backup
    backup_path = self.create_backup(file_path)
                if backup_path is None:
    self.files_skipped += 1
    return False

                # Write modified content
    modified_content = ''.join(modified_lines)
                with open(file_path, 'w', encoding=encoding) as f:
    f.write(modified_content)

                # Validate the fix with AST
                if self.validate_syntax(file_path):
    self.logger.info(
    f"✅ Fixed {file_path}: {changes_made} import statements normalized"
    )
    self.log_success(file_path, changes_made)
    self.files_fixed += 1
    return True
                else:
                    # Restore from backup if syntax is still broken
    shutil.copy2(backup_path, file_path)
    self.logger.warning(
    f"❌ Syntax still invalid after fix, restored: {file_path}"
    )
    self.log_skip(file_path, "Syntax still invalid after fix")
    self.files_skipped += 1
    return False
            else:
    self.logger.info(f"⚪ No indentation changes needed: {file_path}")
    self.files_no_changes += 1
    return True

        except Exception as e:
    self.logger.error(f"Error processing {file_path}: {e}")
    self.log_skip(file_path, f"Processing error: {e}")
    self.files_skipped += 1
    return False

    def validate_syntax(self, file_path):
    """Validate Python syntax using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    ast.parse(content)
    return True
        except Exception:
    return False

    def log_skip(self, file_path, reason):
    """Log skipped files"""
        with open(self.skip_log_file, 'a', encoding='utf-8') as f:
    f.write(f"{datetime.now().isoformat()} - {file_path} - {reason}\n")

    def log_success(self, file_path, changes_count):
    """Log successful fixes"""
        with open(self.success_log_file, 'a', encoding='utf-8') as f:
    f.write(
    f"{datetime.now().isoformat()} - {file_path} - {changes_count} imports fixed\n"
    )

    def get_files_with_indentation_errors(self):
    """Get list of files with indentation errors from the validation report"""
    report_file = "syntax_validation_report_20250727_011815.txt"

        if not os.path.exists(report_file):
    self.logger.error(f"Validation report not found: {report_file}")
    return []

    files_with_errors = []

        try:
            with open(report_file, 'r', encoding='utf-8') as f:
    content = f.read()

            # Extract files with indentation errors
    lines = content.split('\n')
    in_indentation_section = False

            for line in lines:
                if "Indentation Issues:" in line:
    in_indentation_section = True
                elif line.strip().startswith(
    "F-String Issues:"
    ) or line.strip().startswith("Other Issues:"):
    in_indentation_section = False
                elif in_indentation_section and "❌" in line and ".py:" in line:
                    # Extract filename
    parts = line.split("❌")
                    if len(parts) > 1:
    filename_part = parts[1].strip().split(":")[0]
                        if filename_part.endswith('.py'):
    files_with_errors.append(filename_part)

    self.logger.info(
    f"Found {len(files_with_errors)} files with indentation errors"
    )
    return files_with_errors

        except Exception as e:
    self.logger.error(f"Error reading validation report: {e}")
    return []

    def run(self):
    """Run the indentation fixer on all identified files"""
    self.logger.info("🔧 STEP 2: INDENTATION FIXER STARTING")
    self.logger.info("=" * 50)

        # Clear previous logs
        for log_file in [self.skip_log_file, self.success_log_file]:
            if os.path.exists(log_file):
    os.remove(log_file)

        # Get files to fix
    target_files = self.get_files_with_indentation_errors()

        if not target_files:
    self.logger.error("No files with indentation errors found!")
    return

    self.logger.info(
    f"🎯 Target: {len(target_files)} files with indentation issues"
    )
    self.logger.info(f"📁 Backups will be stored in: {self.backup_dir}")

        # Process each file
        for file_path in target_files:
            if os.path.exists(file_path):
    self.logger.info(f"Processing: {file_path}")
    self.fix_indentation(file_path)
            else:
    self.logger.warning(f"File not found: {file_path}")
    self.log_skip(file_path, "File not found")
    self.files_skipped += 1

        # Summary
    self.logger.info("\n📊 INDENTATION FIXER RESULTS:")
    self.logger.info("=" * 40)
    self.logger.info(f"Files processed: {self.files_processed}")
    self.logger.info(f"Files fixed: {self.files_fixed}")
    self.logger.info(f"Files skipped: {self.files_skipped}")
    self.logger.info(f"Files no changes: {self.files_no_changes}")

    success_rate = (
    (self.files_fixed / self.files_processed * 100)
            if self.files_processed > 0
            else 0
    )
    self.logger.info(f"Success rate: {success_rate:.1f}%")

        if self.files_fixed > 0:
    self.logger.info(f"\n🎯 Next: Run syntax validator to measure improvement")
    self.logger.info(
    f"Expected: +{self.files_fixed} files moved to clean status"
    )


def main():
    """Main entry point"""
    fixer = IndentationFixer()
    fixer.run()


if __name__ == "__main__":
    main()
