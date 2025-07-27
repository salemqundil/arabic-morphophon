#!/usr/bin/env python3
"""
🔧 STEP 2B: SIMPLIFIED INDENTATION FIXER
========================================

Simplified approach to fix import statement indentation:
- Fix only indented import/from statements
- No complex validation, just normalize to column 0
- Minimal logging to avoid Unicode issues
- Focus on the specific pattern identified in validation report
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def create_backup(file_path):
    """Create a timestamped backup of the file"""
    backup_dir = Path("backups_indentation_simple")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
    backup_path = backup_dir / f"{Path(file_path).name}_{timestamp}.bak"

    try:
    shutil.copy2(file_path, backup_path)
    return backup_path
    except Exception as e:
    print(f"Error creating backup for {file_path}: {e}")
    return None


def fix_file_indentation(file_path):
    """Fix indentation in a single file"""
    print(f"Processing: {file_path}")

    try:
        # Read file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    modified_lines = []
    changes_made = 0

        for i, line in enumerate(lines):
            # Check if line starts with whitespace and contains import/from
    stripped = line.strip()
            if line.startswith(' ') and (
    stripped.startswith('import ')
    or (stripped.startswith('from ') and ' import ' in stripped)
    ):

                # Remove leading whitespace
    fixed_line = line.lstrip()
                if line != fixed_line:
    changes_made += 1
    print(f"  Line {i+1}: Fixed import indentation")
    modified_lines.append(fixed_line)
            else:
    modified_lines.append(line)

        if changes_made > 0:
            # Create backup
    backup_path = create_backup(file_path)
            if backup_path:
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(modified_lines)
    print(f"  SUCCESS: Fixed {changes_made} import statements")
    return True
            else:
    print(f"  FAILED: Could not create backup")
    return False
        else:
    print(f"  NO CHANGES: No indented imports found")
    return True

    except Exception as e:
    print(f"  ERROR: {e}")
    return False


def get_target_files():
    """Get the specific files identified in the validation report"""
    # Based on the output we saw, these are the files being processed
    target_files = [
    "arabic_function_words_analyzer.py",
    "arabic_interrogative_pronouns_deep_model.py",
    "arabic_interrogative_pronouns_enhanced.py",
    "arabic_interrogative_pronouns_final.py",
    "arabic_interrogative_pronouns_test_analysis.py",
    "arabic_normalizer.py",
    "arabic_phoneme_word_decision_tree.py",
    "arabic_pronouns_analyzer.py",
    "arabic_pronouns_deep_model.py",
    "arabic_relative_pronouns_analyzer.py",
    ]

    # Only return files that actually exist
    existing_files = []
    for file_path in target_files:
        if os.path.exists(file_path):
    existing_files.append(file_path)
        else:
    print(f"File not found: {file_path}")

    return existing_files


def main():
    """Main entry point"""
    print("STEP 2B: SIMPLIFIED INDENTATION FIXER")
    print("=" * 50)

    target_files = get_target_files()
    print(f"Target files: {len(target_files)}")

    success_count = 0

    for file_path in target_files:
        if fix_file_indentation(file_path):
    success_count += 1

    print(f"\nRESULTS:")
    print(f"Files processed: {len(target_files)}")
    print(f"Files successfully processed: {success_count}")
    print(
    f"Success rate: {success_count/len(target_files)*100:.1f}%"
        if target_files
        else "0%"
    )

    if success_count > 0:
    print(f"\nNext: Run syntax validator to check improvements")


if __name__ == "__main__":
    main()
