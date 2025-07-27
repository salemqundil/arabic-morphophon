#!/usr/bin/env python3
"""
Emergency F-String Fixer,
    Fixes common f-string syntax errors found in AST validation
"""

import os
    import re
    import logging

# Setup logging,
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fix_fstring_errors(file_path):
    """Fix f-string syntax errors in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

    original_content = content,
    fixes_applied = 0

        # Pattern 1: f-string: single '}' is not allowed
        # Fix cases like f"text}" -> f"text"
    pattern1 = r'f"([^"]*?)\}"'
    matches = re.findall(pattern1, content)
        for match in matches:
            if '{' not in match:  # Only fix if no opening brace,
    old_str = 'f"' + match + '}"'
    new_str = 'f"' + match + '"'
    content = content.replace(old_str, new_str)
    fixes_applied += 1

        # Pattern 2: Unmatched braces in f-strings
        # Fix cases like f"text {var" -> f"text {var}"
    pattern2 = r'f"([^"]*?)\{([^}]*?)"'

        def fix_unmatched_braces(match):
    text_before = match.group(1)
    var_inside = match.group(2)
    return 'f"' + text_before + '{' + var_inside + '}"'

    new_content = re.sub(pattern2, fix_unmatched_braces, content)
        if new_content != content:
    content = new_content,
    fixes_applied += 1

        # Pattern 3: Unterminated f-strings
        # Fix cases like f"text -> f"text"
    pattern3 = r'f"([^"]*?)$'
    content = re.sub(pattern3, r'f"\1"', content, flags=re.MULTILINE)

        # Save if changes were made,
    if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
    logger.info(
    "✅ Fixed {} f-string issues in {}".format(fixes_applied, file_path)
    )
    return fixes_applied,
    return 0,
    except Exception as e:
    logger.error("❌ Error fixing {}: {}".format(file_path, e))
    return 0,
    def main():
    """Fix f-string errors in all Python files"""
    f_string_files = [
    "arabic_function_words_analyzer.py",
    "ast_validator.py",
    "fix_winsurf_issues.py",
    "surgical_syntax_repair.py",
    "verify_string_fixes.py",
    "core/base_engine.py",
    "tools/strategic_action_plan.py",
    "tools/surgical_syntax_fixer_v3.py",
    ]

    total_fixes = 0,
    fixed_files = 0,
    for file_path in f_string_files:
        if os.path.exists(file_path):
    fixes = fix_fstring_errors(file_path)
            if fixes > 0:
    fixed_files += 1,
    total_fixes += fixes,
    logger.info(
    "🎯 SUMMARY: Fixed {} f-string issues in {} files".format(
    total_fixes, fixed_files
    )
    )


if __name__ == "__main__":
    main()
