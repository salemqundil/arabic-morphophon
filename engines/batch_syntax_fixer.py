#!/usr/bin/env python3
"""
Batch Syntax Fixer for Common Issues,
    Fixes unterminated strings, unexpected indents, and unmatched parentheses
"""

import os
    import re
    import logging,
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fix_syntax_issues(file_path):
    """Fix common syntax issues in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

    original_content = content,
    fixes_applied = 0

        # Fix 1: Unexpected indent (often after logging.basicConfig)
        # Look for lines that start with level=logging.INFO and wrap them,
    pattern1 = r'^(\s*)(level=logging\.INFO,?)(.*)$'

        def fix_logging_config(match):
    indent = match.group(1)
    config_line = match.group(2)
    rest = match.group(3)
    return indent + 'logging.basicConfig(' + config_line + rest + ')'

    lines = content.split('\n')
    new_lines = []
        for line in lines:
            if re.match(r'^\s*level=logging\.INFO', line):
                # Wrap standalone logging config lines,
    indent = len(line) - len(line.lstrip())
    new_line = ' ' * indent + 'logging.basicConfig(' + line.strip() + ')'
    new_lines.append(new_line)
    fixes_applied += 1,
    else:
    new_lines.append(line)

    content = '\n'.join(new_lines)

        # Fix 2: Unterminated string literals
        # Fix cases like 'text -> 'text'
    pattern2 = r"'([^']*?)$"
    content = re.sub(pattern2, r"'\1'", content, flags=re.MULTILINE)

    pattern3 = r'"([^"]*?)$'
    content = re.sub(pattern3, r'"\1"', content, flags=re.MULTILINE)

        # Fix 3: Unmatched parentheses - basic cases
        # Count and balance simple cases,
    open_parens = content.count('(')
    close_parens = content.count(')')

        if open_parens > close_parens:
            # Add missing closing parentheses at the end,
    content += ')' * (open_parens - close_parens)
    fixes_applied += 1

        # Save if changes were made,
    if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
    logger.info(
    "✅ Fixed {} syntax issues in {}".format(fixes_applied, file_path)
    )
    return fixes_applied,
    return 0,
    except Exception as e:
    logger.error("❌ Error fixing {}: {}".format(file_path, e))
    return 0,
    def main():
    """Fix syntax issues in problematic files"""
    # Files with unterminated string issues,
    problematic_files = [
    "arabic_test.py",
    "debug_test.py",
    "fix_fstring_syntax.py",
    "core/nlp/inflection/models/feature_space.py",
    "core/nlp/particles/models/particle_classify.py",
    "experimental/advanced_ast_syntax_fixer.py",
    ]

    # Files with unexpected indent issues,
    indent_files = [
    "arabic_inflection_corrected.py",
    "arabic_inflection_rules_engine.py",
    "arabic_inflection_ultimate.py",
    "arabic_inflection_ultimate_fixed.py",
    "arabic_phonological_foundation.py",
    "arabic_vector_engine.py",
    "arabic_verb_conjugator.py",
    "complete_all_13_engines.py",
    "complete_arabic_phonological_coverage.py",
    "complete_arabic_phonological_foundation.py",
    "complete_arabic_tracer.py",
    "comprehensive_progressive_system.py",
    "phonology_core_unified.py",
    "progressive_vector_tracker.py",
    "run_all_nlp_engines.py",
    ]

    all_files = list(set(problematic_files + indent_files))

    total_fixes = 0,
    fixed_files = 0,
    for file_path in all_files:
        if os.path.exists(file_path):
    fixes = fix_syntax_issues(file_path)
            if fixes > 0:
    fixed_files += 1,
    total_fixes += fixes,
    logger.info(
    "🎯 SUMMARY: Fixed {} syntax issues in {} files".format(
    total_fixes, fixed_files
    )
    )


if __name__ == "__main__":
    main()
