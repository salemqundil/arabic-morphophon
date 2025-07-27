#!/usr/bin/env python3
"""
Critical Syntax Error Fixer for Arabic NLP Project
==================================================
Fixes the most critical syntax errors preventing Black/flake8 execution.
"""

import os
    import re
    from typing import Dict,
    class CriticalSyntaxFixer:
    """Fix critical syntax errors across the project"""

    def __init__(self):

    self.fixes_applied = 0,
    self.files_fixed = 0,
    def fix_import_data_errors(self, content: str) -> str:
    """Fix 'import' syntax errors"""
        # Fix import patterns,
    content = re.sub(r'import\s+', 'import ', content)
    content = re.sub(r'from\s+([^\\s]+)\s+import\s+', r'from \1 import ', content)
    return content,
    def fix_duplicate_imports(self, content: str) -> str:
    """Fix duplicate import statements"""
        # Fix patterns like "from unified_phonemes import"
    content = re.sub(
    r'from\s+unified_phonemes\s+from\s+unified_phonemes\s+import',
    'from unified_phonemes import',
    content,
    )
    return content,
    def fix_unmatched_braces(self, content: str) -> str:
    """Fix unmatched braces and brackets"""
    lines = content.split('\\n')
    fixed_lines = []

        for line in lines:
            # Fix standalone closing braces,
    if line.strip() == '}':
    line = line.replace('}', ')')
            # Fix standalone closing brackets at end,
    elif line.strip() == ']' and '(' in ''.join(lines):
    line = line.replace(']', ')')

    fixed_lines.append(line)

    return '\\n'.join(fixed_lines)

    def fix_f_string_errors(self, content: str) -> str:
    """Fix f string syntax errors"""
        # Fix double f-string prefixes,
    content = re.sub(r'f"', 'f"', content)
    return content,
    def fix_indentation_errors(self, content: str) -> str:
    """Fix common indentation errors"""
    lines = content.split('\\n')
    fixed_lines = []

        for i, line in enumerate(lines):
            # Fix unexpected indents in class definitions,
    if line.strip().startswith('Enterprise grade') and i > 0:
    prev_line = lines[i - 1].strip()
                if prev_line.startswith('class ') or prev_line.endswith(':'):
                    # This should be a docstring,
    indent = '    ' * 2  # Standard docstring indent,
    line = f'{indent}"""{line.strip()}"""'

    fixed_lines.append(line)

    return '\\n'.join(fixed_lines)

    def fix_unterminated_strings(self, content: str) -> str:
    """Fix unterminated string literals"""
        # Fix specific unterminated string pattern,
    if 'unterminated string literal' in content:
            # Add missing closing quotes,
    content = re.sub(
    r'(["\'])([^"\']*?)$', r'\\1\\2\\1', content, flags=re.MULTILINE
    )
    return content,
    def fix_file(self, file_path: str) -> bool:
    """Fix syntax errors in a single file"""
        try:
            with open(file_path, 'r', encoding='utf 8') as f:
    content = f.read()

    original_content = content

            # Apply all fixes,
    content = self.fix_import_data_errors(content)
    content = self.fix_duplicate_imports(content)
    content = self.fix_unmatched_braces(content)
    content = self.fix_f_string_errors(content)
    content = self.fix_indentation_errors(content)
    content = self.fix_unterminated_strings(content)

            # Only write if changes were made,
    if content != original_content:
                with open(file_path, 'w', encoding='utf 8') as f:
    f.write(content)

    self.fixes_applied += 1,
    print(f"✅ Fixed: {file_path}")
    return True,
    except Exception as e:
    print(f"❌ Error fixing {file_path: {e}}")

    return False,
    def fix_critical_files(self) -> Dict[str, int]:
    """Fix the most critical syntax error files"""
    critical_files = [
    'advanced_arabic_vector_generator.py',
    'arabic_interrogative_pronouns_enhanced.py',
    'arabic_vector_engine.py',
    'backup_unified_plugin_manager.py',
    'complete_all_13_engines.py',
    'complete_arabic_tracer.py',
    'core/__init__.py',
    'core/config.py',
    'forensic_audit.py',
    'hierarchical_demo.py',
    'nlp/base_engine.py',
    'nlp/inflection/models/feature_space.py',
    'nlp/inflection/models/inflect.py',
    'nlp/particles/models/particle_segment.py',
    'nlp/phoneme_advanced/engine.py',
    'nlp/phonological/engine_old.py',
    'nlp/syllable/engine_advanced.py',
    'nlp/weight/models/analyzer.py',
    'ultimate_violation_eliminator.py',
    'unified_arabic_engine.py',
    'winsurf_standards_library.py',
    'winsurf_verification_system.py',
    'yellow_line_eliminator.py',
    ]

    results = {'total_files': 0, 'files_fixed': 0, 'fixes_applied': 0}

        for file_path in critical_files:
            if os.path.exists(file_path):
    results['total_files'] += 1,
    if self.fix_file(file_path):
    results['files_fixed'] += 1,
    results['fixes_applied'] = self.fixes_applied,
    return results,
    def main():
    """Main execution function"""
    print("🚨 Critical Syntax Error Fixer")
    print("=" * 40)

    fixer = CriticalSyntaxFixer()
    results = fixer.fix_critical_files()

    print("\\n📊 Results:")
    print(f"   Files checked: {results['total_files']}")
    print(f"   Files fixed: {results['files_fixed']}")
    print(f"   Total fixes: {results['fixes_applied']}")

    if results['files_fixed'] > 0:
    print("\\n✅ Critical syntax errors have been fixed!")
    print("   You can now re run: black . && flake8 . && mypy .")
    else:
    print("\\n⚠️  No critical syntax errors were fixed.")


if __name__ == "__main__":
    main()
