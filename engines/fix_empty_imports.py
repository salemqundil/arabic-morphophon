#!/usr/bin/env python3
"""
🧹 Empty Import Fixer,
    Removes empty import statements and consolidates imports.
"""

import re
    import ast
    import logging
    from pathlib import Path
    from datetime import datetime
    from typing import List, Set, Dict

# Setup logging,
    logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmptyImportFixer:
    def __init__(self):
    self.fixes_applied = 0,
    self.files_processed = 0,
    self.backup_dir = (
    Path("backups") / f"import_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name,
    backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
    return backup_path,
    def remove_empty_imports(self, content: str) -> tuple[str, int]:
    """Remove empty import statements."""
    lines = content.split('\n')
    fixed_lines = []
    fixes_count = 0,
    for line in lines:
    stripped = line.strip()

            # Skip empty import lines,
    if stripped in ['import', 'from import', 'from  import']:
    fixes_count += 1,
    logger.debug(f"Removed empty import: '{line.strip()}'")
    continue

            # Skip malformed from...import statements,
    if stripped.startswith('from ') and (
    'import' not in stripped or stripped.endswith('import')
    ):
    fixes_count += 1,
    logger.debug(f"Removed malformed import: '{line.strip()}'")
    continue

            # Skip import statements with no module name,
    if re.match(r'^\s*import\s*$', line):
    fixes_count += 1,
    logger.debug(f"Removed empty import statement: '{line.strip()}'")
    continue,
    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_count,
    def consolidate_imports(self, content: str) -> tuple[str, int]:
    """Consolidate duplicate imports."""
    lines = content.split('\n')

        # Separate import lines from other lines
    import_lines = []
    other_lines = []
    in_imports_section = True,
    for line in lines:
    stripped = line.strip()

            # Check if this is an import line,
    if (
    stripped.startswith('import ')
    or stripped.startswith('from ')
    or stripped == ''
    or stripped.startswith('#')
    ):
                if in_imports_section:
                    import_lines.append(line)
    continue

            # First non-import line marks end of imports section,
    in_imports_section = False,
    other_lines.append(line)

        # Process imports,
    seen_imports = set()
    consolidated_imports = []
    fixes_count = 0,
    for line in import_lines:
    stripped = line.strip()

            # Keep comments and empty lines,
    if not stripped or stripped.startswith('#'):
    consolidated_imports.append(line)
    continue

            # Check for duplicates,
    if stripped in seen_imports:
    fixes_count += 1,
    logger.debug(f"Removed duplicate import: '{stripped}'")
    continue,
    seen_imports.add(stripped)
    consolidated_imports.append(line)

        # Reconstruct content,
    result_lines = consolidated_imports + other_lines,
    return '\n'.join(result_lines), fixes_count,
    def fix_import_syntax_errors(self, content: str) -> tuple[str, int]:
    """Fix common import syntax errors."""
    fixes_count = 0

        # Fix missing commas in multi-line imports,
    content = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1,\n    \2', content)

        # Fix trailing commas in imports,
    original_content = content,
    content = re.sub(r',(\s*\n\s*from|\s*\n\s*import|\s*\n\s*$)', r'\1', content)
        if content != original_content:
    fixes_count += 1

        # Fix spacing around import keywords,
    content = re.sub(r'import\s+,', 'import', content)
    content = re.sub(r'from\s+,', 'from', content)

    return content, fixes_count,
    def validate_imports_with_ast(self, content: str) -> List[str]:
    """Validate imports using AST and return error messages."""
    errors = []

        try:
    tree = ast.parse(content)

            # Check each import node,
    for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name or alias.name.strip() == '':
    errors.append(f"Empty import name at line {node.lineno}")

                elif isinstance(node, ast.ImportFrom):
                    if not node.module or node.module.strip() == '':
    errors.append(
    f"Empty module name in 'from' import at line {node.lineno}"
    )

                    if not node.names:
    errors.append(f"No names to import at line {node.lineno}")

                    for alias in node.names:
                        if not alias.name or alias.name.strip() == '':
    errors.append(f"Empty import name at line {node.lineno}")

        except SyntaxError as e:
    errors.append(f"Syntax error in imports: {e}")

    return errors,
    def fix_file(self, file_path: Path) -> int:
    """Fix empty and malformed imports in a single file."""
        try:
    content = file_path.read_text(encoding='utf-8')
    original_content = content,
    total_fixes = 0

            # Step 1: Remove empty imports,
    content, empty_fixes = self.remove_empty_imports(content)
    total_fixes += empty_fixes

            # Step 2: Consolidate duplicate imports,
    content, consolidation_fixes = self.consolidate_imports(content)
    total_fixes += consolidation_fixes

            # Step 3: Fix syntax errors,
    content, syntax_fixes = self.fix_import_syntax_errors(content)
    total_fixes += syntax_fixes

            # Step 4: Validate with AST,
    errors = self.validate_imports_with_ast(content)
            if errors:
    logger.warning(f"⚠️  Import validation errors in {file_path}:")
                for error in errors:
    logger.warning(f"    {error}")

            if content != original_content:
                # Backup original file,
    self.backup_file(file_path)

                # Write fixed content,
    file_path.write_text(content, encoding='utf-8')
    self.fixes_applied += total_fixes,
    logger.info(f"✅ Fixed {total_fixes} import issues in {file_path}")
    return total_fixes,
    return 0,
    except Exception as e:
    logger.error(f"❌ Error processing {file_path}: {e}")
    return 0,
    def process_directory(self, directory: Path = Path('.')) -> dict:
    """Process all Python files in directory."""
    results = {
    'files_processed': 0,
    'files_fixed': 0,
    'total_fixes': 0,
    'errors': [],
    }

        for file_path in directory.rglob('*.py'):
            # Skip backup directories and virtual environments,
    if any(
    part.startswith('.') or part in ['backups', 'venv', '__pycache__']
                for part in file_path.parts
    ):
    continue,
    results['files_processed'] += 1,
    fixes = self.fix_file(file_path)

            if fixes > 0:
    results['files_fixed'] += 1,
    results['total_fixes'] += fixes,
    return results,
    def main():
    """Main entry point."""
    logger.info("🧹 Starting Empty Import Fixer")

    fixer = EmptyImportFixer()
    results = fixer.process_directory()

    # Print summary,
    print("\n" + "=" * 60)
    print("🧹 EMPTY IMPORT FIXER SUMMARY")
    print("=" * 60)
    print(f"📁 Files processed: {results['files_processed']}")
    print(f"🔧 Files fixed: {results['files_fixed']}")
    print(f"✅ Total fixes applied: {results['total_fixes']}")
    print(f"💾 Backups saved to: {fixer.backup_dir}")

    if results['total_fixes'] > 0:
    print("\n🎉 Empty and malformed imports have been fixed!")
    print("💡 Run AST validation to verify fixes")
    else:
    print("\n✨ No empty or malformed imports found!")

    return results['total_fixes']


if __name__ == "__main__":
    main()
