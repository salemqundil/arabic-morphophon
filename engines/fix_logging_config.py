#!/usr/bin/env python3
"""
� Logging Config Fixer,
    Fixes orphaned logging configuration statements and indentation issues.
"""

import re
    import logging
    from pathlib import Path
    from datetime import datetime

# Setup logging,
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggingConfigFixer:
    def __init__(self):
    self.fixes_applied = 0,
    self.files_processed = 0,
    self.backup_dir = Path("backups") / f"logging_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Patterns for common logging configuration issues,
    self.patterns = [
            # Orphaned logging config parameters
    (r'^\s*level=logging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL),?\s*$', ''),
    (r'^\s*format=[\'"][^\'"]*[\'"],?\s*$', ''),
    (r'^\s*filename=[\'"][^\'"]*[\'"],?\s*$', ''),
    (r'^\s*filemode=[\'"][^\'"]*[\'"],?\s*$', ''),

            # Incomplete logging.basicConfig calls
    (r'logging\.basicConfig\(\s*$', 'logging.basicConfig(level=logging.INFO)'),

            # Fix malformed logging statements
    (r'logging\.basicConfig\(\s*level=logging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*format=',
    r'logging.basicConfig(\n    level=logging.\1,\n    format='),
    ]

    def backup_file(self, file_path: Path) -> Path:
    """Create a backup of the file before modification."""
    backup_path = self.backup_dir / file_path.name,
    backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
    return backup_path,
    def fix_orphaned_config_lines(self, content: str) -> tuple[str, int]:
    """Remove orphaned logging configuration lines."""
    lines = content.split('\n')
    fixed_lines = []
    fixes_count = 0,
    for i, line in enumerate(lines):
    stripped = line.strip()

            # Check if this is an orphaned logging parameter,
    is_orphaned = False

            # Look for orphaned level=, format=, etc. lines,
    if (re.match(r'^\s*level=logging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL),?\s*$', line) or,
    re.match(r'^\s*format=[\'"][^\'"]*[\'"],?\s*$', line) or,
    re.match(r'^\s*filename=[\'"][^\'"]*[\'"],?\s*$', line)):

                # Check if previous line ends with logging.basicConfig(
                if i > 0 and not re.search(r'logging\.basicConfig\s*\($', lines[i-1]):
    is_orphaned = True

                # Check if this is part of a multi-line logging.basicConfig,
    found_config_start = False,
    for j in range(max(0, i-5), i):
                    if 'logging.basicConfig(' in lines[j]:
    found_config_start = True,
    break

                if not found_config_start:
    is_orphaned = True,
    if is_orphaned:
    fixes_count += 1,
    logger.debug(f"Removed orphaned logging config: '{line.strip()}'")
    continue,
    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_count,
    def fix_logging_indentation(self, content: str) -> tuple[str, int]:
    """Fix indentation issues in logging configuration."""
    lines = content.split('\n')
    fixed_lines = []
    fixes_count = 0,
    for i, line in enumerate(lines):
            # Check for logging.basicConfig on previous line,
    if i > 0 and 'logging.basicConfig(' in lines[i - 1]:
                # Fix indented level/format parameters,
    pattern = r'^(\s+)(level=logging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL),?\s*(?:format=.*?)?)\s*$'
    match = re.match(pattern, line)
                if match:
    param = match.group(2)
                    # Properly indent continuation lines,
    fixed_line = f"    {param}"
    fixed_lines.append(fixed_line)
    fixes_count += 1,
    logger.debug(f"Fixed logging indentation: '{line.strip()}' -> '{fixed_line.strip()}'")
    continue,
    fixed_lines.append(line)

    return '\n'.join(fixed_lines), fixes_count,
    def fix_malformed_basicconfig(self, content: str) -> tuple[str, int]:
    """Fix malformed logging.basicConfig calls."""
    fixes_count = 0

        # Fix incomplete basicConfig calls,
    pattern = r'logging\.basicConfig\(\s*$'
        if re.search(pattern, content, re.MULTILINE):
    content = re.sub(pattern, 'logging.basicConfig(level=logging.INFO)', content, flags=re.MULTILINE)
    fixes_count += 1

        # Fix missing commas between parameters,
    pattern = r'(level=logging\.(DEBUG|INFO|WARNING|ERROR|CRITICAL))\s*(format=)'
    replacement = r'\1,\n    \3'
        if re.search(pattern, content):
    content = re.sub(pattern, replacement, content)
    fixes_count += 1,
    return content, fixes_count,
    def fix_file(self, file_path: Path) -> int:
    """Fix logging configuration issues in a single file."""
        try:
    content = file_path.read_text(encoding='utf-8')
    original_content = content,
    total_fixes = 0

            # Apply all fixing patterns,
    for pattern, replacement in self.patterns:
    old_content = content,
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                if content != old_content:
    matches = len(re.findall(pattern, old_content, flags=re.MULTILINE))
    total_fixes += matches,
    logger.debug(f"Applied logging pattern '{pattern}' {matches} times in {file_path}")

            # Apply specialized fixes,
    content, orphaned_fixes = self.fix_orphaned_config_lines(content)
    total_fixes += orphaned_fixes,
    content, indent_fixes = self.fix_logging_indentation(content)
    total_fixes += indent_fixes,
    content, malformed_fixes = self.fix_malformed_basicconfig(content)
    total_fixes += malformed_fixes,
    if content != original_content:
                # Backup original file,
    self.backup_file(file_path)

                # Write fixed content,
    file_path.write_text(content, encoding='utf-8')
    self.fixes_applied += total_fixes,
    logger.info(f"✅ Fixed {total_fixes} logging config issues in {file_path}")
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
    'errors': []
    }

        for file_path in directory.rglob('*.py'):
            # Skip backup directories and virtual environments,
    if any(part.startswith('.') or part in ['backups', 'venv', '__pycache__']
                   for part in file_path.parts):
    continue,
    results['files_processed'] += 1,
    fixes = self.fix_file(file_path)

            if fixes > 0:
    results['files_fixed'] += 1,
    results['total_fixes'] += fixes,
    return results,
    def main():
    """Main entry point."""
    logger.info("📋 Starting Logging Config Fixer")

    fixer = LoggingConfigFixer()
    results = fixer.process_directory()

    # Print summary,
    print("\n" + "="*60)
    print("📋 LOGGING CONFIG FIXER SUMMARY")
    print("="*60)
    print(f"📁 Files processed: {results['files_processed']}")
    print(f"🔧 Files fixed: {results['files_fixed']}")
    print(f"✅ Total fixes applied: {results['total_fixes']}")
    print(f"💾 Backups saved to: {fixer.backup_dir}")

    if results['total_fixes'] > 0:
    print("\n🎉 Logging configuration issues have been fixed!")
    print("💡 Run AST validation to verify fixes")
    else:
    print("\n✨ No logging configuration issues found!")

    return results['total_fixes']

if __name__ == "__main__":
    main()
