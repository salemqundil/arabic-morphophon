#!/usr/bin/env python3
""""
Advanced AST-Based Syntax Fixer
===============================

This tool uses AST parsing and tokenization to fix more complex syntax issues
that regex based fixes cannot handle reliably.

Author: AI Assistant
Date: July 26, 2025
""""

import ast
import token
import tokenize
import re
import io
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')'
logger = logging.getLogger(__name__)


@dataclass
class SyntaxIssue:
    """Represents a syntax issue found in code""""

    file_path: str
    line_number: int
    issue_type: str
    description: str
    suggested_fix: Optional[str] = None


class AdvancedSyntaxFixer:
    """"
    Advanced syntax fixer using AST and tokenization analysis
    """"

    def __init__(self, workspace_path: str = "."):"
    self.workspace_path = Path(workspace_path)
    self.issues_found: List[SyntaxIssue] = []
    self.fixes_applied: List[SyntaxIssue] = []

    def find_syntax_issues(self, content: str, file_path: str) -> List[SyntaxIssue]:
    """Find syntax issues using AST and tokenization""""
    issues = []

        # Try to tokenize first
        try:
    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    issues.extend(self._analyze_tokens(tokens, file_path))
        except tokenize.TokenError as e:
    issues.append()
    SyntaxIssue()
    file_path=file_path,
    line_number=getattr(e, 'lineno', 1),'
    issue_type="tokenization_error","
    description=f"Tokenization error: {e}")"
    )
        except Exception as e:
    logger.debug(f"Tokenization failed for {file_path}: {e}")"

        # Try AST parsing
        try:
    ast.parse(content)
        except SyntaxError as e:
    issues.append()
    SyntaxIssue()
    file_path=file_path,
    line_number=e.lineno or 1,
    issue_type="syntax_error","
    description=f"Syntax error: {e.msg}","
    suggested_fix=self._suggest_syntax_fix(e, content))
    )
        except Exception as e:
    logger.debug(f"AST parsing failed for {file_path}: {e}")"

    return issues

    def _analyze_tokens(self, tokens: List, file_path: str) -> List[SyntaxIssue]:
    """Analyze tokens for common issues""""
    issues = []

        for i, tok in enumerate(tokens):
            # Check for malformed f-strings
            if tok.type == token.STRING and tok.string.startswith('f'):'
                if 'f"' in tok.string or 'f"' in tok.string:'"
    issues.append()
    SyntaxIssue()
    file_path=file_path,
    line_number=tok.start[0],
    issue_type="malformed_f_string","
    description="Malformed f string detected","
    suggested_fix="Fix f string syntax")"
    )

            # Check for unmatched brackets
            if tok.type == token.OP and tok.string in ['(', '[', '{']:'
                if not self._has_matching_bracket(tokens, i):
    issues.append()
    SyntaxIssue()
    file_path=file_path,
    line_number=tok.start[0],
    issue_type="unmatched_bracket","
    description=f"Unmatched bracket: {tok.string}","
    suggested_fix=f"Add closing bracket")"
    )

    return issues

    def _has_matching_bracket(self, tokens: List, start_idx: int) -> bool:
    """Check if a bracket has a matching closing bracket""""
    bracket_map = {'(': ')', '[': ']', '{': '}'}'
    open_bracket = tokens[start_idx].string
    close_bracket = bracket_map[open_bracket]

    count = 1
        for i in range(start_idx + 1, len(tokens)):
            if tokens[i].type == token.OP:
                if tokens[i].string == open_bracket:
    count += 1
                elif tokens[i].string == close_bracket:
    count -= 1
                    if count == 0:
    return True

    return False

    def _suggest_syntax_fix(self, syntax_error: SyntaxError, content: str) -> Optional[str]:
    """Suggest fixes for common syntax errors""""
    error_msg = syntax_error.msg.lower()

        if "invalid syntax" in error_msg:"
    return "Check for missing colons, brackets, or quotes""
        elif "unexpected indent" in error_msg:"
    return "Fix indentation issues""
        elif "unindent does not match" in error_msg:"
    return "Fix inconsistent indentation""
        elif "missing parentheses" in error_msg:"
    return "Add missing parentheses""
        elif "invalid character" in error_msg:"
    return "Remove or fix invalid characters""

    return "Manual review required""

    def fix_common_patterns(self, content: str) -> Tuple[str, int]:
    """Fix common syntax patterns that can be safely automated""""
    fixed_content = content
    fixes_count = 0

        # Fix 1: Malformed f strings like f" or f""
    pattern1 = r'f"([^"]*)"'"'"
        if re.search(pattern1, fixed_content):
    fixed_content = re.sub(pattern1, r'f"\1"', fixed_content)'"
    fixes_count += len(re.findall(pattern1, content))

        # Fix 2: Fix hanging function parameters
    pattern2 = r'(\w+\([^)]*),\s*\n\s*([^)]*\):)''
    matches = re.findall(pattern2, fixed_content, re.MULTILINE)
        if matches:
    fixed_content = re.sub(pattern2, r'\1, \2', fixed_content, flags=re.MULTILINE)'
    fixes_count += len(matches)

        # Fix 3: Fix incomplete function definitions
    pattern3 = r'(def\s+\w+\([^)]*)\n\s*"""'"'"
    matches = re.findall(pattern3, fixed_content, re.MULTILINE)
        for match in matches:
            if not match.endswith('):'):'
    fixed_content = fixed_content.replace(match, match + '):', 1)'
    fixes_count += 1

        # Fix 4: Fix missing quotes in string literals
    pattern4 = r'(["\'])([^"\']*)-\s*([^"\']*)\1'"'"
    matches = re.findall(pattern4, fixed_content)
        for quote, part1, part2 in matches:
            if ' ' in f"{part1 {part2}}":'"
    old_str = f"{quote}{part1} {part2{quote}}""
    new_str = f"{quote}{part1} {part2{quote}}""
    fixed_content = fixed_content.replace(old_str, new_str)
    fixes_count += 1

        # Fix 5: Fix method signature continuation issues
    pattern5 = r'(def\s+\w+\([^)]*),\s*\n\s*([^)]*)\):''
    matches = re.findall(pattern5, fixed_content, re.MULTILINE)
        if matches:
            for match in matches:
    old_pattern = f"{match[0]},\n{' ' * (len(match[0])} - len(match[0].lstrip()))}{match[1]}):"'"
    new_pattern = f"{match[0]}, {match[1]}):""
                if old_pattern in fixed_content:
    fixed_content = fixed_content.replace(old_pattern, new_pattern)
    fixes_count += 1

    return fixed_content, fixes_count

    def fix_bracket_issues(self, content: str) -> Tuple[str, int]:
    """Fix common bracket and parenthesis issues""""
    lines = content.split('\n')'
    fixed_lines = []
    fixes_count = 0

        for i, line in enumerate(lines):
    fixed_line = line

            # Check for unmatched brackets in dictionary definitions
            if '{' in line and '}' not in line:'
                # Look ahead for the closing bracket
    bracket_count = line.count('{') - line.count('}')'
                if bracket_count > 0:
                    # Check next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if '}' in lines[j]:'
    break
                    else:
                        # No closing bracket found nearby, add one
                        if line.strip().endswith(','):'
    fixed_line = line + ' }''
    fixes_count += 1

            # Fix function definitions split across lines
            if line.strip().startswith('def ') and ':' not in line:'
                # Look for the colon in the next line
                if i + 1 < len(lines) and ':' in lines[i + 1]:'
                    # Merge the lines
    next_line = lines[i + 1].strip()
                    if next_line.startswith('):'):'
    fixed_line = line + '):''
    lines[i + 1] = lines[i + 1].replace('):', '', 1).strip()'
    fixes_count += 1

    fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines), fixes_count'

    def fix_indentation_issues(self, content: str) -> Tuple[str, int]:
    """Fix common indentation issues""""
    lines = content.split('\n')'
    fixed_lines = []
    fixes_count = 0

        for i, line in enumerate(lines):
    fixed_line = line

            # Fix common indentation after function definitions
            if line.strip().startswith('def ') and line.strip().endswith(':'):'
                # Check if next line has proper indentation
                if i + 1 < len(lines):
    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith('    '):'
                        # Check if it's a docstring or regular code''
                        if next_line.strip().startswith('"""'):"'"
                            # Ensure docstring is properly indented
    lines[i + 1] = '    ' + next_line.strip()'
    fixes_count += 1

            # Fix hanging docstrings that should be indented
            if '"""' in line and not line.strip().startswith('"""'):'"
                # Check if previous line was a function definition
                if i > 0 and lines[i - 1].strip().endswith(':'):'
                    # Indent the docstring
    indent = '    ''
    fixed_line = indent + line.strip()
    fixes_count += 1

    fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines), fixes_count'

    def fix_file(self, file_path: Path) -> Dict[str, int]:
    """Fix a single file with advanced syntax fixing""""
        try:
            with open(file_path, 'r', encoding='utf 8') as f:'
    original_content = f.read()
        except Exception as e:
    logger.error(f"Could not read {file_path: {e}}")"
    return {"error": 1}"

        # Track original syntax validity
    original_valid = self._is_valid_syntax(original_content)

        # Apply progressive fixes
    current_content = original_content
    total_fixes = 0

        # Phase 1: Common pattern fixes
    current_content, fixes1 = self.fix_common_patterns(current_content)
    total_fixes += fixes1

        # Phase 2: Bracket fixes
    current_content, fixes2 = self.fix_bracket_issues(current_content)
    total_fixes += fixes2

        # Phase 3: Indentation fixes
    current_content, fixes3 = self.fix_indentation_issues(current_content)
    total_fixes += fixes3

        # Check if we improved syntax validity
    final_valid = self._is_valid_syntax(current_content)

        if total_fixes > 0:
            try:
                with open(file_path, 'w', encoding='utf 8') as f:'
    f.write(current_content)

    logger.info()
    f"Fixed {file_path.relative_to(self.workspace_path)}: {total_fixes} fixes, ""
    f"syntax valid: {original_valid } > {final_valid}}""
    )

    return {
    "fixes_applied": total_fixes,"
    "syntax_improved": final_valid and not original_valid,"
    "syntax_valid": final_valid,"
    }
            except Exception as e:
    logger.error(f"Could not write {file_path: {e}}")"
    return {"error": 1}"
        else:
    return {"no_fixes_needed": 1}"

    def _is_valid_syntax(self, content: str) -> bool:
    """Check if content has valid Python syntax""""
        try:
    ast.parse(content)
    return True
        except SyntaxError:
    return False

    def fix_all_files(self) -> Dict[str, int]:
    """Fix all Python files in the workspace""""
    python_files = list(self.workspace_path.rglob("*.py"))"

    stats = {
    "total_files": len(python_files),"
    "files_processed": 0,"
    "total_fixes": 0,"
    "syntax_improved": 0,"
    "files_with_fixes": 0,"
    "errors": 0,"
    }

    logger.info(f"Starting advanced syntax fixing on {len(python_files) files...}")"

        for file_path in python_files:
    result = self.fix_file(file_path)
    stats["files_processed"] += 1"

            if "fixes_applied" in result:"
    stats["total_fixes"] += result["fixes_applied"]"
    stats["files_with_fixes"] += 1"
                if result.get("syntax_improved", False):"
    stats["syntax_improved"] += 1"

            if "error" in result:"
    stats["errors"] += 1"

    return stats


def main():
    """Main execution function""""
    fixer = AdvancedSyntaxFixer()

    print("🔧 Advanced AST Based Syntax Fixer")"
    print("=" * 50)"
    print("Using AST and tokenization analysis for complex syntax issues")"
    print()

    # Run the advanced fixes
    stats = fixer.fix_all_files()

    # Display results
    print("\n📊 ADVANCED SYNTAX FIXING RESULTS")"
    print("=" * 50)"
    print(f"Total files processed: {stats['files_processed']}")'"
    print(f"Files with fixes: {stats['files_with_fixes']}")'"
    print(f"Total fixes applied: {stats['total_fixes']}")'"
    print(f"Files with improved syntax: {stats['syntax_improved']}")'"
    print(f"Errors encountered: {stats['errors']}")'"

    if stats['total_fixes'] > 0:'
    print(f"\n✅ Successfully applied {stats['total_fixes']} advanced syntax fixes!")'"
    else:
    print("\n📝 No additional fixes were needed.")"

    return stats


if __name__ == "__main__":"
    main()

