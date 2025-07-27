import os
from pathlib import Path
import ast

# Define types of issues we will try to fix
ISSUE_TYPES = {
    "unexpected_indent": "unexpected indent",
    "unterminated_string": "unterminated string literal",
    "invalid_fstring": "f-string",
    "invalid_unicode": "invalid character",
}

# Define a list of invalid characters that commonly cause syntax issues
BAD_UNICODE = ['🎯', '🔥', '🔧', '📊', '🧪', '🔬', '🏥', '🏗', '📋', '→', '،']


def scan_python_files_for_issues(base_path="."):
    results = []
    for path in Path(base_path).rglob("*.py"):
        # Skip backup directories
        if any(
            part.startswith('.') or part in ['backups', 'venv', '__pycache__']
            for part in path.parts
        ):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            results.append(
                {
                    "file": str(path.name),  # Just filename for readability
                    "full_path": str(path),
                    "lineno": e.lineno,
                    "msg": e.msg,
                    "text": (e.text or "").strip()[:50],  # Truncate for display
                    "type": categorize_error(e.msg, e.text),
                }
            )
        except Exception as e:
            results.append(
                {
                    "file": str(path.name),
                    "full_path": str(path),
                    "lineno": "?",
                    "msg": str(e)[:50],
                    "text": "",
                    "type": "unknown",
                }
            )
    return results


def categorize_error(msg, text):
    if not msg:
        return "unknown"
    msg_lower = msg.lower()

    for key, keyword in ISSUE_TYPES.items():
        if keyword in msg_lower:
            return key

    if text:
        for ch in BAD_UNICODE:
            if ch in text:
                return "invalid_unicode"

    return "unknown"


# Run the analysis
print("🔍 Scanning Python files for syntax issues...")
issues = scan_python_files_for_issues(".")

if issues:
    print(f"\n📊 Found {len(issues)} files with syntax issues:")

    # Group by error type
    from collections import Counter

    type_counts = Counter(issue['type'] for issue in issues)

    print("\n📋 Issue breakdown:")
    for issue_type, count in type_counts.most_common():
        print(f"  {issue_type}: {count} files")

    print(f"\n🔍 Sample issues (first 10):")
    for i, issue in enumerate(issues[:10]):
        print(f"{i+1:2d}. {issue['file']}: Line {issue['lineno']} - {issue['type']}")
        print(f"    {issue['msg']}")
        if issue['text']:
            print(f"    Code: {issue['text']}")
        print()

    if len(issues) > 10:
        print(f"... and {len(issues) - 10} more files")

else:
    print("✅ No syntax issues found!")

# Create a summary
print(f"\n📈 Summary:")
print(f"Total files with issues: {len(issues)}")
print(
    f"Most common issue type: {type_counts.most_common(1)[0] if type_counts else 'None'}"
)
