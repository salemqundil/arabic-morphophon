import os
from pathlib import Path
import ast
import pandas as pd

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
                    "file": str(path.name),
                    "full_path": str(path),
                    "lineno": e.lineno,
                    "msg": e.msg,
                    "text": (e.text or "").strip(),
                    "type": categorize_error(e.msg, e.text),
                }
            )
        except Exception as e:
            results.append(
                {
                    "file": str(path.name),
                    "full_path": str(path),
                    "lineno": "?",
                    "msg": str(e),
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

# Create DataFrame
df = pd.DataFrame(issues)

# Display summary statistics
print(f"\n📊 CONTROLLED REPAIR TARGETS ANALYSIS")
print("=" * 50)

if not df.empty:
    print(f"Total files with issues: {len(df)}")
    print(f"\nIssue type distribution:")
    type_counts = df['type'].value_counts()
    for issue_type, count in type_counts.items():
        print(f"  {issue_type}: {count} files ({count/len(df)*100:.1f}%)")

    print(f"\nTop 10 most problematic files:")
    print(df[['file', 'type', 'lineno', 'msg']].head(10).to_string(index=False))

    # Priority repair strategy
    print(f"\n🎯 PRIORITY REPAIR STRATEGY:")
    print(
        f"1. Fix 'unterminated_string' issues first (easiest): {type_counts.get('unterminated_string', 0)} files"
    )
    print(
        f"2. Fix 'invalid_unicode' issues (medium): {type_counts.get('invalid_unicode', 0)} files"
    )
    print(
        f"3. Fix 'unexpected_indent' issues (hardest): {type_counts.get('unexpected_indent', 0)} files"
    )

    # Save detailed report
    output_file = "syntax_issues_detailed_report.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed report saved to: {output_file}")

    # Show sample fixes needed
    print(f"\n🔧 SAMPLE REPAIR PREVIEW:")
    for issue_type in ['unterminated_string', 'invalid_unicode', 'unexpected_indent']:
        subset = df[df['type'] == issue_type]
        if not subset.empty:
            print(f"\n{issue_type.upper()} examples:")
            for _, row in subset.head(3).iterrows():
                print(f"  📄 {row['file']} (Line {row['lineno']}): {row['text'][:50]}")
else:
    print("✅ No syntax issues found!")

print(f"\n📈 RECOMMENDATION:")
if not df.empty:
    easiest_fixes = type_counts.get('unterminated_string', 0)
    total_fixes = len(df)
    print(f"Start with {easiest_fixes} unterminated string fixes for quick wins.")
    print(
        f"This will improve success rate from current to {(120 + easiest_fixes)/357*100:.1f}%"
    )
else:
    print("All files have valid syntax! 🎉")
