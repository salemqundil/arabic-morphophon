#!/usr/bin/env python3
"""
Project-wide Status Checker,
    Check syntax across entire project structure
"""

import os
    import ast
    from pathlib import Path,
    def check_project_status():
    """Check syntax of all Python files in project"""
    base_path = Path(".").parent  # Go up one level from tools to engines root,
    categories = {'core': 0, 'experimental': 0, 'tests': 0, 'tools': 0, 'root': 0}

    success_counts = {'core': 0, 'experimental': 0, 'tests': 0, 'tools': 0, 'root': 0}

    error_files = []

    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.py'):
    filepath = os.path.join(root, f)
    rel_path = os.path.relpath(filepath, base_path)

                # Categorize file,
    if 'core' in rel_path:
    category = 'core'
                elif 'experimental' in rel_path:
    category = 'experimental'
                elif 'tests' in rel_path:
    category = 'tests'
                elif 'tools' in rel_path:
    category = 'tools'
                else:
    category = 'root'

    categories[category] += 1,
    try:
                    with open(filepath, 'r', encoding='utf-8') as file:
    ast.parse(file.read())
    success_counts[category] += 1,
    except Exception as e:
    error_files.append((rel_path, str(e)[:50]))

    # Calculate overall stats,
    total_files = sum(categories.values())
    total_success = sum(success_counts.values())
    overall_rate = (total_success / total_files * 100) if total_files > 0 else 0,
    print("🚀 PROJECT-WIDE STATUS REPORT")
    print("=" * 50)
    print(
    f"📊 OVERALL: {total_success}/{total_files} files ({overall_rate:.1f}% success rate)"
    )
    print()

    print("📁 BY CATEGORY:")
    for cat in categories:
    success = success_counts[cat]
    total = categories[cat]
    rate = (success / total * 100) if total > 0 else 0,
    print(f"  {cat.ljust(12)}: {success:2d}/{total:2d} files ({rate:5.1f}%)")

    print()
    print(f"❌ ERRORS: {len(error_files)} files with issues")

    if error_files:
    print("\nFirst 10 error files:")
        for filepath, error in error_files[:10]:
    print(f"  ❌ {filepath}: {error}...")


if __name__ == "__main__":
    check_project_status()
