#!/usr/bin/env python3
"""
📊 INDENTATION FIX RESULTS SUMMARY
=================================

Creates a comprehensive summary of the massive indentation fix operation,
displaying results in a structured format similar to pandas DataFrame.
"""

import ast
from pathlib import Path
from datetime import datetime


def validate_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "Valid"
    except SyntaxError as e:
        return False, str(e).split('(')[0].strip()
    except Exception as e:
        return False, f"Read Error: {str(e)[:50]}"


def create_summary_report():
    """Create comprehensive summary report"""
    print("📊 INDENTATION FIX OPERATION SUMMARY")
    print("=" * 50)

    base_dir = Path.cwd()

    # Load the fix log to get fix counts
    log_files = list(base_dir.glob("indentation_fix_log_*.txt"))
    fix_data = {}

    if log_files:
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"📋 Loading data from: {latest_log.name}")

        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse fix information
        for line in content.split('\n'):
            if line.startswith('✅ Fixed:'):
                parts = line.split('(')
                if len(parts) > 1:
                    file_part = parts[0].replace('✅ Fixed: ', '').strip()
                    fixes_part = parts[1].replace(' fixes)', '').replace(' fix)', '')
                    try:
                        fix_count = int(fixes_part)
                        file_name = file_part.split('\\')[-1]  # Get just filename
                        fix_data[file_name] = fix_count
                    except ValueError:
                        pass

    # Current validation status
    python_files = [f for f in base_dir.rglob("*.py") if "backup" not in str(f)]

    results = []
    total_fixes = 0
    valid_count = 0

    for py_file in python_files:
        file_name = py_file.name
        is_valid, error_type = validate_syntax(py_file)
        fixes_applied = fix_data.get(file_name, 0)

        results.append(
            {
                'file': file_name,
                'fixes': fixes_applied,
                'status': 'Valid' if is_valid else 'Error',
                'error': error_type if not is_valid else '',
            }
        )

        total_fixes += fixes_applied
        if is_valid:
            valid_count += 1

    # Sort by number of fixes (descending)
    results.sort(key=lambda x: x['fixes'], reverse=True)

    # Summary statistics
    total_files = len(results)
    success_rate = (valid_count / total_files * 100) if total_files > 0 else 0

    print(f"\n📈 OPERATION STATISTICS")
    print(f"   Total files processed: {total_files}")
    print(f"   Total indentation fixes: {total_fixes:,}")
    print(f"   Files with fixes: {len([r for r in results if r['fixes'] > 0])}")
    print(f"   Current valid files: {valid_count}")
    print(f"   Success rate: {success_rate:.1f}%")

    # Top files by fix count
    print(f"\n🔥 TOP 20 FILES BY INDENTATION FIXES:")
    print(f"{'Rank':<4} | {'File':<45} | {'Fixes':<6} | {'Status':<8} | {'Error Type'}")
    print("-" * 90)

    for i, result in enumerate(results[:20], 1):
        status_emoji = "✅" if result['status'] == 'Valid' else "❌"
        error_display = (
            result['error'][:30] + "..."
            if len(result['error']) > 30
            else result['error']
        )

        print(
            f"{i:3}. | {result['file']:<45} | {result['fixes']:6,} | {status_emoji} {result['status']:<6} | {error_display}"
        )

    # Error summary
    error_types = {}
    for result in results:
        if result['status'] != 'Valid':
            error_key = (
                result['error'].split(':')[0]
                if ':' in result['error']
                else result['error']
            )
            error_types[error_key] = error_types.get(error_key, 0) + 1

    if error_types:
        print(f"\n⚠️ ERROR TYPE BREAKDOWN:")
        for error_type, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   {error_type}: {count} files")

    # Files that still need attention
    problematic_files = [
        r for r in results if r['status'] != 'Valid' and r['fixes'] > 0
    ]
    if problematic_files:
        print(f"\n🎯 FIXED FILES STILL WITH ERRORS ({len(problematic_files)}):")
        for result in problematic_files[:10]:
            print(
                f"   📁 {result['file']} ({result['fixes']} fixes) - {result['error'][:50]}"
            )

    # Success stories
    successful_fixes = [r for r in results if r['status'] == 'Valid' and r['fixes'] > 0]
    print(f"\n🎉 SUCCESSFUL FIXES ({len(successful_fixes)} files):")
    print(
        f"   Total fixes in valid files: {sum(r['fixes'] for r in successful_fixes):,}"
    )

    if successful_fixes:
        avg_fixes = sum(r['fixes'] for r in successful_fixes) / len(successful_fixes)
        print(f"   Average fixes per successful file: {avg_fixes:.1f}")

    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = base_dir / f"indentation_fix_summary_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"INDENTATION FIX OPERATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"STATISTICS:\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Total fixes: {total_fixes:,}\n")
        f.write(f"Valid files: {valid_count}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n\n")
        f.write(f"DETAILED RESULTS:\n")
        for i, result in enumerate(results, 1):
            f.write(
                f"{i:3}. {result['file']:<50} | Fixes: {result['fixes']:6,} | Status: {result['status']:<8} | {result['error']}\n"
            )

    print(f"\n📋 Detailed report saved: {report_file.name}")

    return {
        'total_files': total_files,
        'total_fixes': total_fixes,
        'valid_files': valid_count,
        'success_rate': success_rate,
        'successful_fixes': len(successful_fixes),
    }


def main():
    """Main function"""
    summary = create_summary_report()

    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Operation: MASSIVE INDENTATION FIX")
    print(
        f"   Impact: {summary['total_fixes']:,} fixes across {summary['total_files']} files"
    )
    print(
        f"   Success: {summary['valid_files']} valid files ({summary['success_rate']:.1f}%)"
    )
    print(f"   Next: Address remaining syntax issues for full recovery")


if __name__ == "__main__":
    main()
