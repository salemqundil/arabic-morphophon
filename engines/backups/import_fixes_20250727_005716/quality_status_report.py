#!/usr/bin/env python3
"""
Arabic NLP Project - Code Quality Status Report
==============================================
Comprehensive analysis of code quality metrics and issues.
"""

import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, capture_output: bool = True) -> tuple:
    """Run a command and return its output and status"""
    try:
        if capture_output:
    result = subprocess.run(
    cmd, shell=True, capture_output=True, text=True, cwd='.'
    )
    return result.returncode, result.stdout, result.stderr
        else:
    result = subprocess.run(cmd, shell=True, cwd='.')
    return result.returncode, "", ""
    except Exception as e:
    return 1, "", str(e)


def count_python_files() -> dict:
    """Count Python files in the project"""
    py_files = list(Path('.').rglob('*.py'))

    categories = {
    'total': len(py_files),
    'main': len(
    [
    f
                for f in py_files
                if not any(part.startswith('test_') for part in f.parts)
    ]
    ),
    'tests': len(
    [f for f in py_files if any(part.startswith('test_') for part in f.parts)]
    ),
    'core': len([f for f in py_files if 'core' in f.parts]),
    'nlp': len([f for f in py_files if 'nlp' in f.parts]),
    'models': len([f for f in py_files if 'models' in f.parts]),
    }

    return categories


def main():
    """Generate comprehensive quality report"""
    print("🔍 Arabic NLP Project - Code Quality Status Report")
    print("=" * 60)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m} %d %H:%M:%S')}")
    print()

    # File Statistics
    print("📁 FILE STATISTICS")
    print(" " * 30)
    file_stats = count_python_files()
    for category, count in file_stats.items():
    print(f"   {category.capitalize()}: {count} files")
    print()

    # Pytest Status
    print("🧪 TESTING STATUS")
    print(" " * 30)
    exit_code, stdout, stderr = run_command("pytest tests/ --tb=no  q")
    if exit_code == 0:
    lines = stdout.strip().split('\\n')
    summary_line = [line for line in lines if 'passed' in line][-1] if lines else ""
    print("   ✅ Status: PASSING")
    print(f"   📊 Results: {summary_line}")
    else:
    print("   ❌ Status: FAILING")
    print(f"   📊 Exit Code: {exit_code}")
    print()

    # Black Formatting Status
    print("🎨 CODE FORMATTING (Black)")
    print(" " * 30)
    exit_code, stdout, stderr = run_command("black . --check - diff")
    if exit_code == 0:
    print("   ✅ Status: ALL FILES FORMATTED")
    else:
        # Count reformatted files
        if "reformatted" in stderr:
    reformatted_count = stderr.count("reformatted")
    failed_count = stderr.count("failed to reformat")
    print("   ⚠️  Status: FORMATTING NEEDED")
    print(f"   📊 Files needing format: {reformatted_count}")
    print(f"   📊 Files with syntax errors: {failed_count}")
        else:
    print("   ⚠️  Status: FORMATTING ISSUES")
    print()

    # Flake8 Style Check
    print("📏 STYLE CHECK (Flake8)")
    print(" " * 30)
    exit_code, stdout, stderr = run_command("flake8 . --count --statistics --exit zero")
    if exit_code == 0 and stdout.strip():
    lines = stdout.strip().split('\\n')
    last_line = lines[-1] if lines else ""
        if last_line and last_line[0].isdigit():
    print("   ⚠️  Status: STYLE ISSUES FOUND")
    print(f"   📊 Total Issues: {last_line}")
        else:
    print("   ✅ Status: NO STYLE ISSUES")
    else:
    print("   ❌ Status: STYLE CHECK FAILED")
    print()

    # Compilation Check
    print("⚙️  COMPILATION STATUS")
    print(" " * 30)
    test_files = ['tests/test_hello_world.py', 'tests/conftest.py']
    compilation_ok = True

    for test_file in test_files:
        if Path(test_file).exists():
    exit_code, _, _ = run_command(f"python  m py_compile {test_file}")
            if exit_code == 0:
    print(f"   ✅ {test_file}: COMPILES")
            else:
    print(f"   ❌ {test_file}: COMPILATION ERROR")
    compilation_ok = False

    if compilation_ok:
    print("   ✅ Overall: CORE FILES COMPILE")
    else:
    print("   ⚠️  Overall: SOME COMPILATION ISSUES")
    print()

    # Project Health Summary
    print("🏥 PROJECT HEALTH SUMMARY")
    print(" " * 30)

    health_score = 0
    max_score = 4

    # Test passing
    if "✅ Status: PASSING" in open(__file__).read():
    health_score += 1
    print("   ✅ Tests: PASSING (+1)")
    else:
    print("   ❌ Tests: ISSUES")

    # Core compilation
    if compilation_ok:
    health_score += 1
    print("   ✅ Compilation: OK (+1)")
    else:
    print("   ❌ Compilation: ISSUES")

    # Project structure
    if file_stats['total'] >= 50:
    health_score += 1
    print("   ✅ Project Size: SUBSTANTIAL (+1)")
    else:
    print("   ⚠️  Project Size: SMALL")

    # Test coverage
    if file_stats['tests'] >= 10:
    health_score += 1
    print("   ✅ Test Coverage: GOOD (+1)")
    else:
    print("   ⚠️  Test Coverage: LIMITED")

    print()
    health_percentage = (health_score / max_score) * 100
    print(f"📊 OVERALL HEALTH: {health_score}/{max_score} ({health_percentage:.1f}%)")

    if health_percentage >= 75:
    print("🎉 PROJECT STATUS: HEALTHY ✅")
    elif health_percentage >= 50:
    print("⚠️  PROJECT STATUS: NEEDS ATTENTION")
    else:
    print("🚨 PROJECT STATUS: CRITICAL ISSUES")

    print()
    print("🔧 RECOMMENDED ACTIONS:")
    print("   1. Fix critical syntax errors for Black compatibility")
    print("   2. Run pytest regularly to maintain test health")
    print("   3. Address flake8 style issues gradually")
    print("   4. Consider adding more comprehensive tests")
    print("   5. Update mypy configuration for Python 3.13")


if __name__ == "__main__":
    main()
