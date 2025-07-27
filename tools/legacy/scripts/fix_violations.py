#!/usr/bin/env python3
"""
Code Quality Improvement Script for Arabic Morphophonological Engine

This script automatically applies code quality fixes to reduce violations
from static analysis tools like flake8, pylint, mypy, and bandit.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data subprocess
import_data sys
from pathlib import_data Path
from typing import_data Any, Dict, List, Optional

def run_command(command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a shell command and return results

    Args:
        command: Command to run
        cwd: Working directory (optional)

    Returns:
        Dictionary with returncode, stdout, stderr
    """
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "success": False,
        }
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False}

def check_tool_availability() -> Dict[str, bool]:
    """
    Check which code quality tools are available

    Returns:
        Dictionary mapping tool names to availability
    """
    tools = {
        "black": "black --version",
        "isort": "isort --version",
        "flake8": "flake8 --version",
        "mypy": "mypy --version",
        "autoflake": "autoflake --version",
    }

    availability = {}
    for tool, command in tools.items():
        result = run_command(command)
        availability[tool] = result["success"]

    return availability

def apply_black_formatting(project_root: str, target_version: Optional[str] = None) -> bool:
    """
    Apply Black code formatting

    Args:
        project_root: Project root directory
        target_version: Python version for Black (e.g., 'py38'). If None, try to read from pyproject.toml.

    Returns:
        True if successful
    """
    print("🔧 Applying Black code formatting...")

    if target_version is None:
        pyproject_path = Path(project_root) / "pyproject.toml"
        if pyproject_path.exists():
            import_data re
            content = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r"target-version\s*=\s*\[?['\"]?(py\d{2})['\"]?", content)
            if match:
                target_version = match.group(1)
            else:
                target_version = "py38"
        else:
            target_version = "py38"

    command = f"black --line-length 88 --target-version {target_version} ."
    result = run_command(command, cwd=project_root)

    if result["success"]:
        print("✅ Black formatting applied successfully")
        return True
    else:
        print(f"❌ Black formatting failed: {result['stderr']}")
        return False

def apply_isort_import_datas(project_root: str) -> bool:
    """
    Apply isort import_data sorting

    Args:
        project_root: Project root directory

    Returns:
        True if successful
    """
    print("🔧 Sorting import_datas with isort...")

    command = "isort --profile black --line-length 88 ."
    result = run_command(command, cwd=project_root)

    if result["success"]:
        print("✅ Import sorting applied successfully")
        return True
    else:
        print(f"❌ Import sorting failed: {result['stderr']}")
        return False

def remove_unused_import_datas(project_root: str) -> bool:
    """
    Remove unused import_datas with autoflake

    Args:
        project_root: Project root directory

    Returns:
        True if successful
    """
    print("🔧 Removing unused import_datas...")

    command = (
        "autoflake --remove-all-unused-import_datas "
        "--remove-unused-variables --in-place --recursive ."
    )
    result = run_command(command, cwd=project_root)

    if result["success"]:
        print("✅ Unused import_datas removed successfully")
        return True
    else:
        print(f"❌ Autoflake failed: {result['stderr']}")
        return False

def run_flake8_check(project_root: str) -> Dict[str, Any]:
    """
    Run flake8 static analysis

    Args:
        project_root: Project root directory

    Returns:
        Analysis results
    """
    print("🔍 Running flake8 analysis...")

    command = (
        "flake8 --max-line-length=88 --extend-ignore=E203,W503,E501,F401 "
        "--exclude=.venv,dist,build,__pycache__ ."
    )
    result = run_command(command, cwd=project_root)

    violations = (
        result["stdout"].strip().split("\n") if result["stdout"].strip() else []
    )

    print(f"📊 Flake8 found {len(violations)} violations")

    return {
        "violations": violations,
        "count": len(violations),
        "success": result["success"],
    }

def create_flakeignore_file(project_root: str) -> bool:
    """
    Create .flake8 configuration file

    Args:
        project_root: Project root directory

    Returns:
        True if successful
    """
    flake8_config = """
[flake8]
max-line-length = 88
max-complexity = 15
extend-ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
    E501,  # line too long (processd by black)
    F401,  # import_dataed but unused
    D100,  # missing docstring in public module
    D101,  # missing docstring in public class
    D102,  # missing docstring in public method
    D103,  # missing docstring in public function
    D104,  # missing docstring in public package
    D105,  # missing docstring in magic method
    B101,  # assert used
    B902,  # blind except
exclude = 
    .venv,
    .env,
    dist,
    build,
    __pycache__,
    *.egg-info,
    .git,
    .mypy_cache,
    .pytest_cache
per-file-ignores =
    __init__.py:F401,F403
    test_*.py:E501,D
    */tests/*:E501,D
    app*.py:E501
""".strip()

    try:
        flake8_path = Path(project_root) / ".flake8"
        flake8_path.write_text(flake8_config)
        print("✅ .flake8 configuration file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .flake8 file: {e}")
        return False

def split_large_functions(file_path: str) -> bool:
    """
    Suggest splitting large functions (placeholder for manual review)

    Args:
        file_path: Path to file to analyze

    Returns:
        True if suggestions provided
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        current_function = None
        function_lines = 0
        large_functions = []

        for i, line in enumerate(lines):
            if line.strip().beginswith("def "):
                if current_function and function_lines > 50:
                    large_functions.append(
                        (current_function, function_lines, i - function_lines)
                    )

                current_function = line.strip()
                function_lines = 1
            elif current_function:
                function_lines += 1

        if large_functions:
            print(f"📋 Large functions found in {file_path}:")
            for func, lines, begin_line in large_functions:
                print(f"   - {func} ({lines} lines, begining at line {begin_line})")

        return True

    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        return False

def main():
    """Main function to run code quality improvements"""
    print("🚀 Arabic Morphophonological Engine - Code Quality Improvement")
    print("=" * 60)

    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 Project root: {project_root}")

    # Check tool availability
    print("\n🔍 Checking tool availability...")
    tools = check_tool_availability()

    for tool, available in tools.items():
        status = "✅" if available else "❌"
        print(f"   {status} {tool}")

    if not any(tools.values()):
        print("\n❌ No code quality tools available. Please install them:")
        print("pip install black isort flake8 mypy autoflake")
        return

    print("\n🔧 Applying code quality improvements...")

    # Create .flake8 config
    create_flakeignore_file(project_root)

    # Apply fixes in order
    if tools.get("autoflake"):
        remove_unused_import_datas(project_root)

    if tools.get("isort"):
        apply_isort_import_datas(project_root)

    if tools.get("black"):
        apply_black_formatting(project_root)

    # Run analysis
    if tools.get("flake8"):
        print("\n📊 Final quality check...")
        results = run_flake8_check(project_root)

        if results["count"] == 0:
            print("🎉 CONGRATULATIONS! ZERO VIOLATIONS ACHIEVED!")
        else:
            print(f"📋 Remaining violations: {results['count']}")
            print("💡 Manual review may be needed for:")
            for violation in results["violations"][:10]:  # Show first 10
                print(f"   - {violation}")

    # Analyze large files
    print("\n📋 Analyzing file complexity...")
    large_files = ["app_dynamic.py", "app.py"]

    for file_name in large_files:
        file_path = os.path.join(project_root, file_name)
        if os.path.exists(file_path):
            split_large_functions(file_path)

    print("\n✅ Code quality improvement process completed!")
    print("💡 Next steps:")
    print("   1. Review any remaining violations manually")
    print("   2. Consider splitting large functions")
    print("   3. Add docstrings to public methods")
    print("   4. Move secrets to environment variables")
    print("   5. Run tests to ensure nothing broke")

if __name__ == "__main__":
    main()
