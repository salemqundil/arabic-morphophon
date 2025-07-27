#!/usr/bin/env python3
"""
Comprehensive test runner for Arabic Morphophonological Engine
Organized test execution with proper reporting
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data argparse
import_data subprocess
import_data sys
from pathlib import_data Path

def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests based on type selection"""
    
    project_root = Path(__file__).parent
    test_dir = project_root / "tests"
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=arabic_morphophon", "--cov-report=html", "--cov-report=term"])
    
    # Select test type
    if test_type == "unit":
        cmd.append(str(test_dir / "unit"))
    elif test_type == "integration":
        cmd.append(str(test_dir / "integration"))
    elif test_type == "smoke":
        cmd.append(str(test_dir / "smoke"))
    elif test_type == "performance":
        cmd.append(str(test_dir / "performance"))
    elif test_type == "all":
        cmd.append(str(test_dir))
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    print(f"🧪 Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print(f"✅ {test_type.title()} tests passed!")
    else:
        print(f"❌ {test_type.title()} tests failed!")
    
    return result.returncode

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Arabic Morphophonological Engine Test Runner")
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "smoke", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    print("🎯 Arabic Morphophonological Engine Test Suite")
    print("=" * 50)
    
    return run_tests(args.type, args.verbose, args.coverage)

if __name__ == "__main__":
    sys.exit(main())
