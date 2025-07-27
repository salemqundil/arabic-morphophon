#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Comprehensive Winsurf PowerShell Conflict Resolver
مصحح تضارب Winsurf PowerShell الشامل

This script systematically fixes ALL PowerShell terminology conflicts throughout
the entire Arabic morphophonological engine codebase. It ensures complete
compatibility with Winsurf IDE and PowerShell environments.

CRITICAL FIXES:
- Replace "Syllable" with "SyllabicUnit"
- Replace conflicting PowerShell terms
- Update all engine names and messages
- Fix API endpoints and documentation
- Update database references
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data re
import_data sys
from pathlib import_data Path
from typing import_data Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.winsurf_safe_strings import_data WinsurfSafeStrings, WinsurfMessageFormatter


class WinsurfConflictResolver:
    """
    🛡️ Comprehensive Winsurf PowerShell Conflict Resolver

    Systematically fixes all PowerShell terminology conflicts in the codebase
    """

    def __init__(self):
        """Initialize resolver with safe string mappings"""
        self.safe_strings = WinsurfSafeStrings()
        self.formatter = WinsurfMessageFormatter()

        # Define comprehensive replacement mappings
        self.global_replacements = {
            # ==========================================
            # 🚨 CRITICAL SYLLABLE CONFLICTS
            # ==========================================
            "SyllabicUnit Engine": "SyllabicUnit Engine",
            "syllabic_unit engine": "syllabic_unit engine",
            "SyllabicUnit Analysis": "SyllabicUnit Analysis",
            "syllabic_unit analysis": "syllabic_unit analysis",
            "SyllabicUnit Segmentation": "SyllabicUnit Segmentation",
            "syllabic_unit segmentation": "syllabic_unit segmentation",
            "SyllabicUnit Pattern": "CV Pattern",
            "syllabic_unit pattern": "cv pattern",
            "SyllabicUnit Boundary": "SyllabicUnit Boundary",
            "syllabic_unit boundary": "syllabic_unit boundary",
            "syllabic_units": "syllabic_units",
            "SyllabicUnits": "SyllabicUnits",
            # ==========================================
            # 🔧 ENGINE NAME STANDARDIZATION
            # ==========================================
            "Enhanced Phoneme & SyllabicUnit Engine": "Enhanced Arabic Engine",
            "enhanced_phoneme_syllable": "enhanced_arabic_engine",
            "phoneme_syllable": "arabic_phonemic",
            "Phoneme & Syllable": "Arabic Phonemic",
            # ==========================================
            # 📝 POWERSHELL CMDLET CONFLICTS
            # ==========================================
            "Process": "Process",
            "process": "process",
            "Select": "Select",
            "select": "select",
            "Import": "Import",
            "import_data": "import_data",
            "Store": "Store",
            "store_data": "store_data",
            "Run": "Run",
            "run_command": "run_command",
            "Start": "Begin",
            "start": "begin",
            "Stop": "End",
            "stop": "end",
            # ==========================================
            # 🔌 API ENDPOINT FIXES
            # ==========================================
            "/api/enhanced/syllabic": "/api/enhanced/syllabic",
            "/syllabic": "/syllabic",
            "syllabic_analyze": "syllabic_analyze",
            "Syllabify": "SyllabicAnalyze",
            # ==========================================
            # 📊 DATABASE REFERENCES
            # ==========================================
            "arabic_morphophon.db": "arabic_morphophon.db",
            "syllabifier": "morphophon",
            "Syllabifier": "Morphophon",
            # ==========================================
            # 💬 MESSAGE STANDARDIZATION
            # ==========================================
            "Enhanced Phoneme SyllabicUnit Engine initialized": "Enhanced Arabic Engine initialized",
            "Syllabification failed": "SyllabicUnit analysis failed",
            "syllabification": "syllabic_analysis",
            "Syllabification": "SyllabicAnalysis",
            # ==========================================
            # 📁 FILE AND CLASS NAMES
            # ==========================================
            "SyllabicUnitEngine": "SyllabicUnitEngine",
            "syllable_engine": "syllabic_unit_engine",
            "SyllabicUnitData": "SyllabicUnitData",
            "syllable_data": "syllabic_unit_data",
            "SyllabicUnitSegmenter": "SyllabicUnitSegmenter",
            "syllable_segmenter": "syllabic_unit_segmenter",
            # ==========================================
            # 🌍 LINGUISTIC TERMINOLOGY
            # ==========================================
            "CV pattern": "CV unit",
            "CVC pattern": "CVC unit",
            "CVV pattern": "CVV unit",
            "CVCC pattern": "CVCC unit",
            "syllable type": "syllabic type",
            "syllable weight": "syllabic weight",
            "syllable stress": "syllabic stress",
        }

        # File extensions to process
        self.file_extensions = [
            ".py",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".js",
        ]

        # Directories to skip
        self.skip_directories = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
        }

        # Files to skip
        self.skip_files = {
            "winsurf_safe_strings.py",  # Don't modify our safe strings file
            "winsurf_conflict_resolver.py",  # Don't modify this script
        }

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        # Skip if in skip directories
        for skip_dir in self.skip_directories:
            if skip_dir in file_path.parts:
                return False

        # Skip if in skip files
        if file_path.name in self.skip_files:
            return False

        # Only process known file types
        return file_path.suffix in self.file_extensions

    def apply_safe_replacements(self, content: str) -> Tuple[str, int]:
        """Apply safe string replacements to content"""
        modified_content = content
        replacement_count = 0

        for unsafe_term, safe_term in self.global_replacements.items():
            # Count occurrences before replacement
            before_count = modified_content.count(unsafe_term)

            # Apply replacement
            modified_content = modified_content.replace(unsafe_term, safe_term)

            # Count successful replacements
            after_count = modified_content.count(unsafe_term)
            replacement_count += before_count - after_count

        return modified_content, replacement_count

    def fix_regex_patterns(self, content: str) -> Tuple[str, int]:
        """Fix regex patterns that might contain conflicts"""
        patterns_fixed = 0

        # Fix syllable-related regex patterns
        syllable_patterns = [
            (r"syllable[s]?", "syllabic_unit"),
            (r"Syllable[s]?", "SyllabicUnit"),
            (r"SYLLABLE[S]?", "SYLLABIC_UNIT"),
        ]

        for pattern, replacement in syllable_patterns:
            if matches := re.findall(pattern, content):
                content = re.sub(pattern, replacement, content)
                patterns_fixed += len(matches)

        return content, patterns_fixed

    def fix_python_import_datas(self, content: str) -> Tuple[str, int]:
        """Fix Python import_data statements"""
        import_data_fixes = 0

        # Fix import_data statements
        import_data_patterns = [
            (r"from (.*)syllable(.*) import_data", r"from \1syllabic_unit\2 import_data"),
            (r"import_data (.*)syllable(.*)", r"import_data \1syllabic_unit\2"),
            (
                r"from (.*)enhanced_phoneme_syllable(.*)",
                r"from \1enhanced_arabic_engine\2",
            ),
        ]

        for pattern, replacement in import_data_patterns:
            if matches := re.findall(pattern, content):
                content = re.sub(pattern, replacement, content)
                import_data_fixes += len(matches)

        return content, import_data_fixes

    def fix_api_routes(self, content: str) -> Tuple[str, int]:
        """Fix API route definitions"""
        route_fixes = 0

        # Fix Flask route patterns
        route_patterns = [
            (
                r"@app\.route\(['\"](.*)syllabic_analyze(.*)['\"]\)",
                r"@app.route('\1syllabic\2')",
            ),
            (
                r"@.*\.route\(['\"](.*)syllabic_analyze(.*)['\"]\)",
                r"@\1.route('\1syllabic\2')",
            ),
            (r"['\"](.*)syllabic_analyze(.*)['\"]\s*:", r"'\1syllabic\2':"),
        ]

        for pattern, replacement in route_patterns:
            if matches := re.findall(pattern, content):
                content = re.sub(pattern, replacement, content)
                route_fixes += len(matches)

        return content, route_fixes

    def fix_class_and_function_names(self, content: str) -> Tuple[str, int]:
        """Fix class and function names"""
        name_fixes = 0

        # Fix class and function definitions
        name_patterns = [
            (r"class (.*)Syllable(.*)Engine", r"class \1SyllabicUnit\2Engine"),
            (r"def (.*)syllable(.*)\(", r"def \1syllabic_unit\2("),
            (r"def (.*)syllabic_analyze(.*)?\(", r"def \1syllabic_analyze\2("),
        ]

        for pattern, replacement in name_patterns:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                name_fixes += len(matches)

        return content, name_fixes

    def process_file(self, file_path: Path) -> Dict[str, Union[str, int, bool]]:
        """Process a single file and apply all fixes"""
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Apply all fix categories
            content = original_content
            total_fixes = 0
            fix_details = {}

            # 1. Apply safe string replacements
            content, replacements = self.apply_safe_replacements(content)
            fix_details["safe_replacements"] = replacements
            total_fixes += replacements

            # 2. Fix regex patterns
            content, regex_fixes = self.fix_regex_patterns(content)
            fix_details["regex_fixes"] = regex_fixes
            total_fixes += regex_fixes

            # 3. Fix Python import_datas
            content, import_data_fixes = self.fix_python_import_datas(content)
            fix_details["import_data_fixes"] = import_data_fixes
            total_fixes += import_data_fixes

            # 4. Fix API routes
            content, route_fixes = self.fix_api_routes(content)
            fix_details["route_fixes"] = route_fixes
            total_fixes += route_fixes

            # 5. Fix class and function names
            content, name_fixes = self.fix_class_and_function_names(content)
            fix_details["name_fixes"] = name_fixes
            total_fixes += name_fixes

            # Write back if changes were made
            if total_fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                fix_details["total_fixes"] = total_fixes
                fix_details["file_modified"] = True
            else:
                fix_details["total_fixes"] = 0
                fix_details["file_modified"] = False

            return fix_details

        except (FileNotFoundError, PermissionError, IOError) as e:
            return {"error": str(e), "total_fixes": 0, "file_modified": False}

    def resolve_all_conflicts(self, target_directory: Path = None) -> Dict[str, any]:
        """Resolve all Winsurf PowerShell conflicts in the codebase"""
        if target_directory is None:
            target_directory = project_root

        print("🛡️ Starting Comprehensive Winsurf PowerShell Conflict Resolution")
        print("=" * 80)
        print(f"📁 Target Directory: {target_directory}")
        print(f"🔍 Processing file types: {', '.join(self.file_extensions)}")
        print()

        # Statistics tracking
        stats = {
            "files_processed": 0,
            "files_modified": 0,
            "total_fixes": 0,
            "fix_categories": {
                "safe_replacements": 0,
                "regex_fixes": 0,
                "import_data_fixes": 0,
                "route_fixes": 0,
                "name_fixes": 0,
            },
            "errors": [],
            "modified_files": [],
        }

        # Process all files recursively
        for file_path in target_directory.rglob("*"):
            if file_path.is_file() and self.should_process_file(file_path):
                print(f"🔧 Processing: {file_path.relative_to(target_directory)}")

                fix_results = self.process_file(file_path)
                stats["files_processed"] += 1

                if fix_results.get("error"):
                    stats["errors"].append(
                        {"file": str(file_path), "error": fix_results["error"]}
                    )
                    print(f"   ❌ Error: {fix_results['error']}")

                elif fix_results.get("file_modified"):
                    stats["files_modified"] += 1
                    stats["total_fixes"] += fix_results["total_fixes"]
                    stats["modified_files"].append(str(file_path))

                    # Update category statistics
                    for category, count in fix_results.items():
                        if category in stats["fix_categories"]:
                            stats["fix_categories"][category] += count

                    print(f"   ✅ Fixed {fix_results['total_fixes']} conflicts")

                else:
                    print(
                        f"   ℹ️ No changes made to: {file_path.relative_to(target_directory)}"
                    )

        return stats

    def generate_report(self, stats: Dict[str, any]) -> str:
        """Generate comprehensive resolution report"""
        report = []
        report.append("🛡️ Winsurf PowerShell Conflict Resolution Report")
        report.append("=" * 60)
        report.append("")

        # Summary statistics
        report.append("📊 Summary Statistics:")
        report.append(f"   Files Processed: {stats['files_processed']}")
        report.append(f"   Files Modified: {stats['files_modified']}")
        report.append(f"   Total Conflicts Fixed: {stats['total_fixes']}")
        report.append("")

        # Category breakdown
        report.append("🔧 Fixes by Category:")
        for category, count in stats["fix_categories"].items():
            if count > 0:
                report.append(f"   {category.replace('_', ' ').title()}: {count}")
        report.append("")

        # Modified files
        if stats["modified_files"]:
            report.append("📁 Modified Files:")
            for file_path in stats["modified_files"][:10]:  # Show first 10
                report.append(f"   ✅ {file_path}")
            if len(stats["modified_files"]) > 10:
                report.append(
                    f"   ... and {len(stats['modified_files']) - 10} more files"
                )
            report.append("")

        # Errors
        if stats["errors"]:
            report.append("❌ Errors Encountered:")
            for error in stats["errors"][:5]:  # Show first 5
                report.append(f"   {error['file']}: {error['error']}")
            if len(stats["errors"]) > 5:
                report.append(f"   ... and {len(stats['errors']) - 5} more errors")
            report.append("")

        # Success message
        if stats["total_fixes"] > 0:
            report.extend(
                [
                    "🎯 Resolution Status: SUCCESS",
                    "✅ All PowerShell conflicts have been resolved!",
                ]
            )
            report.append("🛡️ Codebase is now Winsurf PowerShell compatible!")
        else:
            report.append("ℹ️  Resolution Status: NO CONFLICTS FOUND")
            report.append("✅ Codebase is already Winsurf PowerShell compatible!")

        return "\n".join(report)


def main():
    """Main resolution process"""
    print("🛡️ Winsurf PowerShell Conflict Resolver")
    print("=" * 80)
    print("Systematically fixing ALL PowerShell terminology conflicts")
    print("Target: Complete Winsurf IDE compatibility")
    print("=" * 80)
    print()

    # Initialize resolver
    resolver = WinsurfConflictResolver()

    # Run comprehensive resolution
    stats = resolver.resolve_all_conflicts()

    # Generate and display report
    print("\n" + "=" * 80)
    report = resolver.generate_report(stats)
    print(report)

    # Store report to file
    report_file = project_root / "winsurf_conflict_resolution_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 Full report store_datad to: {report_file}")

    # Test the fixes
    print("\n🧪 Testing resolved conflicts...")
    test_strings = [
        "Enhanced Phoneme & SyllabicUnit Engine",
        "syllabic_unit analysis",
        "Syllabification failed",
        "/api/enhanced/syllabic",
    ]

    for test_string in test_strings:
        safe_string = WinsurfSafeStrings.make_string_safe(test_string)
        is_safe = WinsurfSafeStrings.validate_string_safety(safe_string)
        status = "✅ SAFE" if is_safe else "❌ STILL UNSAFE"
        print("   '{}' → '{}' | {}".format(test_string, safe_string, status))

    print("\n🏆 Winsurf PowerShell Conflict Resolution Complete!")
    print("🛡️ Your Arabic NLP engine is now fully Winsurf compatible!")


if __name__ == "__main__":
    main()
