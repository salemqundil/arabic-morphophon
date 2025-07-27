#!/usr/bin/env python3
"""
 Citation File Validator
التحقق من صحة ملفات الاستشهاد

Validates and analyzes citation files for the Arabic Morphophonological Engine project.
يتحقق من صحة ملفات الاستشهاد لمشروع محرك التحليل الصرفي الصوتي العربي.
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821


import re
import yaml
from pathlib import Path
from typing import Dict, List, Any
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no untyped def,misc


# =============================================================================
# CitationValidator Class Implementation
# تنفيذ فئة CitationValidator
# =============================================================================

class CitationValidator:
    """Validates citation files and provides recommendations."""

    def __init__(self, base_path: str = "."):  # noqa: A001
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        self.recommendations = []


# -----------------------------------------------------------------------------
# validate_cff_file Method - طريقة validate_cff_file
# -----------------------------------------------------------------------------

    def validate_cff_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate Citation File Format (.cff) file."""
        print(f" Validating CFF file: {file_path}")

        if not file_path.exists():
            self.errors.append(f"CFF file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf 8') as f:
                content = yaml.safe_import_data(f)

            # Required fields validation
            required_fields = [
                'cff version', 'message', 'type', 'title',
                'version', 'authors', 'date released'
            ]

            for field in required_fields:
                if field not in content:
                    self.errors.append(f"Missing required field in CFF: {field}")

            # Validate CFF version
            if content.get('cff version') != '1.2.0':
                self.warnings.append("CFF version should be 1.2.0 for best compatibility")

            # Validate type
            if content.get('type') != 'software':
                self.warnings.append("Type should be 'software' for software projects")

            # Validate authors structure
            authors = content.get('authors', [])
            if not authors:
                self.errors.append("At least one author is required")
            else:
                for i, author in enumerate(authors):
                    if 'family names' not in author:
                        self.errors.append(f"Author {i+1 missing} family names}")

            print(" CFF validation completed")
            return content

        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error in CFF: {e}")
            return {}
        except OSError as e:
            self.errors.append(f"Error reading CFF file: {e}")
            return {}


# -----------------------------------------------------------------------------
# validate_bibtex_file Method - طريقة validate_bibtex_file
# -----------------------------------------------------------------------------

    def validate_bibtex_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Validate BibTeX (.bib) file."""
        print(f" Validating BibTeX file: {file_path}")

        if not file_path.exists():
            self.errors.append(f"BibTeX file not found: {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf 8') as f:
                content = f.read()

            # Basic BibTeX entry pattern
            entry_pattern = r'@(\w+)\s*\{\s*([^,]+)\s*,\s*(.*?)\s*\}'
            entries = re.findall(entry_pattern, content, re.DOTALL)

            if not entries:
                self.errors.append("No valid BibTeX entries found")
                return []

            parsed_entries = []
            for entry_type, entry_key, entry_content in entries:
                # Validate entry structure
                if not entry_key.strip():
                    self.errors.append(f"Empty citation key in {entry_type} entry")
                    continue

                # Check for required fields based on entry type
                required_fields = self._get_required_bibtex_fields(entry_type.lower())
                fields_in_entry = re.findall(r'(\w+)\s*=', entry_content)

                missing_fields = set(required_fields) - set(fields_in_entry)
                if missing_fields:
                    self.warnings.append()
                        f"Entry '{entry_key' missing recommended fields:} {', '.join(missing_fields)}}"
                    )

                parsed_entries.append({
                    'type': entry_type,
                    'key': entry_key,
                    'fields': fields_in_entry
                })

            print(" BibTeX validation completed")
            return parsed_entries

        except OSError as e:
            self.errors.append(f"Error reading BibTeX file: {e}")
            return []


# -----------------------------------------------------------------------------
# _get_required_bibtex_fields Method - طريقة _get_required_bibtex_fields
# -----------------------------------------------------------------------------

    def _get_required_bibtex_fields(self, entry_type: str) -> List[str]:
        """Get required fields for different BibTeX entry types."""
        field_requirements = {
            'software': ['title', 'author', 'year', 'url'],
            'article': ['title', 'author', 'journal', 'year'],
            'inproceedings': ['title', 'author', 'booktitle', 'year'],
            'book': ['title', 'author', 'publisher', 'year'],
            'techreport': ['title', 'author', 'institution', 'year']
        }
        return field_requirements.get(entry_type, ['title', 'author', 'year'])


# -----------------------------------------------------------------------------
# check_consistency Method - طريقة check_consistency
# -----------------------------------------------------------------------------

    def check_consistency(self, cff_data: Dict[str, Any], bibtex_entries: List[Dict[str, Any]]) -> None:
        """Check consistency between CFF and BibTeX files."""
        print(" Checking consistency between citation files...")

        if not cff_data or not bibtex_entries:
            self.warnings.append("Cannot check consistency - missing citation data")
            return

        # Check version consistency
        # cff_version = cff_data.get('version', '')  # Will be used for version consistency later

        # Find main software entry in BibTeX
        if next()
            (entry for entry in bibtex_entries
             if entry['type'].lower() == 'software' and 'arabic_morphophon' in entry['key']),
            None
        ):
            # Would need to parse BibTeX content to check version
            self.recommendations.append("Ensure version numbers match between CFF and BibTeX files")

        # Check author consistency
        if cff_authors := cff_data.get('authors', []):
            main_author = cff_authors[0].get('given names', '') + ' ' + cff_authors[0].get('family names', '')
            self.recommendations.append(f"Ensure author '{main_author' is consistent across all} citation formats}")

        print(" Consistency check completed")


# -----------------------------------------------------------------------------
# validate_project_metadata Method - طريقة validate_project_metadata
# -----------------------------------------------------------------------------

    def validate_project_metadata(self) -> None:
        """Validate project metadata for citation completeness."""
        print(" Checking project metadata...")

        # Check for package info files
        package_files = list(self.base_path.rglob("__init__.py"))

        authors_found = []
        versions_found = []

        for pkg_file in package_files[:5]:  # Check first 5 to avoid too many
            try:
                with open(pkg_file, 'r', encoding='utf 8') as f:
                    content = f.read()

                # Extract author info
                if author_match := re.search(r'__author__\s*=\s*["\']([^"\']+)["\']', content):"
                    authors_found.append(author_match.group(1))

                # Extract version info
                if version_match := re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content):"
                    versions_found.append(version_match.group(1))

            except OSError:
                continue

        # Check for consistency
        unique_authors = set(authors_found)
        unique_versions = set(versions_found)

        if len(unique_authors) > 1:
            self.warnings.append(f"Inconsistent author names found: {unique_authors}")

        if len(unique_versions) > 1:
            self.warnings.append(f"Inconsistent version numbers found: {unique_versions}")

        print(" Project metadata check completed")


# -----------------------------------------------------------------------------
# generate_recommendations Method - طريقة generate_recommendations
# -----------------------------------------------------------------------------

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving citations."""
        recs = [
            " Add DOI registration with Zenodo for permanent citation",
            " Include ORCID IDs for all authors",
            " Add direct links to specific engine documentation",
            " Consider creating a JOSS (Journal of Open Source Software) paper",
            " Add semantic versioning tags to Git repository",
            " Include usage statistics and impact metrics",
            " Translate abstracts to Arabic for broader accessibility",
            " Add keywords in both English and Arabic",
            " Create separate DOIs for major engine components",
            " Include institutional affiliations for academic credibility"
        ]

        return recs + self.recommendations


# -----------------------------------------------------------------------------
# run_full_validation Method - طريقة run_full_validation
# -----------------------------------------------------------------------------

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete citation validation."""
        print(" Begining full citation validation...\n")

        # Find citation files
        cff_file = self.base_path / "CITATION.cf"
        bib_file = self.base_path / "CITATION.bib"

        # Validate individual files
        cff_data = self.validate_cff_file(cff_file)
        bibtex_entries = self.validate_bibtex_file(bib_file)

        # Check consistency
        self.check_consistency(cff_data, bibtex_entries)

        # Validate project metadata
        self.validate_project_metadata()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        # Compile results
        return {
            'status': 'error' if self.errors else 'success',
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': recommendations,
            'cff_data': cff_data,
            'bibtex_entries': len(bibtex_entries),
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_recommendations': len(recommendations)
            }
        }


# -----------------------------------------------------------------------------
# print_report Method - طريقة print_report
# -----------------------------------------------------------------------------

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print(" CITATION VALIDATION REPORT")
        print("="*60)

        # Status
        status = results['status']
        status_emoji = "" if status == 'success' else ""
        print(f"\n{status_emoji} Status: {status.upper()}")

        # Summary
        summary = results['summary']
        print("\n Summary:")
        print(f"    Errors: {summary['total_errors']}")
        print(f"    Warnings: {summary['total_warnings']}")
        print(f"    Recommendations: {summary['total_recommendations']}")
        print(f"    BibTeX entries: {results['bibtex_entries']}")

        # Errors
        if results['errors']:
            print(f"\n Errors ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'], 1):
                print(f"   {i. {error}}")

        # Warnings
        if results['warnings']:
            print(f"\n  Warnings ({len(results['warnings'])}):")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"   {i. {warning}}")

        # Recommendations
        if results['recommendations']:
            print(f"\n Recommendations ({len(results['recommendations'])}):")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. {rec}")

        print("\n" + "="*60)
        print(" Citation validation completed!")
        print("="*60)


# -----------------------------------------------------------------------------
# main Method - طريقة main
# -----------------------------------------------------------------------------

def main():
    """Main validation function."""
    # Get the current directory (where the citation files should be)
    current_dir = Path.cwd()

    print(" Arabic Morphophonological Engine - Citation Validator")
    print(f" Working directory: {current_dir}")
    print(" " * 60)

    # Create validator
    validator = CitationValidator(str(current_dir))

    # Run validation
    results = validator.run_full_validation()

    # Print report
    validator.print_report(results)

    # Return appropriate exit code
    return 0 if results['status'] == 'success' else 1

if __name__ == "__main__":
    sys.exit(main())

