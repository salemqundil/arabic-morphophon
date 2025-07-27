#!/usr/bin/env python3
"""
🚀 Syntax Fix Orchestrator
Master controller that runs all syntax fixers in the correct order.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntaxFixOrchestrator:
    def __init__(self):
        self.start_time = datetime.now()
        self.total_fixes = 0
        self.fixers_run = 0

        # Define fixer modules in execution order
        self.fixers = [
            ('fix_empty_imports.py', 'Empty Import Fixer', '🧹'),
            ('fix_common_fstring_errors.py', 'F-String Error Fixer', '📝'),
            ('fix_logging_config.py', 'Logging Config Fixer', '📋'),
            ('fix_broken_comparisons.py', 'Broken Comparison Fixer', '🔧'),
        ]

        self.results = {}

    def import_fixer_module(self, fixer_path: Path):
        """Dynamically import a fixer module."""
        try:
            spec = importlib.util.spec_from_file_location("fixer", fixer_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"❌ Failed to import {fixer_path}: {e}")
            return None

    def run_fixer(self, fixer_file: str, fixer_name: str, emoji: str) -> dict:
        """Run a single fixer and return results."""
        logger.info(f"\n{emoji} Starting {fixer_name}...")

        fixer_path = Path(fixer_file)
        if not fixer_path.exists():
            logger.error(f"❌ Fixer not found: {fixer_path}")
            return {'status': 'error', 'message': 'Fixer file not found', 'fixes': 0}

        try:
            # Import and run the fixer
            module = self.import_fixer_module(fixer_path)
            if not module:
                return {
                    'status': 'error',
                    'message': 'Failed to import fixer',
                    'fixes': 0,
                }

            # Run the main function
            if hasattr(module, 'main'):
                fixes_applied = module.main()
                self.total_fixes += fixes_applied
                self.fixers_run += 1

                return {
                    'status': 'success',
                    'fixes': fixes_applied,
                    'message': f'Applied {fixes_applied} fixes',
                }
            else:
                logger.error(f"❌ No main() function found in {fixer_file}")
                return {'status': 'error', 'message': 'No main() function', 'fixes': 0}

        except Exception as e:
            logger.error(f"❌ Error running {fixer_name}: {e}")
            return {'status': 'error', 'message': str(e), 'fixes': 0}

    def run_ast_validation(self) -> dict:
        """Run AST validation after all fixes."""
        logger.info("\n🔍 Running AST validation...")

        try:
            # Look for AST validator
            validator_path = Path('ast_validator.py')
            if not validator_path.exists():
                logger.warning("⚠️  AST validator not found, skipping validation")
                return {'status': 'skipped', 'message': 'Validator not found'}

            # Run AST validator
            module = self.import_fixer_module(validator_path)
            if module and hasattr(module, 'main'):
                result = module.main()
                return {'status': 'success', 'result': result}
            else:
                return {'status': 'error', 'message': 'Could not run validator'}

        except Exception as e:
            logger.error(f"❌ Error running AST validation: {e}")
            return {'status': 'error', 'message': str(e)}

    def run_black_check(self) -> dict:
        """Run black formatter check after fixes."""
        logger.info("\n⚫ Running Black formatter check...")

        try:
            # Run black --check to see what would be formatted
            result = subprocess.run(
                ['python', '-m', 'black', '--check', '--diff', '.'],
                capture_output=True,
                text=True,
                cwd=Path('.'),
            )

            if result.returncode == 0:
                return {
                    'status': 'clean',
                    'message': 'All files are properly formatted',
                }
            else:
                # Count files that need formatting
                diff_output = result.stdout
                files_needing_format = len(
                    [
                        line
                        for line in diff_output.split('\n')
                        if line.startswith('--- ') or line.startswith('+++ ')
                    ]
                )

                return {
                    'status': 'needs_formatting',
                    'files_count': files_needing_format
                    // 2,  # Each file has --- and +++ lines
                    'message': f'{files_needing_format // 2} files need formatting',
                }

        except FileNotFoundError:
            return {'status': 'unavailable', 'message': 'Black not installed'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        duration = datetime.now() - self.start_time

        report = [
            "\n" + "=" * 80,
            "🚀 SYNTAX FIX ORCHESTRATOR - FINAL REPORT",
            "=" * 80,
            f"⏱️  Total execution time: {duration.total_seconds():.2f} seconds",
            f"🔧 Fixers executed: {self.fixers_run}/{len(self.fixers)}",
            f"✅ Total fixes applied: {self.total_fixes}",
            "",
            "📊 DETAILED RESULTS:",
            "-" * 40,
        ]

        for fixer_file, fixer_name, emoji in self.fixers:
            if fixer_file in self.results:
                result = self.results[fixer_file]
                status_emoji = "✅" if result['status'] == 'success' else "❌"
                report.append(
                    f"{emoji} {fixer_name}: {status_emoji} {result['message']}"
                )
            else:
                report.append(f"{emoji} {fixer_name}: ⏭️  Skipped")

        # Add validation results
        if 'ast_validation' in self.results:
            ast_result = self.results['ast_validation']
            if ast_result['status'] == 'success':
                report.append(f"🔍 AST Validation: ✅ Completed")
            else:
                report.append(f"🔍 AST Validation: ⚠️  {ast_result['message']}")

        # Add formatting results
        if 'black_check' in self.results:
            black_result = self.results['black_check']
            if black_result['status'] == 'clean':
                report.append(f"⚫ Black Check: ✅ All files properly formatted")
            elif black_result['status'] == 'needs_formatting':
                report.append(f"⚫ Black Check: 📝 {black_result['message']}")
            else:
                report.append(f"⚫ Black Check: ⚠️  {black_result['message']}")

        report.extend(
            [
                "",
                "🎯 NEXT STEPS:",
                "1. Review the backup directories created by each fixer",
                "2. Run 'python -m black .' to format remaining files",
                "3. Run 'python -m pytest' to validate test suite",
                "4. Run 'python -m ruff check' for additional linting",
                "",
                "🎉 Syntax fixing complete!",
            ]
        )

        return "\n".join(report)

    def run_all_fixers(self):
        """Run all syntax fixers in the correct order."""
        logger.info("🚀 Starting Syntax Fix Orchestrator")
        logger.info(f"📁 Working directory: {Path('.').absolute()}")
        logger.info(f"🔧 {len(self.fixers)} fixers scheduled for execution")

        # Run each fixer
        for fixer_file, fixer_name, emoji in self.fixers:
            result = self.run_fixer(fixer_file, fixer_name, emoji)
            self.results[fixer_file] = result

            if result['status'] == 'success':
                logger.info(f"✅ {fixer_name} completed successfully")
            else:
                logger.warning(
                    f"⚠️  {fixer_name} encountered issues: {result['message']}"
                )

        # Run post-fix validation
        self.results['ast_validation'] = self.run_ast_validation()
        self.results['black_check'] = self.run_black_check()

        # Generate and display summary
        summary = self.generate_summary_report()
        print(summary)

        # Return summary stats
        return {
            'total_fixes': self.total_fixes,
            'fixers_run': self.fixers_run,
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'results': self.results,
        }


def main():
    """Main entry point."""
    orchestrator = SyntaxFixOrchestrator()
    return orchestrator.run_all_fixers()


if __name__ == "__main__":
    main()
