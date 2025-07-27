#!/usr/bin/env python3
"""
🔥 COMPREHENSIVE FILE ANALYSIS & HEATMAP GENERATOR
================================================================
Analyzes all Python files to generate:
1. 🔥 Heatmap of most broken files
2. 📊 Severity and fixability ranking
3. 🗂️ Module categorization (core vs experimental vs dead code)
4. 📈 Detailed metrics and recommendations

Usage:
  python comprehensive_file_analysis.py
"""

import ast
import os
import re
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path


class FileAnalyzer:
    """Comprehensive file analysis and categorization system"""

    def __init__(self, root_dir: str = "."):
    self.root_dir = root_dir
    self.files_data = {}
    self.module_categories = {
    'core': [],
    'experimental': [],
    'dead_code': [],
    'test_files': [],
    'tools': [],
    'nlp_engines': [],
    }

        # Core patterns for module classification
    self.core_patterns = [
    r'core/',
    r'nlp/.*/(engine|models)/',
    r'nlp/base_engine\.py',
    r'test_.*\.py$',
    r'tests/',
    ]

    self.experimental_patterns = [
    r'advanced_.*\.py$',
    r'comprehensive_.*\.py$',
    r'enhanced_.*\.py$',
    r'ultimate_.*\.py$',
    r'deep_model\.py$',
    r'generator.*\.py$',
    ]

    self.tool_patterns = [
    r'fix_.*\.py$',
    r'.*_fixer.*\.py$',
    r'surgical_.*\.py$',
    r'emergency_.*\.py$',
    r'analyzer\.py$',
    r'eliminator\.py$',
    r'validator\.py$',
    ]

    def analyze_syntax_errors(self, filepath: str) -> Dict[str, Any]:
    """Analyze syntax errors and categorize them"""
    errors = []
    severity_score = 0

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
    source = f.read()

            # Try to parse with AST
            try:
    ast.parse(source)
    return {
    'syntax_valid': True,
    'errors': [],
    'severity_score': 0,
    'error_types': [],
    'line_count': len(source.splitlines()),
    }
            except SyntaxError as e:
    error_info = {
    'line': e.lineno,
    'message': e.msg,
    'text': e.text.strip() if e.text else '',
    'type': self.categorize_error(e.msg),
    }
    errors.append(error_info)
    severity_score = self.calculate_severity(e.msg)

        except Exception as e:
    errors.append(
    {
    'line': 0,
    'message': f"File read error: {str(e)}",
    'text': '',
    'type': 'file_error',
    }
    )
    severity_score = 10

    error_types = [error['type'] for error in errors]

    return {
    'syntax_valid': False,
    'errors': errors,
    'severity_score': severity_score,
    'error_types': error_types,
    'line_count': len(source.splitlines()) if 'source' in locals() else 0,
    }

    def categorize_error(self, error_msg: str) -> str:
    """Categorize syntax error types"""
    error_msg_lower = error_msg.lower()

        if any(word in error_msg_lower for word in ['f-string', 'invalid decimal']):
    return 'fstring_error'
        elif any(
    word in error_msg_lower for word in ['unmatched', 'parenthesis', 'bracket']
    ):
    return 'bracket_error'
        elif any(
    word in error_msg_lower for word in ['unterminated', 'string literal']
    ):
    return 'string_error'
        elif any(word in error_msg_lower for word in ['indent', 'indented']):
    return 'indentation_error'
        elif any(word in error_msg_lower for word in ['invalid syntax', 'expected']):
    return 'syntax_error'
        elif any(word in error_msg_lower for word in ['eof', 'unexpected']):
    return 'structure_error'
        else:
    return 'other_error'

    def calculate_severity(self, error_msg: str) -> int:
    """Calculate severity score (1-10) based on error type"""
    error_type = self.categorize_error(error_msg)

    severity_map = {
    'fstring_error': 3,  # Usually fixable
    'bracket_error': 4,  # Moderate complexity
    'string_error': 2,  # Often simple fixes
    'indentation_error': 2,  # Usually straightforward
    'syntax_error': 6,  # Can be complex
    'structure_error': 8,  # Often serious
    'other_error': 5,  # Unknown complexity
    }

    return severity_map.get(error_type, 5)

    def calculate_fixability_score(self, file_data: Dict[str, Any]) -> int:
    """Calculate fixability score (1-10, higher = more fixable)"""
        if file_data['syntax_valid']:
    return 10

    base_score = 10

        # Reduce score based on error types
        for error_type in file_data['error_types']:
            if error_type == 'fstring_error':
    base_score -= 1
            elif error_type == 'bracket_error':
    base_score -= 2
            elif error_type == 'string_error':
    base_score -= 1
            elif error_type == 'indentation_error':
    base_score -= 1
            elif error_type == 'syntax_error':
    base_score -= 3
            elif error_type == 'structure_error':
    base_score -= 4
            else:
    base_score -= 2

        # Factor in number of errors
    error_count = len(file_data['errors'])
        if error_count > 5:
    base_score -= 2
        elif error_count > 10:
    base_score -= 4

        # Factor in file size (larger files harder to fix)
    line_count = file_data.get('line_count', 0)
        if line_count > 1000:
    base_score -= 1
        elif line_count > 2000:
    base_score -= 2

    return max(1, min(10, base_score))

    def analyze_code_metrics(self, filepath: str) -> Dict[str, Any]:
    """Analyze code quality metrics"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
    source = f.read()

    lines = source.splitlines()

    metrics = {
    'line_count': len(lines),
    'empty_lines': sum(1 for line in lines if not line.strip()),
    'comment_lines': sum(
    1 for line in lines if line.strip().startswith('#')
    ),
    'function_count': len(
    re.findall(r'^\s*def\s+\w+', source, re.MULTILINE)
    ),
    'class_count': len(
    re.findall(r'^\s*class\s+\w+', source, re.MULTILINE)
    ),
    'import_count': len(
    re.findall(r'^\s*(?:from\s+\w+\s+)?import\s+', source, re.MULTILINE)
    ),
    'complexity_indicators': {
    'nested_functions': len(
    re.findall(r'^\s{4,}def\s+\w+', source, re.MULTILINE)
    ),
    'try_except_blocks': len(
    re.findall(r'^\s*try\s*:', source, re.MULTILINE)
    ),
    'long_lines': sum(1 for line in lines if len(line) > 100),
    'deep_nesting': sum(
    1 for line in lines if len(line) - len(line.lstrip()) > 16
    ),
    },
    }

            # Calculate complexity score
    complexity = (
    metrics['complexity_indicators']['nested_functions'] * 2
    + metrics['complexity_indicators']['try_except_blocks'] * 1
    + metrics['complexity_indicators']['long_lines'] * 0.1
    + metrics['complexity_indicators']['deep_nesting'] * 1.5
    )

    metrics['complexity_score'] = min(10, complexity / 10)

    return metrics

        except Exception as e:
    return {
    'line_count': 0,
    'empty_lines': 0,
    'comment_lines': 0,
    'function_count': 0,
    'class_count': 0,
    'import_count': 0,
    'complexity_indicators': {},
    'complexity_score': 0,
    'error': str(e),
    }

    def categorize_module(self, filepath: str, file_data: Dict[str, Any]) -> str:
    """Categorize module based on patterns and characteristics"""
    rel_path = os.path.relpath(filepath, self.root_dir)

        # Test files
        if any(pattern in rel_path for pattern in ['test_', 'tests/']):
    return 'test_files'

        # Core NLP engines
        if 'nlp/' in rel_path and ('engine.py' in rel_path or '/models/' in rel_path):
    return 'nlp_engines'

        # Core infrastructure
        if any(re.search(pattern, rel_path) for pattern in self.core_patterns):
    return 'core'

        # Tools and fixers
        if any(re.search(pattern, rel_path) for pattern in self.tool_patterns):
    return 'tools'

        # Experimental code
        if any(re.search(pattern, rel_path) for pattern in self.experimental_patterns):
    return 'experimental'

        # Dead code detection
    metrics = file_data.get('code_metrics', {})

        # Heuristics for dead code
        if (
    metrics.get('function_count', 0) == 0
    and metrics.get('class_count', 0) == 0
    and metrics.get('line_count', 0) < 50
    ):
    return 'dead_code'

        # If has many syntax errors and low complexity, might be dead
        if (
    not file_data.get('syntax_valid', True)
    and file_data.get('severity_score', 0) > 5
    and metrics.get('complexity_score', 0) < 2
    ):
    return 'dead_code'

        # Default to experimental for unclassified
    return 'experimental'

    def analyze_all_files(self) -> None:
    """Analyze all Python files in the repository"""
    print("🔍 Scanning Python files...")

    file_count = 0
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
    filepath = os.path.join(root, file)
    rel_path = os.path.relpath(filepath, self.root_dir)

    print(f"  Analyzing: {rel_path}")

                    # Analyze syntax errors
    syntax_analysis = self.analyze_syntax_errors(filepath)

                    # Analyze code metrics
    code_metrics = self.analyze_code_metrics(filepath)

                    # Calculate scores
    fixability_score = self.calculate_fixability_score(syntax_analysis)

                    # Combine all data
    file_data = {
    'filepath': rel_path,
    'syntax_analysis': syntax_analysis,
    'code_metrics': code_metrics,
    'fixability_score': fixability_score,
    'severity_score': syntax_analysis['severity_score'],
    }

                    # Categorize module
    category = self.categorize_module(filepath, file_data)
    file_data['category'] = category

    self.files_data[rel_path] = file_data
    self.module_categories[category].append(rel_path)

    file_count += 1

    print(f"✅ Analyzed {file_count} Python files")

    def generate_heatmap_data(self) -> Dict[str, Any]:
    """Generate heatmap data for visualization"""
    broken_files = []

        for filepath, data in self.files_data.items():
            if not data['syntax_analysis']['syntax_valid']:
    broken_files.append(
    {
    'file': filepath,
    'severity': data['severity_score'],
    'fixability': data['fixability_score'],
    'error_count': len(data['syntax_analysis']['errors']),
    'error_types': data['syntax_analysis']['error_types'],
    'category': data['category'],
    'line_count': data['code_metrics']['line_count'],
    }
    )

        # Sort by severity (descending) then by fixability (ascending)
    broken_files.sort(key=lambda x: (-x['severity'], x['fixability']))

    return {
    'broken_files': broken_files,
    'total_files': len(self.files_data),
    'broken_count': len(broken_files),
    'success_rate': (
    (len(self.files_data) - len(broken_files)) / len(self.files_data)
    )
    * 100,
    }

    def generate_severity_ranking(self) -> List[Dict[str, Any]]:
    """Generate files ranked by severity and fixability"""
    rankings = []

        for filepath, data in self.files_data.items():
    rankings.append(
    {
    'file': filepath,
    'category': data['category'],
    'severity_score': data['severity_score'],
    'fixability_score': data['fixability_score'],
    'syntax_valid': data['syntax_analysis']['syntax_valid'],
    'error_count': len(data['syntax_analysis']['errors']),
    'line_count': data['code_metrics']['line_count'],
    'complexity': data['code_metrics']['complexity_score'],
    'priority_score': self.calculate_priority_score(data),
    }
    )

        # Sort by priority score (higher = more urgent)
    rankings.sort(key=lambda x: -x['priority_score'])

    return rankings

    def calculate_priority_score(self, file_data: Dict[str, Any]) -> float:
    """Calculate priority score for fixing (higher = more urgent)"""
        if file_data['syntax_analysis']['syntax_valid']:
    return 0  # No need to fix

    base_score = file_data['severity_score']

        # Boost score for high fixability
    fixability_bonus = file_data['fixability_score'] / 2

        # Boost score for core modules
    category_multiplier = {
    'core': 2.0,
    'nlp_engines': 1.8,
    'test_files': 1.5,
    'tools': 1.2,
    'experimental': 0.8,
    'dead_code': 0.3,
    }

    multiplier = category_multiplier.get(file_data['category'], 1.0)

        # Factor in file size (larger files might be more important)
    size_factor = min(2.0, file_data['code_metrics']['line_count'] / 500)

    priority = (base_score + fixability_bonus) * multiplier * size_factor

    return round(priority, 2)

    def generate_report(self) -> str:
    """Generate comprehensive analysis report"""
    heatmap_data = self.generate_heatmap_data()
    severity_ranking = self.generate_severity_ranking()

        # Category statistics
    category_stats = {}
        for category, files in self.module_categories.items():
    broken_in_category = sum(
    1
                for f in files
                if not self.files_data[f]['syntax_analysis']['syntax_valid']
    )
    category_stats[category] = {
    'total': len(files),
    'broken': broken_in_category,
    'success_rate': (
    ((len(files) - broken_in_category) / len(files)) * 100
                    if files
                    else 100
    ),
    }

        # Error type analysis
    error_type_counts = Counter()
        for data in self.files_data.values():
            for error_type in data['syntax_analysis']['error_types']:
    error_type_counts[error_type] += 1

    report = f"""
🔥 COMPREHENSIVE FILE ANALYSIS REPORT
{'='*70}
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATISTICS:
  • Total Python files: {len(self.files_data)}
  • Files with syntax errors: {heatmap_data['broken_count']}
  • Overall success rate: {heatmap_data['success_rate']:.1f}%

🗂️ MODULE CATEGORIZATION:
"""

        for category, stats in category_stats.items():
    report += f"  • {category.upper()}: {stats['total']} files ({stats['broken']} broken, {stats['success_rate']:.1f}% success)\n"

    report += f"""
🔥 TOP 15 MOST CRITICAL FILES (by priority):
"""

        for i, file_info in enumerate(severity_ranking[:15], 1):
            if not file_info['syntax_valid']:
    report += f"  {i:2d}. {file_info['file']:<50} "
    report += f"[{file_info['category']:<12}] "
    report += f"Priority: {file_info['priority_score']:4.1f} "
    report += f"Severity: {file_info['severity_score']}/10 "
    report += f"Fixability: {file_info['fixability_score']}/10\n"

    report += f"""
🎯 ERROR TYPE BREAKDOWN:
"""

        for error_type, count in error_type_counts.most_common():
    report += f"  • {error_type:<20}: {count:3d} occurrences\n"

    report += f"""
📈 FIXABILITY RECOMMENDATIONS:

🟢 HIGH PRIORITY (Fix First):
"""

    high_priority = [
    f
            for f in severity_ranking
            if f['priority_score'] > 8 and not f['syntax_valid']
    ]
        for file_info in high_priority[:10]:
    report += f"  • {file_info['file']:<50} [{file_info['category']}]\n"

    report += f"""
🟡 MEDIUM PRIORITY (Fix After High):
"""

    medium_priority = [
    f
            for f in severity_ranking
            if 4 <= f['priority_score'] <= 8 and not f['syntax_valid']
    ]
        for file_info in medium_priority[:10]:
    report += f"  • {file_info['file']:<50} [{file_info['category']}]\n"

    report += f"""
🔴 LOW PRIORITY (Consider Deprecation):
"""

    low_priority = [
    f
            for f in severity_ranking
            if f['priority_score'] < 4 and not f['syntax_valid']
    ]
        for file_info in low_priority[:10]:
    report += f"  • {file_info['file']:<50} [{file_info['category']}]\n"

    report += f"""
🗂️ DETAILED MODULE BREAKDOWN:

CORE MODULES ({len(self.module_categories['core'])} files):
"""
        for filepath in sorted(self.module_categories['core']):
    status = (
    "✅"
                if self.files_data[filepath]['syntax_analysis']['syntax_valid']
                else "❌"
    )
    report += f"  {status} {filepath}\n"

    report += f"""
NLP ENGINES ({len(self.module_categories['nlp_engines'])} files):
"""
        for filepath in sorted(self.module_categories['nlp_engines']):
    status = (
    "✅"
                if self.files_data[filepath]['syntax_analysis']['syntax_valid']
                else "❌"
    )
    report += f"  {status} {filepath}\n"

    report += f"""
EXPERIMENTAL CODE ({len(self.module_categories['experimental'])} files):
"""
    experimental_broken = [
    f
            for f in self.module_categories['experimental']
            if not self.files_data[f]['syntax_analysis']['syntax_valid']
    ]
        for filepath in sorted(experimental_broken)[
    :15
    ]:  # Show top 15 broken experimental
    priority = next(
    f['priority_score'] for f in severity_ranking if f['file'] == filepath
    )
    report += f"  ❌ {filepath:<50} (Priority: {priority:.1f})\n"

        if len(experimental_broken) > 15:
    report += f"  ... and {len(experimental_broken)} - 15} more broken experimental files\n"

    report += f"""
DEAD CODE CANDIDATES ({len(self.module_categories['dead_code'])} files):
"""
        for filepath in sorted(self.module_categories['dead_code'])[:10]:
    report += f"  🗑️ {filepath}\n"

        if len(self.module_categories['dead_code']) > 10:
    report += (
    f"  ... and {len(self.module_categories['dead_code'])} - 10} more\n"
    )

    return report

    def save_detailed_data(self, filename: str = None) -> str:
    """Save detailed analysis data to JSON"""
        if filename is None:
    filename = f"file_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    export_data = {
    'timestamp': datetime.now().isoformat(),
    'files_data': self.files_data,
    'module_categories': self.module_categories,
    'heatmap_data': self.generate_heatmap_data(),
    'severity_ranking': self.generate_severity_ranking(),
    }

        with open(filename, 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

    return filename


def main():
    print("🔥 Starting Comprehensive File Analysis...")

    analyzer = FileAnalyzer()
    analyzer.analyze_all_files()

    # Generate and display report
    report = analyzer.generate_report()
    print(report)

    # Save detailed data
    json_file = analyzer.save_detailed_data()
    print(f"\n📊 Detailed analysis data saved to: {json_file}")

    # Save text report
    report_file = (
    f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

    print(f"📝 Text report saved to: {report_file}")


if __name__ == "__main__":
    main()
