#!/usr/bin/env python3
"""
📊 TEXT-BASED HEATMAP GENERATOR
================================================================
Creates ASCII-based heatmaps and summary reports for systems
without matplotlib/seaborn dependencies

Usage:
  python text_heatmap_generator.py
"""

import json
import os
from datetime import datetime
from collections import Counter, defaultdict


class TextHeatmapGenerator:
    """Generate text-based visualizations and reports"""

    def __init__(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def create_ascii_heatmap(self) -> str:
        """Create ASCII art heatmap of severity vs fixability"""
        broken_files = self.data['heatmap_data']['broken_files']

        if not broken_files:
            return "No broken files to visualize"

        # Create 10x10 grid for severity (y) vs fixability (x)
        grid = [[0 for _ in range(11)] for _ in range(11)]

        # Fill grid with file counts
        for file_info in broken_files:
            severity = min(10, max(1, file_info['severity']))
            fixability = min(10, max(1, file_info['fixability']))
            grid[severity][fixability] += 1

        # Generate ASCII heatmap
        heatmap = """
🔥 SEVERITY vs FIXABILITY HEATMAP
=====================================
Severity (Y-axis) vs Fixability (X-axis)
Numbers show count of files in each cell

     Fixability Score (1=Hard to Fix → 10=Easy to Fix)
     1  2  3  4  5  6  7  8  9 10
   ┌─────────────────────────────────┐
10 │"""

        # Color mapping for ASCII
        def get_intensity_char(count):
            if count == 0:
                return " ."
            elif count <= 2:
                return f" {count}"
            elif count <= 5:
                return f"●{count}"
            else:
                return f"█{count}"

        # Build heatmap from top to bottom (severity 10 to 1)
        for severity in range(10, 0, -1):
            if severity < 10:
                heatmap += f"{severity:2d} │"
            for fixability in range(1, 11):
                count = grid[severity][fixability]
                heatmap += get_intensity_char(count)
            heatmap += "│\n"
            if severity > 1:
                heatmap += "   │"

        heatmap += """   └─────────────────────────────────┘
   S
   e  Legend: . = 0 files, 1-2 = few files
   v  ● = moderate (3-5 files), █ = many (6+ files)
   e
   r
   i
   t
   y
"""

        return heatmap

    def create_category_breakdown(self) -> str:
        """Create text-based category breakdown"""
        categories = self.data['module_categories']

        breakdown = """
📊 MODULE CATEGORY BREAKDOWN
==============================
"""

        total_files = len(self.data['files_data'])

        for category, files in categories.items():
            broken_count = 0
            working_count = 0

            for filepath in files:
                if filepath in self.data['files_data']:
                    if self.data['files_data'][filepath]['syntax_analysis'][
                        'syntax_valid'
                    ]:
                        working_count += 1
                    else:
                        broken_count += 1

            total_in_category = len(files)
            if total_in_category > 0:
                success_rate = (working_count / total_in_category) * 100

                # Create progress bar
                bar_length = 20
                filled_length = int(bar_length * success_rate / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                breakdown += f"""
{category.upper().replace('_', ' '):<20} [{bar}] {success_rate:5.1f}%
  • Total: {total_in_category:3d}  Working: {working_count:3d}  Broken: {broken_count:3d}"""

        return breakdown

    def create_priority_list(self) -> str:
        """Create text-based priority list"""
        rankings = self.data['severity_ranking']
        broken_files = [f for f in rankings if not f['syntax_valid']][:25]  # Top 25

        priority_list = """
🎯 TOP 25 CRITICAL FILES (Priority Order)
==========================================
Rank File                                    Category      Priority  Severity  Fixability
─────────────────────────────────────────────────────────────────────────────────────────
"""

        for i, file_info in enumerate(broken_files, 1):
            filename = os.path.basename(file_info['file'])[:30]
            category = file_info['category'][:10]
            priority = file_info['priority_score']
            severity = file_info['severity_score']
            fixability = file_info['fixability_score']

            priority_list += f"{i:3d}. {filename:<30} {category:<12} {priority:6.1f}   {severity:7.0f}     {fixability:8.0f}\n"

        return priority_list

    def create_error_analysis(self) -> str:
        """Create detailed error type analysis"""
        error_counts = Counter()
        error_by_category = defaultdict(lambda: Counter())

        for filepath, file_data in self.data['files_data'].items():
            category = file_data['category']
            for error_type in file_data['syntax_analysis']['error_types']:
                error_counts[error_type] += 1
                error_by_category[category][error_type] += 1

        analysis = """
🔍 ERROR TYPE ANALYSIS
=======================
"""

        # Overall error distribution
        analysis += "\nOverall Error Distribution:\n"
        total_errors = sum(error_counts.values())

        for error_type, count in error_counts.most_common():
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            bar_length = 20
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            analysis += f"  {error_type:<20} [{bar}] {count:3d} ({percentage:5.1f}%)\n"

        # Error types by category
        analysis += "\nError Types by Category:\n"
        for category, errors in error_by_category.items():
            if sum(errors.values()) > 0:
                analysis += f"\n  {category.upper().replace('_',} ' ')}:\n"
                for error_type, count in errors.most_common(3):  # Top 3 per category
                    analysis += f"    • {error_type:<18}: {count:3d}\n"

        return analysis

    def create_quick_wins_report(self) -> str:
        """Create focused quick wins report"""
        rankings = self.data['severity_ranking']
        broken_files = [f for f in rankings if not f['syntax_valid']]

        # Define quick wins criteria
        quick_wins = []
        for file_info in broken_files:
            if file_info['fixability_score'] >= 7 and file_info['priority_score'] >= 8:
                quick_wins.append(file_info)

        report = f"""
🚀 QUICK WINS REPORT
====================
Found {len(quick_wins)} high-priority, easy-to-fix files

These files offer the best return on investment:
(High priority + High fixability = Quick wins)

"""

        for i, file_info in enumerate(quick_wins[:15], 1):
            filename = os.path.basename(file_info['file'])
            report += f"{i:2d}. {filename:<45} "
            report += f"Priority: {file_info['priority_score']:4.1f}  "
            report += f"Fixability: {file_info['fixability_score']}/10  "
            report += f"Category: {file_info['category']}\n"

        if len(quick_wins) > 15:
            report += f"\n... and {len(quick_wins)} - 15} more quick wins available\n"

        # Add recommended action
        report += f"""
💡 RECOMMENDED ACTION:
Focus on these {min(len(quick_wins), 5)} files this week for maximum impact with minimal effort.
Each fix will improve your success rate by ~{100/len(self.data['files_data']):.2f}%.
"""

        return report

    def create_dead_code_report(self) -> str:
        """Create dead code analysis report"""
        dead_candidates = []

        for filepath, file_data in self.data['files_data'].items():
            metrics = file_data['code_metrics']

            # Identify dead code candidates
            if (
                metrics.get('line_count', 0) < 30
                and metrics.get('function_count', 0) == 0
                and metrics.get('class_count', 0) == 0
            ):
                dead_candidates.append(
                    {
                        'file': filepath,
                        'lines': metrics.get('line_count', 0),
                        'functions': metrics.get('function_count', 0),
                        'classes': metrics.get('class_count', 0),
                        'valid': file_data['syntax_analysis']['syntax_valid'],
                    }
                )

        dead_candidates.sort(key=lambda x: x['lines'])

        report = f"""
🗑️ DEAD CODE ANALYSIS
======================
Found {len(dead_candidates)} potential dead code files

These files have minimal content and could be candidates for removal:

"""

        bytes_saved = 0
        for i, candidate in enumerate(dead_candidates[:20], 1):
            filename = os.path.basename(candidate['file'])
            status = "✅" if candidate['valid'] else "❌"
            bytes_saved += candidate['lines'] * 50  # Rough estimate

            report += f"{i:2d}. {status} {filename:<35} "
            report += f"{candidate['lines']:3d} lines  "
            report += f"{candidate['functions']} funcs  "
            report += f"{candidate['classes']} classes\n"

        if len(dead_candidates) > 20:
            for candidate in dead_candidates[20:]:
                bytes_saved += candidate['lines'] * 50
            report += f"\n... and {len(dead_candidates)} - 20} more candidates\n"

        report += f"""
📈 CLEANUP IMPACT:
• Removing these files would eliminate {len(dead_candidates)} files from your codebase
• Estimated {bytes_saved/1024:.1f}KB reduction in codebase size
• Reduced maintenance burden and cleaner repository
• Improved focus on core functionality

💡 RECOMMENDATION:
Review and archive/delete files with 0 functions and <10 lines of code.
"""

        return report

    def generate_comprehensive_report(self) -> str:
        """Generate complete text-based analysis report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""
{'='*70}
🔥 COMPREHENSIVE FILE HEATMAP & ANALYSIS REPORT
{'='*70}
📅 Generated: {timestamp}

{self.create_ascii_heatmap()}

{self.create_category_breakdown()}

{self.create_priority_list()}

{self.create_error_analysis()}

{self.create_quick_wins_report()}

{self.create_dead_code_report()}

{'='*70}
📊 SUMMARY STATISTICS
{'='*70}
• Total Python files analyzed: {len(self.data['files_data'])}
• Files with syntax errors: {self.data['heatmap_data']['broken_count']}
• Current success rate: {self.data['heatmap_data']['success_rate']:.1f}%
• Improvement potential: {100 - self.data['heatmap_data']['success_rate']:.1f}%

🎯 NEXT STEPS SUMMARY:
1. Start with Quick Wins (15 high-priority, easy fixes)
2. Clean up Dead Code ({len([f for f in self.data['files_data'].values() if f['code_metrics'].get('line_count', 0) < 30 and f['code_metrics'].get('function_count', 0) == 0])} candidates)
3. Focus on Critical Path (core infrastructure files)
4. Plan Major Projects (complex but high-impact fixes)

💡 Use surgical_syntax_fixer_v3.py for automated improvements!
{'='*70}
"""

        return report


def main():
    # Find the most recent analysis file
    analysis_files = [
        f
        for f in os.listdir('.')
        if f.startswith('file_analysis_') and f.endswith('.json')
    ]

    if not analysis_files:
        print(
            "❌ No analysis JSON file found. Run comprehensive_file_analysis.py first."
        )
        return

    # Use the most recent file
    latest_file = sorted(analysis_files)[-1]
    print(f"📊 Using analysis file: {latest_file}")

    try:
        generator = TextHeatmapGenerator(latest_file)
        report = generator.generate_comprehensive_report()

        print(report)

        # Save the report
        filename = f"text_heatmap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📝 Complete text heatmap report saved to: {filename}")

    except Exception as e:
        print(f"❌ Error generating text heatmap: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
