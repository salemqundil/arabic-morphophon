#!/usr/bin/env python3
"""
🎯 STRATEGIC ACTION PLAN GENERATOR
================================================================
Generates actionable recommendations based on file analysis data

Usage:
  python strategic_action_plan.py
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


class StrategicActionPlanner:
    """Generate strategic action plans for code improvement"""

    def __init__(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
    self.data = json.load(f)

    self.recommendations = {
    'immediate': [],
    'short_term': [],
    'medium_term': [],
    'long_term': [],
    'deprecation': [],
    }

    def analyze_critical_path(self) -> List[Dict[str, Any]]:
    """Identify critical path files that block other improvements"""
    rankings = self.data['severity_ranking']
    broken_files = [f for f in rankings if not f['syntax_valid']]

        # Core infrastructure files (highest impact)
    core_blockers = []
        for file_info in broken_files:
            if (
    file_info['category'] in ['core', 'nlp_engines']
    or 'base_engine' in file_info['file']
    or 'engine.py' in file_info['file']
    ):
    core_blockers.append(file_info)

        # Sort by priority
    core_blockers.sort(key=lambda x: -x['priority_score'])

    return core_blockers

    def categorize_by_effort_impact(self) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize files by effort vs impact matrix"""
    rankings = self.data['severity_ranking']
    broken_files = [f for f in rankings if not f['syntax_valid']]

    categories = {
    'quick_wins': [],  # High impact, low effort
    'major_projects': [],  # High impact, high effort
    'fill_ins': [],  # Low impact, low effort
    'questionable': [],  # Low impact, high effort
    }

        for file_info in broken_files:
    impact = self.calculate_impact_score(file_info)
    effort = self.calculate_effort_score(file_info)

    file_info['impact_score'] = impact
    file_info['effort_score'] = effort

            if impact >= 7 and effort <= 4:
    categories['quick_wins'].append(file_info)
            elif impact >= 7 and effort > 4:
    categories['major_projects'].append(file_info)
            elif impact < 7 and effort <= 4:
    categories['fill_ins'].append(file_info)
            else:
    categories['questionable'].append(file_info)

        # Sort each category by priority
        for category in categories.values():
    category.sort(key=lambda x: -x['priority_score'])

    return categories

    def calculate_impact_score(self, file_info: Dict[str, Any]) -> int:
    """Calculate business/technical impact score (1-10)"""
    base_score = 5

        # Category impact multipliers
    category_impact = {
    'core': 10,
    'nlp_engines': 9,
    'test_files': 7,
    'tools': 6,
    'experimental': 4,
    'dead_code': 1,
    }

    base_score = category_impact.get(file_info['category'], 5)

        # File size factor (larger files more impactful)
        if file_info['line_count'] > 1000:
    base_score += 1
        elif file_info['line_count'] > 2000:
    base_score += 2

        # Complexity factor
        if file_info.get('complexity', 0) > 5:
    base_score += 1

        # Critical keywords in filename
    critical_keywords = ['base_engine', 'core', 'api', 'main', 'engine.py']
        if any(keyword in file_info['file'].lower() for keyword in critical_keywords):
    base_score += 2

    return min(10, max(1, base_score))

    def calculate_effort_score(self, file_info: Dict[str, Any]) -> int:
    """Calculate effort required to fix (1=easy, 10=hard)"""
    base_effort = 5

        # Invert fixability score (higher fixability = lower effort)
    base_effort = 11 - file_info['fixability_score']

        # Adjust for file size
        if file_info['line_count'] > 1000:
    base_effort += 1
        elif file_info['line_count'] > 2000:
    base_effort += 2

        # Adjust for error count
        if file_info['error_count'] > 5:
    base_effort += 1
        elif file_info['error_count'] > 10:
    base_effort += 2

        # Adjust for complexity
        if file_info.get('complexity', 0) > 7:
    base_effort += 1

    return min(10, max(1, base_effort))

    def identify_dead_code_candidates(self) -> List[Dict[str, Any]]:
    """Identify files that should be considered for deletion"""
    dead_candidates = []

        for filepath, file_data in self.data['files_data'].items():
    metrics = file_data['code_metrics']

            # Criteria for dead code
    is_dead_candidate = (
                # Very small files with no functions/classes
    (
    metrics.get('line_count', 0) < 30
    and metrics.get('function_count', 0) == 0
    and metrics.get('class_count', 0) == 0
    )
    or
                # Files with severe syntax errors and low complexity
    (
    not file_data['syntax_analysis']['syntax_valid']
    and file_data['severity_score'] > 6
    and metrics.get('complexity_score', 0) < 2
    )
    or
                # Files already categorized as dead code
    file_data['category'] == 'dead_code'
    )

            if is_dead_candidate:
    dead_candidates.append(
    {
    'file': filepath,
    'category': file_data['category'],
    'line_count': metrics.get('line_count', 0),
    'function_count': metrics.get('function_count', 0),
    'class_count': metrics.get('class_count', 0),
    'syntax_valid': file_data['syntax_analysis']['syntax_valid'],
    'severity_score': file_data['severity_score'],
    'reason': self.get_dead_code_reason(file_data, metrics),
    }
    )

    return sorted(dead_candidates, key=lambda x: x['line_count'])

    def get_dead_code_reason(
    self, file_data: Dict[str, Any], metrics: Dict[str, Any]
    ) -> str:
    """Get reason why file is considered dead code"""
        if metrics.get('line_count', 0) < 30 and metrics.get('function_count', 0) == 0:
    return "Empty or minimal content"
        elif (
    not file_data['syntax_analysis']['syntax_valid']
    and file_data['severity_score'] > 6
    ):
    return "Severe syntax errors, low value"
        elif file_data['category'] == 'dead_code':
    return "Categorized as unused"
        else:
    return "Low complexity, questionable value"

    def generate_action_plan(self) -> str:
    """Generate comprehensive strategic action plan"""
    critical_path = self.analyze_critical_path()
    effort_impact = self.categorize_by_effort_impact()
    dead_code = self.identify_dead_code_candidates()

        # Calculate summary statistics
    total_files = len(self.data['files_data'])
    broken_files = len(
    [
    f
                for f in self.data['files_data'].values()
                if not f['syntax_analysis']['syntax_valid']
    ]
    )
    success_rate = ((total_files - broken_files) / total_files) * 100

    plan = f"""
🎯 STRATEGIC ACTION PLAN FOR CODE IMPROVEMENT
{'='*70}
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CURRENT STATE ASSESSMENT:
  • Total files: {total_files}
  • Broken files: {broken_files}
  • Success rate: {success_rate:.1f}%
  • Improvement potential: {100 - success_rate:.1f}%

🚨 CRITICAL PATH ANALYSIS:
These files block other improvements and should be fixed first:
"""

        for i, file_info in enumerate(critical_path[:5], 1):
    plan += f"""
  {i}. {file_info['file']}
     • Category: {file_info['category'].title()}
     • Priority Score: {file_info['priority_score']:.1f}
     • Impact: High (blocks {file_info['category']} functionality)
     • Estimated Effort: {'Low' if file_info['fixability_score'] >= 7 else 'Medium' if file_info['fixability_score'] >= 4 else 'High'}
"""

    plan += f"""
🎯 EFFORT VS IMPACT MATRIX:

🟢 PHASE 1: QUICK WINS (High Impact, Low Effort) - Start Here!
   Target: Fix {len(effort_impact['quick_wins'])} files in 1-2 weeks
"""

        for i, file_info in enumerate(effort_impact['quick_wins'][:8], 1):
    plan += f"""   {i}. {os.path.basename(file_info['file']):<40} [Impact: {file_info['impact_score']}/10, Effort: {file_info['effort_score']}/10]"""

        if len(effort_impact['quick_wins']) > 8:
    plan += (
    f"\n   ... and {len(effort_impact['quick_wins'])} - 8} more quick wins"
    )

    plan += f"""

🔵 PHASE 2: MAJOR PROJECTS (High Impact, High Effort) - Plan Carefully
   Target: Fix {len(effort_impact['major_projects'])} files in 1-2 months
"""

        for i, file_info in enumerate(effort_impact['major_projects'][:5], 1):
    plan += f"""   {i}. {os.path.basename(file_info['file']):<40} [Impact: {file_info['impact_score']}/10, Effort: {file_info['effort_score']}/10]"""

        if len(effort_impact['major_projects']) > 5:
    plan += f"\n   ... and {len(effort_impact['major_projects'])} - 5} more major projects"

    plan += f"""

🟡 PHASE 3: FILL-INS (Low Impact, Low Effort) - When Time Permits
   Target: Fix {len(effort_impact['fill_ins'])} files gradually
"""

        for i, file_info in enumerate(effort_impact['fill_ins'][:5], 1):
    plan += f"""   {i}. {os.path.basename(file_info['file']):<40} [Impact: {file_info['impact_score']}/10, Effort: {file_info['effort_score']}/10]"""

    plan += f"""

🗑️ DEPRECATION CANDIDATES ({len(dead_code)} files):
   Recommend deletion or archival to reduce maintenance burden:
"""

        for file_info in dead_code[:10]:
    plan += f"""   • {file_info['file']:<50} ({file_info['reason']})"""

        if len(dead_code) > 10:
    plan += f"\n   ... and {len(dead_code)} - 10} more candidates"

    plan += f"""

🚫 AVOID FOR NOW (Low Impact, High Effort):
   {len(effort_impact['questionable'])} files - Consider deprecation instead of fixing
"""

        for file_info in effort_impact['questionable'][:5]:
    plan += f"""   • {os.path.basename(file_info['file']):<40} [Impact: {file_info['impact_score']}/10, Effort: {file_info['effort_score']}/10]"""

    plan += f"""

📈 SUCCESS METRICS & MILESTONES:

🎯 30-Day Goal (Phase 1 Complete):
   • Fix all {len(effort_impact['quick_wins'])} quick wins
   • Expected success rate improvement: +{len(effort_impact['quick_wins']) / total_files * 100:.1f}%
   • Target success rate: {success_rate + (len(effort_impact['quick_wins']) / total_files * 100):.1f}%

🎯 90-Day Goal (Phase 2 Major Projects):
   • Complete {min(3, len(effort_impact['major_projects']))} major projects
   • Expected additional improvement: +{min(3, len(effort_impact['major_projects'])) / total_files * 100:.1f}%
   • Target success rate: {success_rate + ((len(effort_impact['quick_wins']) + min(3, len(effort_impact['major_projects']))) / total_files * 100):.1f}%

🎯 6-Month Goal (Full Cleanup):
   • Complete all critical fixes
   • Remove {len(dead_code)} dead code files
   • Target success rate: 85%+
   • Reduced technical debt significantly

🛠️ RECOMMENDED TOOLING STRATEGY:

1. Enhanced Surgical Fixers:
   • Continue using surgical_syntax_fixer_v3.py for automated fixes
   • Target specific error patterns for quick wins
   • Maintain comprehensive backup strategy

2. Incremental Approach:
   • Fix 3-5 files per day from quick wins list
   • Weekly review of progress against milestones
   • Monthly reassessment of priorities

3. Quality Gates:
   • All fixes must pass existing test suite
   • New fixes should include test coverage
   • Code review for major projects

📋 NEXT IMMEDIATE ACTIONS:

1. 🔥 START WITH CRITICAL PATH:
   Fix these 3 files in next 48 hours:
"""

        for i, file_info in enumerate(critical_path[:3], 1):
    plan += f"""   {i}. {file_info['file']} (Priority: {file_info['priority_score']:.1f})"""

    plan += f"""

2. 🧹 CLEAN UP DEAD CODE:
   Delete these obvious dead code files today:
"""

    obvious_dead = [
    f for f in dead_code if f['line_count'] < 10 and f['function_count'] == 0
    ][:5]
        for file_info in obvious_dead:
    plan += f"""   • {file_info['file']} ({file_info['line_count']} lines, {file_info['reason']})"""

    plan += f"""

3. 🎯 FOCUS ON QUICK WINS:
   Target these high-value, low-effort fixes this week:
"""

        for i, file_info in enumerate(effort_impact['quick_wins'][:5], 1):
    plan += f"""   {i}. {file_info['file']} (Impact: {file_info['impact_score']}, Effort: {file_info['effort_score']})"""

    plan += f"""

💡 SUCCESS FACTORS:
   • Maintain daily progress tracking
   • Use automated tools for consistent fixes
   • Preserve test suite functionality throughout
   • Regular backup and version control
   • Focus on sustainable improvement rate

🎉 EXPECTED OUTCOMES:
   • 30-day: {len(effort_impact['quick_wins'])+ len(obvious_dead)} files improved/removed
   • 90-day: {success_rate + 15:.1f}% success rate achieved
   • 6-month: World-class Arabic NLP codebase with <90% success rate
   • Long-term: Maintainable, scalable architecture ready for production
"""

    return plan

    def save_action_plan(self, plan: str, filename: str = None) -> str:
    """Save action plan to file"""
        if filename is None:
    filename = (
    f"strategic_action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

        with open(filename, 'w', encoding='utf-8') as f:
    f.write(plan)

    return filename


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
    planner = StrategicActionPlanner(latest_file)
    action_plan = planner.generate_action_plan()

    print(action_plan)

        # Save the plan
    filename = planner.save_action_plan(action_plan)
    print(f"\n📝 Strategic action plan saved to: {filename}")

    except Exception as e:
    print(f"❌ Error generating action plan: {e}")
        import traceback

    traceback.print_exc()


if __name__ == "__main__":
    main()
