#!/usr/bin/env python3
"""
🔍 FOCUSED DEPENDENCY ANALYSIS REPORT
===================================

Advanced analysis of module dependencies with specific focus on:
- Core module structure analysis
- High-coupling identification
- Architectural improvement suggestions
- Import dependency visualization
"""

import json
from pathlib import Path
from datetime import datetime


def load_dependency_data():
    """Load the most recent dependency analysis data"""
    # Find the most recent JSON file
    json_files = list(Path('.').glob('dependency_graph_data_*.json'))
    if not json_files:
    print("❌ No dependency data found. Run circular_import_analyzer.py first.")
    return None

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"📊 Loading dependency data from: {latest_file}")

    with open(latest_file, 'r') as f:
    return json.load(f)


def analyze_core_structure(data):
    """Analyze the core module structure"""
    print("\n🏗️ CORE MODULE STRUCTURE ANALYSIS")
    print("=" * 50)

    modules = data['modules']
    dependencies = data['dependencies']
    complexity = data['complexity']

    # Group modules by directory structure
    structure = {}
    for module in modules:
    parts = module.split('.')
        if len(parts) > 1:
    base = parts[0]
            if base not in structure:
    structure[base] = []
    structure[base].append(module)
        else:
            if 'root' not in structure:
    structure['root'] = []
    structure['root'].append(module)

    print(f"📁 Directory Structure Overview:")
    for base, mods in sorted(structure.items()):
    print(f"  {base}/: {len(mods)} modules")
        if base in ['core', 'nlp', 'tests', 'tools']:
            # Show top modules in key directories
    top_mods = sorted(
    mods,
    key=lambda m: complexity.get(m, {}).get('total_coupling', 0),
    reverse=True,
    )[:3]
            for mod in top_mods:
    coupling = complexity.get(mod, {}).get('total_coupling', 0)
                if coupling > 0:
    print(f"    📦 {mod}: {coupling} connections")

    return structure


def identify_architectural_patterns(data):
    """Identify architectural patterns and anti-patterns"""
    print("\n🏛️ ARCHITECTURAL PATTERN ANALYSIS")
    print("=" * 50)

    complexity = data['complexity']
    dependencies = data['dependencies']

    # Find modules with high fan-in (many dependents)
    stable_modules = {}
    unstable_modules = {}

    for module, stats in complexity.items():
    fan_in = stats.get('imported_by', 0)
    fan_out = stats.get('imports', 0)
    instability = stats.get('instability', 0)

        if fan_in > 3:  # Many modules depend on this
    stable_modules[module] = {
    'dependents': fan_in,
    'dependencies': fan_out,
    'instability': instability,
    }

        if fan_out > 3 and fan_in > 1:  # High coupling
    unstable_modules[module] = {
    'dependents': fan_in,
    'dependencies': fan_out,
    'instability': instability,
    }

    print("🎯 STABLE MODULES (High Fan-In - Good for Interfaces)")
    for module, stats in sorted(
    stable_modules.items(), key=lambda x: x[1]['dependents'], reverse=True
    )[:10]:
    print(f"  📌 {module}")
    print(
    f"     Dependents: {stats['dependents']} | Dependencies: {stats['dependencies']} | Instability: {stats['instability']:.2f}"
    )

    print("\n⚠️ HIGH COUPLING MODULES (Need Refactoring)")
    for module, stats in sorted(
    unstable_modules.items(), key=lambda x: x[1]['dependencies'], reverse=True
    )[:5]:
    print(f"  🔗 {module}")
    print(
    f"     Dependents: {stats['dependents']} | Dependencies: {stats['dependencies']} | Instability: {stats['instability']:.2f}"
    )

    return stable_modules, unstable_modules


def suggest_refactoring_strategies(data, stable_modules, unstable_modules):
    """Generate specific refactoring suggestions"""
    print("\n🛠️ TARGETED REFACTORING STRATEGIES")
    print("=" * 50)

    dependencies = data['dependencies']
    complexity = data['complexity']

    suggestions = []

    # Strategy 1: Extract Common Interfaces
    highly_depended = [m for m, s in stable_modules.items() if s['dependents'] > 10]
    if highly_depended:
    suggestions.append(
    {
    'priority': 'HIGH',
    'type': 'Interface Extraction',
    'title': 'Extract Stable Interfaces',
    'description': f'Create abstract base classes for modules with many dependents',
    'modules': highly_depended,
    'action': 'Create ABC interfaces to reduce direct coupling',
    }
    )

    # Strategy 2: Split Large Modules
    large_modules = [
    (m, s) for m, s in complexity.items() if s.get('total_coupling', 0) > 15
    ]
    if large_modules:
    suggestions.append(
    {
    'priority': 'HIGH',
    'type': 'Module Decomposition',
    'title': 'Split Large Modules',
    'description': f'Break down modules with excessive coupling',
    'modules': [m for m, s in large_modules[:5]],
    'action': 'Split into focused, single-responsibility modules',
    }
    )

    # Strategy 3: Create Facade Pattern
    tool_modules = [
    m
        for m in data['modules']
        if any(part in m.lower() for part in ['fix', 'tool', 'util', 'helper'])
    ]
    if len(tool_modules) > 10:
    suggestions.append(
    {
    'priority': 'MEDIUM',
    'type': 'Facade Pattern',
    'title': 'Consolidate Tool Modules',
    'description': f'Create facade for {len(tool_modules)} utility modules',
    'modules': tool_modules[:10],
    'action': 'Create unified interface for tools and utilities',
    }
    )

    # Strategy 4: Dependency Injection
    unstable_high_coupling = [
    m for m, s in unstable_modules.items() if s['instability'] > 0.7
    ]
    if unstable_high_coupling:
    suggestions.append(
    {
    'priority': 'MEDIUM',
    'type': 'Dependency Injection',
    'title': 'Implement Dependency Injection',
    'description': f'Reduce coupling in unstable modules',
    'modules': unstable_high_coupling[:5],
    'action': 'Use dependency injection to invert control',
    }
    )

    return suggestions


def generate_focused_report(
    data, structure, stable_modules, unstable_modules, suggestions
):
    """Generate a focused architectural report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"focused_dependency_analysis_{timestamp}.md"

    report_content = f"""# 🔍 FOCUSED DEPENDENCY ANALYSIS REPORT

**Generated:** {datetime.now().isoformat()}
**Analysis Type:** Architectural Structure and Refactoring Guidance

## 📊 EXECUTIVE SUMMARY

- **Total Modules:** {len(data['modules'])}
- **Stable Modules (High Fan-In):** {len(stable_modules)}
- **High Coupling Modules:** {len(unstable_modules)}
- **Refactoring Opportunities:** {len(suggestions)}

## 🏗️ MODULE STRUCTURE OVERVIEW

"""

    for base, mods in sorted(structure.items()):
    report_content += f"### {base}/ Directory\n"
    report_content += f"- **Modules:** {len(mods)}\n"

        # Calculate directory-level metrics
    dir_complexity = sum(
    data['complexity'].get(m, {}).get('total_coupling', 0) for m in mods
    )
    avg_complexity = dir_complexity / len(mods) if mods else 0

    report_content += f"- **Total Coupling:** {dir_complexity}\n"
    report_content += f"- **Average Coupling:** {avg_complexity:.1f}\n\n"

    report_content += """## 🎯 STABLE MODULES (Interface Candidates)

These modules have high fan-in (many dependents) and should be considered for interface extraction:

| Module | Dependents | Dependencies | Instability | Recommendation |
|--------|------------|--------------|-------------|----------------|
"""

    for module, stats in sorted(
    stable_modules.items(), key=lambda x: x[1]['dependents'], reverse=True
    )[:10]:
    stability = (
    "Very Stable"
            if stats['instability'] < 0.2
            else "Stable" if stats['instability'] < 0.5 else "Moderate"
    )
    recommendation = "Extract Interface" if stats['dependents'] > 5 else "Monitor"

    report_content += f"| `{module}` | {stats['dependents']} | {stats['dependencies']} | {stats['instability']:.2f} | {recommendation} |\n"

    report_content += """

## ⚠️ HIGH COUPLING MODULES (Refactoring Priority)

These modules have excessive coupling and should be refactored:

| Module | Dependents | Dependencies | Total Coupling | Action Required |
|--------|------------|--------------|----------------|-----------------|
"""

    for module, stats in sorted(
    unstable_modules.items(), key=lambda x: x[1]['dependencies'], reverse=True
    )[:10]:
    total = stats['dependents'] + stats['dependencies']
    action = (
    "Split Module"
            if total > 20
            else "Reduce Dependencies" if stats['dependencies'] > 10 else "Monitor"
    )

    report_content += f"| `{module}` | {stats['dependents']} | {stats['dependencies']} | {total} | {action} |\n"

    report_content += "\\n## 🛠️ REFACTORING STRATEGIES\\n\\n"

    for i, suggestion in enumerate(suggestions, 1):
    priority_emoji = "🔴" if suggestion['priority'] == 'HIGH' else "🟡"
    report_content += f"### {priority_emoji} Strategy {i}: {suggestion['title']} ({suggestion['priority']})\n\n"
    report_content += f"**Type:** {suggestion['type']}\\n"
    report_content += f"**Description:** {suggestion['description']}\\n\\n"

        if suggestion.get('modules'):
    report_content += "**Target Modules:**\\n"
            for module in suggestion['modules'][:5]:  # Show first 5
    report_content += f"- `{module}`\\n"
            if len(suggestion['modules']) > 5:
    report_content += f"- ... and {len(suggestion['modules']) - 5} more\\n"
    report_content += "\\n"

    report_content += f"**Action:** {suggestion['action']}\\n\\n"

    report_content += """## 🎨 ARCHITECTURAL IMPROVEMENTS

### 1. 🏛️ Layer Architecture
```
┌─────────────────────────────────────────┐
│                API Layer                │  ← Entry points, facades
├─────────────────────────────────────────┤
│              Service Layer              │  ← Business logic
├─────────────────────────────────────────┤
│             Core Domain                 │  ← Stable interfaces
├─────────────────────────────────────────┤
│          Infrastructure Layer           │  ← I/O, utilities, tools
└─────────────────────────────────────────┘
```

### 2. 🔧 Dependency Injection Pattern
```python
# Instead of direct imports
class ArabicProcessor:
    def __init__(self, phonology_engine=None, morphology_engine=None):
    self.phonology = phonology_engine or DefaultPhonologyEngine()
    self.morphology = morphology_engine or DefaultMorphologyEngine()
```

### 3. 🎭 Facade Pattern for Tools
```python
# Unified interface for all tools
class ArabicNLPToolkit:
    def __init__(self):
    self._syntax_fixer = SyntaxFixer()
    self._validator = Validator()
    self._analyzer = Analyzer()

    def fix_syntax(self, files):
    return self._syntax_fixer.fix(files)

    def validate(self, files):
    return self._validator.validate(files)
```

### 4. 📦 Interface Segregation
```python
# Split large interfaces into focused ones
from abc import ABC, abstractmethod

class PhonologyProcessor(ABC):
    @abstractmethod
    def process_phonemes(self, text): pass

class MorphologyProcessor(ABC):
    @abstractmethod
    def analyze_morphology(self, text): pass
```

## 📈 SUCCESS METRICS

Track these metrics to measure refactoring success:

- **Coupling Reduction:** Target <10 total coupling per module
- **Interface Stability:** Keep instability <0.3 for core modules
- **Dependency Direction:** Dependencies should flow toward stable modules
- **Test Coverage:** Maintain >80% coverage during refactoring

## 🎯 IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
1. Extract interfaces for top 5 stable modules
2. Create facade for tool modules
3. Set up dependency injection framework

### Phase 2: Refactoring (Week 3-4)
1. Split highest coupling modules
2. Implement interface segregation
3. Add comprehensive tests

### Phase 3: Optimization (Week 5-6)
1. Fine-tune dependencies
2. Optimize module boundaries
3. Performance testing

---

*This analysis provides actionable guidance for improving the codebase architecture. Focus on high-priority items first for maximum impact.*
"""

    with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_content)

    return report_file


def main():
    """Main analysis function"""
    print("🔍 FOCUSED DEPENDENCY ANALYSIS")
    print("=" * 40)

    # Load dependency data
    data = load_dependency_data()
    if not data:
    return

    # Analyze core structure
    structure = analyze_core_structure(data)

    # Identify architectural patterns
    stable_modules, unstable_modules = identify_architectural_patterns(data)

    # Generate refactoring suggestions
    suggestions = suggest_refactoring_strategies(data, stable_modules, unstable_modules)

    # Generate focused report
    report_file = generate_focused_report(
    data, structure, stable_modules, unstable_modules, suggestions
    )

    print(f"\n🎉 FOCUSED ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"📋 Report Generated: {report_file}")
    print(f"🏗️ Stable Modules: {len(stable_modules)}")
    print(f"⚠️ High Coupling: {len(unstable_modules)}")
    print(f"🛠️ Refactoring Strategies: {len(suggestions)}")

    # Show top recommendations
    if suggestions:
    print(f"\n🔥 TOP PRIORITY:")
        for suggestion in suggestions[:2]:
    priority_emoji = "🔴" if suggestion['priority'] == 'HIGH' else "🟡"
    print(f"  {priority_emoji} {suggestion['title']}")


if __name__ == "__main__":
    main()
