#!/usr/bin/env python3
"""
🔁 CIRCULAR IMPORT DEPENDENCY ANALYZER
=====================================

Comprehensive tool to analyze Python module dependencies:
- Scans all Python files for import statements
- Builds dependency graph
- Detects circular references
- Suggests refactoring strategies
- Generates visual dependency map

Features:
- Static analysis (no code execution)
- Handles both direct and indirect circular imports
- Analyzes relative and absolute imports
- Provides actionable refactoring suggestions
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import json


class CircularImportAnalyzer:
    def __init__(self):
    self.imports = defaultdict(set)  # module -> set of imported modules
    self.reverse_imports = defaultdict(
    set
    )  # module -> set of modules that import it
    self.file_to_module = {}  # file path -> module name
    self.module_to_file = {}  # module name -> file path
    self.circular_chains = []
    self.all_files = []

    print("🔁 CIRCULAR IMPORT DEPENDENCY ANALYZER")
    print("=" * 50)

    def get_module_name(self, file_path):
    """Convert file path to module name"""
        try:
            # Convert to relative path from current directory
    rel_path = os.path.relpath(file_path)

            # Remove .py extension
            if rel_path.endswith('.py'):
    rel_path = rel_path[:-3]

            # Convert path separators to dots
    module_name = rel_path.replace(os.sep, '.')

            # Remove __init__ from module names
            if module_name.endswith('.__init__'):
    module_name = module_name[:-9]

    return module_name
        except Exception:
    return file_path

    def extract_imports_from_file(self, file_path):
    """Extract all import statements from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

            # Parse with AST
    tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Relative import - try to resolve it
    current_module = self.get_module_name(file_path)
    current_parts = current_module.split('.')

                            if node.level <= len(current_parts):
    base_parts = (
    current_parts[: -node.level]
                                    if node.level > 0
                                    else current_parts
    )
                                if node.module:
    full_module = '.'.join(base_parts + [node.module])
                                else:
    full_module = '.'.join(base_parts)
                                imports.add(full_module.split('.')[0])
                        else:
                            # Absolute import
                            imports.add(node.module.split('.')[0])

    return imports

        except Exception as e:
    print(f"⚠️ Error parsing {file_path}: {e}")
    return set()

    def scan_python_files(self):
    """Scan all Python files in the current directory"""
    print("🔍 Step 1: Scanning Python files...")

    python_files = []
        for root, dirs, files in os.walk('.'):
            # Skip common directories to avoid
    skip_dirs = {
    '__pycache__',
    '.git',
    'backup',
    'quarantine',
    'build',
    'dist',
    '.venv',
    'venv',
    }
    dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
    file_path = os.path.join(root, file)
    python_files.append(file_path)

    self.all_files = python_files
    print(f"Found {len(python_files)} Python files")
    return python_files

    def build_dependency_graph(self):
    """Build the complete dependency graph"""
    print("🏗️ Step 2: Building dependency graph...")

    python_files = self.scan_python_files()

        # Build module mappings
        for file_path in python_files:
    module_name = self.get_module_name(file_path)
    self.file_to_module[file_path] = module_name
    self.module_to_file[module_name] = file_path

        # Extract imports for each file
        for file_path in python_files:
    module_name = self.file_to_module[file_path]
            imports = self.extract_imports_from_file(file_path)

            # Filter to only include local modules
    local_imports = set()
            for imp in imports:
                # Check if this import corresponds to a local module
                for local_module in self.module_to_file.keys():
                    if local_module.startswith(imp) or imp in local_module:
    local_imports.add(local_module)
    break
                # Also check exact matches
                if imp in self.module_to_file:
    local_imports.add(imp)

    self.imports[module_name] = local_imports

            # Build reverse mapping
            for imported_module in local_imports:
    self.reverse_imports[imported_module].add(module_name)

    print(f"Built dependency graph with {len(self.imports)} modules")

    def detect_circular_imports(self):
    """Detect circular import chains using DFS"""
    print("🔍 Step 3: Detecting circular imports...")

    visited = set()
    rec_stack = set()
    path = []

        def dfs(module, path, visited, rec_stack):
    """DFS to detect cycles"""
    visited.add(module)
    rec_stack.add(module)
    path.append(module)

            for neighbor in self.imports.get(module, set()):
                if neighbor not in visited:
                    if dfs(neighbor, path, visited, rec_stack):
    return True
                elif neighbor in rec_stack:
                    # Found a cycle
    cycle_start = path.index(neighbor)
    cycle = path[cycle_start:] + [neighbor]
    self.circular_chains.append(cycle)
    return True

    path.pop()
    rec_stack.remove(module)
    return False

        # Check all modules
        for module in self.imports.keys():
            if module not in visited:
    dfs(module, [], visited, rec_stack)

        # Remove duplicate cycles
    unique_cycles = []
        for cycle in self.circular_chains:
            # Normalize cycle (start with alphabetically first module)
    min_idx = cycle.index(min(cycle[:-1]))
    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            if normalized not in unique_cycles:
    unique_cycles.append(normalized)

    self.circular_chains = unique_cycles
    print(f"Detected {len(self.circular_chains)} circular import chains")

    def analyze_module_complexity(self):
    """Analyze module complexity and coupling"""
    complexity_analysis = {}

        for module in self.imports.keys():
            imports_count = len(self.imports.get(module, set()))
            imported_by_count = len(self.reverse_imports.get(module, set()))

            # Calculate coupling metrics
    fan_out = imports_count  # Dependencies
    fan_in = imported_by_count  # Dependents

            # Instability metric (I = fan_out / (fan_in + fan_out))
    total_coupling = fan_in + fan_out
    instability = fan_out / total_coupling if total_coupling > 0 else 0

    complexity_analysis[module] = {
    'imports': imports_count,
    'imported_by': imported_by_count,
    'fan_in': fan_in,
    'fan_out': fan_out,
    'instability': instability,
    'total_coupling': total_coupling,
    }

    return complexity_analysis

    def suggest_refactoring(self):
    """Generate refactoring suggestions"""
    suggestions = []
    complexity = self.analyze_module_complexity()

        # Analyze circular imports
        if self.circular_chains:
    suggestions.append(
    {
    'type': 'circular_imports',
    'priority': 'HIGH',
    'title': 'Break Circular Import Chains',
    'description': f'Found {len(self.circular_chains)} circular import chains that need resolution',
    'actions': [
    'Extract common functionality into separate modules',
    'Use dependency injection instead of direct imports',
    'Move imports inside functions where possible',
    'Consider using import hooks or lazy imports',
    ],
    }
    )

        # Identify highly coupled modules
    high_coupling = [
    (m, c) for m, c in complexity.items() if c['total_coupling'] > 5
    ]
        if high_coupling:
    high_coupling.sort(key=lambda x: x[1]['total_coupling'], reverse=True)
    suggestions.append(
    {
    'type': 'high_coupling',
    'priority': 'MEDIUM',
    'title': 'Reduce Module Coupling',
    'description': f'Found {len(high_coupling)} modules with high coupling',
    'modules': [m for m, c in high_coupling[:5]],
    'actions': [
    'Split large modules into smaller, focused modules',
    'Extract interfaces to reduce direct dependencies',
    'Use dependency inversion principle',
    'Consider facade pattern for complex subsystems',
    ],
    }
    )

        # Identify unstable modules
    unstable = [
    (m, c)
            for m, c in complexity.items()
            if c['instability'] > 0.8 and c['imported_by'] > 2
    ]
        if unstable:
    suggestions.append(
    {
    'type': 'instability',
    'priority': 'MEDIUM',
    'title': 'Stabilize Core Modules',
    'description': f'Found {len(unstable)} unstable modules that are heavily depended upon',
    'modules': [m for m, c in unstable[:5]],
    'actions': [
    'Reduce outgoing dependencies from stable modules',
    'Extract stable interfaces',
    'Move volatile code to dependent modules',
    ],
    }
    )

    return suggestions

    def generate_dependency_graph_data(self):
    """Generate data for dependency graph visualization"""
    nodes = []
    edges = []
    complexity = self.analyze_module_complexity()

        # Create nodes
        for module in self.imports.keys():
    size = complexity[module]['total_coupling'] * 2 + 10
    color = (
    'red'
                if any(module in chain for chain in self.circular_chains)
                else 'blue'
    )

    nodes.append(
    {
    'id': module,
    'label': module.split('.')[-1],  # Show only last part
    'size': size,
    'color': color,
    'title': f"{module}\\nImports: {complexity[module]['imports']}\\nImported by: {complexity[module]['imported_by']}",
    }
    )

        # Create edges
        for module, imports in self.imports.items():
            for imported_module in imports:
                if imported_module in self.module_to_file:  # Only local modules
    edges.append(
    {'from': module, 'to': imported_module, 'arrows': 'to'}
    )

    return {'nodes': nodes, 'edges': edges}

    def generate_report(self):
    """Generate comprehensive analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"circular_import_analysis_{timestamp}.md"

    complexity = self.analyze_module_complexity()
    suggestions = self.suggest_refactoring()
    graph_data = self.generate_dependency_graph_data()

    report_content = f"""# 🔁 CIRCULAR IMPORT DEPENDENCY ANALYSIS

**Generated:** {datetime.now().isoformat()}
**Files Analyzed:** {len(self.all_files)}
**Modules Found:** {len(self.imports)}

## 📊 SUMMARY

- **Total Python Files:** {len(self.all_files)}
- **Local Modules:** {len(self.imports)}
- **Circular Import Chains:** {len(self.circular_chains)}
- **High Coupling Modules:** {len([m for m, c in complexity.items() if c['total_coupling'] > 5])}

## 🔴 CIRCULAR IMPORT CHAINS

"""

        if self.circular_chains:
            for i, chain in enumerate(self.circular_chains, 1):
    report_content += f"### Chain {i}\n\n```\n"
                for j in range(len(chain) - 1):
    report_content += f"{chain[j]} → {chain[j+1]}\n"
    report_content += "```\n\n"
        else:
    report_content += "✅ **No circular imports detected!**\n\n"

    report_content += """## 📈 MODULE COMPLEXITY ANALYSIS

| Module | Imports | Imported By | Total Coupling | Instability |
|--------|---------|-------------|----------------|-------------|
"""

        # Sort by total coupling
    sorted_modules = sorted(
    complexity.items(), key=lambda x: x[1]['total_coupling'], reverse=True
    )
        for module, stats in sorted_modules[:20]:  # Top 20
    report_content += f"| `{module}` | {stats['imports']} | {stats['imported_by']} | {stats['total_coupling']} | {stats['instability']:.2f} |\n"

    report_content += "\n## 🛠️ REFACTORING SUGGESTIONS\n\n"

        for suggestion in suggestions:
    priority_emoji = "🔴" if suggestion['priority'] == 'HIGH' else "🟡"
    report_content += f"### {priority_emoji} {suggestion['title']} ({suggestion['priority']})\n\n"
    report_content += f"{suggestion['description']}\n\n"

            if 'modules' in suggestion:
    report_content += "**Affected Modules:**\n"
                for module in suggestion['modules']:
    report_content += f"- `{module}`\n"
    report_content += "\n"

    report_content += "**Recommended Actions:**\n"
            for action in suggestion['actions']:
    report_content += f"- {action}\n"
    report_content += "\n"

    report_content += f"""## 📦 DEPENDENCY GRAPH DATA

The following data can be used with graph visualization tools:

```json
{json.dumps(graph_data, indent=2)}
```

## 🔧 TOOLS AND TECHNIQUES

### Breaking Circular Imports:

1. **Dependency Injection:**
   ```python
   # Instead of direct import
   def process_data(processor=None):
       if processor is None:
           from .processor import DataProcessor
    processor = DataProcessor()
   ```

2. **Lazy Imports:**
   ```python
   def get_processor():
       from .processor import DataProcessor
       return DataProcessor()
   ```

3. **Interface Extraction:**
   ```python
   # Create abstract base class
   from abc import ABC, abstractmethod

   class ProcessorInterface(ABC):
       @abstractmethod
       def process(self, data): pass
   ```

4. **Module Restructuring:**
   - Extract common functionality into separate modules
   - Move shared constants/types to dedicated modules
   - Use configuration objects instead of direct imports

## ✅ NEXT STEPS

1. **Address circular imports** (if any) as highest priority
2. **Reduce coupling** in highly connected modules
3. **Extract interfaces** for stable module boundaries
4. **Consider architectural patterns** like dependency injection
5. **Use tools** like `import-linter` for ongoing monitoring

---

*Analysis completed successfully. Use this report to guide module refactoring efforts.*
"""

        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_content)

        # Also save JSON data for tools
    json_file = f"dependency_graph_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(
    {
    'modules': list(self.imports.keys()),
    'dependencies': {k: list(v) for k, v in self.imports.items()},
    'circular_chains': self.circular_chains,
    'complexity': complexity,
    'graph_data': graph_data,
    },
    f,
    indent=2,
    )

    return report_file, json_file

    def run(self):
    """Execute the complete analysis"""
    print("🔍 Starting circular import analysis...")

        # Build dependency graph
    self.build_dependency_graph()

        # Detect circular imports
    self.detect_circular_imports()

        # Generate report
    print("📋 Generating analysis report...")
    report_file, json_file = self.generate_report()

        # Summary
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"📁 Files analyzed: {len(self.all_files)}")
    print(f"📦 Modules found: {len(self.imports)}")
    print(f"🔴 Circular chains: {len(self.circular_chains)}")
    print(f"📋 Report: {report_file}")
    print(f"📊 Data: {json_file}")

        if self.circular_chains:
    print(f"\n⚠️ CIRCULAR IMPORTS DETECTED:")
            for i, chain in enumerate(self.circular_chains, 1):
    chain_str = " → ".join(chain)
    print(f"  {i}. {chain_str}")
        else:
    print(f"\n✅ No circular imports detected!")

    return report_file, json_file


def main():
    """Main entry point"""
    analyzer = CircularImportAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
