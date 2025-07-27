#!/usr/bin/env python3
"""
📊 VISUAL DEPENDENCY GRAPH GENERATOR
===================================

Creates visual representations of module dependencies:
- HTML interactive graph using vis.js
- ASCII text graph for terminal viewing
- Dependency matrix visualization
- Module clustering analysis
"""

import json
import html
from pathlib import Path
from datetime import datetime


def load_dependency_data():
    """Load the most recent dependency analysis data"""
    json_files = list(Path('.').glob('dependency_graph_data_*.json'))
    if not json_files:
        print("❌ No dependency data found. Run circular_import_analyzer.py first.")
        return None

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    with open(latest_file, 'r') as f:
        return json.load(f)


def generate_html_graph(data):
    """Generate interactive HTML graph using vis.js"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"dependency_graph_{timestamp}.html"

    # Filter to most important modules to avoid clutter
    complexity = data['complexity']
    important_modules = {
        m for m, stats in complexity.items() if stats.get('total_coupling', 0) > 0
    }

    # Prepare nodes and edges for vis.js
    nodes = []
    edges = []

    for module in important_modules:
        stats = complexity.get(module, {})
        size = min(50, max(10, stats.get('total_coupling', 0) * 3))

        # Color coding
        if stats.get('imported_by', 0) > 10:
            color = '#ff4444'  # High fan-in (red)
        elif stats.get('imports', 0) > 5:
            color = '#ffaa00'  # High fan-out (orange)
        elif stats.get('total_coupling', 0) > 0:
            color = '#4444ff'  # Normal (blue)
        else:
            color = '#888888'  # Isolated (gray)

        # Module category for grouping
        if 'core' in module:
            group = 1
        elif 'test' in module:
            group = 2
        elif 'tool' in module:
            group = 3
        elif any(x in module for x in ['fix', 'syntax', 'fixer']):
            group = 4
        else:
            group = 0

        nodes.append(
            {
                'id': module,
                'label': module.split('.')[-1][:20],  # Show last part, truncated
                'title': f"{module}\\nImports: {stats.get('imports', 0)}\\nImported by: {stats.get('imported_by', 0)}",
                'size': size,
                'color': color,
                'group': group,
            }
        )

    # Add edges
    for module, imports in data['dependencies'].items():
        if module in important_modules:
            for imported in imports:
                if imported in important_modules:
                    edges.append(
                        {
                            'from': module,
                            'to': imported,
                            'arrows': 'to',
                            'color': {'opacity': 0.6},
                        }
                    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Module Dependency Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #mynetworkid {{ width: 100%; height: 800px; border: 1px solid lightgray; }}
        .controls {{ margin: 10px 0; }}
        .legend {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
        .legend-item {{ margin: 5px 0; }}
        .color-box {{ display: inline-block; width: 20px; height: 20px; margin-right: 10px; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <h1>🔁 Module Dependency Graph</h1>
    <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>
    <p><strong>Modules:</strong> {len(nodes)} | <strong>Dependencies:</strong> {len(edges)}</p>

    <div class="legend">
        <h3>Legend:</h3>
        <div class="legend-item"><span class="color-box" style="background: #ff4444;"></span>High Fan-In (Many Dependents)</div>
        <div class="legend-item"><span class="color-box" style="background: #ffaa00;"></span>High Fan-Out (Many Dependencies)</div>
        <div class="legend-item"><span class="color-box" style="background: #4444ff;"></span>Normal Coupling</div>
        <div class="legend-item"><span class="color-box" style="background: #888888;"></span>Isolated Module</div>
    </div>

    <div class="controls">
        <button onclick="network.fit()">Fit to Screen</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="exportNetwork()">Export as PNG</button>
    </div>

    <div id="mynetworkid"></div>

    <div id="info">
        <h3>Click on a node to see details</h3>
        <div id="nodeInfo"></div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes, indent=2)});
        var edges = new vis.DataSet({json.dumps(edges, indent=2)});

        var container = document.getElementById('mynetworkid');
        var data = {{ nodes: nodes, edges: edges }};

        var options = {{
            groups: {{
                0: {{ color: {{ background: '#97C2FC', border: '#2B7CE9' }} }},
                1: {{ color: {{ background: '#FFAB91', border: '#FF5722' }} }},
                2: {{ color: {{ background: '#C8E6C9', border: '#4CAF50' }} }},
                3: {{ color: {{ background: '#FFE082', border: '#FFC107' }} }},
                4: {{ color: {{ background: '#F8BBD9', border: '#E91E63' }} }}
            }},
            physics: {{
                stabilization: {{ iterations: 100 }},
                barnesHut: {{ gravitationalConstant: -2000, springConstant: 0.001, springLength: 200 }}
            }},
            interaction: {{ hover: true }},
            nodes: {{
                borderWidth: 2,
                shadow: true,
                font: {{ size: 14, color: '#000000' }}
            }},
            edges: {{
                width: 2,
                shadow: true,
                smooth: {{ type: 'continuous' }}
            }}
        }};

        var network = new vis.Network(container, data, options);

        // Click event
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                document.getElementById('nodeInfo').innerHTML =
                    '<h4>' + node.id + '</h4>' +
                    '<p>' + node.title.replace(/\\\\n/g, '<br>') + '</p>';
            }}
        }});

        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}

        function exportNetwork() {{
            var canvas = document.querySelector('#mynetworkid canvas');
            var link = document.createElement('a');
            link.download = 'dependency_graph.png';
            link.href = canvas.toDataURL();
            link.click();
        }}
    </script>
</body>
</html>"""

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_file


def generate_ascii_graph(data):
    """Generate ASCII text representation of dependencies"""
    complexity = data['complexity']
    dependencies = data['dependencies']

    # Get top modules by coupling
    top_modules = sorted(
        [
            (m, stats)
            for m, stats in complexity.items()
            if stats.get('total_coupling', 0) > 0
        ],
        key=lambda x: x[1]['total_coupling'],
        reverse=True,
    )[:20]

    ascii_graph = []
    ascii_graph.append("📊 ASCII DEPENDENCY GRAPH (Top 20 Modules)")
    ascii_graph.append("=" * 60)
    ascii_graph.append("")

    for i, (module, stats) in enumerate(top_modules):
        coupling = stats.get('total_coupling', 0)
        imports = stats.get('imports', 0)
        imported_by = stats.get('imported_by', 0)

        # Visual representation
        bar_length = min(40, coupling * 2)
        bar = "█" * bar_length + "░" * (40 - bar_length)

        ascii_graph.append(f"{i+1:2}. {module[:35]:<35}")
        ascii_graph.append(f"    [{bar}] {coupling:3}")
        ascii_graph.append(
            f"    ↑{imported_by:2} dependents | ↓{imports:2} dependencies"
        )
        ascii_graph.append("")

    return "\\n".join(ascii_graph)


def generate_dependency_matrix(data):
    """Generate dependency matrix visualization"""
    dependencies = data['dependencies']
    complexity = data['complexity']

    # Get top modules
    top_modules = [
        m
        for m, stats in sorted(
            complexity.items(),
            key=lambda x: x[1].get('total_coupling', 0),
            reverse=True,
        )[:15]
    ]

    matrix = []
    matrix.append("📋 DEPENDENCY MATRIX (Top 15 Modules)")
    matrix.append("=" * 80)
    matrix.append("")

    # Header
    header = (
        "Module".ljust(25) + " | " + " ".join(f"{i:2}" for i in range(len(top_modules)))
    )
    matrix.append(header)
    matrix.append("-" * len(header))

    # Module index
    matrix.append("Module Index:")
    for i, module in enumerate(top_modules):
        matrix.append(f"{i:2}. {module}")
    matrix.append("")

    # Matrix rows
    for i, module in enumerate(top_modules):
        row = f"{module[:23]:<23} | "
        for j, target in enumerate(top_modules):
            if target in dependencies.get(module, []):
                row += " ●"  # Dependency exists
            elif i == j:
                row += " ■"  # Self
            else:
                row += " ·"  # No dependency
        matrix.append(row)

    matrix.append("")
    matrix.append("Legend: ● = imports, ■ = self, · = no dependency")

    return "\\n".join(matrix)


def generate_cluster_analysis(data):
    """Analyze module clusters and suggest groupings"""
    dependencies = data['dependencies']
    complexity = data['complexity']

    # Simple clustering based on naming patterns
    clusters = {}
    for module in data['modules']:
        if '.' in module:
            base = module.split('.')[0]
        else:
            # Cluster by naming pattern
            if any(word in module.lower() for word in ['test', 'spec']):
                base = 'testing'
            elif any(word in module.lower() for word in ['fix', 'repair', 'syntax']):
                base = 'tools'
            elif any(
                word in module.lower() for word in ['arabic', 'phonology', 'morphology']
            ):
                base = 'nlp'
            else:
                base = 'misc'

        if base not in clusters:
            clusters[base] = []
        clusters[base].append(module)

    analysis = []
    analysis.append("🔍 MODULE CLUSTER ANALYSIS")
    analysis.append("=" * 40)
    analysis.append("")

    for cluster_name, modules in sorted(clusters.items()):
        if not modules:
            continue

        total_coupling = sum(
            complexity.get(m, {}).get('total_coupling', 0) for m in modules
        )
        avg_coupling = total_coupling / len(modules) if modules else 0

        analysis.append(f"📦 {cluster_name.upper()} Cluster")
        analysis.append(f"   Modules: {len(modules)}")
        analysis.append(f"   Total Coupling: {total_coupling}")
        analysis.append(f"   Average Coupling: {avg_coupling:.1f}")

        # Show top modules in cluster
        cluster_modules = [
            (m, complexity.get(m, {}).get('total_coupling', 0)) for m in modules
        ]
        cluster_modules.sort(key=lambda x: x[1], reverse=True)

        analysis.append("   Top modules:")
        for m, c in cluster_modules[:3]:
            if c > 0:
                analysis.append(f"     • {m}: {c} connections")
        analysis.append("")

    return "\\n".join(analysis)


def main():
    """Main function to generate all visualizations"""
    print("📊 VISUAL DEPENDENCY GRAPH GENERATOR")
    print("=" * 45)

    # Load data
    data = load_dependency_data()
    if not data:
        return

    # Generate interactive HTML graph
    print("🌐 Generating interactive HTML graph...")
    html_file = generate_html_graph(data)
    print(f"   ✅ Created: {html_file}")

    # Generate ASCII graph
    print("📄 Generating ASCII dependency graph...")
    ascii_graph = generate_ascii_graph(data)
    print(ascii_graph)

    # Generate dependency matrix
    print("\\n📋 Generating dependency matrix...")
    matrix = generate_dependency_matrix(data)
    print(matrix)

    # Generate cluster analysis
    print("\\n🔍 Generating cluster analysis...")
    cluster_analysis = generate_cluster_analysis(data)
    print(cluster_analysis)

    # Save text outputs to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = f"dependency_analysis_{timestamp}.txt"

    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(ascii_graph + "\\n\\n")
        f.write(matrix + "\\n\\n")
        f.write(cluster_analysis)

    print(f"\\n🎉 VISUALIZATION COMPLETE!")
    print("=" * 35)
    print(f"🌐 Interactive Graph: {html_file}")
    print(f"📄 Text Analysis: {text_file}")
    print(f"💡 Open {html_file} in browser for interactive exploration")


if __name__ == "__main__":
    main()
