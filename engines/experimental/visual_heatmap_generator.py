#!/usr/bin/env python3
"""
🔥 VISUAL HEATMAP GENERATOR
================================================================
Creates visual heatmaps and charts for file analysis data,
    Usage:
  python visual_heatmap_generator.py
"""

import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os,
    class VisualHeatmapGenerator:
    """Generate visual heatmaps and charts for file analysis"""

    def __init__(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
    self.data = json.load(f)

    self.output_dir = f"analysis_visuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(self.output_dir, exist_ok=True)

        # Set up plotting style,
    plt.style.use('default')
    sns.set_palette("husl")

    def create_severity_heatmap(self):
    """Create severity vs fixability heatmap"""
    broken_files = self.data['heatmap_data']['broken_files']

        if not broken_files:
    print("No broken files to visualize")
    return

        # Prepare data,
    df = pd.DataFrame(broken_files)

        # Create heatmap data,
    heatmap_data = df.pivot_table(
    values='error_count',
    index='severity',
    columns='fixability',
    aggfunc='count',
    fill_value=0,
    )

        # Create the plot,
    plt.figure(figsize=(12, 8))
    sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    cbar_kws={'label': 'Number of Files'},
    )

    plt.title(
    'File Severity vs Fixability Heatmap\n(Color intensity = number of files)',
    fontsize=16,
    fontweight='bold',
    )
    plt.xlabel('Fixability Score (1=Hard to Fix, 10=Easy to Fix)', fontsize=12)
    plt.ylabel('Severity Score (1=Minor Issues, 10=Critical Issues)', fontsize=12)

    plt.tight_layout()
    plt.savefig(
    f"{self.output_dir}/severity_fixability_heatmap.png",
    dpi=300,
    bbox_inches='tight',
    )
    plt.close()

    print(
    f"✅ Severity heatmap saved to: {self.output_dir}/severity_fixability_heatmap.png"
    )

    def create_category_breakdown_chart(self):
    """Create category breakdown charts"""
    categories = self.data['module_categories']

        # Calculate stats for each category,
    category_stats = []
        for category, files in categories.items():
    broken_count = 0,
    total_count = len(files)

            for filepath in files:
                if filepath in self.data['files_data']:
                    if not self.data['files_data'][filepath]['syntax_analysis'][
    'syntax_valid'
    ]:
    broken_count += 1,
    if total_count > 0:
    success_rate = ((total_count - broken_count) / total_count) * 100,
    category_stats.append(
    {
    'category': category.replace('_', ' ').title(),
    'total': total_count,
    'broken': broken_count,
    'working': total_count - broken_count,
    'success_rate': success_rate,
    }
    )

        # Create subplots,
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Total files by category,
    df_stats = pd.DataFrame(category_stats)
    df_stats.plot(x='category', y='total', kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Total Files by Category', fontweight='bold')
    ax1.set_ylabel('Number of Files')
    ax1.tick_params(axis='x', rotation=45)

        # 2. Success rate by category,
    df_stats.plot(
    x='category', y='success_rate', kind='bar', ax=ax2, color='lightgreen'
    )
    ax2.set_title('Success Rate by Category', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 100)

        # 3. Broken vs Working files,
    categories_list = df_stats['category'].tolist()
    working_counts = df_stats['working'].tolist()
    broken_counts = df_stats['broken'].tolist()

    width = 0.35,
    x = np.arange(len(categories_list))

    ax3.bar(
    x - width / 2, working_counts, width, label='Working', color='lightgreen'
    )
    ax3.bar(x + width / 2, broken_counts, width, label='Broken', color='lightcoral')
    ax3.set_title('Working vs Broken Files by Category', fontweight='bold')
    ax3.set_ylabel('Number of Files')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories_list, rotation=45)
    ax3.legend()

        # 4. Error type distribution,
    error_counts = {}
        for filepath, file_data in self.data['files_data'].items():
            for error_type in file_data['syntax_analysis']['error_types']:
    error_counts[error_type] = error_counts.get(error_type, 0) + 1,
    if error_counts:
    error_types = list(error_counts.keys())
    error_values = list(error_counts.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))
    ax4.pie(error_values, labels=error_types, autopct='%1.1f%%', colors=colors)
    ax4.set_title('Error Type Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(
    f"{self.output_dir}/category_breakdown.png", dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(
    f"✅ Category breakdown saved to: {self.output_dir}/category_breakdown.png"
    )

    def create_priority_matrix(self):
    """Create priority matrix scatter plot"""
    broken_files = self.data['heatmap_data']['broken_files']

        if not broken_files:
    return,
    df = pd.DataFrame(broken_files)

        # Create scatter plot,
    plt.figure(figsize=(14, 10))

        # Color by category,
    categories = df['category'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, colors))

        for category in categories:
    cat_data = df[df['category'] == category]
    plt.scatter(
    cat_data['fixability'],
    cat_data['severity'],
    c=[category_colors[category]],
    label=category.replace('_', ' ').title(),
    s=cat_data['error_count'] * 20,  # Size by error count,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5,
    )

        # Add annotations for high priority files,
    high_priority = df[df['severity'] >= 6]
        for _, row in high_priority.iterrows():
    plt.annotate(
    os.path.basename(row['file']),
    (row['fixability'], row['severity']),
    xytext=(5, 5),
    textcoords='offset points',
    fontsize=8,
    alpha=0.8,
    )

    plt.xlabel('Fixability Score (Higher = Easier to Fix)', fontsize=12)
    plt.ylabel('Severity Score (Higher = More Critical)', fontsize=12)
    plt.title(
    'File Priority Matrix\n(Bubble size = number of errors)',
    fontsize=16,
    fontweight='bold',
    )

        # Add quadrant lines,
    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels,
    plt.text(
    2,
    8,
    'High Severity\nLow Fixability\n(Avoid)',
    ha='center',
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )
    plt.text(
    8,
    8,
    'High Severity\nHigh Fixability\n(Fix First)',
    ha='center',
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )
    plt.text(
    2,
    2,
    'Low Severity\nLow Fixability\n(Ignore)',
    ha='center',
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )
    plt.text(
    8,
    2,
    'Low Severity\nHigh Fixability\n(Quick Wins)',
    ha='center',
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
    f"{self.output_dir}/priority_matrix.png", dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f"✅ Priority matrix saved to: {self.output_dir}/priority_matrix.png")

    def create_top_files_chart(self):
    """Create chart of top problematic files"""
    rankings = self.data['severity_ranking']
    broken_files = [f for f in rankings if not f['syntax_valid']][:20]  # Top 20,
    if not broken_files:
    return,
    df = pd.DataFrame(broken_files)

        # Create horizontal bar chart,
    plt.figure(figsize=(14, 10))

        # Sort by priority score,
    df_sorted = df.sort_values('priority_score', ascending=True)

        # Create bars colored by category,
    categories = df_sorted['category'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, colors))

    bars = plt.barh(
    range(len(df_sorted)),
    df_sorted['priority_score'],
    color=[category_colors[cat] for cat in df_sorted['category']],
    )

        # Customize,
    plt.yticks(
    range(len(df_sorted)), [os.path.basename(f) for f in df_sorted['file']]
    )
    plt.xlabel('Priority Score (Higher = More Urgent)', fontsize=12)
    plt.title('Top 20 Most Critical Files to Fix', fontsize=16, fontweight='bold')

        # Add value labels on bars,
    for i, (bar, priority) in enumerate(zip(bars, df_sorted['priority_score'])):
    plt.text(
    bar.get_width() + 0.1,
    bar.get_y() + bar.get_height() / 2,
    f'{priority:.1f}',
    va='center',
    fontsize=8,
    )

        # Add legend,
    legend_elements = [
    plt.Rectangle(
    (0, 0),
    1,
    1,
    facecolor=category_colors[cat],
    label=cat.replace('_', ' ').title(),
    )
            for cat in categories
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
    f"{self.output_dir}/top_critical_files.png", dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(
    f"✅ Top critical files chart saved to: {self.output_dir}/top_critical_files.png"
    )

    def create_module_health_dashboard(self):
    """Create comprehensive module health dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # 1. Overall health gauge,
    total_files = self.data['heatmap_data']['total_files']
    broken_files = self.data['heatmap_data']['broken_count']
    success_rate = self.data['heatmap_data']['success_rate']

        # Create gauge chart,
    theta = np.linspace(0, np.pi, 100)
    r = 1,
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ax1.plot(x, y, 'k-', linewidth=2)
    ax1.fill_between(
    x,
    0,
    y,
    where=(theta <= np.pi * success_rate / 100),
    alpha=0.7,
    color='green',
    label=f'Working: {success_rate:.1f}%',
    )
    ax1.fill_between(
    x,
    0,
    y,
    where=(theta > np.pi * success_rate / 100),
    alpha=0.7,
    color='red',
    label=f'Broken: {100-success_rate:.1f}%',
    )

        # Add needle,
    needle_angle = np.pi * success_rate / 100,
    needle_x = 0.8 * np.cos(needle_angle)
    needle_y = 0.8 * np.sin(needle_angle)
    ax1.arrow(
    0,
    0,
    needle_x,
    needle_y,
    head_width=0.1,
    head_length=0.1,
    fc='black',
    ec='black',
    linewidth=3,
    )

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title(
    f'Overall Project Health\n{total_files} total files', fontweight='bold'
    )
    ax1.legend()
    ax1.axis('off')

        # 2. Error type trends,
    error_counts = {}
        for filepath, file_data in self.data['files_data'].items():
            for error_type in file_data['syntax_analysis']['error_types']:
    error_counts[error_type] = error_counts.get(error_type, 0) + 1,
    if error_counts:
    ax2.bar(error_counts.keys(), error_counts.values(), color='lightcoral')
    ax2.set_title('Error Types Distribution', fontweight='bold')
    ax2.set_ylabel('Number of Occurrences')
    ax2.tick_params(axis='x', rotation=45)

        # 3. Fixability distribution,
    fixability_scores = []
        for filepath, file_data in self.data['files_data'].items():
            if not file_data['syntax_analysis']['syntax_valid']:
    fixability_scores.append(file_data['fixability_score'])

        if fixability_scores:
    ax3.hist(
    fixability_scores,
    bins=10,
    alpha=0.7,
    color='skyblue',
    edgecolor='black',
    )
    ax3.set_title('Fixability Score Distribution', fontweight='bold')
    ax3.set_xlabel('Fixability Score (1=Hard, 10=Easy)')
    ax3.set_ylabel('Number of Files')
    ax3.axvline(
    x=np.mean(fixability_scores),
    color='red',
    linestyle='--',
    label=f'Mean: {np.mean(fixability_scores):.1f}',
    )
    ax3.legend()

        # 4. File size vs error correlation,
    file_sizes = []
    error_counts_per_file = []

        for filepath, file_data in self.data['files_data'].items():
            if not file_data['syntax_analysis']['syntax_valid']:
    file_sizes.append(file_data['code_metrics']['line_count'])
    error_counts_per_file.append(
    len(file_data['syntax_analysis']['errors'])
    )

        if file_sizes and error_counts_per_file:
    ax4.scatter(file_sizes, error_counts_per_file, alpha=0.6, color='orange')
    ax4.set_xlabel('File Size (Lines of Code)')
    ax4.set_ylabel('Number of Errors')
    ax4.set_title('File Size vs Error Count', fontweight='bold')

            # Add trend line,
    if len(file_sizes) > 1:
    z = np.polyfit(file_sizes, error_counts_per_file, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(file_sizes), p(sorted(file_sizes)), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(
    f"{self.output_dir}/module_health_dashboard.png",
    dpi=300,
    bbox_inches='tight',
    )
    plt.close()

    print(
    f"✅ Module health dashboard saved to: {self.output_dir}/module_health_dashboard.png"
    )

    def generate_all_visuals(self):
    """Generate all visualization charts"""
    print(f"🎨 Generating visual analysis charts...")
    print(f"📁 Output directory: {self.output_dir}")

        try:
    self.create_severity_heatmap()
    self.create_category_breakdown_chart()
    self.create_priority_matrix()
    self.create_top_files_chart()
    self.create_module_health_dashboard()

    print(f"\n🎉 All visualizations generated successfully!")
    print(f"📂 Check the '{self.output_dir}' directory for all charts")

        except Exception as e:
    print(f"❌ Error generating visuals: {e}")
            import traceback,
    traceback.print_exc()


def main():
    # Find the most recent analysis file,
    analysis_files = [
    f,
    for f in os.listdir('.')
        if f.startswith('file_analysis_') and f.endswith('.json')
    ]

    if not analysis_files:
    print(
    "❌ No analysis JSON file found. Run comprehensive_file_analysis.py first."
    )
    return

    # Use the most recent file,
    latest_file = sorted(analysis_files)[-1]
    print(f"📊 Using analysis file: {latest_file}")

    try:
    generator = VisualHeatmapGenerator(latest_file)
    generator.generate_all_visuals()
    except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("📦 Install with: pip install matplotlib seaborn pandas numpy")
    except Exception as e:
    print(f"❌ Error: {e}")
        import traceback,
    traceback.print_exc()


if __name__ == "__main__":
    main()
