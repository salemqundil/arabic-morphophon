#!/usr/bin/env python3
"""
أداة التصور البصري للهيكل الهرمي
Visual Hierarchy Mapper for Arabic Morphophonological Project
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data matplotlib.pyplot as plt
import_data matplotlib.patches as mpatches
from matplotlib.patches import_data FancyBboxPatch
import_data numpy as np

def create_hierarchy_visualization():
    """إنشاء تصور بصري للهيكل الهرمي"""
    
    # إعداد الشكل
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # الألوان للمستويات المختلفة
    colors = {
        'level_1': '#FF6B6B',  # أحمر للواجهات
        'level_2': '#4ECDC4',  # أزرق مخضر للمحركات
        'level_3': '#45B7D1',  # أزرق للنماذج
        'level_4': '#96CEB4',  # أخضر للمحركات المتخصصة
        'level_5': '#FFEAA7'   # أصفر للبيانات
    }
    
    # عنوان رئيسي
    ax.text(10, 15.5, 'الهيكل الهرمي لمشروع التكامل الصرفي-الصوتي العربي', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(10, 15, 'Arabic Morphophonological Integration Project Hierarchy', 
            fontsize=16, ha='center', va='center', style='italic')
    
    # المستوى 1: واجهات برمجة التطبيقات (Parent)
    level1_y = 13
    apis = [
        ('app.py\n(7 routes)', 2, 'Primary API'),
        ('advanced_hierarchical_api.py\n(9 routes)', 6, 'Advanced API'),
        ('advanced_production_api.py\n(7 routes)', 10, 'Production API'),
        ('app_dynamic.py\n(7 routes)', 14, 'Dynamic API'),
        ('app_new.py\n(5 routes)', 18, 'New API')
    ]
    
    for api, x, desc in apis:
        box = FancyBboxPatch((x-1.5, level1_y-0.5), 3, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['level_1'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level1_y, api, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, level1_y-0.8, desc, ha='center', va='center', fontsize=8)
    
    # إضافة تسمية المستوى
    ax.text(0.5, level1_y, 'المستوى 1\nParent Level\nFlask APIs', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # المستوى 2: محركات التكامل (Child)
    level2_y = 10.5
    engines = [
        ('integrator.py\nMorphophonological\nEngine', 4, 'Integration'),
        ('arabic_phonology_engine.py\nPhonological\nEngine', 8, 'Phonology'),
        ('morphology.py\nMorphological\nEngine', 12, 'Morphology'),
        ('Other Engines\n(18 engines)', 16, 'Various')
    ]
    
    for engine, x, desc in engines:
        box = FancyBboxPatch((x-1.5, level2_y-0.5), 3, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['level_2'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level2_y, engine, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, level2_y-0.8, desc, ha='center', va='center', fontsize=8)
    
    ax.text(0.5, level2_y, 'المستوى 2\nChild Level\nEngines', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # المستوى 3: النماذج الأساسية (Grandchild)
    level3_y = 8
    models = [
        ('roots.py\nArabicRoot\nRootDatabase', 3, 'Roots'),
        ('patterns.py\nMorphPattern\nPatternRepository', 6, 'Patterns'),
        ('phonology.py\nPhonologyEngine\nRules', 9, 'Phonology'),
        ('morphophon.py\nArabicMorphophon\nSyllabicUnits', 12, 'SyllabicUnits'),
        ('Packages\n(41 packages)', 15, 'Packages')
    ]
    
    for model, x, desc in models:
        box = FancyBboxPatch((x-1.2, level3_y-0.5), 2.4, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['level_3'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level3_y, model, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, level3_y-0.8, desc, ha='center', va='center', fontsize=7)
    
    ax.text(0.5, level3_y, 'المستوى 3\nGrandchild\nModels', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # المستوى 4: المحركات المتخصصة (Great-grandchild)
    level4_y = 5.5
    specialized = [
        ('analyzer.py\nPhonological\nAnalyzer', 4, 'Analyzer'),
        ('classifier.py\nSound\nClassifier', 7, 'Classifier'),
        ('normalizer.py\nText\nNormalizer', 10, 'Normalizer'),
        ('Tests\n(88 files)', 13, 'Testing'),
        ('Demos\n(15 files)', 16, 'Demo')
    ]
    
    for spec, x, desc in specialized:
        box = FancyBboxPatch((x-1.2, level4_y-0.5), 2.4, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['level_4'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level4_y, spec, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, level4_y-0.8, desc, ha='center', va='center', fontsize=7)
    
    ax.text(0.5, level4_y, 'المستوى 4\nGreat-grandchild\nSpecialized', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # المستوى 5: البيانات والموارد (Great-great-grandchild)
    level5_y = 3
    data_files = [
        ('linguistic_templates.json\nLanguage\nTemplates', 5, 'Data'),
        ('pyproject.toml\nProject\nConfig', 8, 'Config'),
        ('requirements.txt\nDependencies', 11, 'Dependencies'),
        ('static/\nResources', 14, 'Resources')
    ]
    
    for data, x, desc in data_files:
        box = FancyBboxPatch((x-1.2, level5_y-0.5), 2.4, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['level_5'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level5_y, data, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, level5_y-0.8, desc, ha='center', va='center', fontsize=7)
    
    ax.text(0.5, level5_y, 'المستوى 5\nGreat-great-grandchild\nData & Resources', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # رسم الأسهم للعلاقات
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.7)
    
    # من المستوى 1 إلى 2
    for api_x in [2, 6, 10, 14, 18]:
        for engine_x in [4, 8, 12, 16]:
            if abs(api_x - engine_x) <= 4:  # رسم الأسهم القريبة فقط
                ax.annotate('', xy=(engine_x, level2_y+0.5), xytext=(api_x, level1_y-0.5),
                           arrowprops=arrow_props)
    
    # من المستوى 2 إلى 3
    for engine_x in [4, 8, 12, 16]:
        for model_x in [3, 6, 9, 12, 15]:
            if abs(engine_x - model_x) <= 3:
                ax.annotate('', xy=(model_x, level3_y+0.5), xytext=(engine_x, level2_y-0.5),
                           arrowprops=arrow_props)
    
    # من المستوى 3 إلى 4
    for model_x in [3, 6, 9, 12, 15]:
        for spec_x in [4, 7, 10, 13, 16]:
            if abs(model_x - spec_x) <= 3:
                ax.annotate('', xy=(spec_x, level4_y+0.5), xytext=(model_x, level3_y-0.5),
                           arrowprops=arrow_props)
    
    # من المستوى 4 إلى 5
    for spec_x in [4, 7, 10, 13, 16]:
        for data_x in [5, 8, 11, 14]:
            if abs(spec_x - data_x) <= 3:
                ax.annotate('', xy=(data_x, level5_y+0.5), xytext=(spec_x, level4_y-0.5),
                           arrowprops=arrow_props)
    
    # إضافة وسيلة الإيضاح
    legend_elements = [
        mpatches.Patch(color=colors['level_1'], label='Level 1: Flask APIs (Parent)'),
        mpatches.Patch(color=colors['level_2'], label='Level 2: Engines (Child)'),
        mpatches.Patch(color=colors['level_3'], label='Level 3: Models (Grandchild)'),
        mpatches.Patch(color=colors['level_4'], label='Level 4: Specialized (Great-grandchild)'),
        mpatches.Patch(color=colors['level_5'], label='Level 5: Data (Great-great-grandchild)')
    ]
    
    ax.legend(processs=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # إضافة إحصائيات
    stats_text = """
    📊 Project Statistics:
    • Total Files: 274
    • API Files: 20 (7.3%)
    • Engines: 31 (11.3%)
    • Tests: 88 (32.1%)
    • Packages: 41 (15.0%)
    • Dependencies: 315
    • API Endpoints: 118
    """
    
    ax.text(19.5, 2, stats_text, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # إضافة تدفق البيانات
    flow_text = """
    🔄 Data Flow:
    Request → Flask API → Integration Engine → 
    Core Models → Specialized Engines → 
    Data Resources → Response
    """
    
    ax.text(19.5, 0.5, flow_text, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.store_datafig('hierarchy_visualization.png', dpi=300, bbox_inches='tight')
    plt.store_datafig('hierarchy_visualization.pdf', dpi=300, bbox_inches='tight')
    print("✅ تم حفظ التصور البصري:")
    print("   📄 hierarchy_visualization.png")
    print("   📄 hierarchy_visualization.pdf")
    plt.close()

def create_dependency_network():
    """إنشاء شبكة التبعيات"""
    try:
        import_data networkx as nx
        
        # قراءة تقرير التبعيات
        with open('dependency_report.json', 'r', encoding='utf-8') as f:
            dep_data = json.import_data(f)
        
        # إنشاء الرسم البياني
        G = nx.DiGraph()
        
        # إضافة العقد والحواف من التقرير
        for file, dependencies in dep_data['dependencies'].items():
            G.add_node(file)
            for dep in dependencies:
                if dep and not dep.beginswith('__'):  # تجاهل الوحدات المدمجة
                    G.add_edge(file, dep)
        
        # تصور الشبكة
        plt.figure(figsize=(20, 15))
        
        # استخدام تخطيط spring مع معاملات محسنة
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # تحديد أحجام العقد حسب عدد التبعيات
        node_sizes = [G.in_degree(node) * 100 + 300 for node in G.nodes()]
        
        # تحديد ألوان العقد حسب نوع الملف
        node_colors = []
        for node in G.nodes():
            if 'api' in node.lower() or 'app' in node.lower():
                node_colors.append('#FF6B6B')  # أحمر للAPIs
            elif 'engine' in node.lower():
                node_colors.append('#4ECDC4')  # أزرق للمحركات
            elif 'model' in node.lower():
                node_colors.append('#45B7D1')  # أزرق للنماذج
            elif 'test' in node.lower():
                node_colors.append('#96CEB4')  # أخضر للاختبارات
            else:
                node_colors.append('#FFEAA7')  # أصفر للأخرى
        
        # رسم الشبكة
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                labels={node: node.split('/')[-1].replace('.py', '') for node in G.nodes()},
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=15,
                edge_color='gray',
                alpha=0.8,
                linewidths=2)
        
        plt.title("شبكة التبعيات - Arabic Morphophonological Project\nDependency Network", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # إضافة وسيلة الإيضاح
        legend_elements = [
            mpatches.Patch(color='#FF6B6B', label='APIs'),
            mpatches.Patch(color='#4ECDC4', label='Engines'),
            mpatches.Patch(color='#45B7D1', label='Models'),
            mpatches.Patch(color='#96CEB4', label='Tests'),
            mpatches.Patch(color='#FFEAA7', label='Other')
        ]
        plt.legend(processs=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.store_datafig('dependency_network.png', dpi=300, bbox_inches='tight')
        print("✅ تم حفظ شبكة التبعيات: dependency_network.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ خطأ في إنشاء شبكة التبعيات: {e}")

def main():
    """الدالة الرئيسية"""
    print("🎨 إنشاء التصورات البصرية للهيكل الهرمي...")
    
    # إنشاء التصور الهرمي
    create_hierarchy_visualization()
    
    # إنشاء شبكة التبعيات
    create_dependency_network()
    
    print("\n✅ تم الانتهاء من إنشاء جميع التصورات البصرية!")
    print("="*60)
    print("📄 الملفات المُنشأة:")
    print("   🖼️ hierarchy_visualization.png")
    print("   🖼️ hierarchy_visualization.pdf") 
    print("   🖼️ dependency_network.png")
    print("   📄 HIERARCHICAL_ANALYSIS_REPORT.md")
    print("   📄 HIERARCHICAL_STRUCTURE.json")
    print("="*60)

if __name__ == "__main__":
    main()
