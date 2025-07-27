#!/usr/bin/env python3
"""
أداة تحليل الهيكل الهرمي لمشروع التكامل الصرفي-الصوتي العربي
Project Hierarchy Analysis Tool for Arabic Morphophonological Integration

هذه الأداة تحلل:
1. هيكل الملفات والمجلدات
2. العلاقات بين الوحدات
3. خريطة الاستيرادات
4. التبعيات بين المحركات
5. نقاط الدخول (Entry Points)
6. واجهات برمجة التطبيقات (APIs)
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data ast
import_data json
import_data os
import_data re
from collections import_data defaultdict, deque
from pathlib import_data Path
from typing import_data Dict, List, Set, Any
import_data networkx as nx
import_data matplotlib.pyplot as plt
from dataclasses import_data dataclass, field

@dataclass
class FileInfo:
    """معلومات ملف Python"""
    path: str
    name: str
    type: str  # 'module', 'package', 'script', 'api', 'test'
    import_datas: List[str] = field(default_factory=list)
    from_import_datas: Dict[str, List[str]] = field(default_factory=dict)
    store_datas: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    flask_routes: List[str] = field(default_factory=list)
    parent: str = ""
    children: List[str] = field(default_factory=list)
    description: str = ""

class ProjectHierarchyAnalyzer:
    """محلل الهيكل الهرمي للمشروع"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.files_info: Dict[str, FileInfo] = {}
        self.dependency_graph = nx.DiGraph()
        self.api_endpoints: Dict[str, List[str]] = defaultdict(list)
        self.engines: Dict[str, List[str]] = defaultdict(list)
        
    def analyze_project(self):
        """تحليل المشروع بالكامل"""
        print("🔄 بدء تحليل المشروع...")
        
        # 1. جمع معلومات الملفات
        self._collect_files_info()
        
        # 2. تحليل الاستيرادات والتبعيات
        self._analyze_dependencies()
        
        # 3. تحليل المحركات والواجهات
        self._analyze_engines_and_apis()
        
        # 4. بناء الهيكل الهرمي
        self._build_hierarchy()
        
        # 5. إنشاء التقارير
        self._generate_reports()
        
        print("✅ تم الانتهاء من التحليل!")
        
    def _collect_files_info(self):
        """جمع معلومات الملفات"""
        print("📂 جمع معلومات الملفات...")
        
        for py_file in self.project_root.rglob("*.py"):
            # تجاهل ملفات البيئة الافتراضية
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            relative_path = py_file.relative_to(self.project_root)
            file_info = self._analyze_python_file(py_file)
            file_info.path = str(relative_path)
            
            self.files_info[str(relative_path)] = file_info
            
    def _analyze_python_file(self, file_path: Path) -> FileInfo:
        """تحليل ملف Python واحد"""
        file_info = FileInfo(
            path=str(file_path),
            name=file_path.stem,
            type=self._determine_file_type(file_path)
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # تحليل AST
            tree = ast.parse(content)
            
            # استخراج الاستيرادات
            self._extract_import_datas(tree, file_info)
            
            # استخراج الكلاسات والدوال
            self._extract_classes_and_functions(tree, file_info)
            
            # استخراج Flask routes
            self._extract_flask_routes(content, file_info)
            
            # استخراج الوصف
            file_info.description = self._extract_description(content)
            
        except Exception as e:
            print(f"⚠️ خطأ في تحليل {file_path}: {e}")
            
        return file_info
        
    def _determine_file_type(self, file_path: Path) -> str:
        """تحديد نوع الملف"""
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()
        
        if name == '__init__.py':
            return 'package'
        elif 'test' in name or 'test' in parent:
            return 'test'
        elif 'app' in name or 'api' in name or 'web' in name:
            return 'api'
        elif 'engine' in name or 'engine' in parent:
            return 'engine'
        elif 'demo' in name:
            return 'demo'
        elif 'setup' in name or 'config' in name:
            return 'config'
        else:
            return 'module'
            
    def _extract_import_datas(self, tree: ast.AST, file_info: FileInfo):
        """استخراج الاستيرادات"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_info.import_datas.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                file_info.from_import_datas[module] = names
                
    def _extract_classes_and_functions(self, tree: ast.AST, file_info: FileInfo):
        """استخراج الكلاسات والدوال"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                file_info.classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                # تجاهل الدوال الداخلية
                if not any(isinstance(parent, (ast.ClassDef, ast.FunctionDef)) 
                          for parent in ast.walk(tree)):
                    file_info.functions.append(node.name)
                    
    def _extract_flask_routes(self, content: str, file_info: FileInfo):
        """استخراج Flask routes"""
        # البحث عن decorators للـ Flask
        route_patterns = [
            r"@app\.route\(['\"]([^'\"]+)['\"]",
            r"@api\.route\(['\"]([^'\"]+)['\"]",
            r"@bp\.route\(['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in route_patterns:
            matches = re.findall(pattern, content)
            file_info.flask_routes.extend(matches)
            
    def _extract_description(self, content: str) -> str:
        """استخراج الوصف من docstring"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.beginswith('"""') or stripped.beginswith("'''"):
                # العثور على نهاية docstring
                for j in range(i, len(lines)):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        if j > i:  # docstring متعدد الأسطر
                            return '\n'.join(lines[i:j+1]).strip('"""').strip("'''").strip()
                        else:  # docstring سطر واحد
                            return stripped.strip('"""').strip("'''").strip()
        return ""
        
    def _analyze_dependencies(self):
        """تحليل التبعيات والاستيرادات"""
        print("🔗 تحليل التبعيات...")
        
        for file_path, file_info in self.files_info.items():
            # إضافة العقدة للرسم البياني
            self.dependency_graph.add_node(file_path, **file_info.__dict__)
            
            # إضافة الحواف للتبعيات
            for import_dataed_module in file_info.import_datas:
                self._add_dependency_edge(file_path, import_dataed_module)
                
            for module, names in file_info.from_import_datas.items():
                self._add_dependency_edge(file_path, module)
                
    def _add_dependency_edge(self, from_file: str, to_module: str):
        """إضافة حافة تبعية"""
        # محاولة العثور على الملف المقابل للوحدة
        target_file = self._resolve_module_to_file(to_module)
        if target_file and target_file in self.files_info:
            self.dependency_graph.add_edge(from_file, target_file)
            
    def _resolve_module_to_file(self, module_name: str) -> str:
        """تحويل اسم الوحدة إلى مسار ملف"""
        # تحويل نقاط إلى مسارات
        parts = module_name.split('.')
        
        # البحث في الملفات الموجودة
        for file_path in self.files_info.keys():
            path_parts = Path(file_path).parts
            
            # مطابقة أجزاء المسار
            if len(parts) <= len(path_parts):
                match = True
                for i, part in enumerate(parts):
                    if i < len(path_parts) and part not in path_parts[i]:
                        match = False
                        break
                if match:
                    return file_path
                    
        return ""
        
    def _analyze_engines_and_apis(self):
        """تحليل المحركات والواجهات"""
        print("⚙️ تحليل المحركات والواجهات...")
        
        for file_path, file_info in self.files_info.items():
            # تصنيف المحركات
            if file_info.type == 'engine' or 'engine' in file_info.name.lower():
                category = self._categorize_engine(file_info)
                self.engines[category].append(file_path)
                
            # تصنيف APIs
            if file_info.flask_routes:
                self.api_endpoints[file_path] = file_info.flask_routes
                
    def _categorize_engine(self, file_info: FileInfo) -> str:
        """تصنيف المحرك"""
        name = file_info.name.lower()
        path = file_info.path.lower()
        
        if 'morpho' in name or 'morpho' in path:
            return 'Morphological Engine'
        elif 'phono' in name or 'phono' in path:
            return 'Phonological Engine'
        elif 'syllabic_unit' in name or 'syllabic_unit' in path:
            return 'SyllabicAnalysis Engine'
        elif 'pattern' in name or 'pattern' in path:
            return 'Pattern Engine'
        elif 'root' in name or 'root' in path:
            return 'Root Engine'
        elif 'integrat' in name or 'integrat' in path:
            return 'Integration Engine'
        else:
            return 'Other Engine'
            
    def _build_hierarchy(self):
        """بناء الهيكل الهرمي"""
        print("🏗️ بناء الهيكل الهرمي...")
        
        # تحديد العلاقات الهرمية بناءً على المسارات
        for file_path, file_info in self.files_info.items():
            path_obj = Path(file_path)
            
            # تحديد الوالد
            if len(path_obj.parts) > 1:
                parent_path = str(path_obj.parent / "__init__.py")
                if parent_path in self.files_info:
                    file_info.parent = parent_path
                    self.files_info[parent_path].children.append(file_path)
                    
    def _generate_reports(self):
        """إنشاء التقارير"""
        print("📊 إنشاء التقارير...")
        
        # تقرير الهيكل الهرمي
        self._generate_hierarchy_report()
        
        # تقرير التبعيات
        self._generate_dependency_report()
        
        # تقرير المحركات
        self._generate_engines_report()
        
        # تقرير APIs
        self._generate_api_report()
        
        # رسم بياني للتبعيات
        self._generate_dependency_graph()
        
    def _generate_hierarchy_report(self):
        """تقرير الهيكل الهرمي"""
        report = {
            "project_structure": {},
            "summary": {
                "total_files": len(self.files_info),
                "file_types": defaultdict(int),
                "packages": [],
                "entry_points": []
            }
        }
        
        # إحصائيات أنواع الملفات
        for file_info in self.files_info.values():
            report["summary"]["file_types"][file_info.type] += 1
            
        # العثور على نقاط الدخول
        entry_points = [
            path for path, info in self.files_info.items()
            if info.type in ['api', 'demo'] or 
            any(route for route in info.flask_routes)
        ]
        report["summary"]["entry_points"] = entry_points
        
        # الحزم الرئيسية
        packages = [
            path for path, info in self.files_info.items()
            if info.type == 'package'
        ]
        report["summary"]["packages"] = packages
        
        # بناء الهيكل الشجري
        report["project_structure"] = self._build_tree_structure()
        
        with open('project_hierarchy_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
    def _build_tree_structure(self) -> Dict:
        """بناء الهيكل الشجري"""
        tree = {}
        
        for file_path, file_info in self.files_info.items():
            path_parts = Path(file_path).parts
            current = tree
            
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # إضافة معلومات الملف
            filename = path_parts[-1]
            current[filename] = {
                "type": file_info.type,
                "classes": file_info.classes,
                "functions": file_info.functions,
                "flask_routes": file_info.flask_routes,
                "description": file_info.description[:100] + "..." if len(file_info.description) > 100 else file_info.description
            }
            
        return tree
        
    def _generate_dependency_report(self):
        """تقرير التبعيات"""
        report = {
            "dependencies": {},
            "circular_dependencies": [],
            "dependency_levels": {},
            "most_dependent": [],
            "least_dependent": []
        }
        
        # تحليل التبعيات
        for file_path, file_info in self.files_info.items():
            deps = list(file_info.import_datas) + list(file_info.from_import_datas.keys())
            report["dependencies"][file_path] = deps
            
        # العثور على التبعيات الدائرية
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            report["circular_dependencies"] = cycles
        except:
            report["circular_dependencies"] = []
            
        # ترتيب حسب عدد التبعيات
        dependency_counts = [
            (path, len(list(self.dependency_graph.predecessors(path))))
            for path in self.dependency_graph.nodes()
        ]
        dependency_counts.sort(key=lambda x: x[1], reverse=True)
        
        report["most_dependent"] = dependency_counts[:10]
        report["least_dependent"] = dependency_counts[-10:]
        
        with open('dependency_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
    def _generate_engines_report(self):
        """تقرير المحركات"""
        report = {
            "engines_by_category": dict(self.engines),
            "engine_details": {},
            "integration_points": []
        }
        
        # تفاصيل كل محرك
        for category, engine_files in self.engines.items():
            for engine_file in engine_files:
                if engine_file in self.files_info:
                    info = self.files_info[engine_file]
                    report["engine_details"][engine_file] = {
                        "category": category,
                        "classes": info.classes,
                        "functions": info.functions,
                        "description": info.description
                    }
                    
        # نقاط التكامل
        integration_files = [
            path for path, info in self.files_info.items()
            if 'integrat' in info.name.lower() or 'integrat' in info.path.lower()
        ]
        report["integration_points"] = integration_files
        
        with open('engines_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
    def _generate_api_report(self):
        """تقرير واجهات برمجة التطبيقات"""
        report = {
            "api_files": dict(self.api_endpoints),
            "total_endpoints": sum(len(routes) for routes in self.api_endpoints.values()),
            "endpoint_details": {}
        }
        
        # تفاصيل نقاط النهاية
        for file_path, routes in self.api_endpoints.items():
            if file_path in self.files_info:
                info = self.files_info[file_path]
                report["endpoint_details"][file_path] = {
                    "routes": routes,
                    "functions": info.functions,
                    "description": info.description
                }
                
        with open('api_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
    def _generate_dependency_graph(self):
        """إنشاء رسم بياني للتبعيات"""
        try:
            plt.figure(figsize=(20, 15))
            
            # تحديد المواضع
            pos = nx.spring_layout(self.dependency_graph, k=1, iterations=50)
            
            # رسم العقد
            node_colors = []
            for node in self.dependency_graph.nodes():
                if node in self.files_info:
                    file_type = self.files_info[node].type
                    if file_type == 'api':
                        node_colors.append('red')
                    elif file_type == 'engine':
                        node_colors.append('blue')
                    elif file_type == 'package':
                        node_colors.append('green')
                    elif file_type == 'test':
                        node_colors.append('orange')
                    else:
                        node_colors.append('gray')
                else:
                    node_colors.append('lightgray')
                    
            nx.draw(self.dependency_graph, pos, 
                   node_color=node_colors,
                   node_size=300,
                   with_labels=True,
                   labels={node: Path(node).stem for node in self.dependency_graph.nodes()},
                   font_size=8,
                   arrows=True,
                   arrowsize=10,
                   edge_color='gray',
                   alpha=0.7)
                   
            plt.title("مخطط التبعيات - Arabic Morphophonological Project")
            plt.legend(['API', 'Engine', 'Package', 'Test', 'Other'], 
                      loc='upper right')
            plt.tight_layout()
            plt.store_datafig('dependency_graph.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ تم حفظ الرسم البياني في dependency_graph.png")
            
        except Exception as e:
            print(f"⚠️ خطأ في إنشاء الرسم البياني: {e}")
            
    def print_summary(self):
        """طباعة ملخص التحليل"""
        print("\n" + "="*80)
        print("📊 ملخص تحليل الهيكل الهرمي للمشروع")
        print("="*80)
        
        print(f"\n📂 إجمالي الملفات: {len(self.files_info)}")
        
        # إحصائيات أنواع الملفات
        type_counts = defaultdict(int)
        for info in self.files_info.values():
            type_counts[info.type] += 1
            
        print("\n📋 أنواع الملفات:")
        for file_type, count in sorted(type_counts.items()):
            print(f"   {file_type}: {count}")
            
        # المحركات
        print(f"\n⚙️ إجمالي المحركات: {sum(len(engines) for engines in self.engines.values())}")
        for category, engines in self.engines.items():
            print(f"   {category}: {len(engines)}")
            
        # APIs
        total_endpoints = sum(len(routes) for routes in self.api_endpoints.values())
        print(f"\n🌐 ملفات API: {len(self.api_endpoints)}")
        print(f"   إجمالي نقاط النهاية: {total_endpoints}")
        
        # التبعيات
        print(f"\n🔗 إجمالي التبعيات: {self.dependency_graph.number_of_edges()}")
        
        # العثور على الملفات الرئيسية
        entry_points = [
            path for path, info in self.files_info.items()
            if info.type in ['api', 'demo'] or info.flask_routes
        ]
        
        print(f"\n🚪 نقاط الدخول الرئيسية: {len(entry_points)}")
        for entry in entry_points[:10]:  # أول 10
            info = self.files_info.get(entry, FileInfo(path=entry, name="", type=""))
            print(f"   📄 {Path(entry).name} ({info.type})")
            if info.flask_routes:
                print(f"      Routes: {', '.join(info.flask_routes[:3])}{'...' if len(info.flask_routes) > 3 else ''}")
                
        print("\n" + "="*80)
        print("✅ تم حفظ التقارير التفصيلية:")
        print("   📄 project_hierarchy_report.json")
        print("   📄 dependency_report.json") 
        print("   📄 engines_report.json")
        print("   📄 api_report.json")
        print("   🖼️ dependency_graph.png")
        print("="*80)

def main():
    """الدالة الرئيسية"""
    project_root = "."
    
    analyzer = ProjectHierarchyAnalyzer(project_root)
    analyzer.analyze_project()
    analyzer.print_summary()

if __name__ == "__main__":
    main()
