#!/usr/bin/env python3
"""
🏗️ Arabic NLP Engine Architecture Visualizer
Complete Classes Tree and Engine Pipeline Analysis

This module provides a comprehensive visualization of the Arabic NLP engine
architecture from phonology through noun pluralization, showing the complete
inheritance hierarchy and processing pipeline.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data import_datalib.util
import_data inspect
import_data json
import_data os
import_data sys
from dataclasses import_data dataclass
from pathlib import_data Path
from typing import_data Any, Dict, List, Optional, Type

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@dataclass
class EngineInfo:
    """Information about an NLP engine"""
    name: str
    path: str
    engine_class: Optional[Type] = None
    base_classes: List[str] = None
    methods: List[str] = None
    dependencies: List[str] = None
    description: str = ""
    models: List[str] = None

class EngineArchitectureAnalyzer:
    """Analyzes and visualizes the complete engine architecture"""
    
    def __init__(self):
        self.engines_path = Path("engines/nlp")
        self.discovered_engines = {}
        self.architecture_tree = {}
        
    def discover_all_engines(self) -> Dict[str, EngineInfo]:
        """Discover all available engines in the system"""
        
        engines = {}
        
        if not self.engines_path.exists():
            print(f"❌ Engines path not found: {self.engines_path}")
            return engines
        
        for engine_dir in self.engines_path.iterdir():
            if engine_dir.is_dir() and not engine_dir.name.beginswith('__'):
                engine_file = engine_dir / "engine.py"
                if engine_file.exists():
                    try:
                        engine_info = self._analyze_engine(engine_dir.name, engine_file)
                        engines[engine_dir.name] = engine_info
                    except Exception as e:
                        print(f"⚠️ Failed to analyze {engine_dir.name}: {e}")
        
        return engines
    
    def _analyze_engine(self, engine_name: str, engine_file: Path) -> EngineInfo:
        """Analyze a specific engine file"""
        
        # Read engine file content
        try:
            with open(engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read {engine_file}: {e}")
            content = ""
        
        # Extract basic information
        engine_info = EngineInfo(
            name=engine_name,
            path=str(engine_file),
            base_classes=[],
            methods=[],
            dependencies=[],
            models=[]
        )
        
        # Try to import_data and inspect the engine
        try:
            spec = import_datalib.util.spec_from_file_location(f"{engine_name}_engine", engine_file)
            if spec and spec.import_dataer:
                module = import_datalib.util.module_from_spec(spec)
                spec.import_dataer.exec_module(module)
                
                # Find engine classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name.endswith('Engine') and not name.beginswith('Base'):
                        engine_info.engine_class = obj
                        engine_info.base_classes = [base.__name__ for base in obj.__mro__[1:]]
                        engine_info.methods = [method for method, _ in inspect.getmembers(obj, inspect.ismethod)]
                        break
                        
        except Exception as e:
            print(f"Could not import_data {engine_name}: {e}")
        
        # Extract description from docstring
        if '"""' in content:
            try:
                begin = content.find('"""') + 3
                end = content.find('"""', begin)
                if end > begin:
                    engine_info.description = content[begin:end].strip()[:200] + "..."
            except:
                pass
        
        # Find model files
        models_dir = engine_file.parent / "models"
        if models_dir.exists():
            engine_info.models = [f.stem for f in models_dir.glob("*.py") if not f.name.beginswith('__')]
        
        return engine_info
    
    def create_processing_pipeline(self) -> List[str]:
        """Define the typical processing pipeline order"""
        return [
            "phonology",          # 1. Phonological analysis
            "phoneme",           # 2. Phoneme segmentation  
            "phoneme_advanced",  # 3. Advanced phoneme analysis
            "phonological",      # 4. Phonological rules
            "syllabic_unit",          # 5. SyllabicUnit segmentation
            "morphology",        # 6. Morphological analysis
            "frozen_root",       # 7. Root extraction/analysis
            "weight",            # 8. Pattern/weight analysis
            "derivation",        # 9. Derivational morphology
            "inflection",        # 10. Inflectional morphology (verbs)
            "particles",         # 11. Grammatical particles
            "grammatical_particles", # 12. Advanced particles
            "full_pipeline"      # 13. Complete pipeline integration
        ]
    
    def visualize_architecture(self):
        """Create comprehensive visualization of the engine architecture"""
        
        print("🏗️ ARABIC NLP ENGINE ARCHITECTURE ANALYSIS")
        print("=" * 80)
        
        # Discover all engines
        engines = self.discover_all_engines()
        pipeline_order = self.create_processing_pipeline()
        
        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"   Total Engines Found: {len(engines)}")
        print(f"   Engine Directories: {list(engines.keys())}")
        
        # Show processing pipeline
        print(f"\n🔄 PROCESSING PIPELINE ORDER:")
        print("-" * 50)
        
        for i, engine_name in enumerate(pipeline_order, 1):
            if engine_name in engines:
                engine = engines[engine_name]
                status = "✅ Available"
                class_name = engine.engine_class.__name__ if engine.engine_class else "Unknown"
                models_count = len(engine.models) if engine.models else 0
            else:
                status = "❌ Missing"
                class_name = "N/A"
                models_count = 0
            
            print(f"   {i:2d}. {engine_name:<20} → {class_name:<25} [{models_count} models] {status}")
        
        # Show detailed engine information
        print(f"\n🏷️ DETAILED ENGINE ANALYSIS:")
        print("=" * 80)
        
        for engine_name in pipeline_order:
            if engine_name in engines:
                self._print_engine_details(engines[engine_name])
        
        # Show classes hierarchy
        print(f"\n🌳 CLASSES INHERITANCE TREE:")
        print("=" * 80)
        self._print_inheritance_tree(engines)
        
        # Show engine dependencies
        print(f"\n🔗 ENGINE INTERDEPENDENCIES:")
        print("=" * 80)
        self._analyze_dependencies(engines)
        
        return engines
    
    def _print_engine_details(self, engine: EngineInfo):
        """Print detailed information about an engine"""
        
        print(f"\n🎯 {engine.name.upper()} ENGINE")
        print("-" * 50)
        print(f"   Path: {engine.path}")
        
        if engine.engine_class:
            print(f"   Main Class: {engine.engine_class.__name__}")
            print(f"   Base Classes: {' → '.join(engine.base_classes) if engine.base_classes else 'None'}")
        
        if engine.description:
            print(f"   Description: {engine.description}")
        
        if engine.models:
            print(f"   Models ({len(engine.models)}): {', '.join(engine.models)}")
        
        # Show key methods
        if engine.engine_class:
            methods = []
            for name, method in inspect.getmembers(engine.engine_class, inspect.isfunction):
                if not name.beginswith('_'):
                    methods.append(name)
            
            if methods:
                print(f"   Public Methods ({len(methods)}): {', '.join(methods[:8])}")
                if len(methods) > 8:
                    print(f"                                    ... and {len(methods) - 8} more")
    
    def _print_inheritance_tree(self, engines: Dict[str, EngineInfo]):
        """Print the inheritance hierarchy tree"""
        
        # Group engines by their base classes
        inheritance_map = {}
        
        for engine_name, engine in engines.items():
            if engine.engine_class:
                class_name = engine.engine_class.__name__
                base_classes = engine.base_classes
                
                if base_classes:
                    primary_base = base_classes[0] if base_classes else "object"
                    if primary_base not in inheritance_map:
                        inheritance_map[primary_base] = []
                    inheritance_map[primary_base].append((engine_name, class_name))
                else:
                    if "object" not in inheritance_map:
                        inheritance_map["object"] = []
                    inheritance_map["object"].append((engine_name, class_name))
        
        # Print inheritance tree
        for base_class, engines_list in inheritance_map.items():
            print(f"\n📁 {base_class}")
            for engine_name, class_name in engines_list:
                print(f"   ├── {class_name} ({engine_name})")
        
        # Show common base classes
        common_bases = {}
        for engine_name, engine in engines.items():
            if engine.base_classes:
                for base in engine.base_classes:
                    if base not in common_bases:
                        common_bases[base] = []
                    common_bases[base].append(engine_name)
        
        print(f"\n🔗 COMMON BASE CLASSES:")
        for base_class, engine_list in common_bases.items():
            if len(engine_list) > 1:
                print(f"   {base_class}: {', '.join(engine_list)}")
    
    def _analyze_dependencies(self, engines: Dict[str, EngineInfo]):
        """Analyze dependencies between engines"""
        
        dependencies = {}
        
        # Read engine files to find import_datas
        for engine_name, engine in engines.items():
            try:
                with open(engine.path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find import_datas from other engines
                engine_import_datas = []
                for other_engine in engines.keys():
                    if other_engine != engine_name:
                        if f"engines.nlp.{other_engine}" in content or f"from engines.nlp.{other_engine}" in content:
                            engine_import_datas.append(other_engine)
                
                if engine_import_datas:
                    dependencies[engine_name] = engine_import_datas
                    
            except Exception as e:
                print(f"Could not analyze dependencies for {engine_name}: {e}")
        
        # Print dependency graph
        if dependencies:
            for engine, deps in dependencies.items():
                print(f"   {engine} depends on: {', '.join(deps)}")
        else:
            print("   No direct engine dependencies found (engines are well-isolated)")
    
    def generate_api_endpoints(self, engines: Dict[str, EngineInfo]):
        """Generate Flask API endpoints for each engine"""
        
        print(f"\n🌐 FLASK API ENDPOINTS:")
        print("=" * 80)
        
        # Base endpoints
        print(f"📍 SYSTEM ENDPOINTS:")
        print(f"   GET  /api/nlp/status              → System health check")
        print(f"   GET  /api/nlp/engines             → List all engines")
        print(f"   GET  /api/nlp/pipeline            → Processing pipeline info")
        
        # Engine-specific endpoints
        print(f"\n📍 ENGINE-SPECIFIC ENDPOINTS:")
        for engine_name in engines.keys():
            print(f"   POST /api/nlp/{engine_name}/analyze    → Main analysis endpoint")
            print(f"   GET  /api/nlp/{engine_name}/info       → Engine information")
            print(f"   GET  /api/nlp/{engine_name}/models     → Available models")
            print(f"   POST /api/nlp/{engine_name}/reimport_data     → Hot reimport_data engine")
        
        # Pipeline endpoints
        print(f"\n📍 PIPELINE ENDPOINTS:")
        print(f"   POST /api/nlp/pipeline/full       → Complete pipeline analysis")
        print(f"   POST /api/nlp/pipeline/custom     → Custom pipeline (specify engines)")
        print(f"   POST /api/nlp/pipeline/batch      → Batch processing")
        
        # Utility endpoints
        print(f"\n📍 UTILITY ENDPOINTS:")
        print(f"   GET  /api/nlp/config              → System configuration")
        print(f"   GET  /api/nlp/metrics             → Performance metrics")
        print(f"   GET  /api/nlp/cache/stats         → Cache statistics")

def create_usage_examples():
    """Create usage examples for the engine architecture"""
    
    print(f"\n💻 USAGE EXAMPLES:")
    print("=" * 80)
    
    # Individual engine usage
    print(f"🔧 INDIVIDUAL ENGINE USAGE:")
    print(f"""
# Phonology Engine
from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
phonology = PhonologyEngine()
result = phonology.analyze("كتاب")  # → phonemes, syllabic_units

# SyllabicUnit Engine  
from engines.nlp.syllabic_unit.engine import_data SyllabicUnitEngine
syllabic_unit = SyllabicUnitEngine()
result = syllabic_unit.analyze("مدرسة")  # → syllabic_unit segmentation

# Morphology Engine
from engines.nlp.morphology.engine import_data MorphologyEngine
morphology = MorphologyEngine()
result = morphology.analyze("الكتاب")  # → root, pattern, affixes

# Inflection Engine
from engines.nlp.inflection.engine import_data InflectionEngine
inflection = InflectionEngine()
result = inflection.conjugate(('ك','ت','ب'), 'present', 3, 'masc', 'singular')
""")
    
    # Pipeline usage
    print(f"🔄 PIPELINE USAGE:")
    print(f"""
# Complete Pipeline
from engines.nlp.full_pipeline.engine import_data FullPipelineEngine
pipeline = FullPipelineEngine()
result = pipeline.analyze("يكتب الطالب الواجب", 
                         target_engines=['phonology', 'syllabic_unit', 'morphology'],
                         enable_parallel=True)

# Custom Pipeline
engines_to_use = ['phonology', 'syllabic_unit', 'inflection']
result = pipeline.analyze("كتب", target_engines=engines_to_use)
""")
    
    # Flask API usage
    print(f"🌐 FLASK API USAGE:")
    print(f"""
# Single Engine Analysis
curl -X POST http://localhost:5000/api/nlp/phonology/analyze \\
     -H "Content-Type: application/json" \\
     -d '{{"text": "كتاب", "detailed": true}}'

# Pipeline Analysis
curl -X POST http://localhost:5000/api/nlp/pipeline/full \\
     -H "Content-Type: application/json" \\
     -d '{{"text": "يدرس الطالب", "engines": ["phonology", "morphology"]}}'

# Engine Information
curl http://localhost:5000/api/nlp/engines
""")

def main():
    """Main function to demonstrate the complete architecture"""
    
    analyzer = EngineArchitectureAnalyzer()
    
    # Analyze architecture
    engines = analyzer.visualize_architecture()
    
    # Generate API information
    analyzer.generate_api_endpoints(engines)
    
    # Show usage examples
    create_usage_examples()
    
    print(f"\n✅ ARCHITECTURE ANALYSIS COMPLETE!")
    print(f"📊 Found {len(engines)} engines in the processing pipeline")
    print(f"🎯 Complete Arabic NLP system ready for production deployment")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
