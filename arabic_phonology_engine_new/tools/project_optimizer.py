"""
Arabic Phonology Engine - Project Optimization Script
Clean up duplicates, organize files, and optimize structure
"""

import shutil
from pathlib import Path
from typing import List, Dict
import hashlib

class ProjectOptimizer:
    """Optimize and clean up the project structure."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.duplicates_found: List[str] = []
        self.cleaned_files: List[str] = []
        
    def optimize_project(self) -> None:
        """Execute complete project optimization."""
        print("🔧 OPTIMIZING ARABIC PHONOLOGY ENGINE")
        print("=" * 50)
        
        self._remove_duplicates()
        self._clean_empty_directories()
        self._organize_test_files()
        self._validate_structure()
        
        print("\n✅ OPTIMIZATION COMPLETE!")
        self._generate_report()
    
    def _remove_duplicates(self) -> None:
        """Remove duplicate files based on content hash."""
        print("\n🗑️  Removing duplicate files...")
        
        file_hashes: Dict[str, Path] = {}
        duplicates: List[Path] = []
        
        for file_path in self.project_root.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if content_hash in file_hashes:
                        duplicates.append(file_path)
                        print(f"  🔍 Duplicate found: {file_path.name}")
                    else:
                        file_hashes[content_hash] = file_path
                        
                except Exception as e:
                    print(f"  ⚠️  Could not process {file_path}: {e}")
        
        # Remove duplicates
        for duplicate in duplicates:
            try:
                duplicate.unlink()
                self.duplicates_found.append(str(duplicate))
                print(f"  ✅ Removed: {duplicate.name}")
            except Exception as e:
                print(f"  ❌ Failed to remove {duplicate}: {e}")
    
    def _clean_empty_directories(self) -> None:
        """Remove empty directories."""
        print("\n📁 Cleaning empty directories...")
        
        for dir_path in list(self.project_root.rglob("*")):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print(f"  🗑️  Removed empty directory: {dir_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Could not remove {dir_path}: {e}")
    
    def _organize_test_files(self) -> None:
        """Organize test files into proper directories."""
        print("\n🧪 Organizing test files...")
        
        test_patterns = {
            "unit": ["test_engine", "test_analyzer", "test_phoneme", "test_core"],
            "integration": ["test_api", "test_pipeline", "test_workflow"],
            "performance": ["test_performance", "test_load", "test_benchmark"]
        }
        
        for test_file in self.project_root.glob("test_*.py"):
            if test_file.is_file():
                moved = False
                for category, patterns in test_patterns.items():
                    if any(pattern in test_file.stem for pattern in patterns):
                        target_dir = self.project_root / "tests" / category
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / test_file.name
                        
                        if not target_path.exists():
                            shutil.move(str(test_file), str(target_path))
                            print(f"  📋 Moved {test_file.name} to {category}/")
                            moved = True
                            break
                
                if not moved:
                    # Default to unit tests
                    target_dir = self.project_root / "tests" / "unit"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / test_file.name
                    
                    if not target_path.exists():
                        shutil.move(str(test_file), str(target_path))
                        print(f"  📋 Moved {test_file.name} to unit/ (default)")
    
    def _validate_structure(self) -> None:
        """Validate the final project structure."""
        print("\n✅ Validating project structure...")
        
        required_dirs = [
            "src/arabic_phonology/core",
            "src/arabic_phonology/analysis", 
            "src/arabic_phonology/data",
            "src/arabic_phonology/utils",
            "src/arabic_phonology/web",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "docs",
            "scripts",
            "tools",
            "config"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ❌ Missing: {dir_path}")
                
        required_files = [
            "src/arabic_phonology/__init__.py",
            "src/arabic_phonology/core/__init__.py",
            "src/arabic_phonology/analysis/__init__.py",
            "src/arabic_phonology/data/__init__.py",
            "src/arabic_phonology/utils/__init__.py",
            "src/arabic_phonology/web/__init__.py",
            "tests/__init__.py",
            "tests/conftest.py",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ Missing: {file_path}")
    
    def _generate_report(self) -> None:
        """Generate optimization report."""
        print("\n📊 OPTIMIZATION REPORT")
        print("-" * 30)
        print(f"Duplicates removed: {len(self.duplicates_found)}")
        print(f"Files cleaned: {len(self.cleaned_files)}")
        
        if self.duplicates_found:
            print("\n🗑️  Removed duplicates:")
            for duplicate in self.duplicates_found[:10]:  # Show first 10
                print(f"  - {Path(duplicate).name}")
            if len(self.duplicates_found) > 10:
                print(f"  ... and {len(self.duplicates_found) - 10} more")


def main() -> None:
    """Main optimization function."""
    optimizer = ProjectOptimizer("arabic_phonology_engine_new")
    try:
        optimizer.optimize_project()
        print("\n🎉 SUCCESS: Project structure optimized!")
        print("Ready for development and deployment.")
    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")


if __name__ == "__main__":
    main()
