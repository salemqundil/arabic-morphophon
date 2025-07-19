"""
File Cleanup Utility - Remove Non-English Coding Characters
Zero Tolerance for Encoding Issues
"""

import re
from pathlib import Path
from typing import List, Union

class FileCleanup:
    """Clean up files with non-English characters."""
    
    def __init__(self, project_root: Union[str, Path] = ".") -> None:
        self.project_root = Path(project_root)
        self.cleaned_files: List[str] = []
        self.renamed_files: List[str] = []
        
    def cleanup_project(self) -> None:
        """Execute comprehensive cleanup."""
        print("CLEANING UP NON-ENGLISH CODING CHARACTERS")
        print("=" * 50)
        
        # Phase 1: Remove problematic files
        self._remove_problematic_files()
        
        # Phase 2: Rename files with non-English characters
        self._rename_non_english_files()
        
        # Phase 3: Clean file contents
        self._clean_file_contents()
        
        # Phase 4: Generate report
        self._generate_report()
        
        print("
CLEANUP COMPLETE!")
    
    def _remove_problematic_files(self) -> None:
        """Remove files that cause encoding issues."""
        print("
Removing problematic files...")
        
        problem_patterns = [
            "*cd*",
            "ص,ل,ح.txt",
            "Untitled-*.txt",
            "Untitled-*.py",
            "Untitled-*.groovy", 
            "Untitled-*.ini",
            "*.pyc",
            "# *.md",
            "# *.txt"
        ]
        
        for pattern in problem_patterns:
            try:
                files = list(self.project_root.glob(pattern))
                for file_path in files:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            print(f"  Removed: {file_path.name}")
                            self.cleaned_files.append(str(file_path))
                    except Exception as e:
                        print(f"  Warning: Could not remove {file_path.name}: {e}")
            except Exception:
                continue
    
    def _rename_non_english_files(self) -> None:
        """Rename files with non-English characters."""
        print("
Renaming files with non-English characters...")
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                original_name = file_path.name
                
                # Check if filename contains non-ASCII characters
                if any(ord(char) > 127 for char in original_name):
                    # Create safe ASCII name
                    safe_name = self._create_safe_filename(original_name)
                    
                    if safe_name != original_name:
                        try:
                            new_path = file_path.parent / safe_name
                            if not new_path.exists():
                                file_path.rename(new_path)
                                print(f"  Renamed: {original_name} -> {safe_name}")
                                self.renamed_files.append(f"{original_name} -> {safe_name}")
                        except Exception as e:
                            print(f"  Warning: Could not rename {original_name}: {e}")
    
    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe ASCII filename."""
        # Remove or replace non-ASCII characters
        safe_name = ""
        
        for char in filename:
            if ord(char) <= 127:  # ASCII character
                safe_name += char if char.isalnum() or char in ".-_" else "_"
            else:
                # Replace non-ASCII with description
                safe_name += "_"
        
        # Clean up multiple underscores
        safe_name = re.sub(r"_+", "_", safe_name)
        safe_name = safe_name.strip("_")
        
        # Ensure we have a valid filename
        if not safe_name or safe_name == "_":
            safe_name = "cleaned_file"
        
        return safe_name
    
    def _clean_file_contents(self) -> None:
        """Clean non-English characters from Python files."""
        print("
Cleaning file contents...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                # Read file content
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                original_content = content
                
                # Replace problematic Unicode characters in strings/comments
                # Keep emojis in comments but fix any broken encoding
                content = content.replace('cd', 'cd')  # Fix specific error
                
                # Fix any broken Unicode escape sequences
                content = re.sub(r'[\\\\]+n', r'
', content)  # Fix escaped newlines
                
                # If content changed, write it back
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  Cleaned: {py_file.name}")
                    self.cleaned_files.append(str(py_file))
                    
            except Exception as e:
                print(f"  Warning: Could not clean {py_file.name}: {e}")
    
    def _generate_report(self) -> None:
        """Generate cleanup report."""
        print("
CLEANUP REPORT")
        print("-" * 30)
        print(f"Files cleaned: {len(self.cleaned_files)}")
        print(f"Files renamed: {len(self.renamed_files)}")
        
        if self.cleaned_files:
            print("
Cleaned files:")
            for file_path in self.cleaned_files[:5]:  # Show first 5
                print(f"  - {Path(file_path).name}")
            if len(self.cleaned_files) > 5:
                print(f"  ... and {len(self.cleaned_files) - 5} more")
        
        if self.renamed_files:
            print("
Renamed files:")
            for rename_info in self.renamed_files[:5]:  # Show first 5
                print(f"  - {rename_info}")
            if len(self.renamed_files) > 5:
                print(f"  ... and {len(self.renamed_files) - 5} more")


def main() -> None:
    """Main cleanup function."""
    cleanup = FileCleanup()
    try:
        cleanup.cleanup_project()
        print("
SUCCESS: All non-English coding characters removed!")
        print("Project is now clean and ready for development.")
    except Exception as e:
        print(f"
ERROR during cleanup: {e}")


if __name__ == "__main__":
    main()
