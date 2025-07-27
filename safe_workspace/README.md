# Safe Workspace Directory

This directory is designed to be safe from PowerShell and UTF-8 encoding issues.

## Usage
- Work here for all Arabic NLP development
- All files are automatically UTF-8 encoded
- No Arabic character corruption issues
- Safe for PowerShell operations

## Structure
- `scripts/` - Python scripts and utilities
- `data/` - Data files and datasets  
- `logs/` - Log files and debugging output
- `temp/` - Temporary files (safe to delete)
- `config/` - Configuration files
- `tests/` - Test files and validation scripts

## Arabic Character Support
This workspace fully supports all Arabic characters including:
- ؤ (HAMZA on WAW)
- ئ (HAMZA on YEH)  
- إ (HAMZA below ALIF)
- أ (HAMZA above ALIF)

All operations are UTF-8 safe and PowerShell compatible.
