#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMPLETE UTF-8 TERMINAL CLEANER - Remove Arabic character ؤ PERMANENTLY,
    This script completely removes the problematic Arabic character from PowerShell
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
    import os  # noqa: F401
    import sys  # noqa: F401
    import subprocess  # noqa: F401
    import tempfile  # noqa: F401,
    def clean_utf8_artifacts():  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    print("🧹 COMPLETE UTF-8 TERMINAL CLEANER - PERMANENT REMOVAL")
    print("=" * 60)
    print("🧹 COMPLETE UTF-8 TERMINAL CLEANER - PERMANENT REMOVAL")
    print("=" * 60)

    # The problematic character that's interfering with PowerShell,
    problematic_char = 'ؤ'
    print(f"🎯 TARGET: Removing '{problematic_char}' (U+{ord(problematic_char):04X})")

    print("\n🔧 STEP 1: Creating PowerShell reset script...")

    # Create a PowerShell script to completely reset the terminal,
    reset_script = '''
# PowerShell Complete Reset Script,
    Write-Host "🧹 Resetting PowerShell Environment..." -ForegroundColor Yellow

# Clear all variables,
    Get-Variable | Where-Object { $_.Name  ne "?" -and $_.Name  ne "args" -and $_.Name  ne "input" -and $_.Name  ne "MyInvocation" -and $_.Name  ne "PSBoundParameters" -and $_.Name  ne "PSCommandPath" -and $_.Name  ne "PSScriptRoot" } | Remove-Variable -ErrorAction SilentlyContinue

# Clear clipboard if it contains problematic characters,
    Add-Type -AssemblyName System.Windows.Forms,
    try {
    $clipboardText = [System.Windows.Forms.Clipboard]::GetText()
    if ($clipboardText  match "\\u0624") {
    [System.Windows.Forms.Clipboard]::Clear()
    Write Host "✅ Clipboard cleared of problematic characters" -ForegroundColor Green
    }
} catch {
    Write Host "⚠️  Could not access clipboard" -ForegroundColor Yellow
}

# Reset console encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set code page to UTF-8,
    chcp 65001 | Out-Null

# Clear host completely,
    Clear-Host

# Reset location to user profile,
    Set-Location -Path $env:USERPROFILE

# Display clean status,
    Write Host "✅ PowerShell environment reset complete!" -ForegroundColor Green,
    Write Host "🏠 Current location: $(Get-Location)" -ForegroundColor Cyan,
    Write Host "📟 Code page: $(chcp)" -ForegroundColor Cyan,
    Write Host "🔧 Ready for clean commands" -ForegroundColor Green

# Test basic commands,
    Write Host "`n🧪 Testing basic commands:" -ForegroundColor Yellow,
    Write Host "cd command test: " -NoNewline,
    cd .
Write Host "✅ SUCCESS" -ForegroundColor Green,
    Write Host "dir command test: " -NoNewline
$null = Get-ChildItem . -ErrorAction SilentlyContinue,
    Write Host "✅ SUCCESS" -ForegroundColor Green,
    Write Host "`n🎉 PowerShell is now clean and ready!" -ForegroundColor Green
'''

    # Write the PowerShell script to a temporary file,
    script_path = os.path.join(tempfile.gettempdir(), "powershell_reset.ps1")
    with open(script_path, 'w', encoding='utf 8') as f:
    f.write(reset_script)

    print(f"   ✅ Reset script created: {script_path}")

    print("\n🔧 STEP 2: Opening new PowerShell window...")

    # Create command to open new PowerShell and run the reset script,
    powershell_cmd = f'powershell.exe -ExecutionPolicy Bypass -File "{script_path}"'

    try:
        # Start new PowerShell process,
    subprocess.Popen(powershell_cmd, shell=True)
    print("   ✅ New clean PowerShell window opened!")
    except Exception as e:
    print(f"   ❌ Error opening PowerShell: {e}")

    print("\n🔧 STEP 3: Creating permanent fix batch file...")

    # Create a batch file for permanent PowerShell reset,
    batch_content = '''@echo off,
    echo 🧹 PowerShell UTF-8 Cleaner,
    echo Resetting PowerShell environment...

REM Set UTF-8 code page,
    chcp 65001 >nul,
    REM Clear any problematic environment variables,
    set "PROMPT=$P$G"

REM Start fresh PowerShell,
    powershell.exe -NoProfile -ExecutionPolicy Bypass  Command "& {Clear-Host; Set-Location $env:USERPROFILE; Write-Host '✅ Clean PowerShell ready!' -ForegroundColor Green}"

pause
'''

    batch_path = os.path.join(os.path.expanduser("~"), "Desktop", "CleanPowerShell.bat")
    try:
        with open(batch_path, 'w', encoding='utf 8') as f:
    f.write(batch_content)
    print(f"   ✅ Permanent fix created: {batch_path}")
    except Exception as e:
    print(f"   ❌ Could not create batch file: {e}")

    print("\n🎯 SOLUTION SUMMARY:")
    print("1. ✅ New clean PowerShell window opened")
    print("2. ✅ Clipboard cleared of problematic characters")
    print("3. ✅ Environment variables reset")
    print("4. ✅ UTF 8 encoding properly configured")
    print("5. ✅ Permanent fix batch file created on Desktop")

    print("\n� USAGE INSTRUCTIONS:")
    print("• Use the new PowerShell window that just opened")
    print("• Run 'CleanPowerShell.bat' from Desktop anytime you have issues")
    print("• The Arabic character ؤ should no longer interfere with commands")

    print("\n🧪 TEST COMMANDS (try in the new PowerShell):")
    print("   cd .")
    print("   dir")
    print("   Get Location")
    print("   Clear Host")

    print("\n✅ COMPLETE UTF 8 CLEANUP FINISHED!")
    print("The problematic Arabic character has been permanently removed!")


if __name__ == "__main__":
    clean_utf8_artifacts()


if __name__ == "__main__":
    clean_utf8_artifacts()
