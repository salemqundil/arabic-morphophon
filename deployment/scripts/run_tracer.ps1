# PowerShell Script for Arabic Word Tracer
# سكريبت PowerShell لمتتبع الكلمات العربية

param(
    [string]$Host = "localhost",
    [int]$Port = 5000,
    [switch]$Debug,
    [switch]$MockEngines,
    [switch]$SetupOnly,
    [switch]$Help
)

function Show-Help {
    Write-Host "🔍 متتبع الكلمات العربية - Arabic Word Tracer" -ForegroundColor Cyan
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "استخدام / Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_tracer.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "الخيارات / Options:" -ForegroundColor Yellow
    Write-Host "  -Host <address>    عنوان الخادم (افتراضي: localhost)"
    Write-Host "  -Port <number>     منفذ الخادم (افتراضي: 5000)"
    Write-Host "  -Debug             تفعيل وضع التطوير"
    Write-Host "  -MockEngines       استخدام محركات محاكاة"
    Write-Host "  -SetupOnly         إعداد فقط بدون تشغيل"
    Write-Host "  -Help              عرض هذه المساعدة"
    Write-Host ""
    Write-Host "أمثلة / Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_tracer.ps1                    # تشغيل بسيط"
    Write-Host "  .\run_tracer.ps1 -Debug             # وضع التطوير"
    Write-Host "  .\run_tracer.ps1 -MockEngines       # محركات تجريبية"
    Write-Host "  .\run_tracer.ps1 -Port 8080         # منفذ مخصص"
}

function Test-Dependencies {
    Write-Host "🔧 فحص التبعيات..." -ForegroundColor Yellow
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Host "❌ Python غير مثبت أو غير متاح في PATH" -ForegroundColor Red
        Write-Host "يرجى تثبيت Python من https://python.org" -ForegroundColor Red
        return $false
    }
    
    # Check pip packages
    $requiredPackages = @("flask", "flask-cors")
    $missingPackages = @()
    
    foreach ($package in $requiredPackages) {
        try {
            python -c "import $($package.Replace('-', '_'))" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ $package متوفر" -ForegroundColor Green
            } else {
                $missingPackages += $package
            }
        } catch {
            $missingPackages += $package
        }
    }
    
    if ($missingPackages.Count -gt 0) {
        Write-Host "❌ حزم مفقودة: $($missingPackages -join ', ')" -ForegroundColor Red
        Write-Host "تثبيت الحزم المفقودة..." -ForegroundColor Yellow
        
        try {
            python -m pip install $missingPackages
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ تم تثبيت الحزم بنجاح" -ForegroundColor Green
            } else {
                Write-Host "❌ فشل في تثبيت الحزم" -ForegroundColor Red
                return $false
            }
        } catch {
            Write-Host "❌ خطأ في تثبيت الحزم: $_" -ForegroundColor Red
            return $false
        }
    }
    
    return $true
}

function Setup-Environment {
    Write-Host "📁 إعداد البيئة..." -ForegroundColor Yellow
    
    # Create necessary directories
    $directories = @("logs", "cache", "sample_data")
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "📁 تم إنشاء مجلد: $dir" -ForegroundColor Green
        }
    }
    
    # Create sample data
    $sampleWords = @("كتاب", "يكتب", "مكتبة", "كتابة", "مكتوب", "درس", "يدرس", "مدرسة", "دراسة", "مدروس")
    $sampleFile = "sample_data/test_words.txt"
    
    if (!(Test-Path $sampleFile)) {
        $sampleWords | Out-File -FilePath $sampleFile -Encoding UTF8
        Write-Host "📝 تم إنشاء ملف الكلمات التجريبية: $sampleFile" -ForegroundColor Green
    }
    
    Write-Host "✅ تم إعداد البيئة بنجاح" -ForegroundColor Green
}

function Start-Application {
    param(
        [string]$HostAddress,
        [int]$PortNumber,
        [bool]$DebugMode,
        [bool]$UseMockEngines
    )
    
    Write-Host "🚀 بدء تشغيل متتبع الكلمات العربية..." -ForegroundColor Cyan
    Write-Host ""
    
    # Set environment variables
    if ($UseMockEngines) {
        $env:USE_MOCK_ENGINES = "true"
        Write-Host "🎭 تم تفعيل المحركات التجريبية" -ForegroundColor Yellow
    }
    
    if ($DebugMode) {
        $env:FLASK_ENV = "development"
        Write-Host "🐛 تم تفعيل وضع التطوير" -ForegroundColor Yellow
    }
    
    # Build arguments
    $args = @()
    if ($HostAddress -ne "localhost") {
        $args += "--host", $HostAddress
    }
    if ($PortNumber -ne 5000) {
        $args += "--port", $PortNumber
    }
    if ($DebugMode) {
        $args += "--debug"
    }
    if ($UseMockEngines) {
        $args += "--mock-engines"
    }
    
    Write-Host "🌐 الرابط: http://$HostAddress`:$PortNumber" -ForegroundColor Green
    Write-Host "📱 يمكن الوصول للواجهة من أي متصفح" -ForegroundColor Green
    Write-Host "⏹️  للإيقاف: اضغط Ctrl+C" -ForegroundColor Yellow
    Write-Host "=" * 50
    
    try {
        # Check if run_tracer.py exists
        if (Test-Path "run_tracer.py") {
            python run_tracer.py @args
        } elseif (Test-Path "arabic_word_tracer_app.py") {
            Write-Host "⚠️ استخدام التطبيق مباشرة..." -ForegroundColor Yellow
            python arabic_word_tracer_app.py
        } else {
            Write-Host "❌ لم يتم العثور على ملفات التطبيق" -ForegroundColor Red
            Write-Host "تأكد من وجود run_tracer.py أو arabic_word_tracer_app.py" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ خطأ في تشغيل التطبيق: $_" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Open-Browser {
    param([string]$Url)
    
    Start-Sleep -Seconds 2
    try {
        Start-Process $Url
        Write-Host "🌐 تم فتح المتصفح على: $Url" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ لم يتم فتح المتصفح تلقائياً. افتح الرابط يدوياً: $Url" -ForegroundColor Yellow
    }
}

# Main execution
function Main {
    # Show banner
    Write-Host ""
    Write-Host "🔍 متتبع الكلمات العربية" -ForegroundColor Cyan
    Write-Host "Arabic Word Tracer" -ForegroundColor Cyan
    Write-Host "=" * 50
    
    # Handle help
    if ($Help) {
        Show-Help
        return
    }
    
    # Check dependencies
    if (!(Test-Dependencies)) {
        Write-Host "❌ فشل في فحص التبعيات" -ForegroundColor Red
        return
    }
    
    # Setup environment
    Setup-Environment
    
    # Setup only mode
    if ($SetupOnly) {
        Write-Host "✅ تم الإعداد بنجاح!" -ForegroundColor Green
        Write-Host "لتشغيل الخادم، استخدم:" -ForegroundColor Yellow
        Write-Host ".\run_tracer.ps1 -Host $Host -Port $Port" -ForegroundColor Yellow
        return
    }
    
    # Start application
    $url = "http://$Host`:$Port"
    
    # Open browser in background job
    if ($Host -eq "localhost" -or $Host -eq "127.0.0.1") {
        Start-Job -ScriptBlock { 
            param($Url)
            Start-Sleep -Seconds 3
            Start-Process $Url
        } -ArgumentList $url | Out-Null
    }
    
    # Start the application
    if (Start-Application -HostAddress $Host -PortNumber $Port -DebugMode $Debug -UseMockEngines $MockEngines) {
        Write-Host "👋 تم إيقاف الخادم" -ForegroundColor Yellow
    }
}

# Set console encoding for Arabic text
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Run main function
Main
