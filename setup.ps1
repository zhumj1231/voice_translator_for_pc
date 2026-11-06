param([string]$Python = 'py')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonArgs = @()
if ($Python -eq 'py') { $pythonArgs = @('-3.12') }
& $Python @pythonArgs -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32, 'Python 3.12 x64 required'"
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64, including Tcl/Tk and the py launcher.' }
& $Python @pythonArgs -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath config.json)) {
    Copy-Item -LiteralPath config.example.json -Destination config.json
}
Write-Host 'Python environment ready. No model weights or runtime executables were downloaded.'
& .\.venv\Scripts\python.exe -X utf8 check_setup.py
if ($LASTEXITCODE -ne 0) {
    Write-Host 'See docs/DEPLOYMENT.md for model and runtime downloads; then rerun check_setup.py.'
}
