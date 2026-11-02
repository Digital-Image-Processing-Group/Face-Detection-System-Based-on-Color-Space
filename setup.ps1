$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -c 'import tkinter; root=tkinter.Tk(); root.update(); root.destroy()'
if ($LASTEXITCODE -ne 0) { throw 'Tkinter GUI is unavailable' }
Write-Host 'Setup complete. Open run.bat to start.'
