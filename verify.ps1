$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -t . -v
if ($LASTEXITCODE -ne 0) { throw 'Checks failed' }
