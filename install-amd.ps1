$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = 'D:/Codex/tools/uv-python'
$env:UV_CACHE_DIR = 'D:/Codex/tools/uv-cache'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Instale uv antes de executar este script: https://docs.astral.sh/uv/getting-started/installation/'
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python 3.10 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente Python.' }
}
uv pip install --python .venv/Scripts/python.exe -r requirements-amd.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as dependencias AMD.' }
& ./.venv/Scripts/python.exe download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Falha ao baixar os modelos.' }
Write-Host 'Ambiente preparado. Execute ./start-amd.ps1 para abrir o CharacterGen.'
