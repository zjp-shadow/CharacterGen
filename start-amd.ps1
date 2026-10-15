$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:CHARACTERGEN_DEVICE = 'directml'
$env:CHARACTERGEN_MODEL_DIR = 'D:/Codex/models/CharacterGen'
$env:HF_HOME = 'D:/Codex/models/CharacterGen/huggingface'
$env:GRADIO_TEMP_DIR = 'D:/Codex/outputs/CharacterGen/tmp'
$env:CHARACTERGEN_OUTPUT_DIR = 'D:/Codex/outputs/CharacterGen'
$env:TORCH_HOME = 'D:/Codex/models/CharacterGen/torch'
$env:MPLCONFIGDIR = 'D:/Codex/models/CharacterGen/matplotlib'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'
& ./.venv/Scripts/python.exe webui.py --skip-download @args
exit $LASTEXITCODE
