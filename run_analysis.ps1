param(
    [Parameter(Mandatory=$true,Position=0)]
    [string]$Stage,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$PythonExecutable = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { 'python' }
& $PythonExecutable (Join-Path $PSScriptRoot 'analysis\run_pipeline.py') $Stage @Arguments
exit $LASTEXITCODE
