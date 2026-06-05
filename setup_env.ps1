$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$CodexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

function Test-PythonCandidate {
    param(
        [string]$Executable,
        [string[]]$Arguments = @()
    )

    try {
        & $Executable @Arguments -c "import sys; print(sys.version)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-WindowsStoreAlias {
    param([string]$Executable)
    return $Executable -like "*\Microsoft\WindowsApps\*" -or $Executable -like "*PythonSoftwareFoundation.Python*"
}

function Get-BasePython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and !(Test-WindowsStoreAlias $python.Source) -and (Test-PythonCandidate $python.Source)) {
        return @($python.Source)
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py -and !(Test-WindowsStoreAlias $py.Source) -and (Test-PythonCandidate $py.Source @("-3"))) {
        return @($py.Source, "-3")
    }

    if ((Test-Path $CodexPython) -and (Test-PythonCandidate $CodexPython)) {
        return @($CodexPython)
    }

    throw "No Python executable was found. Install Python 3.10+ or run this from Codex where bundled Python is available."
}

$basePython = @(Get-BasePython)
$basePythonExe = $basePython[0]
$basePythonArgs = @()
if ($basePython.Length -gt 1) {
    $basePythonArgs = $basePython[1..($basePython.Length - 1)]
}

$createVenv = !(Test-Path $VenvPython)
if (!$createVenv -and (Test-Path (Join-Path $VenvDir "pyvenv.cfg"))) {
    $venvConfig = Get-Content (Join-Path $VenvDir "pyvenv.cfg") -Raw
    if ($venvConfig -like "*Microsoft\WindowsApps*" -or $venvConfig -like "*PythonSoftwareFoundation.Python*") {
        Write-Host "Existing virtual environment uses the broken Windows Store Python alias; recreating it."
        $createVenv = $true
    }
}
if (!$createVenv -and !(Test-PythonCandidate $VenvPython)) {
    Write-Host "Existing virtual environment is not runnable; recreating it."
    $createVenv = $true
}

if ($createVenv) {
    Write-Host "Creating virtual environment at $VenvDir"
    & $basePythonExe @basePythonArgs -m venv --clear $VenvDir
}

Write-Host "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing project dependencies"
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host ""
Write-Host "Environment ready."
Write-Host "Run demos with:"
Write-Host "  .\.venv\Scripts\python.exe run_all_demos.py"
Write-Host ""
Write-Host "Run training with:"
Write-Host "  .\.venv\Scripts\python.exe train_all_models_parallel.py"
