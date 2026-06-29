<#
.SYNOPSIS
    Build the Thalamus Windows desktop app (pywebview + PyInstaller).

.DESCRIPTION
    Windows counterpart of build.sh. Produces dist\Thalamus\Thalamus.exe — a
    thin shell (desktop-app/thalamus_app.py) that launches the FastAPI backend
    (server.py) and the UI proxy (launcher_ui.py) from the repo's .venv and
    shows the UI in a WebView2 window.

    Like build.sh, the backend itself is NOT frozen into the exe: server.py runs
    from the repo using the repo's .venv, located at runtime via the
    thalamus_path.conf written next to the exe by this script.

.NOTES
    Run on Windows 10/11 with Python 3.11+ on PATH (the `py` launcher) and the
    WebView2 runtime present (it ships with Windows 11 and Chromium Edge).
        powershell -ExecutionPolicy Bypass -File desktop-app\build.ps1

.PARAMETER ThalamusPath
    Repo root to bake into thalamus_path.conf. Defaults to the parent of this
    script. Override when the exe will run against a repo at a different path.
#>
param([string]$ThalamusPath)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoDir = if ($ThalamusPath) {
    (Resolve-Path $ThalamusPath).Path
} else {
    (Resolve-Path (Join-Path $ScriptDir "..")).Path
}
$AppName = "Thalamus"

Write-Host "========================================="
Write-Host "  Thalamus Windows .exe builder (pywebview)"
Write-Host "========================================="
Write-Host "  repo: $RepoDir"

# 1. Locate (or create) the repo's virtualenv
$VenvPy = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating .venv ..."
    & py -3 -m venv (Join-Path $RepoDir ".venv")
}

# 2. Install runtime + desktop/build dependencies into the venv
Write-Host "Installing dependencies ..."
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r (Join-Path $RepoDir "requirements.txt")
& $VenvPy -m pip install -r (Join-Path $RepoDir "requirements-desktop.txt")

# 3. Generate the Windows icon (.ico) if missing
$IconIco = Join-Path $ScriptDir "assets\icon.ico"
if (-not (Test-Path $IconIco)) {
    Write-Host "Generating icon.ico ..."
    & $VenvPy (Join-Path $ScriptDir "generate_icon.py")
}

# 4. Bundle the shell with PyInstaller.
#    --onedir (not --onefile): --onefile unpacks to %TEMP% on every launch and
#    reliably trips Windows Defender's self-extraction heuristic.
#    --collect-all webview: pull in pywebview's WebView2 backend + data files.
Write-Host "Running PyInstaller ..."
Push-Location $ScriptDir
try {
    & $VenvPy -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --onedir `
        --name $AppName `
        --icon "assets\icon.ico" `
        --collect-all webview `
        --add-data "launcher_ui.py;." `
        --add-data "index.html;." `
        "thalamus_app.py"
} finally {
    Pop-Location
}

# 5. Write thalamus_path.conf next to the exe (UTF-8, no BOM) so the shell can
#    find the repo + its .venv at runtime.
$DistDir = Join-Path $ScriptDir "dist\$AppName"
$ConfPath = Join-Path $DistDir "thalamus_path.conf"
[System.IO.File]::WriteAllText($ConfPath, $RepoDir, [System.Text.UTF8Encoding]::new($false))

Write-Host ""
Write-Host "========================================="
Write-Host "  Build complete!"
Write-Host ""
Write-Host "  Output:  $DistDir\$AppName.exe"
Write-Host "  Repo:    $RepoDir"
Write-Host ""
Write-Host "  Run:     & `"$DistDir\$AppName.exe`""
Write-Host "  (Wrap dist\$AppName in an Inno Setup installer to distribute.)"
Write-Host "========================================="
