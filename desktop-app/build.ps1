<#
.SYNOPSIS
    Build a self-contained, single-file Thalamus.exe with PyInstaller.

.DESCRIPTION
    Produces ONE portable executable — dist\Thalamus-PyInstaller.exe — with the
    Python runtime, the FastAPI backend (server.py + core/routes/config/...),
    pywebview/WebView2, and every dependency baked in. No _internal folder, no
    .venv, no thalamus_path.conf: copy the single .exe anywhere and run it.

    This is the most popular packager and the smoothest path for pywebview.
    Trade-off vs. Nuitka (build_nuitka.ps1): --onefile unpacks to %TEMP% on each
    launch, so cold start is ~1-3s slower, and an UNSIGNED onefile is the classic
    Defender/SmartScreen false-positive magnet. The durable fix for both AV and
    trust prompts is Authenticode code-signing the finished .exe.

.NOTES
    Windows 10/11, Python 3.11+ on PATH (the `py` launcher), WebView2 runtime
    present (ships with Windows 11 and Chromium Edge).

        powershell -ExecutionPolicy Bypass -File desktop-app\build.ps1
        powershell -ExecutionPolicy Bypass -File desktop-app\build.ps1 -Console

.PARAMETER ThalamusPath
    Repo root. Defaults to the parent of this script.

.PARAMETER Console
    Build a console exe (keeps a terminal window) so Python tracebacks are
    visible while debugging. Omit for the windowed product build.
#>
param(
    [string]$ThalamusPath,
    [switch]$Console
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoDir = if ($ThalamusPath) {
    (Resolve-Path $ThalamusPath).Path
} else {
    (Resolve-Path (Join-Path $ScriptDir "..")).Path
}
$AppName = "Thalamus-PyInstaller"

Write-Host "========================================="
Write-Host "  Thalamus single-file build (PyInstaller)"
Write-Host "  repo:    $RepoDir"
Write-Host "  console: $($Console.IsPresent)"
Write-Host "========================================="

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

# 4. Freeze the WHOLE app into one file.
#    --paths $RepoDir          : resolve `import server` and the local packages.
#    --collect-all <pkg>       : pull dynamically/lazily imported deps PyInstaller's
#                                static analysis misses (uvicorn's auto loop/protocol
#                                modules; h2/hpack/hyperframe behind httpx[http2];
#                                certifi's CA bundle; pydantic_core; webview's
#                                WebView2 backend).
#    --hidden-import clr / webview.platforms.winforms + --collect-all pythonnet,
#    clr_loader: pywebview opens its Windows window through WebView2 via pythonnet
#    (the `clr` module), loaded dynamically — without these the frozen exe crashes
#    on launch with "Unhandled exception in script". This is THE fix that made the
#    single-file build run.
#    --collect-submodules <pkg>: insurance for the app's own packages.
$pyiArgs = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onefile",
    "--name", $AppName,
    "--icon", "assets\icon.ico",
    "--paths", $RepoDir,
    "--add-data", "index.html;.",
    "--collect-all", "webview",
    "--collect-all", "pythonnet",
    "--collect-all", "clr_loader",
    "--collect-all", "uvicorn",
    "--collect-all", "httpx",
    "--collect-all", "httpcore",
    "--collect-all", "h2",
    "--collect-all", "hpack",
    "--collect-all", "hyperframe",
    "--collect-all", "anyio",
    "--collect-all", "certifi",
    "--collect-all", "pydantic",
    "--collect-all", "pydantic_core",
    "--collect-all", "google.protobuf",
    "--hidden-import", "clr",
    "--hidden-import", "webview.platforms.winforms",
    "--collect-submodules", "routes",
    "--collect-submodules", "core",
    "--collect-submodules", "config",
    "--collect-submodules", "claude_code",
    "--collect-submodules", "proto",
    "--collect-submodules", "utils"
)
if (-not $Console) { $pyiArgs += "--windowed" }
$pyiArgs += "thalamus_app.py"

Write-Host "Running PyInstaller ..."
Push-Location $ScriptDir
try {
    & $VenvPy @pyiArgs
} finally {
    Pop-Location
}

$ExePath = Join-Path $ScriptDir "dist\$AppName.exe"
Write-Host ""
Write-Host "========================================="
Write-Host "  Build complete!"
Write-Host "  Output: $ExePath"
Write-Host "  Single file — copy it anywhere and run. No _internal folder."
Write-Host "========================================="
