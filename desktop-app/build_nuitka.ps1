<#
.SYNOPSIS
    Build a self-contained, single-file Thalamus.exe with Nuitka.

.DESCRIPTION
    Nuitka counterpart of build.ps1. Compiles the Python app to a real native
    binary instead of bundling a Python interpreter, then wraps it as one file:
    dist-nuitka\Thalamus-Nuitka.exe. Like the PyInstaller build it is fully
    portable (no _internal folder, no .venv, no thalamus_path.conf).

    Why build both: Nuitka's compiled binary typically starts faster than a
    PyInstaller --onefile exe (which re-unpacks to %TEMP% on every launch) and
    trips antivirus heuristics less often. The costs are a much slower build and
    a required C compiler.

    KNOWN RISK — pywebview backend: pywebview's Windows window uses WebView2 via
    pythonnet (`clr`) and ships .NET DLLs as package data. That combination is
    the thing most likely to be missed by a compiler. If the window fails to
    open, iterate on the webview / pythonnet include flags below; the PyInstaller
    build (build.ps1) is the proven-working fallback for pywebview.

.NOTES
    Windows 10/11, Python 3.11+ (`py` launcher), WebView2 runtime present.
    Needs a C compiler: this script passes --mingw64 so Nuitka auto-downloads a
    private MinGW64 (with --assume-yes-for-downloads). Drop --mingw64 to use a
    local MSVC (Visual Studio Build Tools) instead, which Nuitka also supports.

        powershell -ExecutionPolicy Bypass -File desktop-app\build_nuitka.ps1
        powershell -ExecutionPolicy Bypass -File desktop-app\build_nuitka.ps1 -Console

.PARAMETER ThalamusPath
    Repo root. Defaults to the parent of this script.

.PARAMETER Console
    Keep a console window (force) so tracebacks are visible while debugging.
    Omit for the windowed product build (console disabled).
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
$AppName = "Thalamus-Nuitka"

Write-Host "========================================="
Write-Host "  Thalamus single-file build (Nuitka)"
Write-Host "  repo:    $RepoDir"
Write-Host "  console: $($Console.IsPresent)"
Write-Host "========================================="

# 1. Locate (or create) the repo's virtualenv
$VenvPy = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating .venv ..."
    & py -3 -m venv (Join-Path $RepoDir ".venv")
}

# 2. Install runtime + desktop deps, plus Nuitka itself
Write-Host "Installing dependencies ..."
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r (Join-Path $RepoDir "requirements.txt")
& $VenvPy -m pip install -r (Join-Path $RepoDir "requirements-desktop.txt")
& $VenvPy -m pip install "nuitka>=2.4"

# 3. Generate the Windows icon (.ico) if missing
$IconIco = Join-Path $ScriptDir "assets\icon.ico"
if (-not (Test-Path $IconIco)) {
    Write-Host "Generating icon.ico ..."
    & $VenvPy (Join-Path $ScriptDir "generate_icon.py")
}

# 4. Compile to one file. Nuitka follows imports from thalamus_app.py, but the
#    --include-* flags force in the dynamically/lazily imported pieces its static
#    analysis can miss, and the package data (WebView2 DLLs, certifi CA bundle).
$ConsoleMode = if ($Console) { "force" } else { "disable" }

$nuitkaArgs = @(
    "-m", "nuitka",
    "--standalone", "--onefile",
    "--assume-yes-for-downloads",
    "--mingw64",
    "--windows-console-mode=$ConsoleMode",
    "--windows-icon-from-ico=desktop-app\assets\icon.ico",
    "--output-dir=desktop-app\dist-nuitka",
    "--output-filename=$AppName.exe",
    "--include-data-files=desktop-app\index.html=index.html",
    # App code (server.py is a top-level module; the rest are packages)
    "--include-module=server",
    "--include-module=launcher_ui",
    "--include-package=routes",
    "--include-package=core",
    "--include-package=config",
    "--include-package=claude_code",
    "--include-package=proto",
    "--include-package=utils",
    # Runtime deps with dynamic/lazy imports
    "--include-package=uvicorn",
    "--include-package=anyio",
    "--include-package=h2",
    "--include-package=hpack",
    "--include-package=hyperframe",
    "--include-package=pydantic",
    "--include-package=pydantic_core",
    # GUI backend + data files (the risky bit — see KNOWN RISK above). The
    # pywebview WebView2 window loads through pythonnet (the `clr` module) and the
    # winforms backend — the same dependency that the PyInstaller build needed to
    # stop crashing on launch. pythonnet under Nuitka is still the most likely
    # iteration point; if the window won't open, this group is where to tweak.
    "--include-package=webview",
    "--include-package-data=webview",
    "--include-module=webview.platforms.winforms",
    "--include-module=clr",
    "--include-package=clr_loader",
    "--include-package=httpx",
    "--include-package=httpcore",
    "--include-package-data=certifi",
    "desktop-app\thalamus_app.py"
)

# Run from the repo root with the repo on PYTHONPATH so `import server` and the
# local packages resolve during Nuitka's compile-time import analysis.
$env:PYTHONPATH = $RepoDir
Write-Host "Running Nuitka (first build downloads MinGW + can take several minutes) ..."
Push-Location $RepoDir
try {
    & $VenvPy @nuitkaArgs
} finally {
    Pop-Location
    Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
}

$ExePath = Join-Path $ScriptDir "dist-nuitka\$AppName.exe"
Write-Host ""
Write-Host "========================================="
Write-Host "  Build complete!"
Write-Host "  Output: $ExePath"
Write-Host "  Single file — copy it anywhere and run. No _internal folder."
Write-Host "========================================="
