param(
    [string]$RomPath = "",
    [switch]$AcceptLicense
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) {
    Write-Host "[info] $msg"
}

Write-Info "Checking for Python 3.8..."
& py -3.8-64 -c "import sys; print(sys.version)" | Out-Null

Write-Info "Verifying Python is 64-bit..."
$is64 = & py -3.8-64 -c "import sys; print(int(sys.maxsize > 2**32))"
if ($is64 -ne "1") {
    Write-Error "Python 3.8 x64 is required. Install 64-bit Python 3.8 and retry."
}

$venvPath = Join-Path $PSScriptRoot ".venv-rl2"
if (Test-Path $venvPath) {
    Write-Info "Removing existing venv at $venvPath"
    Remove-Item -Recurse -Force $venvPath
}
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating venv at $venvPath"
    & py -3.8-64 -m venv $venvPath
} else {
    Write-Info "Venv already exists at $venvPath"
}

Write-Info "Activating venv"
& "$venvPath\Scripts\Activate.ps1"

Write-Info "Upgrading pip"
python -m pip install --upgrade pip

Write-Info "Installing legacy deps for keras-rl2"
pip install gym==0.17.3
pip install pyglet==1.5.0
pip install Pillow==9.5.0
pip install numpy==1.19.5
pip install h5py==3.1.0

Write-Info "Installing TensorFlow 2.5.3 (and removing newer TensorFlow if present)"
pip uninstall -y tensorflow-intel tensorflow
pip install tensorflow==2.5.3

Write-Info "Installing keras-rl2 (uses tensorflow.keras)"
pip uninstall -y Keras
pip install --no-deps keras-rl2==1.0.5

Write-Info "Installing atari-py"
pip install atari-py==0.2.9

if ($AcceptLicense) {
    Write-Info "Installing AutoROM to download Atari ROMs"
    pip install autorom autorom[accept-rom-license]
    Write-Info "Downloading ROMs (AutoROM)"
    python -c "from AutoROM.AutoROM import cli; cli.main(['--accept-license'])"
    $autoRomDir = Join-Path $env:VIRTUAL_ENV "Lib\\site-packages\\AutoROM\\roms"
    if (Test-Path $autoRomDir) {
        Write-Info "Importing ROMs from $autoRomDir into atari-py"
        python -m atari_py.import_roms $autoRomDir
    } else {
        Write-Info "AutoROM ROM directory not found at $autoRomDir"
    }
} elseif ($RomPath -ne "") {
    if (-not (Test-Path $RomPath)) {
        Write-Error "RomPath '$RomPath' does not exist."
    }
    Write-Info "Importing ROMs from $RomPath"
    python -m atari_py.import_roms $RomPath
} else {
    Write-Info "No RomPath provided; skip ROM import."
    Write-Info "Run later with: .\\install_keras_rl2.ps1 -AcceptLicense"
    Write-Info "Or import your own ROMs: .\\install_keras_rl2.ps1 -RomPath \"C:\\path\\to\\roms\""
}

Write-Info "Done. Activate with: $venvPath\\Scripts\\Activate.ps1"
