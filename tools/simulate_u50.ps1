[CmdletBinding()]
param(
    [ValidateSet('snn')][string]$Kernel = 'snn',
    [string]$VivadoPath = 'vivado',
    [string]$LicensePath = '',
    [string]$Part = 'xcu50-fsvh2104-2-e'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command $VivadoPath -ErrorAction SilentlyContinue)) { throw "Vivado not found: $VivadoPath" }
if ($LicensePath) {
    if (-not (Test-Path -LiteralPath $LicensePath)) { throw "License not found: $LicensePath" }
    $env:XILINXD_LICENSE_FILE = (Get-Item -LiteralPath $LicensePath).FullName
}
$kernels = @($Kernel)
foreach ($name in $kernels) {
    $runId = '{0}_{1}' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'), $PID
    $buildDir = Join-Path $root "build\u50\simulation\$name`_$runId"
    & $VivadoPath -mode batch -notrace -source (Join-Path $root 'vivado\u50\simulate_u50.tcl') -tclargs -kernel $name -part $Part -build_dir $buildDir
    if ($LASTEXITCODE -ne 0) { throw "Vivado $name simulation failed with exit code $LASTEXITCODE" }
}
