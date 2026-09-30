[CmdletBinding()]
param(
    [ValidateSet('xo', 'xclbin')][string]$Mode = 'xo',
    [ValidateSet('snn')][string]$Kernel = 'snn',
    [ValidateSet('lif', 'izhikevich', 'hodgkin_huxley')][string]$HiddenNeuronModel = 'lif',
    [ValidateSet('lif', 'izhikevich', 'hodgkin_huxley')][string]$OutputNeuronModel = 'lif',
    [ValidateRange(1,65535)][int]$InputNeurons = 784,
    [ValidateRange(1,65535)][int]$HiddenNeurons = 256,
    [ValidateRange(1,65535)][int]$OutputNeurons = 10,
    [ValidateRange(1,65535)][int]$NumEvents = 12544,
    [ValidateRange(0,65535)][int]$HiddenExecutorCount = 0,
    [ValidateRange(0,65535)][int]$OutputExecutorCount = 0,
    [ValidateSet(0, 4, 5, 6)][int]$InterfaceMinor = 0,
    [string]$Platform = '',
    [string]$Part = 'xcu50-fsvh2104-2-e',
    [string]$BuildDir = '',
    [string]$VivadoPath = 'vivado',
    [string]$VppPath = 'v++',
    [string]$LicensePath = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $root "build\u50\$Kernel" }
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $root $BuildDir }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$artifactDir = Join-Path $BuildDir 'artifacts'
New-Item -ItemType Directory -Force $artifactDir | Out-Null
$xo = Join-Path $artifactDir "$Kernel`_accelerator.xo"
$tclBuildDir = $BuildDir.Replace('\', '/')
$tclXo = $xo.Replace('\', '/')
$models = @{lif=0; izhikevich=1; hodgkin_huxley=2}
if (-not (Get-Command $VivadoPath -ErrorAction SilentlyContinue)) { throw "Vivado not found: $VivadoPath" }
if ($LicensePath) {
    if (-not (Test-Path -LiteralPath $LicensePath)) { throw "License not found: $LicensePath" }
    $env:XILINXD_LICENSE_FILE = (Get-Item -LiteralPath $LicensePath).FullName
}
$env:SNN_REPO_ROOT = $root
& $VivadoPath -mode batch -notrace -source (Join-Path $root 'vivado\u50\package_u50_kernel.tcl') -tclargs `
    -kernel $Kernel -output $tclXo -build_dir $tclBuildDir -part $Part `
    -hidden_neuron_model $models[$HiddenNeuronModel] -output_neuron_model $models[$OutputNeuronModel] `
    -input_neurons $InputNeurons -hidden_neurons $HiddenNeurons -output_neurons $OutputNeurons -num_events $NumEvents `
    -hidden_executor_count $HiddenExecutorCount -output_executor_count $OutputExecutorCount `
    -interface_minor $InterfaceMinor
if ($LASTEXITCODE -ne 0) { throw "XO packaging failed with exit code $LASTEXITCODE" }
if ($Mode -eq 'xo') { Write-Host "Created $xo"; return }
if (-not $Platform) { throw '-Platform is required for -Mode xclbin (use xbutil examine on the U50 PC).' }
if (-not (Get-Command $VppPath -ErrorAction SilentlyContinue)) { throw "v++ not found: $VppPath" }
$xclbin = Join-Path $artifactDir "$Kernel`_accelerator.xclbin"
$temp = Join-Path $BuildDir 'vpp'
New-Item -ItemType Directory -Force $temp | Out-Null
& $VppPath --link --target hw --platform $Platform --kernel_frequency 100 `
    --temp_dir $temp --output $xclbin $xo
if ($LASTEXITCODE -ne 0) { throw "v++ link failed with exit code $LASTEXITCODE" }
Write-Host "Created $xclbin"
