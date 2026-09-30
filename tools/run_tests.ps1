[CmdletBinding()]
param([switch]$Training)
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'run_checks.py'))
if ($Training) { $arguments += '--training' }
& python @arguments
exit $LASTEXITCODE
