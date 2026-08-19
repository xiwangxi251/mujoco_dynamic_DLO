param(
  [string]$Python = "python",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$DemoArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectDir
try {
  & $Python ".\run_grasp.py" @DemoArgs
} finally {
  Pop-Location
}
