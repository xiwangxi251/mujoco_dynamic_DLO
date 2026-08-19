param(
  [string]$Python = "python",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TestArgs
)

$ErrorActionPreference = "Stop"
$RlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $RlDir
Push-Location $ProjectDir
try {
  & $Python -m rl.test_rl @TestArgs
} finally {
  Pop-Location
}
