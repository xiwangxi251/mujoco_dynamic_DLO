param(
  [string]$Python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe",
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
