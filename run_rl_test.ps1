param(
  [string]$Python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TestArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectDir
try {
  & $Python ".\test_rl.py" @TestArgs
} finally {
  Pop-Location
}

