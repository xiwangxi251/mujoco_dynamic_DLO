param(
  [string]$Python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectDir
try {
  & $Python ".\train_rl.py" @TrainArgs
} finally {
  Pop-Location
}

