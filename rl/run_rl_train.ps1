param(
  [string]$Python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
$RlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $RlDir
Push-Location $ProjectDir
try {
  & $Python -m rl.train_rl @TrainArgs
} finally {
  Pop-Location
}
