param(
  [string]$ThirdPartyRoot = "third_party"
)

$gitDirs = Get-ChildItem -Path $ThirdPartyRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq ".git" }
if (-not $gitDirs) {
  Write-Host "No nested .git directories found under $ThirdPartyRoot"
  exit 0
}

foreach ($gitDir in $gitDirs) {
  Remove-Item -LiteralPath $gitDir.FullName -Recurse -Force
  Write-Host "Removed nested repo metadata: $($gitDir.FullName)"
}
