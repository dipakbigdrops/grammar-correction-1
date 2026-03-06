param(
    [string]$BaseUrl = "http://localhost:8000",
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"
$testsPassed = 0
$testsFailed = 0

function Test-Endpoint {
    param([string]$Name, [scriptblock]$Test)
    Write-Host "Test: $Name"
    try {
        & $Test
        $script:testsPassed++
        Write-Host "  PASS"
        return $true
    } catch {
        $script:testsFailed++
        Write-Host "  FAIL: $_"
        return $false
    }
}

Write-Host "Testing Grammar API at $BaseUrl"
Write-Host ""

Test-Endpoint "GET /" {
    $r = Invoke-RestMethod -Uri "$BaseUrl/" -Method Get -TimeoutSec 10
    if (-not $r.message) { throw "No message in response" }
}

Test-Endpoint "GET /health" {
    $r = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 15
    if ($r.status -eq $null) { throw "No status in response" }
}

$testHtmlPath = Join-Path $PSScriptRoot "..\test_sample.html"
if (-not (Test-Path $testHtmlPath)) {
    $testHtmlContent = @"
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<p>This is a test. I has a grammar error and need corection.</p>
</body>
</html>
"@
    Set-Content -Path $testHtmlPath -Value $testHtmlContent -Encoding UTF8
}

Test-Endpoint "POST /process (HTML file)" {
    $filePath = (Resolve-Path $testHtmlPath).Path
    $uri = "$BaseUrl/process"
    $form = @{
        file = Get-Item -Path $filePath
    }
    $r = Invoke-RestMethod -Uri $uri -Method Post -Form $form -TimeoutSec $TimeoutSec
    if ($r.status -ne "SUCCESS") { throw "Process returned status: $($r.status)" }
    if (-not $r.result.corrected_text) { throw "No corrected_text in result" }
}

Write-Host ""
Write-Host "Results: $testsPassed passed, $testsFailed failed"
if ($testsFailed -gt 0) { exit 1 }
exit 0
