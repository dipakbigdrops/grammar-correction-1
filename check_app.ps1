# Run after: docker compose up -d (or after starting the API)
# Usage: .\check_app.ps1   or   .\check_app.ps1 -BaseUrl http://localhost:8000
param([string]$BaseUrl = "http://localhost:8000")

$ErrorActionPreference = "Stop"
$failed = 0

function Test-Endpoint($Name, $Url, $ExpectStatus = 200, $TimeoutSec = 20) {
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        if ($r.StatusCode -ne $ExpectStatus) {
            Write-Host "FAIL $Name : status $($r.StatusCode) (expected $ExpectStatus)"
            return 1
        }
        Write-Host "OK   $Name"
        return 0
    } catch {
        Write-Host "FAIL $Name : $($_.Exception.Message)"
        return 1
    }
}

function Test-Json($Name, $Url, $Key, $ExpectedValue, $TimeoutSec = 15) {
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        $j = $r.Content | ConvertFrom-Json
        $v = $j.$Key
        if ($v -ne $ExpectedValue) {
            Write-Host "FAIL $Name : $Key = '$v' (expected '$ExpectedValue')"
            return 1
        }
        Write-Host "OK   $Name ($Key=$v)"
        return 0
    } catch {
        Write-Host "FAIL $Name : $($_.Exception.Message)"
        return 1
    }
}

Write-Host "Checking API at $BaseUrl"
Write-Host ""

$failed += Test-Endpoint "GET /" "$BaseUrl/" 200
$failed += Test-Json "GET /health" "$BaseUrl/health" "version" "1.0.0" 60
$failed += Test-Endpoint "GET /docs" "$BaseUrl/docs" 200
$failed += Test-Json "GET /task/any" "$BaseUrl/task/any" "task_id" "any"
$failed += Test-Json "GET /metrics" "$BaseUrl/metrics" "status" "operational"

Write-Host ""
if ($failed -eq 0) {
    Write-Host "All checks passed."
    exit 0
} else {
    Write-Host "$failed check(s) failed."
    exit 1
}
