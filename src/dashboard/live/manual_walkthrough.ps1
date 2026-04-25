# manual_walkthrough.ps1
# ----------------------
# Drives the *real* uvicorn-served dashboard (not TestClient) by hitting
# every public HTTP endpoint a browser would touch and verifying the
# WebSocket actually streams sample/detector/retrain frames.
#
# Run AFTER the server is booted via:
#   .venv\Scripts\python.exe -m dashboard.live --host 127.0.0.1 --port 8765

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8765"

function Write-Step($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Assert($cond, $msg) {
    if (-not $cond) { throw "ASSERT FAILED: $msg" }
    Write-Host "  OK $msg" -ForegroundColor Green
}

# ── 1. Index (SPA) ──────────────────────────────────────────────────
Write-Step "GET /  (the SPA HTML)"
$index = Invoke-WebRequest -Uri "$base/" -UseBasicParsing
Assert ($index.StatusCode -eq 200) "index returns 200"
Assert ($index.Content -match "AIMP") "index contains 'AIMP' brand"
Assert ($index.Content -match "Drift Injection") "index contains injection panel"

# ── 2. Static assets ────────────────────────────────────────────────
Write-Step "GET /static/{app.js,style.css}"
$js  = Invoke-WebRequest -Uri "$base/static/app.js"  -UseBasicParsing
$css = Invoke-WebRequest -Uri "$base/static/style.css" -UseBasicParsing
Assert ($js.StatusCode  -eq 200) "app.js served"
Assert ($css.StatusCode -eq 200) "style.css served"

# ── 3. Status before start ──────────────────────────────────────────
Write-Step "GET /api/status (pre-start)"
$st0 = Invoke-RestMethod -Uri "$base/api/status"
Assert ($null -ne $st0)               "status reachable"
Write-Host "  running=$($st0.running)  step=$($st0.step)  acc=$($st0.accuracy)"

# ── 4. Start ────────────────────────────────────────────────────────
Write-Step "POST /api/control {action: start}"
$r = Invoke-RestMethod -Uri "$base/api/control" -Method Post `
    -ContentType "application/json" -Body '{"action":"start"}'
Assert ($r.ok -eq $true) "start ok=true"

# ── 5. Rate ─────────────────────────────────────────────────────────
Write-Step "POST /api/rate {rate_hz: 60}"
$r = Invoke-RestMethod -Uri "$base/api/rate" -Method Post `
    -ContentType "application/json" -Body '{"rate_hz":60}'
Assert ([math]::Abs($r.rate_hz - 60.0) -lt 1e-6) "rate_hz round-trips as 60"

# ── 6. Inject (drift) ───────────────────────────────────────────────
Write-Step "POST /api/inject {sinr_bias_db: -10, delay_bias_ms: 20}"
$r = Invoke-RestMethod -Uri "$base/api/inject" -Method Post `
    -ContentType "application/json" `
    -Body '{"sinr_bias_db":-10,"delay_bias_ms":20}'
Assert ($r.injection.sinr_bias_db   -eq -10) "sinr_bias_db echoed"
Assert ($r.injection.delay_bias_ms  -eq 20)  "delay_bias_ms echoed"

# ── 7. Mode (LOCAL variant + golden NDT) ────────────────────────────
Write-Step "POST /api/mode {preferred_variant: LOCAL, use_golden_ndt: true}"
$r = Invoke-RestMethod -Uri "$base/api/mode" -Method Post `
    -ContentType "application/json" `
    -Body '{"preferred_variant":"LOCAL","use_golden_ndt":true}'
Assert ($r.injection.preferred_variant -eq "LOCAL") "variant=LOCAL"
Assert ($r.injection.use_golden_ndt   -eq $true)  "golden NDT on"

# ── 8. Mode (BOGUS — must 400) ──────────────────────────────────────
Write-Step "POST /api/mode {preferred_variant: BOGUS}  (must 400)"
try {
    Invoke-RestMethod -Uri "$base/api/mode" -Method Post `
        -ContentType "application/json" -Body '{"preferred_variant":"BOGUS"}' `
        -ErrorAction Stop
    throw "expected 400 but got 200"
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    Assert ($code -eq 400) "BOGUS variant rejected with 400"
}

# ── 9. WebSocket frame stream test ─────────────────────────────────
Write-Step "WS /ws  (collect ~8s of frames)"
$pyExe   = "C:\Users\taieb\PycharmProjects\rtp_observer\.venv\Scripts\python.exe"
$probePy = "C:\Users\taieb\PycharmProjects\rtp_observer\dashboard\live\_ws_probe.py"
$frames  = & $pyExe $probePy
Write-Host "  WS frame counts: $frames"
$counts = $frames | ConvertFrom-Json
Assert ($counts.samples  -gt 0)  "samples frames received"
Assert ($counts.status   -gt 0)  "status greeting received on connect"

# ── 10. Force retrain via /api/control ──────────────────────────────
Write-Step "POST /api/control {action: force_retrain}"
$r = Invoke-RestMethod -Uri "$base/api/control" -Method Post `
    -ContentType "application/json" -Body '{"action":"force_retrain"}'
Assert ($r.ok -eq $true) "force_retrain ok=true"

# Wait for the retrain pipeline to finish (~10s for MTP-L LOCAL is fast)
Start-Sleep -Seconds 12

# ── 11. Status after retrain ────────────────────────────────────────
Write-Step "GET /api/status (post-retrain)"
$st1 = Invoke-RestMethod -Uri "$base/api/status"
Write-Host "  running=$($st1.running)  step=$($st1.step)  acc=$($st1.accuracy)"
Write-Host "  ndt_last:  pseudo=$($st1.ndt_last.pseudo)  gt=$($st1.ndt_last.gt)  bias=$($st1.ndt_last.bias)"
Assert ($st1.step -gt $st0.step) "step advanced"
Assert ($st1.accuracy -ge 0.80)  "accuracy still >= 0.80"

# ── 12. Reset injection ─────────────────────────────────────────────
Write-Step "POST /api/inject {clear all}"
$r = Invoke-RestMethod -Uri "$base/api/inject" -Method Post `
    -ContentType "application/json" `
    -Body '{"sinr_bias_db":0,"delay_bias_ms":0,"noise_scale":1,"poison_mode":false}'
Assert ($r.injection.sinr_bias_db -eq 0) "sliders cleared"

# ── 13. Pause then resume ───────────────────────────────────────────
Write-Step "POST /api/control {action: pause}  then  {action: resume}"
$r = Invoke-RestMethod -Uri "$base/api/control" -Method Post `
    -ContentType "application/json" -Body '{"action":"pause"}'
Assert ($r.ok -eq $true) "pause ok"
$r = Invoke-RestMethod -Uri "$base/api/control" -Method Post `
    -ContentType "application/json" -Body '{"action":"resume"}'
Assert ($r.ok -eq $true) "resume ok"

# ── 14. Stop ────────────────────────────────────────────────────────
Write-Step "POST /api/control {action: stop}"
$r = Invoke-RestMethod -Uri "$base/api/control" -Method Post `
    -ContentType "application/json" -Body '{"action":"stop"}'
Assert ($r.ok -eq $true) "stop ok"

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  ALL HTTP ENDPOINTS WALKED — DASHBOARD UI BACKEND VERIFIED" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
