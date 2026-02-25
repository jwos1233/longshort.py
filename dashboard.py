"""
Macro Quadrant Strategy Dashboard
===================================

Railway-deployable web dashboard showing:
  - Current regime & quadrant scores
  - Current target allocations with constituent breakdown
  - Historical backtest performance (equity curve, metrics)
  - Cached data to avoid repeated yfinance downloads

Usage:
    python dashboard.py

Environment Variables:
    DASHBOARD_USER  : Username for login (default: admin)
    DASHBOARD_PASS  : Password for login (default: changeme)
    DASHBOARD_SECRET: Secret key for sessions (auto-generated if not set)
    PORT            : Server port (default: 8000, Railway sets this)
"""

import os
import sys
import json
import time
import asyncio
import secrets
import hashlib
import traceback
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI, Request, Form, Depends, HTTPException,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

# ── Configuration ────────────────────────────────────────────────────
USERNAME = os.getenv("DASHBOARD_USER", "admin")
PASSWORD = os.getenv("DASHBOARD_PASS", "changeme")
SECRET_KEY = os.getenv("DASHBOARD_SECRET", secrets.token_hex(32))

# Cache TTLs (seconds)
SIGNAL_CACHE_TTL = 15 * 60       # 15 minutes
BACKTEST_CACHE_TTL = 6 * 60 * 60  # 6 hours

app = FastAPI(title="Macro Quadrant Dashboard")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# ── Simple In-Memory Cache ───────────────────────────────────────────
_cache: dict = {}


def cache_get(key: str):
    """Return cached (value, age_seconds) or (None, None) if expired/missing."""
    entry = _cache.get(key)
    if entry is None:
        return None, None
    ts, ttl, val = entry
    age = time.time() - ts
    if age > ttl:
        return None, None
    return val, age


def cache_set(key: str, value, ttl: int):
    _cache[key] = (time.time(), ttl, value)


def cache_clear():
    _cache.clear()


# ── Auth helpers ─────────────────────────────────────────────────────
def verify_session(request: Request) -> bool:
    return request.session.get("authenticated", False)


def require_auth(request: Request):
    if not verify_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True


# ── Signal generation (runs in thread pool) ──────────────────────────
def _build_ticker_to_quads() -> dict:
    """Build ticker -> [quadrants] mapping including constituent stocks."""
    from config import QUAD_ALLOCATIONS, EXPAND_TO_CONSTITUENTS, BTC_PROXY_BASKET
    from etf_mapper import ETF_CONSTITUENTS

    mapping: dict = {}
    for quad, allocations in QUAD_ALLOCATIONS.items():
        for etf in allocations.keys():
            mapping.setdefault(etf, []).append(quad)
            if EXPAND_TO_CONSTITUENTS and etf in ETF_CONSTITUENTS:
                for stock in ETF_CONSTITUENTS[etf]:
                    if quad not in mapping.get(stock, []):
                        mapping.setdefault(stock, []).append(quad)
    # BTC proxy tickers inherit Q1 (they replace BTC-USD which is Q1)
    for proxy in BTC_PROXY_BASKET.keys():
        if 'Q1' not in mapping.get(proxy, []):
            mapping.setdefault(proxy, []).append('Q1')
    return mapping


def _generate_signals_sync() -> dict:
    """Run SignalGenerator and return JSON-safe result dict."""
    from signal_generator import SignalGenerator
    from ticker_names import TICKER_NAMES

    sg = SignalGenerator(
        momentum_days=20,
        ema_period=50,
        vol_lookback=30,
        max_positions=10,
        atr_stop_loss=2.0,
        atr_period=14,
        ema_smoothing_period=20,
    )
    raw = sg.generate_signals()

    # Build JSON-safe payload
    quad_scores = {k: round(float(v), 4) for k, v in raw["quadrant_scores"].items()}
    top1, top2 = raw["top_quadrants"]

    # Build full ticker → quadrant mapping (includes constituent stocks)
    ticker_quads = _build_ticker_to_quads()

    # Target weights sorted by weight descending
    sorted_weights = sorted(raw["target_weights"].items(), key=lambda x: x[1], reverse=True)
    positions = []
    for ticker, weight in sorted_weights:
        quads = ticker_quads.get(ticker, [])
        positions.append({
            "ticker": ticker,
            "name": TICKER_NAMES.get(ticker, ""),
            "weight": round(float(weight), 6),
            "weight_pct": round(float(weight) * 100, 2),
            "quadrants": quads,
        })

    excluded = []
    for ticker, info in raw.get("excluded_below_ema", {}).items():
        excluded.append({
            "ticker": ticker,
            "name": TICKER_NAMES.get(ticker, ""),
            "price": info.get("price"),
            "ema": info.get("ema"),
            "quadrant": info.get("quadrant", ""),
        })

    return {
        "regime": f"{top1} + {top2}",
        "top_quadrants": [top1, top2],
        "quad_scores": quad_scores,
        "positions": positions,
        "excluded": excluded,
        "total_leverage": round(float(raw["total_leverage"]), 4),
        "num_positions": len(positions),
        "price_date": str(raw.get("price_date", "")),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def _run_backtest_sync(months: int = 18) -> dict:
    """Run production backtest and return JSON-safe result dict."""
    from quad_portfolio_backtest import QuadrantPortfolioBacktest

    today = date.today()
    end_dt = datetime.combine(today - timedelta(days=1), datetime.min.time())
    start_dt = end_dt - timedelta(days=int(months * 30.5) + 100)

    bt = QuadrantPortfolioBacktest(
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=50000,
        momentum_days=20,
        max_positions=10,
        atr_stop_loss=2.0,
        atr_period=14,
        ema_smoothing_period=20,
    )
    results = bt.run_backtest()

    # Equity curve (downsample to daily for JSON)
    pv = bt.portfolio_value
    equity_dates = []
    equity_values = []
    if pv is not None and len(pv) > 0:
        for ts, val in pv.items():
            d = ts.date() if hasattr(ts, "date") else ts
            equity_dates.append(str(d))
            equity_values.append(round(float(val), 2))

    # Quad history
    quad_timeline = []
    if bt.quad_history is not None and len(bt.quad_history) > 0:
        for ts, row in bt.quad_history.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            quad_timeline.append({
                "date": str(d),
                "top1": row.get("Top1", ""),
                "top2": row.get("Top2", ""),
            })

    return {
        "total_return": round(float(results["total_return"]), 2),
        "annual_return": round(float(results["annual_return"]), 2),
        "sharpe": round(float(results["sharpe"]), 2),
        "max_drawdown": round(float(results["max_drawdown"]), 2),
        "annual_vol": round(float(results["annual_vol"]), 2),
        "final_value": round(float(results["final_value"]), 2),
        "initial_capital": 50000,
        "months": months,
        "start_date": str(start_dt.date()),
        "end_date": str(end_dt.date()),
        "equity_curve": {"dates": equity_dates, "values": equity_values},
        "quad_timeline": quad_timeline,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ── Health check (no auth, responds instantly) ──────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Background startup: pre-warm signal cache ───────────────────────
@app.on_event("startup")
async def startup_warm_cache():
    """Pre-warm signal cache in background so first page load is fast."""
    async def _warm():
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, _generate_signals_sync)
            cache_set("signals", result, SIGNAL_CACHE_TTL)
            print("[startup] Signal cache pre-warmed successfully")
        except Exception as e:
            print(f"[startup] Signal cache pre-warm failed: {e}")
    asyncio.create_task(_warm())


# ── API Routes ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if not verify_session(request):
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if verify_session(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return HTMLResponse(content=LOGIN_HTML)


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == USERNAME and password == PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url="/dashboard", status_code=302)
    return HTMLResponse(
        content=LOGIN_HTML.replace("</form>", '<p class="error">Invalid credentials</p></form>')
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not verify_session(request):
        return RedirectResponse(url="/login", status_code=302)
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/signals")
async def api_signals(request: Request, _: bool = Depends(require_auth)):
    cached, age = cache_get("signals")
    if cached is not None:
        cached["_cached"] = True
        cached["_cache_age_s"] = round(age, 1)
        return cached

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _generate_signals_sync)
        cache_set("signals", result, SIGNAL_CACHE_TTL)
        result["_cached"] = False
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/signals/refresh")
async def api_signals_refresh(request: Request, _: bool = Depends(require_auth)):
    _cache.pop("signals", None)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _generate_signals_sync)
        cache_set("signals", result, SIGNAL_CACHE_TTL)
        result["_cached"] = False
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/backtest")
async def api_backtest(request: Request, months: int = 18, _: bool = Depends(require_auth)):
    cache_key = f"backtest_{months}"
    cached, age = cache_get(cache_key)
    if cached is not None:
        cached["_cached"] = True
        cached["_cache_age_s"] = round(age, 1)
        return cached

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_backtest_sync, months)
        cache_set(cache_key, result, BACKTEST_CACHE_TTL)
        result["_cached"] = False
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/backtest/refresh")
async def api_backtest_refresh(request: Request, months: int = 18, _: bool = Depends(require_auth)):
    cache_key = f"backtest_{months}"
    _cache.pop(cache_key, None)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_backtest_sync, months)
        cache_set(cache_key, result, BACKTEST_CACHE_TTL)
        result["_cached"] = False
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/cache")
async def api_cache_status(request: Request, _: bool = Depends(require_auth)):
    info = {}
    now = time.time()
    for key, (ts, ttl, _val) in _cache.items():
        age = now - ts
        info[key] = {
            "age_s": round(age, 1),
            "ttl_s": ttl,
            "expires_in_s": round(max(0, ttl - age), 1),
            "expired": age > ttl,
        }
    return info


@app.post("/api/cache/clear")
async def api_cache_clear(request: Request, _: bool = Depends(require_auth)):
    cache_clear()
    return {"status": "cleared"}


# ── HTML Templates ───────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login - Macro Quadrant Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.login-container{background:#1a1a24;padding:2rem;border-radius:12px;border:1px solid #2a2a3a;width:100%;max-width:400px;margin:1rem}
h1{text-align:center;margin-bottom:1.5rem;color:#fff;font-size:1.5rem}
.form-group{margin-bottom:1rem}
label{display:block;margin-bottom:.5rem;color:#888;font-size:.875rem}
input{width:100%;padding:.75rem;border:1px solid #2a2a3a;border-radius:6px;background:#0a0a0f;color:#e0e0e0;font-size:1rem}
input:focus{outline:none;border-color:#4a9eff}
button{width:100%;padding:.75rem;background:#4a9eff;color:#fff;border:none;border-radius:6px;font-size:1rem;cursor:pointer;margin-top:.5rem}
button:hover{background:#3a8eef}
.error{color:#ff6b6b;text-align:center;margin-top:1rem;font-size:.875rem}
.logo{text-align:center;font-size:2rem;margin-bottom:.5rem}
</style>
</head>
<body>
<div class="login-container">
  <div class="logo">&#x1F4C8;</div>
  <h1>Macro Quadrant Dashboard</h1>
  <form method="POST" action="/login">
    <div class="form-group"><label for="username">Username</label><input type="text" id="username" name="username" required autocomplete="username"></div>
    <div class="form-group"><label for="password">Password</label><input type="password" id="password" name="password" required autocomplete="current-password"></div>
    <button type="submit">Sign In</button>
  </form>
</div>
</body>
</html>
"""

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Macro Quadrant Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg0:#0a0a0f;--bg1:#1a1a24;--bg2:#2a2a3a;--bg3:#3a3a4a;--tx0:#e0e0e0;--tx1:#aaa;--tx2:#666;--accent:#4a9eff;--green:#4ade80;--red:#f87171;--yellow:#fbbf24;--q1:#4ade80;--q2:#fbbf24;--q3:#f97316;--q4:#60a5fa}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg0);color:var(--tx0);min-height:100vh}
a{color:var(--accent);text-decoration:none}
.header{background:var(--bg1);border-bottom:1px solid var(--bg2);padding:.75rem 1.5rem;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:10}
.header h1{font-size:1.1rem;display:flex;align-items:center;gap:.5rem}
.header-right{display:flex;align-items:center;gap:1rem}
.header-right a{color:var(--tx1);font-size:.85rem}
.tabs{display:flex;gap:0;background:var(--bg1);border-bottom:1px solid var(--bg2);padding:0 1.5rem;overflow-x:auto}
.tab{padding:.75rem 1.25rem;cursor:pointer;font-size:.85rem;color:var(--tx1);border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s}
.tab:hover{color:var(--tx0)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.container{max-width:1400px;margin:0 auto;padding:1.25rem}
.tab-content{display:none}
.tab-content.active{display:block}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;margin-bottom:1.25rem}
.card{background:var(--bg1);border:1px solid var(--bg2);border-radius:8px;padding:1rem}
.card-label{color:var(--tx1);font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.35rem}
.card-value{font-size:1.3rem;font-weight:700}
.card-sub{font-size:.75rem;color:var(--tx1);margin-top:.25rem}

/* Quad scores */
.quad-scores{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:1.25rem}
.quad-card{background:var(--bg1);border:1px solid var(--bg2);border-radius:8px;padding:.75rem;position:relative;overflow:hidden}
.quad-card.top{border-color:var(--accent)}
.quad-name{font-weight:700;font-size:.9rem;margin-bottom:.25rem}
.quad-desc{font-size:.7rem;color:var(--tx1);margin-bottom:.5rem}
.quad-score{font-size:1.5rem;font-weight:700}
.quad-bar{height:4px;border-radius:2px;margin-top:.5rem;background:var(--bg2)}
.quad-bar-fill{height:100%;border-radius:2px;transition:width .4s}
.quad-badge{position:absolute;top:.5rem;right:.5rem;background:var(--accent);color:#000;font-size:.6rem;padding:.15rem .4rem;border-radius:4px;font-weight:700}

/* Table */
.panel{background:var(--bg1);border:1px solid var(--bg2);border-radius:8px;overflow:hidden;margin-bottom:1.25rem}
.panel-header{background:var(--bg2);padding:.6rem 1rem;display:flex;justify-content:space-between;align-items:center}
.panel-header h2{font-size:.85rem;font-weight:600}
.panel-header .badge{font-size:.7rem;background:var(--bg3);padding:.2rem .5rem;border-radius:4px;color:var(--tx1)}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{text-align:left;padding:.5rem .75rem;color:var(--tx1);font-weight:500;font-size:.7rem;text-transform:uppercase;letter-spacing:.03em;border-bottom:1px solid var(--bg2)}
td{padding:.5rem .75rem;border-bottom:1px solid var(--bg2)}
tr:last-child td{border-bottom:none}
.weight-bar{display:inline-block;height:12px;border-radius:2px;min-width:2px;vertical-align:middle;margin-right:.5rem}

/* Chart */
.chart-wrap{background:var(--bg1);border:1px solid var(--bg2);border-radius:8px;padding:1rem;margin-bottom:1.25rem}
.chart-wrap canvas{width:100%!important}

/* Status / loading */
.loader{display:inline-block;width:16px;height:16px;border:2px solid var(--bg2);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}
.status-msg{padding:2rem;text-align:center;color:var(--tx1)}
.btn{padding:.4rem .9rem;border:1px solid var(--bg2);border-radius:6px;background:var(--bg1);color:var(--tx0);cursor:pointer;font-size:.8rem;transition:all .15s}
.btn:hover{border-color:var(--tx1);background:var(--bg2)}
.btn-sm{padding:.25rem .6rem;font-size:.7rem}
.positive{color:var(--green)}.negative{color:var(--red)}

/* Responsive */
@media(max-width:768px){
  .quad-scores{grid-template-columns:repeat(2,1fr)}
  .cards{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:480px){
  .quad-scores{grid-template-columns:1fr}
  .cards{grid-template-columns:1fr}
}
</style>
</head>
<body>

<header class="header">
  <h1>&#x1F4C8; Macro Quadrant Dashboard</h1>
  <div class="header-right">
    <span id="cacheStatus" style="font-size:.7rem;color:var(--tx2)"></span>
    <a href="/logout">Logout</a>
  </div>
</header>

<div class="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="allocations">Allocations</div>
  <div class="tab" data-tab="performance">Performance</div>
</div>

<div class="container">

  <!-- ═══════ OVERVIEW TAB ═══════ -->
  <div id="tab-overview" class="tab-content active">
    <div id="signalLoading" class="status-msg"><span class="loader"></span> Loading signals...</div>
    <div id="signalContent" style="display:none">

      <div class="cards" id="overviewCards"></div>

      <div class="quad-scores" id="quadScores"></div>

      <div class="panel">
        <div class="panel-header">
          <h2>Target Positions</h2>
          <div>
            <span id="posCount" class="badge"></span>
            <button class="btn btn-sm" onclick="refreshSignals()" style="margin-left:.5rem">Refresh</button>
          </div>
        </div>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>Ticker</th><th>Weight</th><th style="width:40%">Allocation</th><th>Quadrant</th></tr></thead>
            <tbody id="positionsTable"></tbody>
          </table>
        </div>
      </div>

      <div class="panel" id="excludedPanel" style="display:none">
        <div class="panel-header"><h2>Excluded (Below EMA)</h2></div>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>Ticker</th><th>Price</th><th>50d EMA</th><th>Gap</th><th>Quadrant</th></tr></thead>
            <tbody id="excludedTable"></tbody>
          </table>
        </div>
      </div>

    </div>
  </div>

  <!-- ═══════ ALLOCATIONS TAB ═══════ -->
  <div id="tab-allocations" class="tab-content">
    <div id="allocLoading" class="status-msg"><span class="loader"></span> Loading allocations...</div>
    <div id="allocContent" style="display:none">

      <div class="panel">
        <div class="panel-header"><h2>Allocation by Quadrant</h2></div>
        <div style="padding:1rem" id="quadAllocChart"></div>
      </div>

      <div class="panel">
        <div class="panel-header"><h2>Full Position Breakdown</h2></div>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>#</th><th>Ticker</th><th>Weight %</th><th style="width:45%">Bar</th><th>Quadrant</th></tr></thead>
            <tbody id="fullAllocTable"></tbody>
          </table>
        </div>
      </div>

    </div>
  </div>

  <!-- ═══════ PERFORMANCE TAB ═══════ -->
  <div id="tab-performance" class="tab-content">
    <div id="btLoading" class="status-msg"><span class="loader"></span> Running backtest (this may take a minute)...</div>
    <div id="btContent" style="display:none">

      <div class="cards" id="btCards"></div>

      <div class="chart-wrap">
        <canvas id="equityChart" height="300"></canvas>
      </div>

      <div class="panel">
        <div class="panel-header">
          <h2>Regime Timeline</h2>
          <div>
            <button class="btn btn-sm" onclick="refreshBacktest()">Refresh</button>
          </div>
        </div>
        <div style="overflow-x:auto;max-height:300px;overflow-y:auto">
          <table>
            <thead><tr><th>Date</th><th>Top Quadrant</th><th>Second Quadrant</th></tr></thead>
            <tbody id="quadTimeline"></tbody>
          </table>
        </div>
      </div>

    </div>
  </div>

</div>

<script>
// ── State ──
let signalData = null;
let btData = null;
let equityChart = null;

const QUAD_COLORS = {Q1:'#4ade80',Q2:'#fbbf24',Q3:'#f97316',Q4:'#60a5fa'};
const QUAD_LABELS = {Q1:'Goldilocks',Q2:'Reflation',Q3:'Stagflation',Q4:'Deflation'};
const QUAD_ARROWS = {Q1:'Growth \u2191  Inflation \u2193',Q2:'Growth \u2191  Inflation \u2191',Q3:'Growth \u2193  Inflation \u2191',Q4:'Growth \u2193  Inflation \u2193'};

// ── Tabs ──
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    // Lazy-load
    if (t.dataset.tab === 'performance' && !btData) loadBacktest();
    if (t.dataset.tab === 'allocations' && signalData) renderAllocations(signalData);
  });
});

// ── Signals ──
async function loadSignals() {
  try {
    const r = await fetch('/api/signals');
    if (!r.ok) throw new Error(await r.text());
    signalData = await r.json();
    renderOverview(signalData);
    updateCacheStatus(signalData);
  } catch(e) {
    document.getElementById('signalLoading').innerHTML = '<span style="color:var(--red)">Error: ' + e.message + '</span>';
  }
}

async function refreshSignals() {
  document.getElementById('signalContent').style.display = 'none';
  document.getElementById('signalLoading').style.display = '';
  document.getElementById('signalLoading').innerHTML = '<span class="loader"></span> Refreshing signals...';
  try {
    const r = await fetch('/api/signals/refresh', {method:'POST'});
    if (!r.ok) throw new Error(await r.text());
    signalData = await r.json();
    renderOverview(signalData);
    updateCacheStatus(signalData);
  } catch(e) {
    document.getElementById('signalLoading').innerHTML = '<span style="color:var(--red)">Error: ' + e.message + '</span>';
  }
}

function updateCacheStatus(data) {
  const el = document.getElementById('cacheStatus');
  if (data._cached) {
    const m = Math.floor(data._cache_age_s / 60);
    el.textContent = 'Cached ' + m + 'm ago';
  } else {
    el.textContent = 'Fresh data';
  }
}

function renderOverview(d) {
  document.getElementById('signalLoading').style.display = 'none';
  document.getElementById('signalContent').style.display = '';

  // Cards
  const lev = d.total_leverage;
  const levClass = lev > 2 ? 'positive' : '';
  document.getElementById('overviewCards').innerHTML = `
    <div class="card">
      <div class="card-label">Current Regime</div>
      <div class="card-value">${d.regime}</div>
      <div class="card-sub">${QUAD_LABELS[d.top_quadrants[0]] || ''} + ${QUAD_LABELS[d.top_quadrants[1]] || ''}</div>
    </div>
    <div class="card">
      <div class="card-label">Positions</div>
      <div class="card-value">${d.num_positions}</div>
      <div class="card-sub">${d.excluded.length} excluded</div>
    </div>
    <div class="card">
      <div class="card-label">Total Leverage</div>
      <div class="card-value ${levClass}">${lev.toFixed(2)}x</div>
    </div>
    <div class="card">
      <div class="card-label">Price Date</div>
      <div class="card-value" style="font-size:1rem">${d.price_date}</div>
      <div class="card-sub">Yesterday's close</div>
    </div>
  `;

  // Quad scores
  const scores = d.quad_scores;
  const maxAbs = Math.max(...Object.values(scores).map(Math.abs), 1);
  let qhtml = '';
  for (const q of ['Q1','Q2','Q3','Q4']) {
    const s = scores[q] || 0;
    const isTop = d.top_quadrants.includes(q);
    const pct = Math.max(5, (Math.abs(s) / maxAbs) * 100);
    const col = QUAD_COLORS[q];
    qhtml += `
      <div class="quad-card ${isTop?'top':''}">
        ${isTop ? '<span class="quad-badge">ACTIVE</span>' : ''}
        <div class="quad-name" style="color:${col}">${q}</div>
        <div class="quad-desc">${QUAD_ARROWS[q]}</div>
        <div class="quad-score ${s>=0?'positive':'negative'}">${s>=0?'+':''}${s.toFixed(2)}%</div>
        <div class="quad-bar"><div class="quad-bar-fill" style="width:${pct}%;background:${col}"></div></div>
      </div>
    `;
  }
  document.getElementById('quadScores').innerHTML = qhtml;

  // Positions table
  document.getElementById('posCount').textContent = d.num_positions + ' positions';
  const maxW = d.positions.length > 0 ? d.positions[0].weight : 1;
  let phtml = '';
  for (const p of d.positions) {
    const barW = (p.weight / maxW) * 100;
    const col = p.quadrants.length > 0 ? (QUAD_COLORS[p.quadrants[0]] || 'var(--accent)') : 'var(--accent)';
    const nameSpan = p.name ? `<span style="color:var(--tx1);font-weight:400;margin-left:.4rem;font-size:.75rem">${p.name}</span>` : '';
    phtml += `<tr>
      <td><strong>${p.ticker}</strong>${nameSpan}</td>
      <td>${p.weight_pct.toFixed(2)}%</td>
      <td><span class="weight-bar" style="width:${barW}%;background:${col}"></span></td>
      <td style="color:var(--tx1)">${p.quadrants.join(', ')}</td>
    </tr>`;
  }
  document.getElementById('positionsTable').innerHTML = phtml;

  // Excluded
  if (d.excluded.length > 0) {
    document.getElementById('excludedPanel').style.display = '';
    let ehtml = '';
    for (const e of d.excluded) {
      const gap = e.ema && e.price ? ((e.price - e.ema) / e.ema * 100).toFixed(2) : '-';
      const eName = e.name ? `<span style="color:var(--tx1);font-weight:400;margin-left:.4rem;font-size:.75rem">${e.name}</span>` : '';
      ehtml += `<tr>
        <td><strong>${e.ticker}</strong>${eName}</td>
        <td>${e.price ? '$'+e.price.toFixed(2) : '-'}</td>
        <td>${e.ema ? '$'+e.ema.toFixed(2) : '-'}</td>
        <td class="negative">${gap}%</td>
        <td style="color:var(--tx1)">${e.quadrant}</td>
      </tr>`;
    }
    document.getElementById('excludedTable').innerHTML = ehtml;
  } else {
    document.getElementById('excludedPanel').style.display = 'none';
  }
}

// ── Allocations ──
function renderAllocations(d) {
  document.getElementById('allocLoading').style.display = 'none';
  document.getElementById('allocContent').style.display = '';

  // Group by quadrant
  const byQuad = {};
  let unassigned = [];
  for (const p of d.positions) {
    if (p.quadrants.length > 0) {
      const q = p.quadrants[0];
      if (!byQuad[q]) byQuad[q] = [];
      byQuad[q].push(p);
    } else {
      unassigned.push(p);
    }
  }

  // Quad alloc summary
  let qaHtml = '<div style="display:flex;gap:1rem;flex-wrap:wrap">';
  for (const q of ['Q1','Q2','Q3','Q4']) {
    const items = byQuad[q] || [];
    const totalW = items.reduce((s,p) => s + p.weight, 0);
    if (totalW === 0) continue;
    const col = QUAD_COLORS[q];
    qaHtml += `<div style="flex:1;min-width:200px;background:var(--bg0);border-radius:6px;padding:.75rem;border-left:3px solid ${col}">
      <div style="font-weight:700;color:${col};margin-bottom:.25rem">${q} - ${QUAD_LABELS[q]}</div>
      <div style="font-size:1.2rem;font-weight:700">${(totalW*100).toFixed(1)}%</div>
      <div style="font-size:.7rem;color:var(--tx1);margin-top:.25rem">${items.length} positions</div>
      <div style="margin-top:.5rem;font-size:.75rem">${items.map(p=>'<span style="margin-right:.5rem">'+p.ticker+'</span>').join('')}</div>
    </div>`;
  }
  qaHtml += '</div>';
  document.getElementById('quadAllocChart').innerHTML = qaHtml;

  // Full table
  const maxW = d.positions.length > 0 ? d.positions[0].weight : 1;
  let fhtml = '';
  d.positions.forEach((p, i) => {
    const barW = (p.weight / maxW) * 100;
    const col = p.quadrants.length > 0 ? (QUAD_COLORS[p.quadrants[0]] || 'var(--accent)') : 'var(--accent)';
    const nameSpan = p.name ? `<span style="color:var(--tx1);font-weight:400;margin-left:.4rem;font-size:.75rem">${p.name}</span>` : '';
    fhtml += `<tr>
      <td>${i+1}</td>
      <td><strong>${p.ticker}</strong>${nameSpan}</td>
      <td>${p.weight_pct.toFixed(2)}%</td>
      <td><div style="background:var(--bg0);border-radius:2px;height:16px;position:relative"><div style="height:100%;width:${barW}%;background:${col};border-radius:2px"></div></div></td>
      <td style="color:var(--tx1)">${p.quadrants.join(', ')}</td>
    </tr>`;
  });
  document.getElementById('fullAllocTable').innerHTML = fhtml;
}

// ── Backtest ──
async function loadBacktest() {
  try {
    const r = await fetch('/api/backtest?months=18');
    if (!r.ok) throw new Error(await r.text());
    btData = await r.json();
    renderBacktest(btData);
  } catch(e) {
    document.getElementById('btLoading').innerHTML = '<span style="color:var(--red)">Error: ' + e.message + '</span>';
  }
}

async function refreshBacktest() {
  document.getElementById('btContent').style.display = 'none';
  document.getElementById('btLoading').style.display = '';
  document.getElementById('btLoading').innerHTML = '<span class="loader"></span> Running backtest...';
  try {
    const r = await fetch('/api/backtest/refresh?months=18', {method:'POST'});
    if (!r.ok) throw new Error(await r.text());
    btData = await r.json();
    renderBacktest(btData);
  } catch(e) {
    document.getElementById('btLoading').innerHTML = '<span style="color:var(--red)">Error: ' + e.message + '</span>';
  }
}

function renderBacktest(d) {
  document.getElementById('btLoading').style.display = 'none';
  document.getElementById('btContent').style.display = '';

  const retClass = d.total_return >= 0 ? 'positive' : 'negative';
  const ddClass = 'negative';

  document.getElementById('btCards').innerHTML = `
    <div class="card"><div class="card-label">Total Return</div><div class="card-value ${retClass}">${d.total_return.toFixed(1)}%</div><div class="card-sub">${d.months}mo backtest</div></div>
    <div class="card"><div class="card-label">Ann. Return</div><div class="card-value ${retClass}">${d.annual_return.toFixed(1)}%</div></div>
    <div class="card"><div class="card-label">Sharpe Ratio</div><div class="card-value">${d.sharpe.toFixed(2)}</div></div>
    <div class="card"><div class="card-label">Max Drawdown</div><div class="card-value ${ddClass}">${d.max_drawdown.toFixed(1)}%</div></div>
    <div class="card"><div class="card-label">Volatility</div><div class="card-value">${d.annual_vol.toFixed(1)}%</div></div>
    <div class="card"><div class="card-label">Final Value</div><div class="card-value">$${d.final_value.toLocaleString()}</div><div class="card-sub">from $${d.initial_capital.toLocaleString()}</div></div>
  `;

  // Equity chart
  const ec = d.equity_curve;
  if (equityChart) equityChart.destroy();

  // Build quad color bands for background
  const quadMap = {};
  (d.quad_timeline || []).forEach(q => { quadMap[q.date] = q.top1; });

  const bgColors = ec.dates.map(dt => {
    const q = quadMap[dt];
    if (q === 'Q1') return 'rgba(74,222,128,0.07)';
    if (q === 'Q2') return 'rgba(251,191,36,0.07)';
    if (q === 'Q3') return 'rgba(249,115,22,0.07)';
    if (q === 'Q4') return 'rgba(96,165,250,0.07)';
    return 'transparent';
  });

  const ctx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ec.dates,
      datasets: [{
        label: 'Portfolio Value ($)',
        data: ec.values,
        borderColor: '#4a9eff',
        backgroundColor: 'rgba(74,158,255,0.08)',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {mode:'index', intersect:false},
      plugins: {
        legend: {display:false},
        tooltip: {
          callbacks: {
            label: ctx => '$' + ctx.parsed.y.toLocaleString(),
            title: items => items[0].label,
          }
        }
      },
      scales: {
        x: {
          grid:{color:'rgba(255,255,255,0.03)'},
          ticks:{color:'#666',maxTicksLimit:8,font:{size:10}},
        },
        y: {
          grid:{color:'rgba(255,255,255,0.05)'},
          ticks:{color:'#666',callback:v=>'$'+v.toLocaleString(),font:{size:10}},
        }
      }
    }
  });

  // Quad timeline (last 30 entries)
  const timeline = (d.quad_timeline || []).slice(-60);
  let thtml = '';
  for (const q of timeline.reverse()) {
    const c1 = QUAD_COLORS[q.top1] || 'var(--tx1)';
    const c2 = QUAD_COLORS[q.top2] || 'var(--tx1)';
    thtml += `<tr>
      <td style="font-size:.75rem">${q.date}</td>
      <td><span style="color:${c1};font-weight:600">${q.top1}</span> <span style="font-size:.7rem;color:var(--tx1)">${QUAD_LABELS[q.top1]||''}</span></td>
      <td><span style="color:${c2};font-weight:600">${q.top2}</span> <span style="font-size:.7rem;color:var(--tx1)">${QUAD_LABELS[q.top2]||''}</span></td>
    </tr>`;
  }
  document.getElementById('quadTimeline').innerHTML = thtml;
}

// ── Init ──
loadSignals();
</script>
</body>
</html>
"""


# ── Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    print("=" * 60)
    print("Macro Quadrant Dashboard")
    print("=" * 60)
    print(f"  Username: {USERNAME}")
    print(f"  Password: {'*' * len(PASSWORD)}")
    print(f"  Signal cache TTL:   {SIGNAL_CACHE_TTL // 60}m")
    print(f"  Backtest cache TTL: {BACKTEST_CACHE_TTL // 3600}h")
    print(f"  Server: http://0.0.0.0:{port}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
