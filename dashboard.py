"""
Trading Strategy Web Dashboard
==============================

A simple web interface to remotely trigger and monitor the trading strategy.

Usage:
    python dashboard.py

Then open http://localhost:8000 in your browser.

Environment Variables:
    DASHBOARD_USER: Username for login (default: admin)
    DASHBOARD_PASS: Password for login (default: changeme)
    DASHBOARD_SECRET: Secret key for sessions (auto-generated if not set)
"""

import os
import sys
import asyncio
import subprocess
import signal
import secrets
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

# Configuration
USERNAME = os.getenv("DASHBOARD_USER", "admin")
PASSWORD = os.getenv("DASHBOARD_PASS", "changeme")
SECRET_KEY = os.getenv("DASHBOARD_SECRET", secrets.token_hex(32))
STRATEGY_CMD = [sys.executable, "live_trader_simple.py", "--port", "7496", "--live"]
MAX_LOG_LINES = 1000

app = FastAPI(title="Trading Strategy Dashboard")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Global state
class StrategyState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "idle"  # idle, running, error
        self.logs: deque = deque(maxlen=MAX_LOG_LINES)
        self.last_run: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.websockets: list[WebSocket] = []
        self.positions: dict = {}

state = StrategyState()


# Authentication helpers
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_session(request: Request) -> bool:
    return request.session.get("authenticated", False)

def require_auth(request: Request):
    if not verify_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True


# WebSocket broadcast
async def broadcast_log(message: str):
    """Broadcast log message to all connected WebSocket clients."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    state.logs.append(log_entry)

    for ws in state.websockets[:]:
        try:
            await ws.send_json({"type": "log", "data": log_entry})
        except:
            state.websockets.remove(ws)

async def broadcast_status():
    """Broadcast current status to all connected WebSocket clients."""
    status_data = {
        "type": "status",
        "data": {
            "status": state.status,
            "last_run": state.last_run.isoformat() if state.last_run else None,
            "last_error": state.last_error,
            "positions": state.positions
        }
    }
    for ws in state.websockets[:]:
        try:
            await ws.send_json(status_data)
        except:
            state.websockets.remove(ws)


# Process management
async def read_stream(stream, is_stderr=False):
    """Read from process stream and broadcast to websockets."""
    while True:
        line = await asyncio.get_event_loop().run_in_executor(
            None, stream.readline
        )
        if not line:
            break
        decoded = line.decode('utf-8', errors='replace').rstrip()
        if decoded:
            prefix = "[ERR] " if is_stderr else ""
            await broadcast_log(f"{prefix}{decoded}")

            # Try to parse positions from log output
            if "Current positions:" in decoded or "Final positions:" in decoded:
                try:
                    # Extract position count
                    pass
                except:
                    pass

async def run_strategy():
    """Run the strategy script and stream output."""
    if state.process is not None and state.process.poll() is None:
        await broadcast_log("Strategy is already running!")
        return

    state.status = "running"
    state.last_run = datetime.now()
    state.last_error = None
    await broadcast_status()
    await broadcast_log("=" * 60)
    await broadcast_log("STARTING STRATEGY EXECUTION")
    await broadcast_log(f"Command: {' '.join(STRATEGY_CMD)}")
    await broadcast_log("=" * 60)

    try:
        state.process = subprocess.Popen(
            STRATEGY_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent,
            bufsize=1
        )

        # Read stdout and stderr concurrently
        await asyncio.gather(
            read_stream(state.process.stdout, False),
            read_stream(state.process.stderr, True)
        )

        # Wait for process to complete
        return_code = state.process.wait()

        if return_code == 0:
            state.status = "idle"
            await broadcast_log("=" * 60)
            await broadcast_log("STRATEGY COMPLETED SUCCESSFULLY")
            await broadcast_log("=" * 60)
        else:
            state.status = "error"
            state.last_error = f"Process exited with code {return_code}"
            await broadcast_log("=" * 60)
            await broadcast_log(f"STRATEGY FAILED (exit code: {return_code})")
            await broadcast_log("=" * 60)

    except Exception as e:
        state.status = "error"
        state.last_error = str(e)
        await broadcast_log(f"ERROR: {e}")

    finally:
        state.process = None
        await broadcast_status()


def stop_strategy():
    """Stop the running strategy."""
    if state.process is None:
        return False

    try:
        # Try graceful termination first
        state.process.terminate()
        try:
            state.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            state.process.kill()
            state.process.wait()

        state.status = "idle"
        state.process = None
        return True
    except Exception as e:
        state.last_error = str(e)
        return False


# API Routes
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
    return HTMLResponse(content=LOGIN_HTML.replace("</form>", '<p class="error">Invalid credentials</p></form>'))


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not verify_session(request):
        return RedirectResponse(url="/login", status_code=302)
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
async def get_status(request: Request, _: bool = Depends(require_auth)):
    return {
        "status": state.status,
        "last_run": state.last_run.isoformat() if state.last_run else None,
        "last_error": state.last_error,
        "positions": state.positions,
        "log_count": len(state.logs)
    }


@app.get("/api/logs")
async def get_logs(request: Request, _: bool = Depends(require_auth)):
    return {"logs": list(state.logs)}


@app.post("/api/run")
async def api_run(request: Request, _: bool = Depends(require_auth)):
    if state.status == "running":
        return JSONResponse(
            status_code=400,
            content={"error": "Strategy is already running"}
        )

    # Run in background
    asyncio.create_task(run_strategy())
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop(request: Request, _: bool = Depends(require_auth)):
    if state.status != "running":
        return JSONResponse(
            status_code=400,
            content={"error": "Strategy is not running"}
        )

    success = stop_strategy()
    if success:
        asyncio.create_task(broadcast_log("Strategy stopped by user"))
        asyncio.create_task(broadcast_status())
        return {"status": "stopped"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to stop strategy"}
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Check authentication via query param or cookie
    # For simplicity, we'll verify session from cookies
    session_data = websocket.cookies.get("session")
    if not session_data:
        await websocket.close(code=1008)
        return

    state.websockets.append(websocket)

    try:
        # Send current state on connect
        await websocket.send_json({
            "type": "init",
            "data": {
                "status": state.status,
                "last_run": state.last_run.isoformat() if state.last_run else None,
                "last_error": state.last_error,
                "logs": list(state.logs),
                "positions": state.positions
            }
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text("ping")

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.websockets:
            state.websockets.remove(websocket)


# HTML Templates (embedded for simplicity)
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Trading Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: #1a1a24;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #2a2a3a;
            width: 100%;
            max-width: 400px;
            margin: 1rem;
        }
        h1 {
            text-align: center;
            margin-bottom: 1.5rem;
            color: #fff;
            font-size: 1.5rem;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            color: #888;
            font-size: 0.875rem;
        }
        input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #2a2a3a;
            border-radius: 6px;
            background: #0a0a0f;
            color: #e0e0e0;
            font-size: 1rem;
        }
        input:focus {
            outline: none;
            border-color: #4a9eff;
        }
        button {
            width: 100%;
            padding: 0.75rem;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 0.5rem;
        }
        button:hover {
            background: #3a8eef;
        }
        .error {
            color: #ff6b6b;
            text-align: center;
            margin-top: 1rem;
            font-size: 0.875rem;
        }
        .logo {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">&#x1F4C8;</div>
        <h1>Trading Dashboard</h1>
        <form method="POST" action="/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required autocomplete="username">
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required autocomplete="current-password">
            </div>
            <button type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Strategy Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #1a1a24;
            --bg-tertiary: #2a2a3a;
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --accent: #4a9eff;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #f87171;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--bg-tertiary);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logout-btn {
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: background 0.2s;
        }

        .logout-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .status-card {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
        }

        .status-card label {
            display: block;
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .status-card .value {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.idle { background: var(--text-secondary); animation: none; }
        .status-dot.running { background: var(--success); }
        .status-dot.error { background: var(--error); animation: none; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .btn-primary {
            background: var(--success);
            color: #000;
        }

        .btn-primary:hover:not(:disabled) {
            background: #22c55e;
        }

        .btn-danger {
            background: var(--error);
            color: #000;
        }

        .btn-danger:hover:not(:disabled) {
            background: #ef4444;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .log-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }

        .log-header {
            background: var(--bg-tertiary);
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .log-header h2 {
            font-size: 0.875rem;
            font-weight: 500;
        }

        .log-actions {
            display: flex;
            gap: 0.5rem;
        }

        .log-actions button {
            background: none;
            border: 1px solid var(--bg-tertiary);
            color: var(--text-secondary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
        }

        .log-actions button:hover {
            border-color: var(--text-secondary);
            color: var(--text-primary);
        }

        .log-content {
            height: 400px;
            overflow-y: auto;
            padding: 1rem;
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            font-size: 0.8125rem;
            line-height: 1.6;
        }

        .log-line {
            white-space: pre-wrap;
            word-break: break-all;
        }

        .log-line.error {
            color: var(--error);
        }

        .log-line.success {
            color: var(--success);
        }

        .log-line.highlight {
            color: var(--accent);
        }

        .positions-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 8px;
            margin-top: 1.5rem;
            overflow: hidden;
        }

        .positions-header {
            background: var(--bg-tertiary);
            padding: 0.75rem 1rem;
        }

        .positions-header h2 {
            font-size: 0.875rem;
            font-weight: 500;
        }

        .positions-content {
            padding: 1rem;
        }

        .positions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 0.75rem;
        }

        .position-item {
            background: var(--bg-primary);
            padding: 0.75rem;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .position-ticker {
            font-weight: 600;
        }

        .position-qty {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .no-positions {
            color: var(--text-secondary);
            text-align: center;
            padding: 2rem;
        }

        .connection-status {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .connection-status.connected { border-color: var(--success); }
        .connection-status.disconnected { border-color: var(--error); }

        @media (max-width: 640px) {
            .header {
                flex-direction: column;
                gap: 1rem;
                text-align: center;
            }

            .controls {
                flex-direction: column;
            }

            .btn {
                width: 100%;
                justify-content: center;
            }

            .log-content {
                height: 300px;
                font-size: 0.75rem;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>&#x1F4C8; Trading Strategy Dashboard</h1>
        <a href="/logout" class="logout-btn">Logout</a>
    </header>

    <div class="container">
        <div class="status-bar">
            <div class="status-card">
                <label>Status</label>
                <div class="value">
                    <span class="status-indicator">
                        <span class="status-dot" id="statusDot"></span>
                        <span id="statusText">Loading...</span>
                    </span>
                </div>
            </div>
            <div class="status-card">
                <label>Last Run</label>
                <div class="value" id="lastRun">-</div>
            </div>
            <div class="status-card">
                <label>Last Error</label>
                <div class="value" id="lastError" style="color: var(--error); font-size: 0.875rem;">None</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="runBtn" onclick="runStrategy()">
                &#9654; Run Strategy
            </button>
            <button class="btn btn-danger" id="stopBtn" onclick="stopStrategy()" disabled>
                &#9632; Stop
            </button>
        </div>

        <div class="log-panel">
            <div class="log-header">
                <h2>Live Output</h2>
                <div class="log-actions">
                    <button onclick="clearLogs()">Clear</button>
                    <button onclick="scrollToBottom()">Scroll to Bottom</button>
                </div>
            </div>
            <div class="log-content" id="logContent">
                <div class="log-line" style="color: var(--text-secondary);">Waiting for activity...</div>
            </div>
        </div>

        <div class="positions-panel">
            <div class="positions-header">
                <h2>Current Positions (from logs)</h2>
            </div>
            <div class="positions-content" id="positionsContent">
                <div class="no-positions">No position data available. Run the strategy to see positions.</div>
            </div>
        </div>
    </div>

    <div class="connection-status" id="connectionStatus">
        <span class="status-dot" id="connectionDot"></span>
        <span id="connectionText">Connecting...</span>
    </div>

    <script>
        let ws = null;
        let autoScroll = true;
        const logContent = document.getElementById('logContent');
        const positions = {};

        function formatTime(isoString) {
            if (!isoString) return '-';
            const date = new Date(isoString);
            return date.toLocaleString();
        }

        function updateStatus(data) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const runBtn = document.getElementById('runBtn');
            const stopBtn = document.getElementById('stopBtn');

            statusDot.className = 'status-dot ' + data.status;
            statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);

            runBtn.disabled = data.status === 'running';
            stopBtn.disabled = data.status !== 'running';

            document.getElementById('lastRun').textContent = formatTime(data.last_run);
            document.getElementById('lastError').textContent = data.last_error || 'None';
        }

        function addLog(line) {
            const div = document.createElement('div');
            div.className = 'log-line';

            // Color coding
            if (line.includes('[ERR]') || line.includes('ERROR') || line.includes('FAILED')) {
                div.classList.add('error');
            } else if (line.includes('SUCCESS') || line.includes('COMPLETE') || line.includes('CONFIRMED')) {
                div.classList.add('success');
            } else if (line.includes('===') || line.includes('STEP')) {
                div.classList.add('highlight');
            }

            div.textContent = line;
            logContent.appendChild(div);

            // Parse positions from logs
            parsePositionsFromLog(line);

            if (autoScroll) {
                scrollToBottom();
            }
        }

        function parsePositionsFromLog(line) {
            // Try to extract position info from log lines
            // Format: "ticker: quantity" or similar
            const posMatch = line.match(/^\s*([A-Z]{1,5}):\s*([\d,]+)\s*$/);
            if (posMatch) {
                positions[posMatch[1]] = parseInt(posMatch[2].replace(',', ''));
                updatePositionsDisplay();
            }

            // Clear positions on new run
            if (line.includes('STARTING STRATEGY')) {
                Object.keys(positions).forEach(k => delete positions[k]);
                updatePositionsDisplay();
            }
        }

        function updatePositionsDisplay() {
            const container = document.getElementById('positionsContent');
            const tickers = Object.keys(positions);

            if (tickers.length === 0) {
                container.innerHTML = '<div class="no-positions">No position data available. Run the strategy to see positions.</div>';
                return;
            }

            container.innerHTML = '<div class="positions-grid">' +
                tickers.sort().map(ticker =>
                    `<div class="position-item">
                        <span class="position-ticker">${ticker}</span>
                        <span class="position-qty">${positions[ticker]}</span>
                    </div>`
                ).join('') + '</div>';
        }

        function clearLogs() {
            logContent.innerHTML = '<div class="log-line" style="color: var(--text-secondary);">Logs cleared.</div>';
        }

        function scrollToBottom() {
            logContent.scrollTop = logContent.scrollHeight;
        }

        async function runStrategy() {
            try {
                const response = await fetch('/api/run', { method: 'POST' });
                if (!response.ok) {
                    const data = await response.json();
                    alert(data.error || 'Failed to start strategy');
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        async function stopStrategy() {
            if (!confirm('Are you sure you want to stop the strategy?')) return;

            try {
                const response = await fetch('/api/stop', { method: 'POST' });
                if (!response.ok) {
                    const data = await response.json();
                    alert(data.error || 'Failed to stop strategy');
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connectionStatus').className = 'connection-status connected';
                document.getElementById('connectionDot').className = 'status-dot';
                document.getElementById('connectionDot').style.background = 'var(--success)';
                document.getElementById('connectionText').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                document.getElementById('connectionDot').style.background = 'var(--error)';
                document.getElementById('connectionText').textContent = 'Disconnected - Reconnecting...';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = () => {
                ws.close();
            };

            ws.onmessage = (event) => {
                if (event.data === 'ping') {
                    ws.send('pong');
                    return;
                }

                try {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'init') {
                        // Initial state
                        updateStatus(msg.data);
                        if (msg.data.logs && msg.data.logs.length > 0) {
                            logContent.innerHTML = '';
                            msg.data.logs.forEach(line => addLog(line));
                        }
                    } else if (msg.type === 'status') {
                        updateStatus(msg.data);
                    } else if (msg.type === 'log') {
                        addLog(msg.data);
                    }
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
        }

        // Detect scroll position
        logContent.addEventListener('scroll', () => {
            const atBottom = logContent.scrollHeight - logContent.clientHeight <= logContent.scrollTop + 50;
            autoScroll = atBottom;
        });

        // Initialize
        connectWebSocket();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    print("=" * 60)
    print("Trading Strategy Dashboard")
    print("=" * 60)
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")
    print()
    print(f"Starting server at http://0.0.0.0:{port}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
