"""
AI Business Process Analysis Service - Main Application
FastAPI backend with integrated SPA frontend.
"""

import os
import sys
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.schemas import ASRProgressMessage
from services import asr_service, audio_service, prompt_service

# Import routers
from routers import audio, asr, llm, business, prompts, faq

# ==================== App Setup ====================

app = FastAPI(
    title="AI Business Process Analysis Service",
    description="AI驱动的业务流程分析服务 - 音频ASR + LLM分析 + 可视化生成",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(audio.router)
app.include_router(asr.router)
app.include_router(llm.router)
app.include_router(business.router)
app.include_router(prompts.router)
app.include_router(faq.router)

# ==================== WebSocket Manager ====================

class ConnectionManager:
    """Manage WebSocket connections for real-time progress updates."""

    def __init__(self):
        self.active_connections: Dict[str, list] = {
            "asr_progress": [],
            "download_progress": [],
        }

    async def connect(self, channel: str, websocket: WebSocket):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, channel: str, websocket: WebSocket):
        if channel in self.active_connections:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict):
        if channel in self.active_connections:
            dead = []
            for ws in self.active_connections[channel]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active_connections[channel].remove(ws)

manager = ConnectionManager()


# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/asr/progress")
async def ws_asr_progress(websocket: WebSocket):
    """WebSocket for real-time ASR processing progress."""
    await manager.connect("asr_progress", websocket)

    # Send current state on connect
    state = asr_service.get_processing_state()
    await websocket.send_json({
        "type": "status",
        "is_running": state["is_running"],
        "current": state["current"],
        "total": state["total"],
        "progress": round((state["current"] / max(state["total"], 1)) * 100, 1),
    })

    # Send existing results (only completed ones)
    if state["results"]:
        completed_results = {k: v for k, v in state["results"].items() if v.get("status") == "completed"}
        if completed_results:
            await websocket.send_json({
                "type": "existing_results",
                "results": completed_results
            })

    try:
        while True:
            # Keep connection alive and check for updates
            data = await websocket.receive_text()
            if data == "ping":
                state = asr_service.get_processing_state()
                await websocket.send_json({
                    "type": "pong",
                    "is_running": state["is_running"],
                    "current": state["current"],
                    "total": state["total"],
                    "progress": round((state["current"] / max(state["total"], 1)) * 100, 1),
                })
    except WebSocketDisconnect:
        manager.disconnect("asr_progress", websocket)


@app.websocket("/ws/audio/download-progress")
async def ws_download_progress(websocket: WebSocket):
    """WebSocket for real-time download progress."""
    await manager.connect("download_progress", websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                progress = audio_service.get_download_progress()
                await websocket.send_json({
                    "type": "progress",
                    "data": progress
                })
    except WebSocketDisconnect:
        manager.disconnect("download_progress", websocket)


# ==================== Static File Serving ====================

# Mount data directories for file downloads
@app.get("/api/files/{file_type}/{task_id}/{filename}")
async def serve_generated_file(file_type: str, task_id: str, filename: str):
    """Serve generated files from business_output directory."""
    file_path = os.path.join("./data/business_output", filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    return FileResponse(file_path)


# ==================== Frontend ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main SPA frontend."""
    return HTML_PAGE


# ==================== Startup / Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Ensure data directories exist
    for d in ["./data/audio", "./data/asr_output", "./data/business_output", "./data/prompts"]:
        os.makedirs(d, exist_ok=True)

    # Initialize prompt service (creates default prompts)
    prompt_service._init_default_prompts()

    # Pre-load ASR model in background
    def _preload_model():
        try:
            print("[Startup] Pre-loading ASR model...")
            asr_service.get_model()
            print("[Startup] ASR model loaded successfully")
        except Exception as e:
            print(f"[Startup] ASR model pre-load failed (will load on first use): {e}")

    thread = threading.Thread(target=_preload_model, daemon=True)
    thread.start()

    print(f"[Startup] AI Business Process Analysis Service started")
    print(f"[Startup] API docs: http://localhost:{os.getenv('PORT', '8002')}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("[Shutdown] Service stopped")


# ==================== HTML Frontend ====================

HTML_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Business Process Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
  --bg-primary: #0f1117;
  --bg-secondary: #161822;
  --bg-tertiary: #1c1f2e;
  --bg-card: #1e2132;
  --bg-hover: #252839;
  --bg-input: #141620;
  --border: #2a2d3e;
  --border-active: #4a4f6a;
  --text-primary: #e4e6f0;
  --text-secondary: #989bb5;
  --text-muted: #656880;
  --accent: #00d4aa;
  --accent-glow: rgba(0, 212, 170, 0.15);
  --accent-secondary: #6c72cb;
  --danger: #f44b6c;
  --warning: #f0a84c;
  --success: #00d4aa;
  --info: #5b9cf5;
  --font-display: 'Noto Sans SC', sans-serif;
  --font-mono: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
  --radius: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --shadow: 0 2px 8px rgba(0,0,0,0.3);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.4);
  --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-theme="light"] {
  --bg-primary: #f5f6fa;
  --bg-secondary: #ffffff;
  --bg-tertiary: #eef0f5;
  --bg-card: #ffffff;
  --bg-hover: #f0f1f5;
  --bg-input: #f8f9fc;
  --border: #e2e4ea;
  --border-active: #c4c7d4;
  --text-primary: #1a1c2e;
  --text-secondary: #5a5d72;
  --text-muted: #9497b0;
  --shadow: 0 2px 8px rgba(0,0,0,0.06);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.1);
}

* { margin:0; padding:0; box-sizing:border-box; }
html { font-size:14px; }
body {
  font-family: var(--font-display);
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
  line-height: 1.5;
  transition: background var(--transition), color var(--transition);
}

/* Layout */
.app { display:flex; height:100vh; overflow:hidden; }

/* Sidebar */
.sidebar {
  width: 240px; min-width: 240px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  transition: width var(--transition);
  z-index: 100;
}
.sidebar-header {
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-logo {
  display: flex; align-items: center; gap: 10px;
  font-size: 1.1rem; font-weight: 700; color: var(--accent);
  letter-spacing: -0.3px;
}
.sidebar-logo .icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), #009c7d);
  border-radius: 10px; display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem; color: #fff;
}
.sidebar-subtitle {
  font-size: 0.75rem; color: var(--text-muted);
  margin-top: 4px; padding-left: 46px;
  font-family: var(--font-mono); letter-spacing: 0.5px;
}
.sidebar-nav {
  flex: 1; padding: 12px 8px; overflow-y: auto;
}
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; margin-bottom: 2px;
  border-radius: var(--radius); cursor: pointer;
  font-size: 0.9rem; font-weight: 500;
  color: var(--text-secondary); transition: all var(--transition);
  position: relative; user-select: none;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.nav-item.active {
  background: var(--accent-glow); color: var(--accent);
  font-weight: 600;
}
.nav-item.active::before {
  content: ''; position: absolute; left: 0; top: 50%; transform: translateY(-50%);
  width: 3px; height: 20px; background: var(--accent); border-radius: 0 3px 3px 0;
}
.nav-icon { font-size: 1.2rem; width: 24px; text-align: center; }
.nav-badge {
  margin-left: auto; font-size: 0.7rem; padding: 2px 8px;
  border-radius: 10px; background: var(--accent-glow); color: var(--accent);
  font-family: var(--font-mono); font-weight: 600;
}
.sidebar-footer {
  padding: 12px; border-top: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.theme-toggle {
  background: var(--bg-tertiary); border: 1px solid var(--border);
  color: var(--text-secondary); padding: 6px 12px; border-radius: var(--radius);
  cursor: pointer; font-size: 0.85rem; transition: all var(--transition);
  display: flex; align-items: center; gap: 6px;
}
.theme-toggle:hover { background: var(--bg-hover); color: var(--text-primary); }

/* Main Content */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header {
  padding: 16px 28px; background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.main-header h1 { font-size: 1.2rem; font-weight: 700; }
.main-header .status {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.8rem; color: var(--text-muted);
}
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--success);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.main-content { flex: 1; overflow-y: auto; padding: 24px 28px; }

/* Tab Content */
.tab-content { display: none; }
.tab-content.active { display: block; animation: fadeIn 0.3s ease; }
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Cards */
.card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 20px; margin-bottom: 16px;
  transition: all var(--transition);
}
.card:hover { border-color: var(--border-active); }
.card-header {
  font-size: 0.95rem; font-weight: 600; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
  color: var(--text-primary);
}
.card-header .card-icon { font-size: 1.1rem; }

/* Grid */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
@media (max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }

/* Form Elements */
.form-group { margin-bottom: 14px; }
.form-label {
  display: block; font-size: 0.8rem; font-weight: 600;
  color: var(--text-secondary); margin-bottom: 6px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.form-input, .form-select, .form-textarea {
  width: 100%; padding: 9px 12px;
  background: var(--bg-input); border: 1px solid var(--border);
  border-radius: var(--radius); color: var(--text-primary);
  font-family: var(--font-display); font-size: 0.9rem;
  transition: all var(--transition); outline: none;
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
}
.form-textarea {
  resize: vertical; min-height: 200px;
  font-family: var(--font-mono); font-size: 0.8rem;
  line-height: 1.6; tab-size: 2;
}
.form-select {
  cursor: pointer; appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%23989bb5'%3E%3Cpath d='M6 8L1 3h10z'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 12px center;
  padding-right: 32px;
}
.form-row { display: flex; gap: 12px; align-items: flex-end; }
.form-row > * { flex: 1; }

/* Range slider */
input[type="range"] {
  -webkit-appearance: none; width: 100%; height: 6px;
  background: var(--bg-tertiary); border-radius: 3px; outline: none;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 18px; height: 18px;
  background: var(--accent); border-radius: 50%; cursor: pointer;
  box-shadow: 0 0 8px var(--accent-glow);
}
.range-value {
  font-family: var(--font-mono); font-size: 0.85rem;
  color: var(--accent); font-weight: 600; min-width: 30px; text-align: right;
}

/* Toggle */
.toggle-group { display: flex; align-items: center; gap: 10px; }
.toggle {
  width: 44px; height: 24px; background: var(--bg-tertiary);
  border-radius: 12px; cursor: pointer; position: relative;
  transition: all var(--transition); border: 1px solid var(--border);
}
.toggle.on { background: var(--accent); border-color: var(--accent); }
.toggle::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 18px; height: 18px; background: #fff; border-radius: 50%;
  transition: all var(--transition);
}
.toggle.on::after { left: 22px; }

/* Buttons */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 9px 18px; border: none; border-radius: var(--radius);
  font-family: var(--font-display); font-size: 0.85rem; font-weight: 600;
  cursor: pointer; transition: all var(--transition);
  white-space: nowrap; user-select: none;
}
.btn:active { transform: scale(0.97); }
.btn-primary {
  background: linear-gradient(135deg, var(--accent), #00b894);
  color: #000; font-weight: 700;
}
.btn-primary:hover { box-shadow: 0 4px 16px var(--accent-glow); }
.btn-secondary {
  background: var(--bg-tertiary); color: var(--text-primary);
  border: 1px solid var(--border);
}
.btn-secondary:hover { background: var(--bg-hover); border-color: var(--border-active); }
.btn-danger {
  background: rgba(244,75,108,0.1); color: var(--danger);
  border: 1px solid rgba(244,75,108,0.2);
}
.btn-danger:hover { background: rgba(244,75,108,0.2); }
.btn-sm { padding: 5px 12px; font-size: 0.78rem; }
.btn-lg { padding: 12px 24px; font-size: 0.95rem; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; pointer-events: none; }

/* Upload Zone */
.upload-zone {
  border: 2px dashed var(--border); border-radius: var(--radius-lg);
  padding: 40px; text-align: center; cursor: pointer;
  transition: all var(--transition); background: var(--bg-tertiary);
  position: relative;
}
.upload-zone:hover, .upload-zone.drag-over {
  border-color: var(--accent); background: var(--accent-glow);
}
.upload-zone .upload-icon { font-size: 3rem; margin-bottom: 12px; }
.upload-zone .upload-text { font-size: 1rem; font-weight: 600; margin-bottom: 4px; }
.upload-zone .upload-hint { font-size: 0.8rem; color: var(--text-muted); }

/* Progress Bar */
.progress-bar {
  width: 100%; height: 8px; background: var(--bg-tertiary);
  border-radius: 4px; overflow: hidden;
}
.progress-bar .fill {
  height: 100%; background: linear-gradient(90deg, var(--accent), #00b894);
  border-radius: 4px; transition: width 0.3s ease;
  position: relative;
}
.progress-bar .fill::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  animation: shimmer 2s infinite;
}
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
.progress-info {
  display: flex; justify-content: space-between;
  font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px;
}

/* File List */
.file-list {
  max-height: 300px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: var(--radius);
}
.file-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  transition: background var(--transition); font-size: 0.85rem;
}
.file-item:last-child { border-bottom: none; }
.file-item:hover { background: var(--bg-hover); }
.file-item .file-icon { font-size: 1.1rem; }
.file-item .file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-item .file-size { color: var(--text-muted); font-size: 0.78rem; font-family: var(--font-mono); }
.file-item .file-status {
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
  font-weight: 600; text-transform: uppercase;
}
.status-completed { background: rgba(0,212,170,0.1); color: var(--success); }
.status-processing { background: rgba(91,156,245,0.1); color: var(--info); }
.status-error { background: rgba(244,75,108,0.1); color: var(--danger); }
.status-pending { background: rgba(240,168,76,0.1); color: var(--warning); }

/* URL List */
.url-list { margin-top: 12px; }
.url-item {
  display: flex; align-items: center; gap: 8px; padding: 6px 0;
}
.url-item .url-index {
  font-family: var(--font-mono); font-size: 0.75rem;
  color: var(--text-muted); min-width: 24px;
}
.url-item .url-text {
  flex: 1; font-size: 0.8rem; color: var(--text-secondary);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: var(--font-mono);
}

/* ASR Result Viewer */
.asr-viewer {
  display: grid; grid-template-columns: 280px 1fr; gap: 16px;
  min-height: 400px;
}
.asr-file-list {
  border: 1px solid var(--border); border-radius: var(--radius);
  overflow-y: auto; max-height: 500px; background: var(--bg-card);
}
.asr-file-item {
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: all var(--transition);
  display: flex; align-items: center; gap: 8px; font-size: 0.83rem;
}
.asr-file-item:hover { background: var(--bg-hover); }
.asr-file-item.active { background: var(--accent-glow); border-left: 3px solid var(--accent); }
.asr-content {
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; max-height: 500px; overflow-y: auto;
  background: var(--bg-card); font-family: var(--font-display);
}
.asr-sentence {
  padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
  line-height: 1.7; font-size: 0.9rem;
}
.asr-sentence .sentence-num {
  font-family: var(--font-mono); color: var(--accent);
  font-weight: 600; margin-right: 8px; font-size: 0.78rem;
}
.asr-sentence .speaker-tag {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 0.7rem; font-weight: 600; margin-right: 6px;
  background: var(--accent-glow); color: var(--accent);
}

/* Business Process */
.bp-viewer { min-height: 500px; }
.bp-tabs {
  display: flex; gap: 4px; margin-bottom: 16px;
  background: var(--bg-tertiary); padding: 4px; border-radius: var(--radius);
}
.bp-tab {
  padding: 8px 16px; border: none; background: none;
  cursor: pointer; border-radius: 6px; font-size: 0.83rem;
  font-family: var(--font-display); color: var(--text-secondary);
  transition: all var(--transition); font-weight: 500;
}
.bp-tab.active { background: var(--bg-card); color: var(--text-primary); box-shadow: var(--shadow); }
.bp-content { display: none; }
.bp-content.active { display: block; }
.bp-content .markdown-body {
  font-size: 0.9rem; line-height: 1.8;
  padding: 20px; background: var(--bg-card); border-radius: var(--radius);
  border: 1px solid var(--border);
}
.bp-content .markdown-body h1 { font-size: 1.5rem; margin-bottom: 16px; color: var(--accent); }
.bp-content .markdown-body h2 { font-size: 1.2rem; margin: 20px 0 10px; }
.bp-content .markdown-body h3 { font-size: 1rem; margin: 16px 0 8px; }
.bp-content .markdown-body ul { padding-left: 20px; }
.bp-content .markdown-body li { margin: 4px 0; }
.bp-content .markdown-body strong { color: var(--accent); }
.bp-content .markdown-body em { color: var(--warning); font-style: normal; }

/* Prompt Editor */
.prompt-list-panel {
  border: 1px solid var(--border); border-radius: var(--radius);
  max-height: 400px; overflow-y: auto;
}
.prompt-list-item {
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: all var(--transition);
}
.prompt-list-item:hover { background: var(--bg-hover); }
.prompt-list-item.active { background: var(--accent-glow); border-left: 3px solid var(--accent); }
.prompt-list-item .prompt-name { font-weight: 600; font-size: 0.9rem; }
.prompt-list-item .prompt-desc { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.prompt-list-item .prompt-status {
  display: inline-block; font-size: 0.7rem; padding: 2px 8px;
  border-radius: 10px; margin-top: 4px; font-weight: 600;
}
.status-customized { background: rgba(240,168,76,0.1); color: var(--warning); }
.status-default { background: rgba(0,212,170,0.1); color: var(--success); }

/* Toast */
.toast-container {
  position: fixed; top: 20px; right: 20px; z-index: 9999;
  display: flex; flex-direction: column; gap: 8px;
}
.toast {
  padding: 12px 20px; border-radius: var(--radius);
  font-size: 0.85rem; font-weight: 500;
  animation: slideIn 0.3s ease;
  box-shadow: var(--shadow-lg); max-width: 400px;
  display: flex; align-items: center; gap: 8px;
}
.toast-success { background: #1a3a2e; border: 1px solid var(--success); color: var(--success); }
.toast-error { background: #3a1a1a; border: 1px solid var(--danger); color: var(--danger); }
.toast-info { background: #1a2a3a; border: 1px solid var(--info); color: var(--info); }
@keyframes slideIn {
  from { opacity: 0; transform: translateX(100px); }
  to { opacity: 1; transform: translateX(0); }
}

/* Empty State */
.empty-state {
  text-align: center; padding: 60px 20px; color: var(--text-muted);
}
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.5; }
.empty-state .empty-text { font-size: 1rem; margin-bottom: 4px; }
.empty-state .empty-hint { font-size: 0.8rem; }

/* Modal */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center;
  z-index: 9998; animation: fadeIn 0.2s ease;
}
.modal {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-xl); padding: 24px; max-width: 600px; width: 90%;
  box-shadow: var(--shadow-lg); max-height: 80vh; overflow-y: auto;
}
.modal-header {
  font-size: 1.1rem; font-weight: 700; margin-bottom: 16px;
  display: flex; align-items: center; justify-content: space-between;
}
.modal-close {
  background: none; border: none; color: var(--text-muted);
  cursor: pointer; font-size: 1.2rem; padding: 4px;
}
.modal-close:hover { color: var(--text-primary); }

/* Tags */
.tag {
  display: inline-block; padding: 4px 10px; border-radius: 6px;
  font-size: 0.75rem; font-weight: 600; margin: 2px;
  background: var(--bg-tertiary); color: var(--text-secondary);
  border: 1px solid var(--border);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-active); }

/* Mindmap Tree */
.mindmap-tree { font-family: var(--font-display); line-height: 1.6; }
.mindmap-tree > div { border-left: 2px solid var(--border); padding-left: 8px; margin: 2px 0; }
.mindmap-tree .arrow { display: inline-block; width: 16px; color: var(--accent); }

/* Connection test result */
.test-result {
  padding: 12px 16px; border-radius: var(--radius); margin-top: 10px;
  font-size: 0.85rem; font-family: var(--font-mono);
}
.test-result.success { background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.2); color: var(--success); }
.test-result.error { background: rgba(244,75,108,0.08); border: 1px solid rgba(244,75,108,0.2); color: var(--danger); }
.test-result .latency { color: var(--text-muted); font-size: 0.78rem; margin-top: 4px; }

/* Stats mini cards */
.stat-mini {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px; text-align: center;
}
.stat-mini .stat-value { font-size: 1.8rem; font-weight: 900; color: var(--accent); font-family: var(--font-mono); }
.stat-mini .stat-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

/* Config panel */
.config-panel {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
@media (max-width: 900px) { .config-panel { grid-template-columns: 1fr; } }

/* ASR History */
.asr-history { margin-top: 16px; }
.asr-history-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: background var(--transition);
  font-size: 0.83rem;
}
.asr-history-item:hover { background: var(--bg-hover); }
.asr-history-item .history-time { font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); min-width: 140px; }
.asr-history-item .history-info { flex: 1; }
.asr-history-item .history-count { font-weight: 600; color: var(--accent); }
</style>
</head>
<body>
<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg></div>AI-BP
      </div>
      <div class="sidebar-subtitle">BUSINESS PROCESS</div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-item active" data-tab="audio">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="16" cy="16" r="3"/></svg></span>音频管理
        <span class="nav-badge" id="audioCount">0</span>
      </div>
      <div class="nav-item" data-tab="asr">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/></svg></span>ASR解析
      </div>
      <div class="nav-item" data-tab="llm">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M12 3v4M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/></svg></span>模型配置
      </div>
      <div class="nav-item" data-tab="prompts">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></span>提示词管理
      </div>
      <div class="nav-item" data-tab="business">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></span>生成流程
      </div>
      <div class="nav-item" data-tab="faq">
        <span class="nav-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>FAQ提取
      </div>
    </nav>
    <div class="sidebar-footer">
      <button class="theme-toggle" onclick="toggleTheme()">
        <span id="themeIcon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg></span>
        <span id="themeLabel">浅色</span>
      </button>
      <span style="font-size:0.7rem;color:var(--text-muted)">v1.0.0</span>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <header class="main-header">
      <h1 id="pageTitle">音频管理</h1>
      <div class="status">
        <div class="status-dot"></div>
        <span>系统运行中</span>
      </div>
    </header>
    <div class="main-content" id="mainContent">
      <!-- Tab 1: Audio Management -->
      <div class="tab-content active" id="tab-audio">
        <div class="card">
          <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></span>上传音频文件</div>
          <div class="upload-zone" id="uploadZone">
            <div class="upload-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg></div>
            <div class="upload-text">拖拽音频文件到此处，或点击选择</div>
            <div class="upload-hint">支持 MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, WMA 格式</div>
            <input type="file" id="fileInput" multiple accept=".mp3,.wav,.m4a,.flac,.ogg,.opus,.aac,.wma" style="display:none">
          </div>
          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
            <span class="tag">MP3</span><span class="tag">WAV</span><span class="tag">M4A</span>
            <span class="tag">FLAC</span><span class="tag">OGG</span><span class="tag">OPUS</span>
            <span class="tag">AAC</span><span class="tag">WMA</span>
          </div>
        </div>

        <div class="card">
          <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg></span>通过URL下载音频</div>
          <div id="urlSummary" style="display:none;padding:8px 14px;background:var(--bg-tertiary);border-radius:var(--radius);margin-bottom:8px;font-size:0.85rem;cursor:pointer;" onclick="toggleUrlDetail()">
            <span id="urlSummaryText">已解析 0 条音频URL</span>
            <span style="color:var(--accent);float:right;" id="urlToggleIcon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></span>
          </div>
          <div class="url-list" id="urlList" style="display:none;"></div>
          <div style="margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <button class="btn btn-primary" onclick="openUrlModal()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg> 批量粘贴URL</button>
            <button class="btn btn-primary" id="btnDownloadUrls" onclick="startDownload()" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (0)</button>
            <button class="btn btn-secondary" onclick="clearUrls()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> 清空</button>
            <select class="form-select" id="downloadWorkers" style="width:auto;flex:0 0 auto;">
              <option value="2">2 线程</option>
              <option value="4" selected>4 线程</option>
              <option value="8">8 线程</option>
              <option value="12">12 线程</option>
            </select>
            <select class="form-select" id="downloadRetries" style="width:auto;flex:0 0 auto;">
              <option value="0">不重试</option>
              <option value="1">重试1次</option>
              <option value="3" selected>重试3次</option>
              <option value="5">重试5次</option>
            </select>
          </div>
          <div class="file-list" id="downloadProgressList" style="margin-top:12px;max-height:200px;"></div>
          <div id="downloadCompletedMsg" style="display:none;margin-top:8px;padding:8px 12px;background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);border-radius:var(--radius);font-size:0.8rem;color:var(--success);"></div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></span>音频文件列表
            <button class="btn btn-sm btn-secondary" onclick="loadHistoricalAudio()" style="margin-left:auto;margin-right:8px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> 查看历史音频</button>
            <button class="btn btn-danger btn-sm" onclick="deleteAllAudio()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> 清空全部</button>
          </div>
          <div class="file-list" id="audioFileList"></div>
          <div class="empty-state" id="audioEmpty">
            <div class="empty-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="16" cy="16" r="3"/></svg></div>
            <div class="empty-text">暂无音频文件</div>
            <div class="empty-hint">请上传音频文件或通过URL下载</div>
          </div>
        </div>
      </div>

      <!-- Tab 2: ASR Processing -->
      <div class="tab-content" id="tab-asr">
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg></span>ASR配置</div>
            <div class="form-group">
              <label class="form-label">识别语言</label>
              <select class="form-select" id="asrLanguage">
                <option value="zh" selected>中文 (zh)</option>
                <option value="en">英文 (en)</option>
                <option value="yue">粤语 (yue)</option>
                <option value="ja">日语 (ja)</option>
                <option value="ko">韩语 (ko)</option>
                <option value="auto">自动检测 (auto)</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">并行处理数：<span class="range-value" id="workersValue">4</span></label>
              <input type="range" id="asrWorkers" min="1" max="20" value="4" oninput="document.getElementById('workersValue').textContent=this.value">
            </div>
            <div class="form-group">
              <label class="form-label">逆文本正则化 (ITN)</label>
              <div class="toggle-group">
                <div class="toggle on" id="itnToggle" onclick="this.classList.toggle('on')"></div>
                <span style="font-size:0.85rem;color:var(--text-secondary)">数字转写、日期格式化等</span>
              </div>
            </div>
            <div class="form-group">
              <label class="form-label">批次大小 (秒)</label>
              <input type="number" class="form-input" id="asrBatchSize" value="60" min="10" max="300">
            </div>
            <div style="display:flex;gap:8px;margin-top:16px;">
              <button class="btn btn-primary btn-lg" id="btnStartASR" onclick="startASR()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 开始ASR解析</button>
              <button class="btn btn-danger btn-lg" id="btnCancelASR" onclick="cancelASR()" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> 取消</button>
            </div>
          </div>
          <div class="card">
            <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg></span>处理进度</div>
            <!-- Model Download Progress -->
            <div id="modelDownloadProgress" style="display:none;margin-bottom:12px;padding:10px;background:var(--accent-glow);border-radius:var(--radius);border:1px solid var(--accent);">
              <div style="font-size:0.85rem;font-weight:600;color:var(--accent);margin-bottom:6px;" id="modelDownloadText">正在下载ASR模型...</div>
              <div class="progress-bar" style="height:6px;">
                <div class="fill" id="modelDownloadFill" style="width:0%;background:linear-gradient(90deg, var(--accent), var(--info));"></div>
              </div>
              <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:4px;" id="modelDownloadSubtext">首次使用需要下载模型文件（约280MB），请耐心等待...</div>
            </div>
            <div class="progress-bar" style="margin-bottom:8px;">
              <div class="fill" id="asrProgressFill" style="width:0%"></div>
            </div>
            <div class="progress-info">
              <span id="asrProgressText">等待开始...</span>
              <span id="asrProgressCount">0 / 0</span>
            </div>
            <div style="margin-top:16px;">
              <div class="grid-3" style="margin-bottom:12px;">
                <div class="stat-mini">
                  <div class="stat-value" id="statTotal">0</div>
                  <div class="stat-label">文件总数</div>
                </div>
                <div class="stat-mini">
                  <div class="stat-value" id="statCompleted" style="color:var(--success)">0</div>
                  <div class="stat-label">已完成</div>
                </div>
                <div class="stat-mini">
                  <div class="stat-value" id="statErrors" style="color:var(--danger)">0</div>
                  <div class="stat-label">出错</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span>ASR结果查看器
            <button class="btn btn-sm btn-secondary" id="btnDownloadTxt" onclick="downloadASRTxt()" disabled style="margin-left:auto;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> 下载当前ASR文本 (.txt)</button>
            <button class="btn btn-sm btn-secondary" onclick="renderASRHistorySidebar()" style="margin-left:4px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> 查看历史ASR</button>
          </div>
          <div class="asr-viewer">
            <div class="asr-file-list" id="asrFileList">
              <div class="empty-state" style="padding:40px 10px;">
                <div class="empty-text" style="font-size:0.85rem;">暂无ASR结果</div>
              </div>
            </div>
            <div class="asr-content" id="asrContent">
              <div class="empty-state">
                <div class="empty-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/></svg></div>
                <div class="empty-text">请在左侧选择一个文件查看ASR结果</div>
              </div>
            </div>
          </div>
        </div>
      <!-- ASR History Panel -->
        <div class="card" id="asrHistoryCard" style="display:none;">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></span>ASR历史记录（不限数量）
            <button class="btn btn-sm btn-danger" onclick="clearASRHistory()" style="margin-left:auto;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> 清空历史</button>
          </div>
          <div id="asrHistoryList"></div>
          <div class="empty-state" id="asrHistoryEmpty">
            <div class="empty-text" style="font-size:0.85rem;">暂无历史记录</div>
          </div>
        </div>
      </div>

      <!-- Tab 3: LLM Config -->
      <div class="tab-content" id="tab-llm">
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M12 3v4M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/></svg></span>模型配置</div>
            <div class="form-group">
              <label class="form-label">模型提供商</label>
              <select class="form-select" id="llmProvider" onchange="onProviderChange()">
                <option value="deepseek">DeepSeek</option>
                <option value="openai">OpenAI</option>
                <option value="ollama">Ollama (本地)</option>
                <option value="siliconflow">SiliconFlow (硅基流动)</option>
                <option value="custom">自定义 (OpenAI兼容)</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">模型名称</label>
              <input type="text" class="form-input" id="llmModel" value="deepseek-v4-pro">
            </div>
            <div class="form-group">
              <label class="form-label">API Key</label>
              <input type="password" class="form-input" id="llmApiKey" placeholder="sk-...">
            </div>
            <div class="form-group">
              <label class="form-label">Base URL</label>
              <input type="text" class="form-input" id="llmBaseUrl" value="https://api.deepseek.com">
            </div>
            <div class="form-group">
              <label class="form-label">Temperature：<span class="range-value" id="tempValue">0.7</span></label>
              <input type="range" id="llmTemperature" min="0" max="2" step="0.1" value="0.7" oninput="document.getElementById('tempValue').textContent=this.value">
            </div>
            <div class="form-group">
              <label class="form-label">Max Tokens</label>
              <input type="number" class="form-input" id="llmMaxTokens" value="8192" min="1" max="65536">
            </div>
            <div style="display:flex;gap:8px;">
              <button class="btn btn-primary" onclick="testConnection()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg> 测试连接</button>
              <button class="btn btn-secondary" onclick="saveLLMConfig()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> 保存配置</button>
            </div>
            <div id="testResult"></div>
          </div>
          <div class="card">
            <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg></span>提供商信息</div>
            <div id="providerInfo" style="font-size:0.85rem;line-height:1.8;color:var(--text-secondary);">
              <p><strong>DeepSeek</strong> - 深度求索大模型</p>
              <p>Base URL: <code style="background:var(--bg-tertiary);padding:2px 6px;border-radius:4px;font-family:var(--font-mono);">https://api.deepseek.com</code></p>
              <p>可用模型: deepseek-v4-pro, deepseek-chat, deepseek-coder</p>
              <p style="margin-top:8px;color:var(--text-muted);">DeepSeek V4 Pro 是当前最强大的深度求索模型，支持长上下文和复杂推理任务。</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 4: Business Process -->
      <div class="tab-content" id="tab-business">
        <div class="card">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 00-2.91-.09zM12 15l-3-3a22 22 0 012-3.95A12.88 12.88 0 0122 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 01-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg></span>生成流程
            <button class="btn btn-primary btn-lg" id="btnGenerateBP" onclick="generateBusinessProcess()" style="margin-left:12px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2.4 7.2h7.6l-6 4.8 2.4 7.2-6.4-4.8-6.4 4.8 2.4-7.2-6-4.8h7.6z"/></svg> 生成业务流程</button>
          </div>
          <div id="bpProgress" style="display:none;">
            <div class="progress-bar" style="margin-bottom:8px;"><div class="fill" id="bpProgressFill" style="width:0%"></div></div>
            <div class="progress-info"><span id="bpProgressText">准备中...</span><span id="bpProgressPercent">0%</span></div>
          </div>
        </div>

        <!-- BP Intro Card -->
        <div class="card" id="bpIntroCard">
          <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span>流程可视化类型</div>
          <div class="bp-tabs" style="margin-bottom:12px;">
            <button class="bp-tab active" data-intro="mermaid" onclick="switchIntro('mermaid')">Mermaid流程图</button>
            <button class="bp-tab" data-intro="echarts" onclick="switchIntro('echarts')">ECharts树形图</button>
            <button class="bp-tab" data-intro="mindmap" onclick="switchIntro('mindmap')">思维导图</button>
            <button class="bp-tab" data-intro="summary" onclick="switchIntro('summary')">汇总报告</button>
            <button class="bp-tab" data-intro="drawio" onclick="switchIntro('drawio')">DrawIO</button>
          </div>
          <div id="intro-mermaid" class="intro-content" style="display:block;">
            <div style="display:flex;gap:16px;align-items:flex-start;">
              <div style="flex:1;font-size:0.85rem;line-height:1.7;color:var(--text-secondary);">
                <strong style="color:var(--accent);">Mermaid流程图</strong> 使用标准流程图语法，通过节点和分支连接清晰展示决策路径和话术分支。<br><br>
                • 支持节点形状：矩形(话术)、菱形(决策)、圆角(开始/结束)<br>
                • 支持条件分支标签<br>
                • 适合展示完整的业务逻辑树<br>
              </div>
              <div style="flex:1;background:var(--bg-input);border-radius:var(--radius);padding:12px;font-family:var(--font-mono);font-size:0.7rem;overflow:auto;max-height:200px;">
                <pre style="color:var(--accent);">graph TD
  Start(("开始"))
  Q1["Q1: 开场白"]
  Start --> Q1
  Q1 -->|客户有意向| Q2["Q2: 介绍产品"]
  Q1 -->|客户拒绝| End1(("结束"))
  Q2 -->|确认购买| Q3["Q3: 确认订单"]
  Q3 --> End2(("转人工"))</pre>
              </div>
            </div>
          </div>
          <div id="intro-echarts" class="intro-content" style="display:none;">
            <div style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary);">
              <strong style="color:var(--info);">ECharts树形图</strong> 交互式树形结构，从上到下展示层级关系。<br>
              支持滚轮缩放、拖拽平移、点击节点展开收起，适合深层嵌套业务流程的可视化浏览。
            </div>
          </div>
          <div id="intro-mindmap" class="intro-content" style="display:none;">
            <div style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary);">
              <strong style="color:var(--warning);">思维导图</strong> 可折叠的HTML树形结构，直观展示Markdown标题层级。<br>
              支持XMind导入（下载.xmind文件后用XMind打开，或直接导入Markdown到XMind）。
            </div>
          </div>
          <div id="intro-summary" class="intro-content" style="display:none;">
            <div style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary);">
              <strong style="color:var(--danger);">汇总报告</strong> 包含统计卡片、流程图和树形图的综合仪表板。<br>
              适合打印和分享，配色使用网站主题色 #00a986。
            </div>
          </div>
          <div id="intro-drawio" class="intro-content" style="display:none;">
            <div style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary);">
              <strong style="color:var(--accent-secondary);">DrawIO流程图</strong> 生成标准DrawIO XML文件。<br>
              下载后可使用draw.io在线编辑器打开编辑修改，支持导出为PNG/SVG/PDF等格式。
            </div>
          </div>
        </div>

        <div class="card" id="bpResultCard" style="display:none;">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></span>生成结果
          </div>
          <div class="bp-viewer">
            <div class="bp-tabs">
              <button class="bp-tab active" data-bp="markdown" onclick="switchBPTab('markdown')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg> Markdown文档</button>
              <button class="bp-tab" data-bp="mermaid" onclick="switchBPTab('mermaid')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg> Mermaid流程图</button>
              <button class="bp-tab" data-bp="echarts" onclick="switchBPTab('echarts')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg> ECharts树形图</button>
              <button class="bp-tab" data-bp="mindmap" onclick="switchBPTab('mindmap')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg></span> 思维导图</button>
              <button class="bp-tab" data-bp="summary" onclick="switchBPTab('summary')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg> 汇总报告</button>
              <button class="bp-tab" data-bp="drawio" onclick="switchBPTab('drawio')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg> DrawIO</button>
            </div>
            <div class="bp-content active" id="bp-markdown">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-primary" onclick="copyMarkdown()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg> 复制文档</button>
                <button class="btn btn-sm btn-secondary" onclick="downloadBPFile('markdown')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载.md</button>
              </div>
              <div class="markdown-body" id="bpMarkdownContent"></div>
            </div>
            <div class="bp-content" id="bp-mermaid">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-secondary" onclick="downloadBPFile('mermaid')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载.mmd</button>
                <button class="btn btn-sm btn-secondary" onclick="downloadMermaidPNG()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 下载PNG</button>
                <span style="font-size:0.75rem;color:var(--text-muted);margin-left:8px;">🖱 滚轮缩放 | 拖拽平移</span>
              </div>
              <div id="bpMermaidWrapper" style="width:100%;height:70vh;overflow:auto;border:1px solid var(--border);border-radius:var(--radius);background:#fff;">
                <div class="mermaid" id="bpMermaidContent" style="min-width:100%;min-height:100%;"></div>
              </div>
            </div>
            <div class="bp-content" id="bp-echarts">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-secondary" onclick="downloadBPFile('echarts')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载HTML</button>
                <button class="btn btn-sm btn-secondary" onclick="downloadEchartsPNG()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 下载PNG</button>
              </div>
              <div id="bpEchartsContent" style="width:100%;height:70vh;overflow:auto;"></div>
            </div>
            <div class="bp-content" id="bp-mindmap">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-secondary" onclick="downloadBPFile('xmind')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载XMind</button>
              </div>
              <div id="bpMindmapContent" style="width:100%;height:70vh;overflow:auto;"></div>
            </div>
            <div class="bp-content" id="bp-summary">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-secondary" onclick="downloadBPFile('summary')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载HTML</button>
              </div>
              <iframe id="bpSummaryFrame" style="width:100%;height:70vh;border:none;border-radius:var(--radius);"></iframe>
            </div>
            <div class="bp-content" id="bp-drawio">
              <div style="display:flex;gap:6px;margin-bottom:8px;">
                <button class="btn btn-sm btn-primary" onclick="downloadBPFile('drawio')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载.drawio</button>
                <button class="btn btn-sm btn-secondary" onclick="window.open('https://app.diagrams.net/', '_blank')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg> 在线编辑</button>
              </div>
              <pre id="bpDrawioContent" style="width:100%;height:60vh;overflow:auto;font-family:var(--font-mono);font-size:0.7rem;background:var(--bg-input);padding:16px;border-radius:var(--radius);white-space:pre-wrap;color:var(--text-secondary);"></pre>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 5: Prompt Management -->
      <div class="tab-content" id="tab-prompts">
        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></span>提示词列表</div>
            <div class="prompt-list-panel" id="promptList"></div>
          </div>
          <div class="card">
            <div class="card-header">
              <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.828 2.828 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg></span>编辑提示词
              <span id="promptStatus" style="margin-left:auto;font-size:0.75rem;"></span>
            </div>
            <div id="promptEditorPanel">
              <div class="form-group">
                <label class="form-label" id="promptEditorLabel">请从左侧选择一个提示词</label>
              </div>
              <textarea class="form-textarea" id="promptEditor" placeholder="选择提示词后在此编辑..." disabled></textarea>
              <div style="display:flex;gap:8px;margin-top:12px;">
                <button class="btn btn-primary" id="btnSavePrompt" onclick="savePrompt()" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> 保存</button>
                <button class="btn btn-secondary" id="btnResetPrompt" onclick="resetPrompt()" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg> 重置为默认</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 6: FAQ Extraction -->
      <div class="tab-content" id="tab-faq">
        <div class="card">
          <div class="card-header">
            <span class="card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>FAQ提取（基于ASR结果）
            <button class="btn btn-primary" id="btnExtractFAQ" onclick="extractFAQ()" style="margin-left:auto;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2.4 7.2h7.6l-6 4.8 2.4 7.2-6.4-4.8-6.4 4.8 2.4-7.2-6-4.8h7.6z"/></svg> 提取FAQ</button>
            <button class="btn btn-sm btn-secondary" id="btnDownloadFAQExcel" onclick="downloadFAQExcel()" disabled style="margin-left:8px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载Excel</button>
          </div>
          <div id="faqProgress" style="display:none;margin-bottom:12px;">
            <div class="progress-bar"><div class="fill" id="faqProgressFill" style="width:50%"></div></div>
            <div class="progress-info"><span id="faqProgressText">正在提取FAQ...</span></div>
          </div>
          <div id="faqResults" style="max-height:70vh;overflow-y:auto;"></div>
          <div class="empty-state" id="faqEmpty">
            <div class="empty-icon"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
            <div class="empty-text">点击上方按钮从ASR解析结果中提取FAQ</div>
            <div class="empty-hint">系统将自动识别客户问题和客服解答，并生成Excel文件</div>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- URL Batch Paste Modal -->
<div class="modal-overlay" id="urlModal" style="display:none;">
  <div class="modal">
    <div class="modal-header">
      <span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg> 批量粘贴音频URL</span>
      <button class="modal-close" onclick="closeUrlModal()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
    </div>
    <div class="form-group">
      <label class="form-label">每行输入一个URL，支持HTTP/HTTPS链接</label>
      <textarea class="form-textarea" id="urlBatchInput" placeholder="https://example.com/audio1.mp3&#10;https://example.com/audio2.mp3&#10;https://example.com/audio3.mp3" style="min-height:200px;"></textarea>
    </div>
    <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px;">
      支持格式：MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, WMA
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn btn-primary" onclick="parseBatchUrls()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> 解析并添加URL</button>
      <button class="btn btn-secondary" onclick="closeUrlModal()">取消</button>
    </div>
  </div>
</div>

<!-- Toast container -->
<div class="toast-container" id="toastContainer"></div>

<!-- hidden file input -->
<input type="file" id="hiddenFileInput" multiple accept=".mp3,.wav,.m4a,.flac,.ogg,.opus,.aac,.wma" style="display:none">

<script>
// Initialize Mermaid
if (typeof mermaid !== 'undefined') {
  mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose', flowchart: { useMaxWidth: true, htmlLabels: true } });
}

// ==================== SVG Icons ====================
const I = {
  music: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
  speaker: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>',
  robot: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="11" r="1.5"/><circle cx="15" cy="11" r="1.5"/><path d="M12 16v3M9 19h6"/></svg>',
  chart: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
  doc: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
  upload: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
  folder: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>',
  globe: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>',
  clipboard: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/></svg>',
  download: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
  xclose: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  file: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
  gear: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
  play: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="6 3 20 12 6 21 6 3"/></svg>',
  pause: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="5" y="4" width="5" height="16"/><rect x="14" y="4" width="5" height="16"/></svg>',
  trend: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
  search: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
  save: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
  rocket: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 00-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 012-3.95A12.88 12.88 0 0122 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 01-4 2z"/></svg>',
  spark: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
  link: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>',
  pen: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.828 2.828 0 014 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>',
  reset: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>',
  sun: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
  moon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>',
  check: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
  xcircle: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
  info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  history: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
  trash: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>',
  plus: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>',
};

// ==================== Global State ====================
let currentTab = 'audio';
let currentBPResult = null;
let currentPrompt = null;
let urls = [];
let isDark = false;
let asrPollingInterval = null;
let modelDownloadPollingInterval = null;
let sessionAudioFiles = [];  // Track audio files uploaded/downloaded in this session
let sessionASRResults = [];  // Track ASR results produced in this session

// ==================== Navigation ====================
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', function() {
    switchTab(this.dataset.tab);
  });
});

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  document.querySelector(`.nav-item[data-tab="${tab}"]`).classList.add('active');
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');

  const titles = { audio: '音频管理', asr: 'ASR解析', llm: '模型配置', prompts: '提示词管理', business: '生成流程', faq: 'FAQ提取' };
  document.getElementById('pageTitle').textContent = titles[tab] || tab;

  if (tab === 'audio') { refreshAudioList(); }
  if (tab === 'asr') { renderASRHistory(); startASRPolling(); } else { stopASRPolling(); if (modelDownloadPollingInterval) { clearInterval(modelDownloadPollingInterval); modelDownloadPollingInterval = null; } }
  if (tab === 'llm') loadLLMConfig();
  if (tab === 'business') {}
  if (tab === 'prompts') loadPrompts();
  if (tab === 'faq') loadFAQResults();
}

// ==================== FAQ Extraction ====================
async function extractFAQ() {
  // Collect from DOM + session
  const asrItems = document.querySelectorAll('#asrFileList .asr-file-item');
  asrItems.forEach(el => {
    const fn = el.getAttribute('data-file');
    if (fn && !sessionASRResults.includes(fn)) sessionASRResults.push(fn);
  });
  if (sessionASRResults.length === 0) {
    toast('ASR结果查看器中暂无ASR解析文本，请先完成ASR解析或从历史中引用', 'error');
    switchTab('asr');
    return;
  }
  const btn = document.getElementById('btnExtractFAQ');
  btn.disabled = true; btn.textContent = '提取中...';
  document.getElementById('faqProgress').style.display = 'block';
  try {
    const filenames = sessionASRResults.join(',');
    const r = await apiPost(`/api/faq/extract?use_all_asr=false&asr_filenames=${encodeURIComponent(filenames)}`);
    if (r.success) { toast(r.message, 'success'); let attempts = 0;
      const pi = setInterval(async () => {
        try { const d = await apiGet('/api/faq/results');
          if (d.faqs && d.faqs.length > 0) { clearInterval(pi); renderFAQResults(d.faqs);
            document.getElementById('faqProgress').style.display = 'none';
            document.getElementById('btnDownloadFAQExcel').disabled = false;
            btn.disabled = false; btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2.4 7.2h7.6l-6 4.8 2.4 7.2-6.4-4.8-6.4 4.8 2.4-7.2-6-4.8h7.6z"/></svg> 提取FAQ';
            toast(`成功提取 ${d.faqs.length} 条FAQ`, 'success'); }
          if (++attempts > 120) { clearInterval(pi); btn.disabled = false; }
        } catch(e) {}
      }, 2000);
    } else { toast(r.message, 'error'); btn.disabled = false; document.getElementById('faqProgress').style.display = 'none'; }
  } catch(e) { toast('提取失败: ' + e.message, 'error'); btn.disabled = false; document.getElementById('faqProgress').style.display = 'none'; }
}

function renderFAQResults(faqs) {
  document.getElementById('faqEmpty').style.display = 'none';
  const cats = [...new Set(faqs.map(f => f.category || '其他'))];
  let h = '<div style="margin-bottom:12px;display:flex;gap:6px;flex-wrap:wrap;">';
  cats.forEach(c => h += `<span class="tag" style="background:var(--accent-glow);color:var(--accent);">${escapeHtml(c)}</span>`);
  h += `<span style="font-size:0.8rem;color:var(--text-muted);margin-left:8px;">共 ${faqs.length} 条</span></div>`;
  faqs.forEach((faq, i) => {
    h += `<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:14px;margin-bottom:10px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <span style="background:var(--accent);color:#fff;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700;">#${i+1}</span>
        <span class="tag" style="font-size:0.7rem;">${escapeHtml(faq.category||'其他')}</span>
      </div>
      <div style="font-weight:600;color:var(--accent);margin-bottom:4px;font-size:0.9rem;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        question：${escapeHtml(faq.question||'')}
      </div>
      <div style="color:var(--text-secondary);font-size:0.85rem;line-height:1.7;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:4px;"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        answer：${escapeHtml(faq.answer||'')}
      </div>
    </div>`;
  });
  document.getElementById('faqResults').innerHTML = h;
}

async function loadFAQResults() {
  try { const d = await apiGet('/api/faq/results'); if (d.faqs && d.faqs.length > 0) { renderFAQResults(d.faqs); document.getElementById('btnDownloadFAQExcel').disabled = false; } } catch(e) {}
}

function downloadFAQExcel() { window.open('/api/faq/download-excel', '_blank'); }

// ==================== Theme ====================
function toggleTheme() {
  isDark = !isDark;
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  document.getElementById('themeIcon').innerHTML = isDark ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>' : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
  document.getElementById('themeLabel').textContent = isDark ? '深色' : '浅色';
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

(function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') { isDark = false; toggleTheme(); }
})();

// ==================== Toast ====================
function toast(msg, type) {
  type = type || 'info';
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icons = { success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>', error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>', info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>' };
  el.innerHTML = `<span>${icons[type] || ''}</span> ${msg}`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ==================== API Helpers ====================
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPost(url, data) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPut(url, data) {
  const r = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE' });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ==================== Tab 1: Audio Management ====================
document.addEventListener('DOMContentLoaded', function() {
  // Upload zone
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    uploadFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => {
    uploadFiles(fileInput.files);
    fileInput.value = '';
  });

  refreshAudioList();
});

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;
  const formData = new FormData();
  const fileNames = [];
  for (let f of fileList) {
    formData.append('files', f);
    fileNames.push(f.name);
  }
  try {
    const r = await fetch('/api/audio/upload', { method: 'POST', body: formData });
    const data = await r.json();
    // Track uploaded filenames from API response
    data.forEach(f => {
      if (!sessionAudioFiles.includes(f.filename)) {
        sessionAudioFiles.push(f.filename);
      }
    });
    saveAudioUploadHistory(fileNames);
    toast(`成功上传 ${data.length} 个文件`, 'success');
    refreshAudioList();
  } catch (e) {
    toast('上传失败: ' + e.message, 'error');
  }
}

function openUrlModal() {
  document.getElementById('urlModal').style.display = 'flex';
  document.getElementById('urlBatchInput').focus();
}

function closeUrlModal() {
  document.getElementById('urlModal').style.display = 'none';
}

function parseBatchUrls() {
  const text = document.getElementById('urlBatchInput').value.trim();
  if (!text) { toast('请输入URL', 'error'); return; }

  // Split by newlines, filter empty lines, trim whitespace
  const lines = text.split(/[\r\n]+/).map(l => l.trim()).filter(l => l.length > 0);
  const newUrls = [];
  const invalid = [];

  for (const line of lines) {
    // Try to extract URL from the line (it might have extra text)
    const urlMatch = line.match(/(https?:\/\/[^\s<>"']+)/i);
    if (urlMatch) {
      const url = urlMatch[1];
      // Check if it looks like an audio URL
      if (/\.(mp3|wav|m4a|flac|ogg|opus|aac|wma|wmv|amr|webm)(\?.*)?$/i.test(url) || url.includes('cos.') || url.includes('audio') || url.includes('sound') || url.includes('mp3') || url.includes('voice')) {
        newUrls.push(url);
      } else {
        // Accept any HTTP URL, the download will validate
        newUrls.push(url);
      }
    } else {
      invalid.push(line.substring(0, 50));
    }
  }

  if (newUrls.length === 0) {
    toast('未能从输入中解析出有效URL', 'error');
    return;
  }

  urls = [...urls, ...newUrls];
  document.getElementById('urlBatchInput').value = '';
  document.getElementById('urlModal').style.display = 'none';
  renderUrlList();
  document.getElementById('btnDownloadUrls').disabled = urls.length === 0;
  document.getElementById('btnDownloadUrls').innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (${urls.length})`;

  let msg = `成功解析 ${newUrls.length} 个URL`;
  if (invalid.length > 0) msg += `，${invalid.length} 行无法识别`;
  toast(msg, 'success');
}

function clearUrls() {
  urls = [];
  renderUrlList();
  document.getElementById('btnDownloadUrls').disabled = true;
  document.getElementById('btnDownloadUrls').innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (0)';
}

function toggleUrlDetail() {
  const list = document.getElementById('urlList');
  const icon = document.getElementById('urlToggleIcon');
  if (list.style.display === 'none') {
    list.style.display = 'block';
    icon.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>';
  } else {
    list.style.display = 'none';
    icon.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>';
  }
}

function renderUrlList() {
  const summary = document.getElementById('urlSummary');
  const summaryText = document.getElementById('urlSummaryText');
  const list = document.getElementById('urlList');

  if (urls.length === 0) {
    summary.style.display = 'none';
    list.innerHTML = '';
    list.style.display = 'none';
    return;
  }

  summary.style.display = 'block';
  summaryText.textContent = `已解析 ${urls.length} 条音频URL（点击展开查看详情）`;
  list.style.display = 'none';  // collapsed by default

  list.innerHTML = urls.map((url, i) => `
    <div class="url-item">
      <span class="url-index">#${i+1}</span>
      <span class="url-text">${escapeHtml(url)}</span>
      <button class="btn btn-sm btn-danger" onclick="removeUrl(${i})"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
    </div>
  `).join('');
}

function removeUrl(i) {
  urls.splice(i, 1);
  renderUrlList();
  document.getElementById('btnDownloadUrls').disabled = urls.length === 0;
  document.getElementById('btnDownloadUrls').innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (${urls.length})`;
}

async function startDownload() {
  if (urls.length === 0) return;
  const workers = parseInt(document.getElementById('downloadWorkers').value);
  const retries = parseInt(document.getElementById('downloadRetries').value);
  const btn = document.getElementById('btnDownloadUrls');
  btn.disabled = true;
  btn.textContent = '下载中...';

  try {
    const r = await apiPost('/api/audio/download-urls', { urls: urls, max_workers: workers, max_retries: retries });
    toast(r.message, 'success');
    // Poll for progress and auto-refresh audio list
    let pollCount = 0;
    let lastCompleted = 0;
    let allDone = false;
    let downloadedFiles = [];  // Track actual filenames from this batch
    const pollInterval = setInterval(async () => {
      try {
        const progress = await apiGet('/api/audio/download-progress');
        renderDownloadProgress(progress);
        const values = Object.values(progress);
        const completedCount = values.filter(p => p.status === 'completed').length;
        const totalCount = values.length;
        const errorCount = values.filter(p => p.status === 'error').length;

        // Collect completed filenames
        values.filter(p => p.status === 'completed' && p.filename).forEach(p => {
          if (!downloadedFiles.includes(p.filename)) {
            downloadedFiles.push(p.filename);
          }
        });

        // Auto-populate audio list as files complete
        if (completedCount > lastCompleted) {
          lastCompleted = completedCount;
          // Update session audio files with actual filenames
          downloadedFiles.forEach(f => {
            if (!sessionAudioFiles.includes(f)) sessionAudioFiles.push(f);
          });
          refreshAudioList();
          document.getElementById('downloadCompletedMsg').style.display = 'block';
          document.getElementById('downloadCompletedMsg').innerHTML =
            `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> 已完成 ${completedCount}/${totalCount} 个文件（${errorCount} 个失败），已自动填充到音频列表`;
        }

        // Check if all done
        if (!allDone && completedCount + errorCount >= totalCount && totalCount > 0) {
          allDone = true;
          saveDownloadHistory(urls, completedCount);
          // Final update of session audio files
          downloadedFiles.forEach(f => {
            if (!sessionAudioFiles.includes(f)) sessionAudioFiles.push(f);
          });
          refreshAudioList();
          btn.disabled = false;
          btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (${urls.length})`;
          if (completedCount > 0) {
            toast(`下载完成：成功 ${completedCount} 个，失败 ${errorCount} 个`, completedCount > 0 ? 'success' : 'error');
          }
        }

        pollCount++;
        if (pollCount > 600) { clearInterval(pollInterval); refreshAudioList(); btn.disabled = false; }
      } catch(e) {}
    }, 1000);
  } catch (e) {
    toast('下载失败: ' + e.message, 'error');
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始下载 (${urls.length})`;
  }
}

function renderDownloadProgress(progress) {
  const list = document.getElementById('downloadProgressList');
  if (!progress || Object.keys(progress).length === 0) {
    list.innerHTML = '';
    return;
  }
  const statusLabels = {
    'pending': '等待中', 'downloading': '下载中', 'completed': '已完成',
    'error': '失败', 'retrying': '重试中', 'cancelled': '已取消'
  };
  list.innerHTML = Object.values(progress).map(p => `
    <div class="file-item">
      <span class="file-icon">${p.status === 'completed' ? I.check : p.status === 'error' ? I.xcircle : p.status === 'retrying' ? I.reset : I.history}</span>
      <span class="file-name" style="flex:1;">${escapeHtml(p.filename || p.url)}</span>
      <span style="width:100px;">
        <div class="progress-bar" style="height:4px;"><div class="fill" style="width:${p.progress}%;${p.status === 'retrying' ? 'background:linear-gradient(90deg, var(--warning), #f0a84c);' : ''}"></div></div>
      </span>
      <span class="file-status status-${p.status === 'completed' ? 'completed' : p.status === 'error' ? 'error' : p.status === 'retrying' ? 'processing' : 'processing'}">${statusLabels[p.status] || p.status}</span>
      ${p.error_message ? `<span style="font-size:0.7rem;color:var(--danger);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(p.error_message)}">${escapeHtml(p.error_message)}</span>` : ''}
    </div>
  `).join('');
}

async function refreshAudioList() {
  try {
    const data = await apiGet('/api/audio/list');
    const list = document.getElementById('audioFileList');
    const empty = document.getElementById('audioEmpty');
    document.getElementById('audioCount').textContent = sessionAudioFiles.length;

    // On page refresh (sessionAudioFiles is empty), list is empty
    if (sessionAudioFiles.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      empty.querySelector('.empty-text').textContent = '暂无音频文件';
      empty.querySelector('.empty-hint').textContent = '请上传音频文件或通过URL下载';
      return;
    }

    // Only show files that match sessionAudioFiles by exact filename
    let displayFiles = data.files.filter(f => sessionAudioFiles.includes(f.filename));

    if (displayFiles.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      empty.querySelector('.empty-text').textContent = '暂无音频文件';
      empty.querySelector('.empty-hint').textContent = '请上传音频文件或通过URL下载';
      return;
    }
    empty.style.display = 'none';
    list.innerHTML = displayFiles.map(f => `
      <div class="file-item">
        <span class="file-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="16" cy="16" r="3"/></svg></span>
        <span class="file-name">${escapeHtml(f.filename)}</span>
        <span class="file-size">${formatSize(f.size_bytes)}</span>
        <button class="btn btn-sm btn-primary" onclick="playAudio('${escapeHtml(f.filename)}')" title="播放"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="6 3 20 12 6 21 6 3"/></svg></button>
        <button class="btn btn-sm btn-danger" onclick="deleteAudio('${escapeHtml(f.filename)}')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
      </div>
    `).join('');
  } catch (e) {
    console.error('refreshAudioList:', e);
  }
}

// Audio player - opens in new tab
function playAudio(filename) {
  const audioUrl = `/api/audio/play/${encodeURIComponent(filename)}`;
  window.open(audioUrl, '_blank');
  toast('正在新标签页播放: ' + filename, 'info');
}

function loadHistoricalAudio() {
  // Open a sidebar showing historical audio by rounds
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('audio_upload_history') || '[]');
  } catch(e) {}

  if (history.length === 0) {
    toast('暂无历史音频记录', 'info');
    return;
  }

  // Build sidebar HTML
  let html = '<div class="modal-overlay" id="audioHistoryModal" style="display:flex;" onclick="if(event.target===this)closeAudioHistory()">';
  html += '<div class="modal" style="max-width:700px;max-height:80vh;">';
  html += '<div class="modal-header"><span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> 历史音频上传记录</span>';
  html += '<button class="modal-close" onclick="closeAudioHistory()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button></div>';

  html += '<div style="max-height:60vh;overflow-y:auto;">';
  history.forEach((round, i) => {
    const fileList = (round.files || []).slice(0, 10).map(f => `<span class="tag" style="font-size:0.7rem;">${escapeHtml(f)}</span>`).join(' ');
    const moreFiles = (round.files || []).length > 10 ? `<span class="tag" style="font-size:0.7rem;">...还有${round.files.length - 10}个文件</span>` : '';
    html += `
      <div class="asr-history-item" style="cursor:default;border-bottom:1px solid var(--border);padding:12px 14px;">
        <span class="history-time"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ${escapeHtml(round.time)}</span>
        <div class="history-info" style="flex:1;">
          <div style="font-weight:600;">第 ${i+1} 轮 · <span class="history-count">${round.fileCount}</span> 个音频</div>
          <div style="margin-top:4px;font-size:0.75rem;">${fileList}${moreFiles}</div>
        </div>
        <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();reuseAudioHistory(${i})">${I.rocket} 引用</button>
      </div>
    `;
  });
  html += '</div>';

  html += '<div style="margin-top:16px;display:flex;gap:8px;justify-content:flex-end;">';
  html += '<button class="btn btn-secondary" onclick="closeAudioHistory()">关闭</button>';
  html += '<button class="btn btn-danger btn-sm" onclick="clearAudioHistory()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> 清空历史</button>';
  html += '</div>';
  html += '</div></div>';

  document.body.insertAdjacentHTML('beforeend', html);
}

function closeAudioHistory() {
  const modal = document.getElementById('audioHistoryModal');
  if (modal) modal.remove();
}

function clearAudioHistory() {
  if (!confirm('确定要清空所有音频上传历史记录吗？')) return;
  localStorage.removeItem('audio_upload_history');
  closeAudioHistory();
  toast('音频历史记录已清空', 'success');
}

function reuseAudioHistory(index) {
  let history = [];
  try { history = JSON.parse(localStorage.getItem('audio_upload_history') || '[]'); } catch(e) {}
  if (!history[index]) return;
  const round = history[index];
  // Add the filenames back to session for ASR processing
  if (round.files && round.files.length > 0) {
    round.files.forEach(f => {
      if (!sessionAudioFiles.includes(f)) sessionAudioFiles.push(f);
    });
    closeAudioHistory();
    refreshAudioList();
    toast(`已引用第${index+1}轮历史音频（${round.files.length}个文件），可在ASR解析中处理`, 'success');
  }
}

function saveAudioUploadHistory(files) {
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('audio_upload_history') || '[]');
  } catch(e) {}
  history.unshift({
    time: new Date().toLocaleString('zh-CN'),
    fileCount: files.length,
    files: files.map(f => f.name || f)
  });
  // Keep only last 5 rounds, delete older files from disk
  if (history.length > 5) {
    const removed = history.splice(5);
    removed.forEach(round => {
      (round.files || []).forEach(async fn => {
        try { await apiDelete(`/api/audio/${encodeURIComponent(fn)}`); } catch(e) {}
      });
    });
  }
  localStorage.setItem('audio_upload_history', JSON.stringify(history));
}

async function deleteAudio(filename) {
  try {
    await apiDelete(`/api/audio/${encodeURIComponent(filename)}`);
    toast('已删除: ' + filename, 'success');
    sessionAudioFiles = sessionAudioFiles.filter(f => f !== filename);
    refreshAudioList();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
}

async function deleteAllAudio() {
  if (!confirm('确定要删除所有音频文件吗？此操作不可恢复。')) return;
  try {
    const r = await apiDelete('/api/audio/all/delete');
    toast(r.message, 'success');
    sessionAudioFiles = [];
    refreshAudioList();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
}

// ==================== Tab 2: ASR Processing ====================
async function startASR() {
  // Only process current session audio files
  if (sessionAudioFiles.length === 0) {
    toast('请先上传或下载音频文件到音频文件列表', 'error');
    return;
  }
  try {
    // Start model download progress monitoring
    startModelDownloadMonitor();
    // Pass specific files from current session
    const specificFiles = sessionAudioFiles.join(',');
    const r = await apiPost(`/api/asr/start?audio_dir=./data/audio&specific_files=${encodeURIComponent(specificFiles)}`);
    toast(r.message, 'success');
    document.getElementById('btnStartASR').disabled = true;
    document.getElementById('btnCancelASR').disabled = false;
    asrSessionActive = true;
    startASRPolling();
  } catch (e) {
    toast('启动失败: ' + e.message, 'error');
  }
}

function startModelDownloadMonitor() {
  // Show model download progress UI
  const el = document.getElementById('modelDownloadProgress');
  el.style.display = 'block';
  document.getElementById('modelDownloadText').textContent = '正在检查ASR模型...';
  document.getElementById('modelDownloadFill').style.width = '0%';
  document.getElementById('modelDownloadSubtext').textContent = '检测SenseVoiceSmall模型缓存状态...';

  if (modelDownloadPollingInterval) clearInterval(modelDownloadPollingInterval);
  modelDownloadPollingInterval = setInterval(async () => {
    try {
      const state = await apiGet('/api/asr/model-download-state');
      if (state.is_downloading) {
        // Model is being downloaded - show progress
        el.style.display = 'block';
        document.getElementById('modelDownloadText').textContent = '正在下载ASR模型 (SenseVoiceSmall)...';
        document.getElementById('modelDownloadFill').style.width = state.progress + '%';
        document.getElementById('modelDownloadSubtext').textContent = state.message || '首次使用需要下载模型文件（约900MB），请耐心等待...';
      } else if (state.progress >= 100) {
        // Model is loaded or cached
        document.getElementById('modelDownloadText').textContent = '模型已就绪！';
        document.getElementById('modelDownloadFill').style.width = '100%';
        document.getElementById('modelDownloadSubtext').textContent = state.message || 'ASR模型加载完成，开始处理音频文件...';
        // Hide after 2 seconds
        setTimeout(() => { el.style.display = 'none'; }, 2000);
        clearInterval(modelDownloadPollingInterval);
        modelDownloadPollingInterval = null;
      }
      // If progress is 0 and not downloading, keep showing "checking..."
    } catch(e) {}
  }, 1000);
}

async function cancelASR() {
  try {
    await apiPost('/api/asr/cancel');
    toast('已请求取消ASR处理', 'info');
  } catch (e) {}
}

function startASRPolling() {
  if (asrPollingInterval) return;
  asrPollingInterval = setInterval(pollASRStatus, 1500);
  pollASRStatus();
}

function stopASRPolling() {
  if (asrPollingInterval) {
    clearInterval(asrPollingInterval);
    asrPollingInterval = null;
  }
}

async function pollASRStatus() {
  try {
    const status = await apiGet('/api/asr/status');
    updateASRProgress(status);

    if (status.is_running || status.current > 0) {
      refreshASRResults();
    }

    if (!status.is_running && status.total > 0 && window._lastASRState && window._lastASRState.is_running) {
      // Processing just completed - save history
      document.getElementById('btnStartASR').disabled = false;
      document.getElementById('btnCancelASR').disabled = true;
      asrSessionActive = false;
      refreshASRResults();
      if (modelDownloadPollingInterval) {
        clearInterval(modelDownloadPollingInterval);
        modelDownloadPollingInterval = null;
      }
      document.getElementById('modelDownloadProgress').style.display = 'none';
      // Save directly from current status (contains results)
      if (Object.keys(status.results || {}).length > 0 || status.current > 0) {
        saveASRHistory({
          fileCount: status.total,
          completedCount: status.current,
          results: status.results || {}
        });
      }
    } else if (status.is_running) {
      // Still running
    } else if (!status.is_running && status.total === 0) {
      document.getElementById('btnStartASR').disabled = false;
      document.getElementById('btnCancelASR').disabled = true;
    }
    window._lastASRState = status;
  } catch (e) {}
}

function updateASRProgress(status) {
  document.getElementById('asrProgressFill').style.width = status.progress + '%';
  document.getElementById('asrProgressCount').textContent = `${status.current} / ${status.total}`;
  // Show status_message if available, otherwise derive from state
  const msg = status.status_message || (status.is_running ? '处理中...' : (status.total > 0 ? '处理完成' : '等待开始...'));
  document.getElementById('asrProgressText').textContent = msg;
  document.getElementById('statTotal').textContent = status.total;
  document.getElementById('statCompleted').textContent = status.current;
}

let asrSessionActive = false;

async function refreshASRResults() {
  try {
    const results = await apiGet('/api/asr/results?session_only=true');
    const completedResults = results.filter(r => r.status === 'completed');
    // Always track session results
    completedResults.forEach(r => {
      if (!sessionASRResults.includes(r.filename)) {
        sessionASRResults.push(r.filename);
      }
    });
    // Only update UI during active session
    if (!asrSessionActive) return;
    const list = document.getElementById('asrFileList');
    if (completedResults.length === 0) {
      list.innerHTML = '<div class="empty-state" style="padding:40px 10px;"><div class="empty-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div><div class="empty-text" style="font-size:0.85rem;">等待ASR解析完成...</div><div class="empty-hint" style="font-size:0.75rem;">只有成功解析的音频文件才会显示在此处</div></div>';
      return;
    }
    list.innerHTML = completedResults.map(r => `
      <div class="asr-file-item" onclick="viewASRResult('${escapeHtml(r.filename)}')" data-file="${escapeHtml(r.filename)}">
        <span class="file-icon">${I.check}</span>
        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(r.filename)}</span>
        <span style="font-size:0.7rem;color:var(--text-muted);">${r.sentences ? r.sentences.length + '句' : ''}</span>
      </div>
    `).join('');
  } catch (e) {}
}

let currentASRFile = null;

async function viewASRResult(filename) {
  currentASRFile = filename;
  // Highlight active
  document.querySelectorAll('.asr-file-item').forEach(el => el.classList.remove('active'));
  const active = document.querySelector(`.asr-file-item[data-file="${filename}"]`);
  if (active) active.classList.add('active');

  try {
    const data = await apiGet(`/api/asr/view/${encodeURIComponent(filename)}`);
    const content = document.getElementById('asrContent');
    document.getElementById('btnDownloadTxt').disabled = false;
    if (data.sentences && data.sentences.length > 0) {
      content.innerHTML = data.sentences.map(s => `
        <div class="asr-sentence">
          <span class="sentence-num">part ${s.index}：</span>
          ${s.speaker ? `<span class="speaker-tag">${escapeHtml(s.speaker)}</span>` : ''}
          ${escapeHtml(s.text)}
        </div>
      `).join('');
    } else {
      content.innerHTML = '<div class="empty-state"><div class="empty-text">无识别结果</div></div>';
    }
  } catch (e) {
    toast('获取ASR结果失败: ' + e.message, 'error');
  }
}

function downloadASRTxt() {
  if (!currentASRFile) { toast('请先选择一个ASR结果', 'error'); return; }
  // Fetch the formatted text and trigger download
  apiGet(`/api/asr/view/${encodeURIComponent(currentASRFile)}`).then(data => {
    let text = '';
    if (data.sentences && data.sentences.length > 0) {
      text = data.sentences.map(s => {
        const spk = s.speaker ? `[${s.speaker}] ` : '';
        return `part ${s.index}：${spk}${s.text}`;
      }).join('\n');
    }
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentASRFile.replace(/\.txt$/,'') + '_ASR.txt';
    a.click();
    URL.revokeObjectURL(url);
    toast('下载成功', 'success');
  }).catch(e => toast('下载失败: ' + e.message, 'error'));
}

// ==================== ASR History ====================
function saveASRHistory(taskInfo) {
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('asr_history') || '[]');
  } catch(e) {}
  history.unshift({
    time: new Date().toLocaleString('zh-CN'),
    fileCount: taskInfo.fileCount || 0,
    completedCount: taskInfo.completedCount || 0,
    results: taskInfo.results || {}
  });
  // Keep only last 5, permanently delete older ASR output files
  if (history.length > 5) {
    const removed = history.splice(5);
    removed.forEach(round => {
      const filenames = Object.keys(round.results || {});
      filenames.forEach(async fn => {
        try { await apiDelete(`/api/asr/results/${encodeURIComponent(fn)}`); } catch(e) {}
      });
    });
  }
  localStorage.setItem('asr_history', JSON.stringify(history));
  renderASRHistory();
}

function renderASRHistory() {
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('asr_history') || '[]');
  } catch(e) {}

  const card = document.getElementById('asrHistoryCard');
  const list = document.getElementById('asrHistoryList');
  const empty = document.getElementById('asrHistoryEmpty');

  if (history.length === 0) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';
  empty.style.display = 'none';

  list.innerHTML = history.map((h, i) => `
    <div class="asr-history-item">
      <span class="history-time"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ${escapeHtml(h.time)}</span>
      <span class="history-info">
        完成 <span class="history-count">${h.completedCount}</span> / ${h.fileCount} 个文件
      </span>
      <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation();loadHistoryResults(${i})">${I.search} 查看</button>
      <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();useHistoryForBP(${i})">${I.rocket} 引用生成BP</button>
    </div>
  `).join('');
}

function loadHistoryResults(index) {
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('asr_history') || '[]');
  } catch(e) {}
  if (!history[index]) return;

  const entry = history[index];
  const results = entry.results || {};
  const filenames = Object.keys(results);

  if (filenames.length === 0) {
    toast('该历史记录无ASR结果', 'info');
    return;
  }

  // Populate the ASR file list with history results
  const list = document.getElementById('asrFileList');
  list.innerHTML = filenames.map(fn => {
    const r = results[fn];
    const sentenceCount = (r.sentences && r.sentences.length) ? r.sentences.length : 0;
    return `
      <div class="asr-file-item" onclick="viewHistoryASRResult('${escapeHtml(fn)}', ${index})" data-file="${escapeHtml(fn)}">
        <span class="file-icon">${r.status === 'completed' ? I.check : I.xcircle}</span>
        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(fn)}</span>
        <span style="font-size:0.7rem;color:var(--text-muted);">${sentenceCount}句</span>
      </div>
    `;
  }).join('');

  toast(`已加载历史记录: ${entry.time}`, 'success');
}

function viewHistoryASRResult(filename, historyIndex) {
  let history = [];
  try { history = JSON.parse(localStorage.getItem('asr_history') || '[]'); } catch(e) {}
  if (!history[historyIndex]) return;
  const result = history[historyIndex].results[filename];
  if (!result) return;

  document.querySelectorAll('.asr-file-item').forEach(el => el.classList.remove('active'));
  const active = document.querySelector(`.asr-file-item[data-file="${CSS.escape(filename)}"]`);
  if (active) active.classList.add('active');

  currentASRFile = filename;
  document.getElementById('btnDownloadTxt').disabled = false;
  const content = document.getElementById('asrContent');

  const sentences = result.sentences || [];
  if (sentences.length > 0) {
    content.innerHTML = sentences.map(s => `
      <div class="asr-sentence">
        <span class="sentence-num">part ${s.index}：</span>
        ${s.speaker ? `<span class="speaker-tag">${escapeHtml(s.speaker)}</span>` : ''}
        ${escapeHtml(s.text)}
      </div>
    `).join('');
  } else {
    content.innerHTML = '<div class="empty-state"><div class="empty-text">无识别结果</div></div>';
  }
}

function clearASRHistory() {
  if (!confirm('确定要清空所有ASR历史记录吗？')) return;
  localStorage.removeItem('asr_history');
  renderASRHistory();
  toast('历史记录已清空', 'success');
}

function renderASRHistorySidebar() {
  let history = [];
  try { history = JSON.parse(localStorage.getItem('asr_history') || '[]'); } catch(e) {}
  if (history.length === 0) { toast('暂无ASR历史记录', 'info'); return; }

  let html = '<div class="modal-overlay" id="asrHistorySidebarModal" style="display:flex;" onclick="if(event.target===this)closeASRHistorySidebar()">';
  html += '<div class="modal" style="max-width:750px;max-height:80vh;">';
  html += '<div class="modal-header"><span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ASR历史记录（最多5轮）</span>';
  html += '<button class="modal-close" onclick="closeASRHistorySidebar()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button></div>';
  html += '<div style="max-height:60vh;overflow-y:auto;">';
  history.forEach((h, i) => {
    html += `<div class="asr-history-item" style="border-bottom:1px solid var(--border);padding:12px 14px;">
      <span class="history-time">${I.history} ${escapeHtml(h.time)}</span>
      <span class="history-info" style="flex:1;">完成 <span class="history-count">${h.completedCount}</span> / ${h.fileCount} 个文件</span>
      <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation();loadHistoryResults(${i})">${I.search} 查看</button>
      <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();useASRHistoryForBP(${i})" style="margin-left:4px;">${I.rocket} 引用</button>
    </div>`;
  });
  html += '</div>';
  html += '<div style="margin-top:16px;display:flex;gap:8px;justify-content:flex-end;">';
  html += '<button class="btn btn-secondary" onclick="closeASRHistorySidebar()">关闭</button>';
  html += '<button class="btn btn-danger btn-sm" onclick="clearASRHistory();closeASRHistorySidebar()">清空历史</button>';
  html += '</div></div></div>';
  document.body.insertAdjacentHTML('beforeend', html);
}
function closeASRHistorySidebar() { const m = document.getElementById('asrHistorySidebarModal'); if (m) m.remove(); }
function useASRHistoryForBP(index) {
  let history = [];
  try { history = JSON.parse(localStorage.getItem('asr_history') || '[]'); } catch(e) {}
  if (!history[index]) return;
  const entry = history[index];
  const results = entry.results || {};
  const filenames = Object.keys(results);
  if (filenames.length === 0) { toast('该历史记录无ASR结果', 'info'); return; }

  // Populate session results from history
  sessionASRResults = filenames;

  // Populate ASR file list
  const list = document.getElementById('asrFileList');
  list.innerHTML = filenames.map(fn => {
    const r = results[fn];
    const sentenceCount = (r.sentences && r.sentences.length) ? r.sentences.length : 0;
    return `<div class="asr-file-item" onclick="viewHistoryASRResult('${escapeHtml(fn)}', ${index})" data-file="${escapeHtml(fn)}">
      <span class="file-icon">${r.status === 'completed' ? I.check : I.xcircle}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(fn)}</span>
      <span style="font-size:0.7rem;color:var(--text-muted);">${sentenceCount}句</span>
    </div>`;
  }).join('');

  // Auto-select first file
  if (filenames.length > 0) {
    setTimeout(() => viewHistoryASRResult(filenames[0], index), 200);
  }

  closeASRHistorySidebar();
  switchTab('asr');
  toast(`已引用第${index+1}轮ASR历史（${filenames.length}个文件），可进行生成流程和FAQ提取`, 'success');
}

// Use historical ASR results for BP generation
function useHistoryForBP(index) {
  // Switch to business tab and select the history source
  switchTab('business');
  setTimeout(() => {
    document.getElementById('bpAsrSource').value = 'history_' + index;
    toast('已选择历史ASR结果（第' + (index+1) + '轮），点击"生成业务流程"按钮开始', 'info');
  }, 200);
}

// Update pollASRStatus to save history when processing completes
function onASRComplete() {
  const state = window._lastASRState;
  if (state && state.total > 0 && state.results && Object.keys(state.results).length > 0) {
    saveASRHistory({
      fileCount: state.total,
      completedCount: state.current,
      results: state.results
    });
  }
}

// Save download history when downloads complete
function saveDownloadHistory(urlList, completedCount) {
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem('audio_upload_history') || '[]');
  } catch(e) {}
  history.unshift({
    time: new Date().toLocaleString('zh-CN'),
    fileCount: completedCount,
    files: urlList.map(u => u.split('/').pop().split('?')[0]).slice(0, 20)
  });
  // Keep only last 5
  if (history.length > 5) {
    const removed = history.splice(5);
    removed.forEach(round => {
      (round.files || []).forEach(async fn => {
        try { await apiDelete(`/api/audio/${encodeURIComponent(fn)}`); } catch(e) {}
      });
    });
  }
  localStorage.setItem('audio_upload_history', JSON.stringify(history));
}

// ==================== Tab 3: LLM Config ====================
function loadLLMConfig() {
  const saved = localStorage.getItem('llm_config');
  if (saved) {
    try {
      const cfg = JSON.parse(saved);
      document.getElementById('llmProvider').value = cfg.provider || 'deepseek';
      document.getElementById('llmModel').value = cfg.model || 'deepseek-v4-pro';
      document.getElementById('llmApiKey').value = cfg.api_key || '';
      document.getElementById('llmBaseUrl').value = cfg.base_url || 'https://api.deepseek.com';
      var temp = (cfg.temperature != null) ? cfg.temperature : 0.7;
      document.getElementById('llmTemperature').value = temp;
      document.getElementById('tempValue').textContent = temp;
      document.getElementById('llmMaxTokens').value = cfg.max_tokens || 8192;
    } catch(e) {}
  }
}

function saveLLMConfig() {
  const cfg = {
    provider: document.getElementById('llmProvider').value,
    model: document.getElementById('llmModel').value,
    api_key: document.getElementById('llmApiKey').value,
    base_url: document.getElementById('llmBaseUrl').value,
    temperature: parseFloat(document.getElementById('llmTemperature').value),
    max_tokens: parseInt(document.getElementById('llmMaxTokens').value)
  };
  localStorage.setItem('llm_config', JSON.stringify(cfg));
  toast('配置已保存到本地', 'success');
}

function onProviderChange() {
  const provider = document.getElementById('llmProvider').value;
  const presets = {
    deepseek: { base_url: 'https://api.deepseek.com', model: 'deepseek-v4-pro' },
    openai: { base_url: 'https://api.openai.com/v1', model: 'gpt-4o' },
    ollama: { base_url: 'http://localhost:11434/v1', model: 'qwen3:4b-instruct' },
    siliconflow: { base_url: 'https://api.siliconflow.cn/v1', model: 'deepseek-ai/DeepSeek-V3' },
    custom: { base_url: '', model: '' }
  };
  const p = presets[provider];
  if (p) {
    document.getElementById('llmBaseUrl').value = p.base_url;
    document.getElementById('llmModel').value = p.model;
  }

  const info = {
    deepseek: '<p><strong>DeepSeek</strong> - 深度求索大模型</p><p>Base URL: <code>https://api.deepseek.com</code></p><p>可用模型: deepseek-v4-pro, deepseek-chat, deepseek-coder</p>',
    openai: '<p><strong>OpenAI</strong> - GPT系列模型</p><p>Base URL: <code>https://api.openai.com/v1</code></p><p>可用模型: gpt-4o, gpt-4o-mini, gpt-4-turbo</p>',
    ollama: '<p><strong>Ollama</strong> - 本地大模型</p><p>Base URL: <code>http://localhost:11434/v1</code></p><p>可用模型: qwen3:4b-instruct, qwen2.5:7b, llama3:8b</p>',
    siliconflow: '<p><strong>SiliconFlow</strong> - 硅基流动</p><p>Base URL: <code>https://api.siliconflow.cn/v1</code></p><p>可用模型: DeepSeek-V3, Qwen2.5系列</p>',
    custom: '<p><strong>自定义</strong> - OpenAI兼容接口</p><p>可配置任意兼容OpenAI API的服务地址</p>'
  };
  document.getElementById('providerInfo').innerHTML = info[provider] || '';
}

async function testConnection() {
  const cfg = {
    provider: document.getElementById('llmProvider').value,
    model: document.getElementById('llmModel').value,
    api_key: document.getElementById('llmApiKey').value,
    base_url: document.getElementById('llmBaseUrl').value,
    temperature: parseFloat(document.getElementById('llmTemperature').value),
    max_tokens: parseInt(document.getElementById('llmMaxTokens').value)
  };

  const resultDiv = document.getElementById('testResult');
  resultDiv.innerHTML = '<div style="padding:12px;color:var(--text-muted);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 22h14M5 2h14M17 22v-4.172a2 2 0 00-.586-1.414L12 12l-4.414 4.414A2 2 0 007 17.828V22M7 2v4.172a2 2 0 00.586 1.414L12 12l4.414-4.414A2 2 0 0017 6.172V2"/></svg> 正在测试连接...</div>';

  try {
    const r = await apiPost('/api/llm/test', cfg);
    resultDiv.innerHTML = `
      <div class="test-result ${r.success ? 'success' : 'error'}">
        <strong>${r.success ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> 连接成功' : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> 连接失败'}</strong>
        <div>${escapeHtml(r.message)}</div>
        ${r.latency_ms ? `<div class="latency">延迟: ${r.latency_ms}ms | 模型: ${r.model} | Tokens: ${r.tokens_used}</div>` : ''}
      </div>
    `;
  } catch (e) {
    resultDiv.innerHTML = `<div class="test-result error"><strong><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> 请求失败</strong><div>${escapeHtml(e.message)}</div></div>`;
  }
}

// ==================== Tab 4: Business Process ====================
async function generateBusinessProcess() {
  // Collect filenames from ASR结果查看器 (session + DOM)
  const asrItems = document.querySelectorAll('#asrFileList .asr-file-item');
  const domFilenames = [];
  asrItems.forEach(el => {
    const fn = el.getAttribute('data-file');
    if (fn) domFilenames.push(fn);
  });
  // Merge with session
  domFilenames.forEach(fn => {
    if (!sessionASRResults.includes(fn)) sessionASRResults.push(fn);
  });

  if (sessionASRResults.length === 0) {
    toast('ASR结果查看器中暂无ASR解析文本，请先完成ASR解析或从历史中引用', 'error');
    switchTab('asr');
    return;
  }
  const btn = document.getElementById('btnGenerateBP');
  btn.disabled = true;
  btn.textContent = '生成中...';

  document.getElementById('bpProgress').style.display = 'block';
  document.getElementById('bpProgressFill').style.width = '5%';
  document.getElementById('bpProgressText').textContent = '正在启动（2步LLM调用，约需10-20秒）...';

  try {
    const cfg = JSON.parse(localStorage.getItem('llm_config') || '{}');
    const requestBody = {
      use_all_asr: false,
      asr_filenames: sessionASRResults,
      llm_config: cfg.provider ? cfg : null
    };

    const r = await apiPost('/api/business/generate', requestBody);

    if (r.success && r.data && r.data.task_id) {
      toast('业务流程生成已启动', 'success');
      pollBPTask(r.data.task_id);
    } else {
      toast(r.message || '启动失败', 'error');
      btn.disabled = false;
      btn.textContent = '生成业务流程';
      document.getElementById('bpProgress').style.display = 'none';
    }
  } catch (e) {
    toast('生成失败: ' + e.message, 'error');
    btn.disabled = false;
    btn.textContent = '生成业务流程';
    document.getElementById('bpProgress').style.display = 'none';
  }
}

function pollBPTask(taskId) {
  let attempts = 0;
  const interval = setInterval(async () => {
    try {
      const status = await apiGet(`/api/business/status/${taskId}`);
      document.getElementById('bpProgressFill').style.width = status.progress + '%';
      document.getElementById('bpProgressText').textContent = status.current_step;
      document.getElementById('bpProgressPercent').textContent = status.progress + '%';
      attempts++;

      if (status.status === 'completed') {
        clearInterval(interval);
        document.getElementById('bpProgressFill').style.width = '100%';
        document.getElementById('btnGenerateBP').disabled = false;
        document.getElementById('btnGenerateBP').textContent = '生成业务流程';
        toast('业务流程生成完成！', 'success');
        loadBPResult(taskId);
      } else if (status.status === 'error') {
        clearInterval(interval);
        document.getElementById('btnGenerateBP').disabled = false;
        document.getElementById('btnGenerateBP').textContent = '生成业务流程';
        toast('生成失败: ' + status.error_message, 'error');
      } else if (attempts > 600) {
        clearInterval(interval);
        toast('生成超时，请检查后台状态', 'error');
      }
    } catch (e) {
      if (attempts > 60) clearInterval(interval);
    }
  }, 2000);
}

async function loadBPResult(taskId) {
  try {
    const result = await apiGet(`/api/business/result/${taskId}`);
    currentBPResult = result;
    currentBPResult.task_id = taskId;

    document.getElementById('bpResultCard').style.display = 'block';
    // Hide intro card
    const introCard = document.getElementById('bpIntroCard');
    if (introCard) introCard.style.display = 'none';

    // Render markdown
    if (typeof marked !== 'undefined') {
      marked.setOptions({ breaks: true, gfm: true });
      document.getElementById('bpMarkdownContent').innerHTML = marked.parse(result.markdown || '');
    } else {
      document.getElementById('bpMarkdownContent').innerHTML = '<pre>' + escapeHtml(result.markdown || '') + '</pre>';
    }

    // Render Mermaid using render() API for reliable SVG output
    if (result.mermaid_syntax) {
      if (typeof mermaid !== 'undefined') {
        try {
          const { svg } = await mermaid.render('mermaidSvg', result.mermaid_syntax);
          document.getElementById('bpMermaidContent').innerHTML = svg;
        } catch(err) {
          document.getElementById('bpMermaidContent').innerHTML =
            `<div class="empty-state"><div class="empty-text">Mermaid渲染失败</div><div class="empty-hint">${escapeHtml(err.message || '')}</div></div>`;
        }
      } else {
        document.getElementById('bpMermaidContent').textContent = result.mermaid_syntax;
      }
    } else {
      document.getElementById('bpMermaidContent').innerHTML = '<div class="empty-state"><div class="empty-text">Mermaid流程图数据未生成</div></div>';
    }

    // Render ECharts - check visualization_paths.echarts
    if (result.visualization_paths && result.visualization_paths.echarts) {
      try {
        const echartsData = await apiGet(`/api/business/view/${taskId}/echarts`);
        if (echartsData.content) {
          // Extract script, inject HTML separately, then execute script
          const scriptMatch = echartsData.content.match(/<script>([\s\S]*?)<\/script>/i);
          const htmlContent = echartsData.content.replace(/<script>[\s\S]*?<\/script>/i, '');
          document.getElementById('bpEchartsContent').innerHTML = htmlContent;
          if (scriptMatch && scriptMatch[1]) {
            const s = document.createElement('script');
            s.textContent = scriptMatch[1];
            document.getElementById('bpEchartsContent').appendChild(s);
          }
        }
      } catch(e) { document.getElementById('bpEchartsContent').innerHTML = '<div class="empty-state"><div class="empty-text">ECharts加载失败</div></div>'; }
    } else {
      document.getElementById('bpEchartsContent').innerHTML = '<div class="empty-state"><div class="empty-text">ECharts树形图数据未生成</div><div class="empty-hint">请确保LLM返回了结构化JSON数据</div></div>';
    }

    // Render mindmap as HTML tree (reliable, no external deps)
    if (result.markdown) {
      const treeHtml = markdownToHtmlTree(result.markdown);
      document.getElementById('bpMindmapContent').innerHTML =
        '<div class="mindmap-tree" style="width:100%;height:75vh;overflow:auto;padding:16px;background:var(--bg-input);border-radius:var(--radius);">' +
        treeHtml + '</div>';
    }

    // Load summary
    if (result.visualization_paths && result.visualization_paths.summary) {
      try {
        const summaryData = await apiGet(`/api/business/view/${taskId}/summary`);
        if (summaryData.content) {
          document.getElementById('bpSummaryFrame').srcdoc = summaryData.content;
        }
      } catch(e) { document.getElementById('bpSummaryFrame').srcdoc = '<div class="empty-state"><div class="empty-text">汇总报告加载失败</div></div>'; }
    } else {
      document.getElementById('bpSummaryFrame').srcdoc = '<div class="empty-state"><div class="empty-text">汇总报告数据未生成</div></div>';
    }

    // Load DrawIO
    if (result.visualization_paths && result.visualization_paths.drawio) {
      try {
        const drawioData = await apiGet(`/api/business/view/${taskId}/drawio`);
        if (drawioData.content) {
          document.getElementById('bpDrawioContent').textContent = drawioData.content;
        }
      } catch(e) { document.getElementById('bpDrawioContent').textContent = 'DrawIO加载失败: ' + e.message; }
    } else {
      document.getElementById('bpDrawioContent').textContent = 'DrawIO文件未生成（请点击上方DrawIO按钮下载文件后使用draw.io打开）';
    }

    scrollToBPResult();
  } catch (e) {
    toast('加载结果失败: ' + e.message, 'error');
  }
}

// Convert markdown headings to collapsible HTML tree
function markdownToHtmlTree(md) {
  const lines = md.split('\n');
  const root = { level: 0, text: '业务流程', children: [] };
  const stack = [root];

  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.+)/);
    if (match) {
      const level = match[1].length;
      const text = match[2].trim().replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const node = { level, text, children: [] };

      while (stack.length > 1 && stack[stack.length - 1].level >= level) {
        stack.pop();
      }
      if (stack.length > 0) {
        stack[stack.length - 1].children.push(node);
      }
      stack.push(node);
    }
  }

  // Use the first real heading as root if possible
  let displayRoot = root;
  if (root.children.length === 1 && root.children[0].children.length > 0) {
    displayRoot = root.children[0];
  }

  function renderNode(node, depth) {
    const hasKids = node.children && node.children.length > 0;
    const id = 'mm_' + Math.random().toString(36).substr(2, 8);
    let html = '<div style="margin-left:' + (depth * 20) + 'px;">';
    if (hasKids) {
      html += '<div style="cursor:pointer;font-weight:600;color:var(--accent);padding:4px 0;" onclick="var c=document.getElementById(\'' + id + '\');c.style.display=c.style.display===\'none\'?\'\':\'none\';this.querySelector(\'.arrow\').textContent=c.style.display===\'none\'?\'▶\':\'▼\'">';
      html += '<span class="arrow" style="margin-right:4px;">▼</span>';
      html += '<span style="font-size:' + (16 - depth) + 'px;">' + node.text + '</span>';
      html += '</div>';
      html += '<div id="' + id + '" style="">';
      for (const child of node.children) {
        html += renderNode(child, depth + 1);
      }
      html += '</div>';
    } else {
      html += '<div style="padding:3px 0;color:var(--text-secondary);font-size:' + (13 - depth) + 'px;">';
      html += '<span style="margin-right:4px;">•</span>' + node.text;
      html += '</div>';
    }
    html += '</div>';
    return html;
  }

  return renderNode(displayRoot, 0);
}

function switchBPTab(name) {
  document.querySelectorAll('.bp-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.bp-content').forEach(c => c.classList.remove('active'));
  const tab = document.querySelector(`.bp-tab[data-bp="${name}"]`);
  if (tab) tab.classList.add('active');
  const content = document.getElementById(`bp-${name}`);
  if (content) content.classList.add('active');

  if (name === 'mermaid' && typeof mermaid !== 'undefined') {
    setTimeout(async () => {
      const el = document.getElementById('bpMermaidContent');
      if (el && el.textContent && el.textContent.trim() && !el.querySelector('svg')) {
        try {
          const { svg } = await mermaid.render('mermaidSvg2', el.textContent);
          el.innerHTML = svg;
        } catch(err) {}
      }
    }, 200);
  }
  if (name === 'echarts') {
    setTimeout(() => {
      const el = document.getElementById('bpEchartsContent');
      if (el) {
        const scripts = el.querySelectorAll('script');
        scripts.forEach(s => {
          try { eval(s.textContent); } catch(e) {}
        });
      }
    }, 200);
  }
  if (name === 'mindmap') {
    // No special handling needed
  }
}

function switchIntro(name) {
  document.querySelectorAll('.bp-tab[data-intro]').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.intro-content').forEach(c => c.style.display = 'none');
  const tab = document.querySelector(`.bp-tab[data-intro="${name}"]`);
  if (tab) tab.classList.add('active');
  const content = document.getElementById(`intro-${name}`);
  if (content) content.style.display = 'block';
}

function downloadMermaidPNG() {
  const svg = document.querySelector('#bpMermaidContent svg');
  if (!svg) { toast('请先切换到Mermaid流程图tab查看', 'info'); switchBPTab('mermaid'); return; }
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const svgData = new XMLSerializer().serializeToString(svg);
  const img = new Image();
  img.onload = function() {
    canvas.width = img.width * 2; canvas.height = img.height * 2;
    ctx.scale(2, 2); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    const a = document.createElement('a'); a.download = 'mermaid_flowchart.png'; a.href = canvas.toDataURL('image/png'); a.click();
    toast('PNG下载成功', 'success');
  };
  img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
}

function downloadEchartsPNG() {
  const chartDiv = document.querySelector('#bpEchartsContent canvas');
  if (!chartDiv) { toast('请先切换到ECharts树形图tab查看', 'info'); switchBPTab('echarts'); return; }
  const a = document.createElement('a'); a.download = 'echarts_tree.png'; a.href = chartDiv.toDataURL('image/png'); a.click();
  toast('PNG下载成功', 'success');
}

function downloadBPFile(type) {
  if (!currentBPResult || !currentBPResult.task_id) {
    toast('请先生成业务流程', 'error');
    return;
  }
  const taskId = currentBPResult.task_id;
  if (type === 'drawio') {
    // For DrawIO: open in draw.io online editor
    const drawioUrl = `https://app.diagrams.net/?splash=0&clibs=Uhttps://embedded.diagrams.net/app/vsdx_1.2.0.xml`;
    toast('DrawIO文件已下载，请将下载的文件拖入draw.io编辑器', 'info');
  }
  window.open(`/api/business/download/${taskId}/${type}`, '_blank');
}

function copyMarkdown() {
  if (!currentBPResult || !currentBPResult.markdown) {
    toast('暂无Markdown内容', 'error');
    return;
  }
  navigator.clipboard.writeText(currentBPResult.markdown).then(() => {
    toast('Markdown文档已复制到剪贴板', 'success');
  }).catch(() => {
    toast('复制失败，请手动选择文本复制', 'error');
  });
}

async function refreshBPTasks() {
  try {
    const tasks = await apiGet('/api/business/list');
    const list = document.getElementById('bpTaskList');
    if (tasks.length === 0) {
      list.innerHTML = '';
    } else {
      list.innerHTML = '<div style="font-weight:600;margin-bottom:8px;">历史生成任务</div>' + tasks.map(t => `
        <div class="file-item" style="cursor:pointer;" onclick="loadBPResult('${t.task_id}')">
          <span class="file-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></span>
          <span class="file-name">${t.task_id}</span>
          <span style="font-size:0.75rem;color:var(--text-muted);">${t.created_at}</span>
          <span class="file-status status-${t.status}">${t.status}</span>
        </div>
      `).join('');
    }

    // Populate ASR source dropdown
    const select = document.getElementById('bpAsrSource');
    select.innerHTML = '<option value="current">当前会话ASR结果</option>';
    let history = [];
    try { history = JSON.parse(localStorage.getItem('asr_history') || '[]'); } catch(e) {}
    history.forEach((h, i) => {
      select.innerHTML += `<option value="history_${i}">历史第${i+1}轮 (${h.time} · ${h.completedCount}个文件)</option>`;
    });
  } catch(e) {}
}

// ==================== Tab 5: Prompts ====================
async function loadPrompts() {
  try {
    const prompts = await apiGet('/api/prompts/list');
    const list = document.getElementById('promptList');
    list.innerHTML = prompts.map(p => `
      <div class="prompt-list-item" onclick="selectPrompt('${p.name}')" data-prompt="${p.name}">
        <div class="prompt-name">${escapeHtml(p.display_name)}</div>
        <div class="prompt-desc">${escapeHtml(p.description)}</div>
        <span class="prompt-status ${p.is_customized ? 'status-customized' : 'status-default'}">${p.is_customized ? '已自定义' : '默认'}</span>
      </div>
    `).join('');
  } catch (e) {
    toast('加载提示词失败: ' + e.message, 'error');
  }
}

async function selectPrompt(name) {
  // Highlight
  document.querySelectorAll('.prompt-list-item').forEach(el => el.classList.remove('active'));
  const active = document.querySelector(`.prompt-list-item[data-prompt="${name}"]`);
  if (active) active.classList.add('active');

  try {
    const prompt = await apiGet(`/api/prompts/${name}`);
    currentPrompt = prompt;
    document.getElementById('promptEditorLabel').textContent = prompt.display_name + (prompt.is_customized ? ' (已自定义)' : ' (默认)');
    document.getElementById('promptEditor').value = prompt.content;
    document.getElementById('promptEditor').disabled = false;
    document.getElementById('btnSavePrompt').disabled = false;
    document.getElementById('btnResetPrompt').disabled = false;
    document.getElementById('promptStatus').innerHTML = prompt.is_customized
      ? '<span class="prompt-status status-customized">已自定义</span>'
      : '<span class="prompt-status status-default">默认</span>';
  } catch (e) {
    toast('加载提示词失败: ' + e.message, 'error');
  }
}

async function savePrompt() {
  if (!currentPrompt) return;
  const content = document.getElementById('promptEditor').value;
  try {
    const r = await apiPut(`/api/prompts/${currentPrompt.name}`, { content: content });
    toast(r.message, 'success');
    loadPrompts();
    if (currentPrompt) selectPrompt(currentPrompt.name);
  } catch (e) {
    toast('保存失败: ' + e.message, 'error');
  }
}

async function resetPrompt() {
  if (!currentPrompt) return;
  if (!confirm(`确定要重置"${currentPrompt.display_name}"为默认值吗？`)) return;
  try {
    const r = await apiPost(`/api/prompts/${currentPrompt.name}/reset`);
    toast(r.message, 'success');
    document.getElementById('promptEditor').value = r.data.content;
    loadPrompts();
    selectPrompt(currentPrompt.name);
  } catch (e) {
    toast('重置失败: ' + e.message, 'error');
  }
}

// ==================== Utilities ====================
function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
}
</script>
</body>
</html>'''


# ==================== Main Entry ====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting AI Business Process Analysis Service on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")