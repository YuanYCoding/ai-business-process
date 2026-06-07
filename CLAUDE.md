# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Business Process Analysis Service — a FastAPI backend with an integrated SPA frontend that transforms audio recordings (phone conversations) into structured business process documentation and visualizations. The pipeline: **Audio → ASR → LLM Analysis → Visualizations**.

## Run Commands

```bash
# Start the server (default port 8002)
python main.py

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

No build step, no tests, no linters configured. There is no `requirements.txt` or `pyproject.toml` — dependencies are installed manually via pip.

## Architecture

### Pipeline

```
Audio files (upload/URL download)
  → ASR via SenseVoiceSmall (funasr) with VAD + speaker diarization
    → LLM analysis with structured prompts (DeepSeek/OpenAI/Ollama/SiliconFlow)
      → Output: Markdown document + JSON structured data + Mermaid/ DrawIO/ECharts/HTML visualizations
```

### Directory Layout

```
main.py                  # FastAPI app + ConnectionManager (WebSocket) + entire SPA frontend inlined as HTML_PAGE
models/schemas.py        # All Pydantic models: AudioFile, ASRConfig, LLMConfig, BusinessProcessResult, PromptTemplate, etc.
services/
  audio_service.py       # Upload, multi-threaded URL download (ThreadPoolExecutor), file CRUD
  asr_service.py         # SenseVoiceSmall model management (singleton), batch processing, result persistence
  llm_service.py         # OpenAI-compatible client abstraction, provider presets, streaming, JSON extraction
  business_service.py    # Orchestration: transcript assembly → LLM prompt chain → business flow doc generation
  prompt_service.py      # File-based prompt templates (defaults in prompts/*.txt, user overrides in data/prompts/)
  visualization_service.py # Mermaid, DrawIO XML, ECharts HTML, summary HTML dashboard generation
routers/
  audio.py               # /api/audio/* — upload, URL download, list, delete
  asr.py                 # /api/asr/* — config, start/cancel processing, results
  llm.py                 # /api/llm/* — providers, config, connection test
  business.py            # /api/business/* — generate, status, results, download/view files
  prompts.py             # /api/prompts/* — list, get, save, reset
prompts/                 # Default prompt templates (business_analysis, business_flow, script_generation, visualization)
data/
  audio/                 # Uploaded/downloaded audio files
  asr_output/            # ASR text results
  business_output/       # Generated markdown, JSON, HTML, DrawIO files
  prompts/               # User-customized prompt overrides
```

### Key Design Decisions

- **In-memory state**: Task status, ASR processing state, and results are stored in module-level dicts with `threading.Lock()` — no database. Data is persisted to disk as files (ASR output `.txt`, business output `.md`/`.json`/`.html`).
- **ASR model**: Uses `funasr` AutoModel with `iic/SenseVoiceSmall` + `fsmn-vad` (VAD) + `cam++` (speaker diarization). Model is a singleton, loaded lazily on first use, pre-loaded in background on startup.
- **LLM abstraction**: All providers accessed via OpenAI-compatible API. Provider presets are hardcoded in `llm_service.py`. API keys read from environment variables (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, etc.).
- **SPA frontend**: The entire HTML/CSS/JS frontend is a single Python string literal `HTML_PAGE` in `main.py` (~1300 lines). Uses Mermaid.js, ECharts, and marked.js loaded from CDN. No separate frontend build step.
- **WebSocket real-time updates**: `ConnectionManager` in `main.py` broadcasts ASR progress and download progress to connected clients.

### API Endpoints Summary

| Prefix | Purpose |
|--------|---------|
| `/api/audio/*` | Audio file upload, URL download, list, delete |
| `/api/asr/*` | ASR config, start/cancel, results, status |
| `/api/llm/*` | Provider list, config, connection test |
| `/api/business/*` | Generate business process, task status, results, file download/view |
| `/api/prompts/*` | Prompt template CRUD |
| `/ws/asr/progress` | WebSocket for ASR progress |
| `/ws/audio/download-progress` | WebSocket for download progress |
| `/` | SPA frontend |

### Environment Variables

Configured in `.env` (loaded via `python-dotenv`):
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` — LLM configuration
- `PORT` — server port (default: 8002)
- `HOST` — bind address (default: 0.0.0.0)

### Data Flow for Business Process Generation

1. `business_service.generate_business_process()` is the orchestrator
2. Combines all ASR transcripts into one text blob
3. Calls LLM with `business_analysis` prompt → context analysis
4. Calls LLM with `business_flow` prompt + context → structured markdown
5. Calls LLM with `visualization` prompt + markdown → structured JSON
6. `visualization_service.generate_all_visualizations()` produces Mermaid, DrawIO, ECharts, and summary HTML
7. Results saved to `data/business_output/{task_id}_*`