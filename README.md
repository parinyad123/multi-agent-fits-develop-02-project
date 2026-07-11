# FITS Analysis Multi-Agent System

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** · [ภาษาไทย](README.th.md)

> A multi-agent orchestration backend for automated analysis and scientific interpretation of X-ray astronomy time-series data (FITS files).

Developed for the **Department of Physics, Suranaree University of Technology (SUT)**, in collaboration with the **National Astronomical Research Institute of Thailand (NARIT)**, to support timing analysis of XMM-Newton observations (e.g. the AGN *IRAS 13224-3809*).

---

## Overview

Researchers upload a FITS light-curve file and ask a question in natural language. The system classifies the request, runs the appropriate numerical analyses (statistics, PSD, model fitting), interprets the results in astrophysical terms with an LLM, and returns a synthesized answer with generated plots — while persisting the full conversation and analysis history.

The system is organized as **specialized agents coordinated by a dynamic-routing orchestrator** (a supervisor/router pattern), rather than a single monolithic model.

```
User query + FITS file
        │
        ▼
┌───────────────────────┐
│   Orchestrator        │  async queue · workers · concurrency limits
│  (dynamic routing)    │
└───────────┬───────────┘
            ▼
   ┌────────────────────┐
   │ Classification &   │  intent + parameter extraction (LLM)
   │ Parameter Agent    │
   └─────────┬──────────┘
             │  routing strategy
   ┌─────────┼───────────────────────┐
   ▼         ▼                        ▼
analysis   interpretation          mixed
   │         │                    (analysis → interpretation)
   ▼         ▼                        ▼
┌────────────────┐   ┌────────────────────┐
│ Analysis Agent │   │ Interpretation     │  astrophysical reasoning (LLM)
│ (numerical)    │   │ Agent              │
└────────┬───────┘   └─────────┬──────────┘
         └──────────┬──────────┘
                    ▼
            ┌───────────────┐
            │ Rewrite Agent │  final user-facing synthesis (LLM)
            └───────┬───────┘
                    ▼
          Response + plots + history (PostgreSQL)
```

---

## Key Features

- **Dynamic intent-based routing** — the classifier selects one of three workflows:
  - `analysis` — pure numerical processing
  - `interpretation` — conceptual/astrophysics Q&A
  - `mixed` — analysis followed by physical interpretation
- **FITS time-series analysis** — descriptive statistics, Power Spectral Density (PSD), and **power-law / bending-power-law** model fitting (SciPy), with automatic plot generation.
- **Parameter inheritance** — conversational follow-ups ("*again with A0=5*") reuse and override parameters from previous turns, scoped per file or per session.
- **Async orchestration** — a request queue with background workers and semaphore-based concurrency limits per LLM tier.
- **Dual API (v1 & v2)** — v2 adds pagination, GZIP compression, and **Server-Sent Events (SSE)** for real-time workflow streaming.
- **Full persistence** — PostgreSQL schema (6 tables) covering users, files, sessions, analysis history, and generated plots.
- **Prompt engineering for reliability** — strict structured-output prompts and expertise-adaptive responses (beginner → expert) to reduce hallucination.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API / Web | FastAPI, Uvicorn |
| Database | PostgreSQL (async via `asyncpg` + SQLAlchemy 2.0) |
| LLM orchestration | LangChain, Groq (`langchain-groq`) |
| Numerical | NumPy, SciPy, Astropy/FITS tooling, Matplotlib |
| Packaging | `uv` (PEP 621 `pyproject.toml`) |

### LLM models

The agents run on **Groq-hosted models**, configurable per agent in [`app/core/config.py`](app/core/config.py):

| Agent | Default model |
|-------|--------------|
| Classification & Parameter | `llama-3.3-70b-versatile` |
| Interpretation | `openai/gpt-oss-120b` |
| Rewrite | `llama-3.3-70b-versatile` |

> **History:** the system originated as a hybrid architecture using a locally-hosted **AstroSage-Llama-3.1-8B** model with **OpenAI GPT** for interpretation; the interpretation layer is retained (`app/services/astrosage/`) and now migrated to Groq-served models. OpenAI keys are legacy and no longer required.

---

## Project Structure

```
app/
├── main.py                     # FastAPI app, lifespan, router registration
├── orchestration/
│   └── orchestrator.py         # DynamicWorkflowOrchestrator (queue, routing, persistence)
├── agents/
│   ├── classification_parameter/   # intent + parameter extraction agent
│   ├── analysis/                   # numerical analysis agent + capabilities
│   └── rewrite/                    # final response synthesis + session titles
├── services/
│   ├── astrosage/                  # LLM interpretation client & prompt building
│   └── conversation_service.py     # sessions, messages, workflow persistence
├── tools/                      # statistics, psd, fitting, plotting, FITS loader
├── api/
│   ├── v1/                      # original API (full responses)
│   └── v2/                      # optimized API (pagination, SSE, GZIP)
├── db/                         # SQLAlchemy models + async engine
└── core/                       # config, constants
scripts/
└── init_db.sql                 # PostgreSQL extensions + schema bootstrap
```

---

## Getting Started

### Prerequisites

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (or pip)
- Docker & Docker Compose (for PostgreSQL)
- A **Groq API key** — https://console.groq.com

### 1. Clone & install dependencies

```bash
git clone <repository-url>
cd multi-agent-fits-develop-02-project

uv sync            # or: pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```dotenv
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Database — this project uses host port 5433 (5432 was taken by another service)
POSTGRES_PORT=5433
DATABASE_URL=postgresql+asyncpg://fits_user:fits_password@localhost:5433/fits_analysis_db

# Auth
SECRET_KEY=change_me_to_a_random_secret

# Storage (optional — defaults shown)
FITSFILES_DIR=storage/fitsfiles
PLOTS_DIR=storage/plots
```

### 3. Start the database

```bash
docker compose up -d postgres
```

`scripts/init_db.sql` runs automatically on first launch to create the required extensions (`uuid-ossp`, `pg_trgm`) and schema. Optional pgAdmin UI is available at `http://localhost:5050`.

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8003
```

Application tables are created automatically on startup.

---

## API

| Service | URL |
|---------|-----|
| Swagger UI | http://localhost:8003/docs |
| ReDoc | http://localhost:8003/redoc |
| Health check | http://localhost:8003/health |
| API v1 | `http://localhost:8003/api/v1` |
| API v2 (recommended) | `http://localhost:8003/api/v2` |

### Typical workflow

1. `POST /api/v2/auth/...` — authenticate
2. `POST /api/v2/files` — upload a FITS file
3. `POST /api/v2/analyze` — submit a query → returns a `task_id`
4. `GET /api/v2/analyze/{task_id}/stream` — stream progress via SSE, **or** poll `GET /api/v2/analyze/{task_id}/status`
5. `GET /api/v2/analyze/{task_id}/result` — fetch the final answer + plots

### Example: submit an analysis

**Request** — `POST /api/v2/analyze` (Bearer token required)

```json
{
  "query": "Fit the PSD with a bending power law using 5000 bins and a break frequency of 0.02, then explain how the spectral slopes relate to accretion disk turbulence.",
  "fits_file_id": "eb536ea0-cce7-479f-9153-7b3bc189d71d",
  "session_id": null,
  "user_expertise": "intermediate"
}
```

**Response** — `202` (queued)

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "session_id": "3a7e5c9d-2b4f-4e1a-9c8d-0f1e2d3c4b5a",
  "status": "queued"
}
```

### Example: poll status (lightweight)

**Request** — `GET /api/v2/analyze/{task_id}/status`

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "status": "in_progress",
  "progress": "50%",
  "current_step": "analysis",
  "error": null
}
```

### Example: real-time updates (SSE)

**Request** — `GET /api/v2/analyze/{task_id}/stream`

```
data: {"task_id": "9f1c2b7a...", "status": "in_progress", "progress": "30%", "current_step": "classification"}

data: {"task_id": "9f1c2b7a...", "status": "in_progress", "progress": "70%", "current_step": "astrosage"}

data: {"task_id": "9f1c2b7a...", "status": "completed", "progress": "100%", "current_step": "rewrite"}
```

### Example: fetch the result

**Request** — `GET /api/v2/analyze/{task_id}/result`

```json
{
  "task_id": "9f1c2b7a-1d3e-4a2f-b8c0-5e6d7f8a9b01",
  "status": "completed",
  "content": "The bending power-law fit yields a low-frequency slope of ~1.0 flattening to ~2.3 above the break at f_b ≈ 0.02 Hz. This steepening is consistent with ...",
  "plots": [
    {
      "plot_id": "b2c3d4e5-...",
      "plot_type": "bending_power_law",
      "plot_url": "/api/v1/plots/bending_power_law/9f1c2b7a_bpl.png",
      "title": "Bending Power Law Fit",
      "created_at": "2026-07-11T09:15:42.123456"
    }
  ],
  "completed_at": "2026-07-11T09:15:43.987654"
}
```

---

## Analysis Types

| Type | Description |
|------|-------------|
| `statistics` | Descriptive statistics (mean, median, std, min, max, percentiles) |
| `psd` | Power Spectral Density of the light curve |
| `power_law` | Simple power-law fit: `PSD(f) = A/f^b + n` |
| `bending_power_law` | Bending power-law fit: `PSD(f) = A / [f (1 + (f/f_b)^(sh-1))] + n` |
| `metadata` | FITS header / HDU metadata extraction (always included) |

---

## Testing

```bash
uv run pytest
```

Agent-level test cases live alongside the agents (e.g. `app/agents/rewrite/tests/`, and the `__main__` test harness in the classification agent).

---

## Acknowledgements

- **Department of Physics, Suranaree University of Technology** — project host
- **National Astronomical Research Institute of Thailand (NARIT)** — domain expertise and XMM-Newton data
- Built with FastAPI, LangChain, Groq, SciPy, and PostgreSQL.

---

## License

Released under the [MIT License](LICENSE).

---

## Author

**Parinya Duangklang** — AI Engineer / Research Assistant
[GitHub](https://github.com/parinyad123) · parinya.dg@gmail.com
