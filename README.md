# Intelligent QA System
### Adaptive Multi-Model Question Answering with RoBERTa Complexity Routing

---

## Overview

This project builds an **adaptive Question Answering system** that intelligently routes each user query to the most appropriate LLM based on question complexity and domain — avoiding the unnecessary cost of running a large model for every query.

```
User Question
     │
     ▼
RoBERTa Complexity Classifier (97.17% accuracy)
     │
     ├── Simple ──────────────► Qwen2.5:0.5B                    │ lightweight general model
     │
     └── Complex ─► Domain Detection
                         │
                         ├── Code ──► Qwen2.5-Coder:3B          │ Programming specialist
                         ├── Math ──► Qwen2.5:3B                │ Fast mathematical reasoning
                         └── GK ───► Qwen2.5:3B              │ General knowledge
```

A **Redis cache layer** (with in-memory fallback) wraps every query for instant repeat-response delivery.

---

## System Components

| Component | File(s) | Description |
|---|---|---|
| Complexity Classifier | `src/classifier.py` | RoBERTa-base fine-tuned on 60k questions (97.17% F1) |
| Domain Detector | `src/qa_engine.py` | Keyword-based code/math/GK routing |
| LLM Engine | `src/qa_engine.py` | Ollama HTTP API with timeout & error handling |
| Cache Layer | `src/cache.py` | Redis + in-memory LRU/TTL fallback |
| Main Router | `src/router.py` | Full pipeline orchestrator |
| REST API | `api/main.py` | FastAPI with 6 endpoints |
| Web Dashboard | `web/` | Dark-mode glassmorphism UI |
| Evaluation Suite | `evaluation/benchmark_system.py` | Metrics, latency, resource charts |

---

## Model Routing Table

| Complexity | Domain | Model | Size |
|---|---|---|---|
| Simple | Any | `qwen2.5:1.5b` | Lightweight general model |
| Complex | Code | `qwen2.5-coder:3b` | 1.9 GB |
| Complex | Math | `qwen2.5:3b` | 1.9 GB |
| Complex | General/GK | `qwen2.5:3b` | 1.9 GB |

---

## Setup

### Project Requirements

- Windows, macOS, or Linux
- Python 3.11 or newer
- At least 8 GB RAM for inference; more memory is recommended when running the larger Ollama models
- [Ollama](https://ollama.ai) installed and running
- The trained classifier files in `classification/roberta_final_model/`

The Python packages used by the API, classifier, training scripts, and evaluation suite are listed in
[`requirement.txt`](requirement.txt).

For detailed installation, architecture, operation, troubleshooting, and extension instructions, see the
[`IMPLEMENTATION_MANUAL.md`](IMPLEMENTATION_MANUAL.md).

### Prerequisites

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

Pull the required Ollama models:

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5:3b
```

Verify that Ollama is available before starting the API:

```bash
ollama list
```

Redis is optional. If a Redis server is not available at `localhost:6379`, the application automatically uses its thread-safe in-memory cache.

---

## Usage

### Option 1 — Interactive CLI

```bash
cd "c:\Users\aakif\final year\src"
python router.py
```

Commands:
- Type any question → get routed answer
- `stats` → show cache statistics
- `clear` → flush cache
- `exit` → quit

### Option 2 — FastAPI Web Server + Dashboard

```bash
cd "c:\Users\aakif\final year"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### Option 3 — Evaluation Benchmark

```bash
cd "c:\Users\aakif\final year"
python evaluation/benchmark_system.py
```

Results saved to `evaluation/results/`.

### Windows Path Note

The current model configuration uses the project path
`C:\Users\aakif\final year` in the classifier and related scripts. If the project is moved, update those configuration paths before running inference or evaluation.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ask` | Full QA: classify → route → answer |
| `POST` | `/api/classify` | Complexity classification only |
| `GET` | `/api/health` | System health (RoBERTa, Ollama, cache, CPU/RAM) |
| `GET` | `/api/metrics` | Aggregated query statistics |
| `GET` | `/api/history` | Recent query history |
| `DELETE` | `/api/cache` | Flush cache |

### Example: POST /api/ask

```json
{
  "question": "What is the capital of Japan?",
  "use_cache": true
}
```

**Response:**
```json
{
  "question":     "What is the capital of Japan?",
  "complexity":   "Simple",
  "confidence":   0.9923,
  "domain":       "simple",
  "model":        "qwen2.5:1.5b",
  "answer":       "The capital of Japan is Tokyo.",
  "cache_hit":    false,
  "classify_ms":  14.2,
  "llm_ms":       380.1,
  "total_ms":     394.3,
  "is_error":     false
}
```

---

## Classifier Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 97.17% |
| Precision | 97.71% |
| Recall    | 96.60% |
| F1 Score  | 97.15% |

Trained on 60,000 questions (48k train / 6k val / 6k test), balanced Simple/Complex.

---

## Project Structure

```
final year/
├── classification/
│   ├── roberta_final_model/      ← Trained RoBERTa weights (497 MB)
│   ├── roberta_final.py          ← Training script
│   └── roberta_inference.py      ← Standalone inference test
├── dataset/
│   └── classification_dataset.csv
├── src/
│   ├── classifier.py             ← RoBERTa wrapper
│   ├── qa_engine.py              ← LLM engine + domain routing
│   ├── cache.py                  ← Redis/memory cache
│   └── router.py                 ← Main pipeline
├── api/
│   ├── main.py                   ← FastAPI application
│   └── schemas.py                ← Pydantic schemas
├── web/
│   ├── index.html                ← Dashboard HTML
│   ├── style.css                 ← Dark-mode glassmorphism CSS
│   └── app.js                    ← Dashboard JavaScript
├── evaluation/
│   ├── benchmark_system.py       ← Full benchmark suite
│   └── results/                  ← Generated plots & report
├── requirement.txt               ← Python dependencies
└── README.md
```

---

## Phase 2 — Planned (Cloud + Edge Deployment)

Phase 2 will extend this system into a cloud-deployed, API-key-gated service with an **Edge Intelligence Hub**:

```
Users (API Keys)
      │
      ▼
Edge Gateway
  ├── API key validation & rate limiting
  ├── Token usage metering & billing
  ├── Redis cache (instant repeat responses)
  └── Live monitoring dashboard
      │
      ▼
Cloud GPU Server — LLM Backend
  └── Same RoBERTa + multi-LLM routing as Phase 1
```

> Phase 2 development begins after Phase 1 evaluation is complete.

---

*Final Year Project — Adaptive Multi-Model QA System with RoBERTa Complexity Routing*
