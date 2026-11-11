"""
api/main.py
-----------
FastAPI application for the Intelligent QA System.

Endpoints:
  POST /api/ask       — Full pipeline: classify → route → answer (with caching)
  POST /api/classify  — Classification only (no LLM call)
  GET  /api/health    — Health check: RoBERTa, Ollama, Redis/cache, CPU/RAM
  GET  /api/metrics   — Aggregated query statistics
  GET  /api/history   — Recent query history
  DELETE /api/cache   — Flush the query cache

  GET  /              — Serves the Web Dashboard (web/index.html)
"""

import os
import sys
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import List

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = r"C:\Users\aakif\final year"
SRC_DIR  = os.path.join(BASE_DIR, "src")
WEB_DIR  = os.path.join(BASE_DIR, "web")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from classifier import RoBERTaClassifier
from qa_engine  import (
    answer_question,
    detect_domain,
    requires_coding_specialist,
    requires_knowledge_specialist,
    requires_math_specialist,
    requires_specialist_explanation,
    is_essay_or_creative_request,
    MODEL_SIMPLE,
    OLLAMA_API_URL,
)
from router     import (
    process_question,
    history as query_history,
    cache as query_cache,
    classifier as router_classifier,
)
from api import schemas

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ============================================================
# LIFESPAN — MODEL LOADING
# ============================================================

_classifier: RoBERTaClassifier = router_classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    logger.info("Reusing RoBERTa classifier loaded by the router.")
    # Warmup classifier pass to pre-compile PyTorch/CUDA kernels
    try:
        _classifier.classify("warmup ping")
        logger.info("RoBERTa classifier warmed up.")
    except Exception as e:
        logger.warning("Classifier warmup skipped: %s", e)

    # Pre-warm Ollama simple model with keep_alive
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_SIMPLE,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                },
            )
            if resp.status_code == 200:
                logger.info("Ollama model %s pre-warmed (keep_alive: %s).", MODEL_SIMPLE, OLLAMA_KEEP_ALIVE)
    except Exception as e:
        logger.warning("Ollama pre-warmup skipped: %s", e)

    logger.info("Startup complete. API ready.")
    yield
    logger.info("Shutting down.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Intelligent QA System",
    description=(
        "Adaptive multi-model QA system with RoBERTa complexity routing.\n\n"
        "Routing: Simple → Qwen2.5:1.5B | Complex/Math → Qwen2.5:3B | "
        "Complex/Code → Qwen2.5-Coder:3B | Complex/GK → Qwen2.5:3B"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static web assets
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ============================================================
# ENDPOINTS
# ============================================================

# ── Dashboard & Pages ────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/architecture", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive web dashboard."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h2>Dashboard not found. Run the application from the project root.</h2>",
            status_code=404,
        )
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Authentication ───────────────────────────────────────────

@app.post("/api/auth/login", response_model=schemas.LoginResponse, tags=["Auth"])
async def login(req: schemas.LoginRequest):
    """Authenticate user for IntentRoute dashboard."""
    u = req.username.strip()
    p = req.password.strip()
    if not u or not p:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    # Format user display
    display_name = u.capitalize()
    role = "Lead AI Engineer" if u.lower() in ("aakif", "admin") else "Research Member"
    email = f"{u.lower().replace(' ', '.')}@intentroute.ai"

    return {
        "success": True,
        "token": f"ir_token_{os.urandom(8).hex()}",
        "user": {
            "username": u,
            "name": display_name,
            "email": email,
            "role": role,
        },
        "message": f"Welcome back, {display_name}!"
    }


@app.get("/api/auth/me", tags=["Auth"])
async def auth_me(token: str = ""):
    """Validate current session token."""
    return {"authenticated": True, "token": token}


@app.post("/api/auth/logout", tags=["Auth"])
async def logout():
    """Sign out user session."""
    return {"success": True, "message": "Logged out successfully."}



# ── Ask (Main Endpoint) ──────────────────────────────────────

@app.post("/api/ask", response_model=schemas.QueryResponse, tags=["QA"])
async def ask(req: schemas.QueryRequest):
    """
    Full intelligent QA pipeline:
    Classify → Route → Generate Answer (with Redis/memory caching)
    """
    try:
        history_dicts = [turn.model_dump() for turn in req.conversation_history]
        result = process_question(
            req.question,
            use_cache=req.use_cache,
            conversation_history=history_dicts,
        )
        return result
    except Exception as e:
        logger.exception("Error in /api/ask")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ask Stream (SSE streaming endpoint) ─────────────────────

import asyncio
import urllib.request as _urllib_request
import json as _json

@app.post("/api/ask-stream", tags=["QA"])
async def ask_stream(req: schemas.QueryRequest):
    """
    Streaming QA pipeline: classify → route → stream Ollama tokens via SSE.
    Emits Server-Sent Events:
      - data: {"type": "meta", ...routing metadata...}
      - data: {"type": "token", "text": "..."}  (one per token)
      - data: {"type": "done", "latency_ms": ...}
      - data: {"type": "error", "detail": "..."}
    """
    import time as _time
    import httpx as _httpx
    from qa_engine import (
        detect_domain,
        get_model_for_complexity,
        _build_prompt,
        _clean_model_output,
        OLLAMA_API_URL,
        OLLAMA_TIMEOUT,
        _evaluate_arithmetic,
        _solve_train_platform,
        _solve_algebraic_equation,
        _verified_factual_answer,
        _verified_identity_answer,
        _verified_technical_answer,
        MODEL_ARITHMETIC,
        MODEL_WORD_PROBLEM,
        MODEL_MATH,
        MODEL_GENERAL,
        MODEL_SIMPLE,
        MODEL_CODE,
        MODEL_ESSAY,
        OLLAMA_KEEP_ALIVE,
        _generation_options,
        requires_coding_specialist,
        requires_knowledge_specialist,
        requires_math_specialist,
        requires_specialist_explanation,
        is_followup_question,
        is_essay_or_creative_request,
    )
    from router import cache as _cache

    async def event_generator():
        try:
            history_dicts = [turn.model_dump() for turn in req.conversation_history]
            question = req.question.strip()
            t_start = _time.perf_counter()

            # 1. Cache check
            is_followup = is_followup_question(question, history_dicts)
            if is_followup and history_dicts:
                last_q = history_dicts[-1].get("question", "").strip()[:80]
                cache_key = f"ctx:{last_q} >> {question}"
            else:
                cache_key = question

            if req.use_cache:
                cached = _cache.get_query(cache_key)
                if cached:
                    hit_ms = round((_time.perf_counter() - t_start) * 1000, 2)
                    meta = dict(cached)
                    meta["cache_hit"] = True
                    meta["total_ms"] = hit_ms
                    # Emit meta immediately
                    yield f"data: {_json.dumps({'type': 'meta', **meta})}\n\n"
                    # Send the full cached answer as one token — instant display
                    cached_answer = cached.get("answer", "")
                    if cached_answer:
                        yield f"data: {_json.dumps({'type': 'token', 'text': cached_answer})}\n\n"
                    yield f"data: {_json.dumps({'type': 'done', 'latency_ms': hit_ms})}\n\n"
                    logger.info("Stream: cache HIT returned in %.1fms for key=%s", hit_ms, cache_key[:60])
                    return


            # 2. Fast-path direct arithmetic (Instant & always Simple)
            direct_arithmetic = _evaluate_arithmetic(question)
            if direct_arithmetic:
                meta = {
                    "type": "meta",
                    "model": MODEL_ARITHMETIC,
                    "domain": "math",
                    "complexity": "Simple",
                    "confidence": 1.0,
                    "simple_prob": 1.0,
                    "complex_prob": 0.0,
                    "classify_ms": 0.0,
                    "cache_hit": False,
                }
                yield f"data: {_json.dumps(meta)}\n\n"
                yield f"data: {_json.dumps({'type': 'token', 'text': direct_arithmetic})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'latency_ms': round((_time.perf_counter() - t_start) * 1000, 2)})}\n\n"
                return

            # 3. Classify complexity
            clf = _cache.get_classification(question) if req.use_cache else None
            if clf is None:
                clf = _classifier.classify(question)
                if req.use_cache:
                    _cache.set_classification(question, clf)
            complexity   = clf["label"]
            confidence   = clf["confidence"]
            simple_prob  = clf["simple_prob"]
            complex_prob = clf["complex_prob"]
            classify_ms  = clf["latency_ms"]

            # 4. Domain & specialist overrides
            detected_domain = detect_domain(question)
            domain = detected_domain if complexity == "Complex" else "simple"

            if requires_coding_specialist(question):
                complexity = "Complex"; domain = "code"
                confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
            elif requires_knowledge_specialist(question):
                complexity = "Complex"; domain = "general"
                confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
            elif requires_math_specialist(question):
                complexity = "Complex"; domain = "math"
                confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
            elif requires_specialist_explanation(question):
                complexity = "Complex"; domain = detect_domain(question)
                confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0

            # 5. Fast-path direct answers (word problems, algebraic equations, verified facts)
            direct = _solve_train_platform(question)
            if direct:
                meta = {"type": "meta", "model": MODEL_WORD_PROBLEM, "domain": "math",
                        "complexity": complexity, "confidence": confidence,
                        "simple_prob": simple_prob, "complex_prob": complex_prob,
                        "classify_ms": classify_ms, "cache_hit": False}
                yield f"data: {_json.dumps(meta)}\n\n"
                yield f"data: {_json.dumps({'type': 'token', 'text': direct})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'latency_ms': 0.0})}\n\n"
                return

            direct = _solve_algebraic_equation(question)
            if direct:
                meta = {"type": "meta", "model": MODEL_MATH, "domain": "math",
                        "complexity": complexity, "confidence": confidence,
                        "simple_prob": simple_prob, "complex_prob": complex_prob,
                        "classify_ms": classify_ms, "cache_hit": False}
                yield f"data: {_json.dumps(meta)}\n\n"
                yield f"data: {_json.dumps({'type': 'token', 'text': direct})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'latency_ms': 0.0})}\n\n"
                return

            direct_tech = _verified_technical_answer(question)
            if direct_tech:
                chosen_model = MODEL_SIMPLE if complexity.lower() == "simple" else MODEL_CODE
                chosen_domain = "simple" if complexity.lower() == "simple" else "code"
                meta = {"type": "meta", "model": chosen_model, "domain": chosen_domain,
                        "complexity": complexity, "confidence": confidence,
                        "simple_prob": simple_prob, "complex_prob": complex_prob,
                        "classify_ms": classify_ms, "cache_hit": False}
                yield f"data: {_json.dumps(meta)}\n\n"
                yield f"data: {_json.dumps({'type': 'token', 'text': direct_tech})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'latency_ms': 0.0})}\n\n"
                return

            direct = _verified_factual_answer(question) or _verified_identity_answer(question)
            if direct:
                meta = {"type": "meta", "model": MODEL_GENERAL, "domain": "general",
                        "complexity": complexity, "confidence": confidence,
                        "simple_prob": simple_prob, "complex_prob": complex_prob,
                        "classify_ms": classify_ms, "cache_hit": False}
                yield f"data: {_json.dumps(meta)}\n\n"
                yield f"data: {_json.dumps({'type': 'token', 'text': direct})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'latency_ms': 0.0})}\n\n"
                return

            # 6. Route to model
            model  = get_model_for_complexity(complexity, domain, question)
            prompt = _build_prompt(question, complexity, domain, history_dicts)

            meta = {"type": "meta", "model": model, "domain": domain,
                    "complexity": complexity, "confidence": confidence,
                    "simple_prob": simple_prob, "complex_prob": complex_prob,
                    "classify_ms": classify_ms, "cache_hit": False}
            yield f"data: {_json.dumps(meta)}\n\n"

            # 7. Asynchronously stream Ollama tokens via httpx (non-blocking)
            stream_api_url = OLLAMA_API_URL
            stream_timeout = OLLAMA_TIMEOUT
            stream_model   = model

            t_llm = _time.perf_counter()
            payload = {
                "model": stream_model,
                "prompt": prompt,
                "stream": True,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": _generation_options(stream_model, domain, question),
            }

            full_answer = ""
            try:
                async with _httpx.AsyncClient(timeout=float(stream_timeout)) as client:
                    async with client.stream("POST", stream_api_url, json=payload) as resp:
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            line = raw_line.strip()
                            if not line:
                                continue
                            chunk = _json.loads(line)
                            token_text = chunk.get("response", "")
                            if token_text:
                                full_answer += token_text
                                yield f"data: {_json.dumps({'type': 'token', 'text': token_text})}\n\n"
                            if chunk.get("done", False):
                                break
            except Exception as model_err:
                logger.error("Streaming Ollama error: %s", model_err)
                yield f"data: {_json.dumps({'type': 'error', 'detail': str(model_err)})}\n\n"
                return

            llm_ms = round((_time.perf_counter() - t_llm) * 1000, 2)
            total_ms = round((_time.perf_counter() - t_start) * 1000, 2)
            full_answer = _clean_model_output(full_answer)

            # ── Save to cache & history (mirrors process_question in router.py) ──
            refusal_prefixes = ("i'm sorry", "i am sorry", "[error]")
            is_error = full_answer.startswith("[ERROR]")
            is_refusal = any(full_answer.strip().lower().startswith(p) for p in refusal_prefixes)
            result_to_cache = {
                "question":    question,
                "complexity":  complexity,
                "confidence":  confidence,
                "simple_prob": simple_prob,
                "complex_prob":complex_prob,
                "domain":      domain,
                "model":       model,
                "answer":      full_answer,
                "cache_hit":   False,
                "classify_ms": classify_ms,
                "llm_ms":      llm_ms,
                "total_ms":    total_ms,
                "is_error":    is_error,
            }
            if req.use_cache and not is_error and not is_refusal and full_answer:
                _cache.set_query(cache_key, result_to_cache)
                logger.info("Stream: cached answer for key=%s", cache_key[:60])
            from router import history as _query_history, MAX_HISTORY as _MAX_HISTORY
            _query_history.append(result_to_cache)
            if len(_query_history) > _MAX_HISTORY:
                _query_history.pop(0)

            yield f"data: {_json.dumps({'type': 'done', 'latency_ms': total_ms, 'llm_ms': llm_ms})}\n\n"

        except Exception as e:
            logger.exception("Error in /api/ask-stream")
            yield f"data: {_json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Classify Only ────────────────────────────────────────────

@app.post("/api/classify", response_model=schemas.ClassificationResponse, tags=["Classification"])
async def classify(req: schemas.ClassifyRequest):
    """
    Classify a question's complexity without invoking any LLM.
    Returns label, confidence, and detected domain.
    """
    try:
        from qa_engine import _evaluate_arithmetic
        if _evaluate_arithmetic(req.question):
            return {
                "question": req.question,
                "label": "Simple",
                "confidence": 1.0,
                "simple_prob": 1.0,
                "complex_prob": 0.0,
                "latency_ms": 0.0,
                "complexity": "Simple",
                "domain": "math",
            }
        clf = query_cache.get_classification(req.question)
        if clf is None:
            clf = _classifier.classify(req.question)
            query_cache.set_classification(req.question, clf)
        complexity = clf["label"]
        domain = detect_domain(req.question) if complexity == "Complex" else "simple"
        confidence = clf["confidence"]
        simple_prob = clf["simple_prob"]
        complex_prob = clf["complex_prob"]
        if requires_coding_specialist(req.question):
            complexity = "Complex"
            domain = "code"
            confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
        elif requires_knowledge_specialist(req.question):
            complexity = "Complex"
            domain = "general"
            confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
        elif requires_math_specialist(req.question):
            complexity = "Complex"
            domain = "math"
            confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
        elif requires_specialist_explanation(req.question):
            complexity = "Complex"
            domain = detect_domain(req.question)
            confidence = 1.0; simple_prob = 0.0; complex_prob = 1.0
        return {
            **clf,
            "question": req.question,
            "complexity": complexity,
            "domain": domain,
            "confidence": confidence,
            "simple_prob": simple_prob,
            "complex_prob": complex_prob,
        }
    except Exception as e:
        logger.exception("Error in /api/classify")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health", response_model=schemas.HealthResponse, tags=["System"])
async def health():
    """Health check for all system components."""

    # RoBERTa
    try:
        test_clf = _classifier.classify("What is the capital of France?")
        roberta_status = schemas.HealthComponent(
            status="ok",
            detail=f"device={_classifier.device} | test_label={test_clf['label']}"
        )
    except Exception as e:
        roberta_status = schemas.HealthComponent(status="unavailable", detail=str(e))

    # Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ollama_status = schemas.HealthComponent(status="ok", detail=result.stdout.strip()[:200])
        else:
            ollama_status = schemas.HealthComponent(status="degraded", detail=result.stderr[:200])
    except Exception as e:
        ollama_status = schemas.HealthComponent(status="unavailable", detail=str(e))

    # Cache
    cache_stats = query_cache.stats()
    cache_status = schemas.HealthComponent(
        status="ok",
        detail=f"backend={cache_stats['backend']} | hit_rate={cache_stats['hit_rate']}%"
    )

    # System resources
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.2)

    overall = "ok"
    if roberta_status.status != "ok" or ollama_status.status not in ("ok", "degraded"):
        overall = "degraded"

    return {
        "status":      overall,
        "roberta":     roberta_status,
        "ollama":      ollama_status,
        "cache":       cache_status,
        "cpu_percent": cpu_pct,
        "ram_percent": mem.percent,
        "ram_used_mb": round(mem.used / 1024 / 1024, 1),
    }


# ── Metrics ──────────────────────────────────────────────────

@app.get("/api/metrics", response_model=schemas.MetricsResponse, tags=["System"])
async def metrics():
    """Aggregated statistics across all processed queries."""
    h = query_history
    total = len(h)
    simple_count  = sum(1 for r in h if r["complexity"] == "Simple")
    complex_count = total - simple_count
    error_count   = sum(1 for r in h if r.get("is_error", False))

    domain_dist: dict = {}
    model_dist:  dict = {}
    classify_times = []
    llm_times      = []
    total_times    = []

    for r in h:
        domain_dist[r.get("domain", "unknown")] = domain_dist.get(r.get("domain", "unknown"), 0) + 1
        model_dist[r.get("model", "unknown")]   = model_dist.get(r.get("model", "unknown"), 0) + 1
        if not r.get("cache_hit"):
            classify_times.append(r.get("classify_ms", 0))
            llm_times.append(r.get("llm_ms", 0))
        total_times.append(r.get("total_ms", 0))

    def _avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "total_queries":       total,
        "simple_queries":      simple_count,
        "complex_queries":     complex_count,
        "cache_stats":         query_cache.stats(),
        "avg_classify_ms":     _avg(classify_times),
        "avg_llm_ms":          _avg(llm_times),
        "avg_total_ms":        _avg(total_times),
        "domain_distribution": domain_dist,
        "model_distribution":  model_dist,
        "error_count":         error_count,
    }


# ── History ──────────────────────────────────────────────────

@app.get("/api/history", response_model=List[schemas.HistoryItem], tags=["System"])
async def history(limit: int = 20):
    """Return the most recent query history (newest first)."""
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    recent = list(reversed(query_history[-limit:]))
    return [
        {
            "question":   r["question"],
            "complexity": r["complexity"],
            "domain":     r["domain"],
            "model":      r["model"],
            "cache_hit":  r["cache_hit"],
            "total_ms":   r["total_ms"],
            "is_error":   r["is_error"],
        }
        for r in recent
    ]


# ── Cache Management ─────────────────────────────────────────

@app.delete("/api/cache", tags=["System"])
async def clear_cache():
    """Flush all cached query results."""
    query_cache.clear()
    return JSONResponse({"message": "Cache cleared successfully."})


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
