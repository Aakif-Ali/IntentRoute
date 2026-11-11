"""
src/router.py
-------------
Main Intelligent Question Router for the Adaptive QA System.

Pipeline:
  1. Check Redis/memory cache → instant return on cache hit
  2. Preprocess & classify question complexity (RoBERTa, ~97% accuracy)
  3. Detect domain for complex questions (code / math / general)
  4. Route to appropriate LLM:
       Simple          → Qwen2.5:1.5B                [lightweight general model]
       Complex + Math  → Qwen2.5:3B             [fast mathematical reasoning]
       Complex + Code  → Qwen2.5-Coder:3B       [coding specialist]
       Complex + GK    → Ministral-3:3B          [general knowledge]
  5. Cache result, collect telemetry, return structured response.
"""

import os
import sys
import time
import logging

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router")

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = r"C:\Users\aakif\final year"
SRC_DIR  = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ============================================================
# IMPORTS FROM LOCAL MODULES
# ============================================================

from classifier import RoBERTaClassifier
from qa_engine  import (
    answer_question,
    detect_domain,
    requires_coding_specialist,
    requires_math_specialist,
    requires_specialist_explanation,
    requires_knowledge_specialist,
    is_followup_question,
    _evaluate_arithmetic,
    MODEL_ARITHMETIC,
    MODEL_CODE,
    MODEL_MATH,
)
from cache      import CacheManager

# ============================================================
# INITIALISE SHARED COMPONENTS
# ============================================================

print("=" * 70)
print("  INTELLIGENT QUESTION ROUTING SYSTEM")
print("=" * 70)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("\n[1/3] Loading RoBERTa complexity classifier...")
classifier = RoBERTaClassifier()
print("      [OK] RoBERTa loaded -", classifier.device)

print("\n[2/3] Initialising cache layer...")
cache = CacheManager()
print(f"      [OK] Cache backend: {cache.backend.upper()}")

print("\n[3/3] System ready.\n")
print("=" * 70)

# ============================================================
# IN-MEMORY QUERY HISTORY (for /api/history)
# ============================================================

history: list = []
MAX_HISTORY = 200


# ============================================================
# MAIN ROUTER FUNCTION
# ============================================================

def process_question(question: str, use_cache: bool = True, conversation_history: list = None) -> dict:
    """
    Process a user question through the full routing pipeline.

    Parameters
    ----------
    question              : str   — raw user question
    use_cache             : bool  — whether to check & store in cache
    conversation_history  : list  — prior [{question, answer}, ...] turns for context

    Returns
    -------
    dict with all routing metadata and the generated answer:
        question         : str
        complexity       : str   ("Simple" / "Complex")
        confidence       : float (classifier confidence)
        simple_prob      : float
        complex_prob     : float
        domain           : str   ("simple" / "code" / "math" / "general")
        model            : str   (Ollama model identifier)
        answer           : str
        cache_hit        : bool
        classify_ms      : float (RoBERTa latency)
        llm_ms           : float (LLM latency, 0 on cache hit)
        total_ms         : float
        is_error         : bool
    """
    t_start = time.perf_counter()
    question = question.strip()
    history_turns = conversation_history or []
    is_followup = is_followup_question(question, history_turns)

    # Follow-ups use context-keyed cache; standalone queries use direct question key
    if is_followup and history_turns:
        last_turn = history_turns[-1]
        last_q = last_turn.get("question", "").strip()[:80]
        cache_key = f"ctx:{last_q} >> {question}"
    else:
        cache_key = question

    # --------------------------------------------------------
    # 1. CACHE CHECK
    # --------------------------------------------------------
    if use_cache:
        cached = cache.get_query(cache_key)
        if cached:
            cached_resp = dict(cached)
            cached_resp["cache_hit"] = True
            cached_resp["total_ms"]  = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info("Cache HIT for: %s (key=%s)", question[:60], cache_key[:60])
            history.append(cached_resp)
            if len(history) > MAX_HISTORY:
                history.pop(0)
            return cached_resp

    # --------------------------------------------------------
    # 2. FAST-PATH DIRECT ARITHMETIC (Always Simple)
    # --------------------------------------------------------
    direct_arithmetic = _evaluate_arithmetic(question)
    if direct_arithmetic is not None:
        result = {
            "question":    question,
            "complexity":  "Simple",
            "confidence":  1.0,
            "simple_prob": 1.0,
            "complex_prob":0.0,
            "domain":      "math",
            "model":       MODEL_ARITHMETIC,
            "answer":      direct_arithmetic,
            "cache_hit":   False,
            "classify_ms": 0.0,
            "llm_ms":      0.0,
            "total_ms":    round((time.perf_counter() - t_start) * 1000, 2),
            "is_error":    False,
        }
        if use_cache:
            cache.set_query(cache_key, result)
        history.append(result)
        if len(history) > MAX_HISTORY:
            history.pop(0)
        return result

    # --------------------------------------------------------
    # 3. CLASSIFY COMPLEXITY
    # --------------------------------------------------------
    clf = cache.get_classification(question) if use_cache else None
    if clf is None:
        clf = classifier.classify(question)
        if use_cache:
            cache.set_classification(question, clf)

    complexity   = clf["label"]       # "Simple" or "Complex"
    confidence   = clf["confidence"]
    simple_prob  = clf["simple_prob"]
    complex_prob = clf["complex_prob"]
    classify_ms  = clf["latency_ms"]

    logger.info(
        "Classified: %s (%.2f%%) — classify_ms=%.1f",
        complexity, confidence * 100, classify_ms,
    )

    # --------------------------------------------------------
    # 4. DETECT DOMAIN & SPECIALIST ROUTING
    # --------------------------------------------------------
    detected_domain = detect_domain(question)
    domain = detected_domain if complexity == "Complex" else "simple"

    if requires_coding_specialist(question):
        complexity = "Complex"
        domain = "code"
        confidence = 1.0
        simple_prob = 0.0
        complex_prob = 1.0
        logger.info("Specialist override: coding request -> %s", MODEL_CODE)
    elif requires_knowledge_specialist(question):
        complexity = "Complex"
        domain = "general"
        confidence = 1.0
        simple_prob = 0.0
        complex_prob = 1.0
        logger.info("Specialist override: identity question -> ministral-3:3b")
    elif requires_math_specialist(question):
        complexity = "Complex"
        domain = "math"
        confidence = 1.0
        simple_prob = 0.0
        complex_prob = 1.0
        logger.info("Specialist override: symbolic math -> %s", MODEL_MATH)
    elif requires_specialist_explanation(question):
        complexity = "Complex"
        domain = detect_domain(question)
        confidence = 1.0
        simple_prob = 0.0
        complex_prob = 1.0
        logger.info("Specialist override: multi-step explanation -> %s", domain)


    logger.info("Domain: %s", domain)

    # --------------------------------------------------------
    # 4. GENERATE ANSWER
    # --------------------------------------------------------
    qa = answer_question(
        question=question,
        complexity=complexity,
        domain=domain,
        conversation_history=conversation_history or [],
    )

    # --------------------------------------------------------
    # 5. BUILD RESULT
    # --------------------------------------------------------
    result = {
        "question":    question,
        "complexity":  complexity,
        "confidence":  confidence,
        "simple_prob": simple_prob,
        "complex_prob":complex_prob,
        "domain":      qa["domain"],
        "model":       qa["model"],
        "answer":      qa["answer"],
        "cache_hit":   False,
        "classify_ms": classify_ms,
        "llm_ms":      qa["latency_ms"],
        "total_ms":    round((time.perf_counter() - t_start) * 1000, 2),
        "is_error":    qa["is_error"],
    }

    # --------------------------------------------------------
    # 6. CACHE RESULT (only successful answers)
    # --------------------------------------------------------
    refusal_prefixes = ("i'm sorry", "i am sorry", "[error]")
    is_refusal = any(qa["answer"].strip().lower().startswith(p) for p in refusal_prefixes)
    if use_cache and not qa["is_error"] and not is_refusal:
        cache.set_query(cache_key, result)

    # --------------------------------------------------------
    # 7. STORE IN HISTORY
    # --------------------------------------------------------
    history.append(result)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    return result


# ============================================================
# PRETTY PRINT
# ============================================================

def _print_result(r: dict):
    print("\n" + "=" * 70)
    print("  QUESTION")
    print("=" * 70)
    print(r["question"])

    print("\n── Classification ──────────────────────────────────────────────")
    print(f"  Complexity   : {r['complexity'].upper()}")
    print(f"  Confidence   : {r['confidence'] * 100:.2f}%")
    print(f"  Simple Prob  : {r['simple_prob'] * 100:.2f}%")
    print(f"  Complex Prob : {r['complex_prob'] * 100:.2f}%")

    print("\n── Routing ─────────────────────────────────────────────────────")
    print(f"  Domain       : {r['domain'].upper()}")
    print(f"  Model        : {r['model']}")
    print(f"  Cache Hit    : {'YES ⚡' if r['cache_hit'] else 'NO'}")

    print("\n── Answer ──────────────────────────────────────────────────────")
    print(r["answer"])

    print("\n── Telemetry ───────────────────────────────────────────────────")
    print(f"  Classify     : {r['classify_ms']:.1f} ms")
    print(f"  LLM          : {r['llm_ms']:.1f} ms")
    print(f"  Total        : {r['total_ms']:.1f} ms")
    print("─" * 70)


# ============================================================
# INTERACTIVE CLI LOOP
# ============================================================

if __name__ == "__main__":

    print("\nType a question and press Enter.")
    print("Commands: 'exit'  — quit")
    print("          'stats' — cache statistics")
    print("          'clear' — clear cache")
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if question.lower() == "stats":
            stats = cache.stats()
            print("\n── Cache Statistics ────────────────────────────────────────────")
            for k, v in stats.items():
                print(f"  {k:<15}: {v}")
            print("─" * 70)
            continue

        if question.lower() == "clear":
            cache.clear()
            print("  Cache cleared.\n")
            continue

        try:
            result = process_question(question)
            _print_result(result)
        except Exception as e:
            logger.exception("Unexpected error processing question.")
            print(f"\n[ERROR] {e}\n")