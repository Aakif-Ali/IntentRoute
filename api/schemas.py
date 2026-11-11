"""
api/schemas.py
--------------
Pydantic v2 schemas for the Intelligent QA System FastAPI endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ConversationTurn(BaseModel):
    """A single prior Q&A turn for conversation context."""
    question: str
    answer:   str


class QueryRequest(BaseModel):
    """Request body for POST /api/ask"""
    question: str = Field(
        ...,
        min_length=2,
        max_length=50000,
        description="The user's question",
        examples=["What is the capital of France?"]
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available"
    )
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Prior Q&A turns to maintain conversation context (max 5)"
    )


class ClassifyRequest(BaseModel):
    """Request body for POST /api/classify"""
    question: str = Field(
        ...,
        min_length=2,
        max_length=50000,
        description="The question to classify",
    )


class LoginRequest(BaseModel):
    """Request body for POST /api/auth/login"""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class UserProfile(BaseModel):
    username: str
    name: str
    email: str
    role: str


class LoginResponse(BaseModel):
    success: bool
    token: str
    user: UserProfile
    message: str


class ClassificationResponse(BaseModel):
    """Response for POST /api/classify"""
    question:     str
    complexity:   str          # "Simple" or "Complex"
    confidence:   float        # 0.0 – 1.0
    simple_prob:  float
    complex_prob: float
    domain:       str          # "simple" / "code" / "math" / "general"
    latency_ms:   float


class QueryResponse(BaseModel):
    """Response for POST /api/ask"""
    question:     str
    complexity:   str
    confidence:   float
    simple_prob:  float
    complex_prob: float
    domain:       str
    model:        str
    answer:       str
    cache_hit:    bool
    classify_ms:  float
    llm_ms:       float
    total_ms:     float
    is_error:     bool


class HistoryItem(BaseModel):
    """Single history entry"""
    question:    str
    complexity:  str
    domain:      str
    model:       str
    cache_hit:   bool
    total_ms:    float
    is_error:    bool


class MetricsResponse(BaseModel):
    """Response for GET /api/metrics"""
    total_queries:      int
    simple_queries:     int
    complex_queries:    int
    cache_stats:        Dict[str, Any]
    avg_classify_ms:    float
    avg_llm_ms:         float
    avg_total_ms:       float
    domain_distribution: Dict[str, int]
    model_distribution:  Dict[str, int]
    error_count:         int


class HealthComponent(BaseModel):
    status:  str          # "ok" | "degraded" | "unavailable"
    detail:  Optional[str] = None


class HealthResponse(BaseModel):
    """Response for GET /api/health"""
    status:      str      # "ok" | "degraded"
    roberta:     HealthComponent
    ollama:      HealthComponent
    cache:       HealthComponent
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
