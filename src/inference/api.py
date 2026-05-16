from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import VektorGuard


# Pydantic models
# No empty strings (min_length=1)
# Threshold within valid range (ge=0.0, le=1.0)
class GuardRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Prompt text to classify")
    threshold: float = Field(0.85, ge=0.0, le=1.0, description="Confidence threshold for is_safe()")


class GuardResponse(BaseModel):
    label: str
    confidence: float
    class_id: int
    safe: bool
    action: str
    latency_ms: float


class BatchGuardRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of prompts texts to classify")
    threshold: float = Field(0.85, ge=0.0, le=1.0, description="Confidence threshold for is_safe()")


class BatchGuardResponse(BaseModel):
    results: list[GuardResponse]
    total_latency_ms: float



class HealthResponse(BaseModel):
    status: str
    model: str
    classes: list[str]


# Load model at startup - Lifespan
guard: Optional[VektorGuard] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VektorGuard at startup, release at shutdown."""
    global guard
    guard = VektorGuard()
    yield
    guard = None


# Application
app = FastAPI(
    title="Vektor-Guard API",
    description="Prompt injection and attack category detection using vektor-guard-v2",
    version="2.0.0",
    lifespan=lifespan
)

# API routes
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check - confirms the model is loaded and ready for use."""
    if guard is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model="theinferenceloop/vektor-guard-v2",
        classes=[
            "clean",
            "instruction_override",
            "indirect_injection",
            "jailbreak",
            "tool_call_hijacking"
        ]
    )

@app.post("/v1/guard", response_model=GuardResponse)
def classify(request: GuardRequest):
    """Classify a single prompt and return label, confidence, and safety decision."""
    if guard is None:
        raise HTTPException(status_code=503, description="Model not loaded")
    
    result = guard.predict(request.text)
    safe = result["label"] == "clean" and result["confidence"] >= request.threshold
    action ="allow" if safe else "block"

    return GuardResponse(
        label=result["label"],
        confidence=result["confidence"],
        class_id=result["class_id"],
        safe=safe,
        action=action,
        latency_ms=result["latency_ms"]
    )

@app.post("/v1/guard/batch", response_model=BatchGuardResponse)
def classify_batch(request: BatchGuardRequest):
    """Classify a list of prompts in a single forward pass."""
    if guard is None:
        raise HTTPException(status_code=503, description="Model not loaded")
    
    import time
    start = time.time()

    batch_results = guard.predict_batch(request.texts)

    results = []

    for result in batch_results:
        safe = result["label"] == "clean" and result["confidence"] >= request.threshold
        results.append(GuardResponse(
            label=result["label"],
            confidence=result["confidence"],
            class_id=result["class_id"],
            safe=safe,
            action = "allow" if safe else "block",
            latency_ms=result["latency_ms"]
        ))

    total_latency_ms = round((time.time() - start) * 1000, 2)

    return BatchGuardResponse(
        results=results,
        total_latency_ms=total_latency_ms
    )