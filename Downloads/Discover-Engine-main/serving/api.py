"""
FastAPI serving layer for Discovery Engine Stage 2 (guardrails).

Endpoints:
  POST /recommend
  POST /search
  POST /complete-the-look
  GET  /health
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agents.candidate_agent import CandidateAgent
from agents.complete_the_look_agent import CompleteTheLookAgent
from agents.search_agent import SearchAgent
from guardrails.audit import log_inference_event, new_request_id
from guardrails.cost import estimate_inference_cost_usd
from guardrails.pii import redact_pii
from guardrails.rate_limit import SlidingWindowRateLimiter
from index.faiss_store import FaissStore
from models.discovery_model import DiscoveryModel
from models.item_tower import ItemTower
from serving.schemas import (
    CompleteTheLookRequest,
    CompleteTheLookResponse,
    ExplainItem,
    GuardrailMeta,
    RecommendRequest,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "configs" / "config.yaml"

app = FastAPI(title="Discovery Engine API", version="2.0.0")

_guardrails_cfg: dict[str, Any] = {}
_rate_limiter: SlidingWindowRateLimiter | None = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"embedding_dim": 128, "hidden_dim": 256, "max_history": 10, "temperature": 0.05}


def _init_guardrails(config: dict) -> None:
    global _guardrails_cfg, _rate_limiter
    _guardrails_cfg = config.get("guardrails") or {}
    rpm = int(_guardrails_cfg.get("rate_limit_per_minute", 120))
    _rate_limiter = SlidingWindowRateLimiter(max_requests=rpm, window_sec=60.0)
    if _guardrails_cfg.get("audit_log_enabled", True):
        audit_handler = logging.StreamHandler()
        audit_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("discovery.audit").addHandler(audit_handler)
        logging.getLogger("discovery.audit").setLevel(logging.INFO)


def _sla_ms() -> float:
    return float(_guardrails_cfg.get("latency_sla_ms", 500))


def _cost_overrides() -> dict[str, float] | None:
    raw = _guardrails_cfg.get("inference_cost_usd")
    return dict(raw) if isinstance(raw, dict) else None


def _guardrail_meta(request_id: str, endpoint: str, latency_ms: float, pii_redacted: bool) -> GuardrailMeta:
    return GuardrailMeta(
        request_id=request_id,
        latency_ms=latency_ms,
        sla_met=latency_ms <= _sla_ms(),
        inference_cost_usd=estimate_inference_cost_usd(endpoint, _cost_overrides()),
        pii_redacted=pii_redacted,
    )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request) -> None:
    if _rate_limiter is None:
        return
    result = _rate_limiter.check(_client_ip(request))
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded; retry after {result.retry_after_sec}s",
            headers={"Retry-After": str(int(result.retry_after_sec) + 1)},
        )


def _bootstrap():
    global candidate_agent, search_agent, complete_agent, user_map, item_map

    config = _load_config()
    _init_guardrails(config)
    device = _device()
    logger.info("Using device: %s", device)

    with open(ARTIFACTS / "user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open(ARTIFACTS / "item_map.pkl", "rb") as f:
        item_map = pickle.load(f)
    with open(ARTIFACTS / "user_interactions.pkl", "rb") as f:
        user_interactions = pickle.load(f)
    with open(ARTIFACTS / "user_histories.pkl", "rb") as f:
        user_histories = pickle.load(f)
    with open(ARTIFACTS / "reverse_item_map.pkl", "rb") as f:
        reverse_item_map = pickle.load(f)

    item_emb = np.load(ARTIFACTS / "item_embeddings.npy")
    item_meta = pd.read_json(ARTIFACTS / "item_meta.json").to_dict("records")

    item_tower = ItemTower(output_dim=config.get("embedding_dim", 128))
    model = DiscoveryModel(
        num_items=len(item_map),
        item_embedding_table=torch.from_numpy(item_emb).float(),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        max_history=config.get("max_history", 10),
        item_tower=item_tower,
    )

    ckpt_path = ARTIFACTS / "discovery_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        item_tower.load_state_dict(ckpt["item_tower_state"])
        logger.info("Loaded checkpoint %s", ckpt_path.name)

    model.to(device)
    model.eval()
    item_tower.to(device)

    faiss_store = FaissStore.load(ARTIFACTS / "faiss.index")

    candidate_agent = CandidateAgent(
        model=model,
        faiss_store=faiss_store,
        user_histories=user_histories,
        user_interactions=user_interactions,
        reverse_item_map=reverse_item_map,
        device=device,
        max_history=config.get("max_history", 10),
    )
    search_agent = SearchAgent(item_tower=item_tower, faiss_store=faiss_store, device=device)
    complete_agent = CompleteTheLookAgent(
        faiss_store=faiss_store,
        item_embeddings=item_emb,
        item_meta=item_meta,
    )


candidate_agent = None
search_agent = None
complete_agent = None
user_map: dict = {}
item_map: dict = {}


@app.on_event("startup")
def startup():
    config = _load_config()
    _init_guardrails(config)
    if (ARTIFACTS / "faiss.index").exists():
        _bootstrap()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, Exception):
            return str(obj)
        return obj

    return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})


@app.get("/health")
def health():
    ready = candidate_agent is not None
    return {
        "status": "ok" if ready else "starting",
        "artifacts_loaded": ready,
        "guardrails": {
            "latency_sla_ms": _sla_ms(),
            "rate_limit_per_minute": (_rate_limiter.max_requests if _rate_limiter else None),
        },
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request_body: RecommendRequest, request: Request):
    _enforce_rate_limit(request)
    request_id = new_request_id()
    t0 = time.perf_counter()

    if candidate_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded; run make train && make index")

    try:
        raw = candidate_agent.recommend(
            request_body.user_id, user_map, request_body.top_k, explain=request_body.explain
        )
        outcome = "ok"
    except KeyError as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if _guardrails_cfg.get("audit_log_enabled", True):
            log_inference_event(
                request_id=request_id,
                endpoint="recommend",
                client_ip=_client_ip(request),
                outcome="not_found",
                latency_ms=latency_ms,
                extra={"user_id": str(request_body.user_id)},
            )
        raise HTTPException(status_code=404, detail=str(e))

    latency_ms = raw["latency_ms"]
    guardrails = _guardrail_meta(request_id, "recommend", latency_ms, pii_redacted=False)
    if _guardrails_cfg.get("audit_log_enabled", True):
        log_inference_event(
            request_id=request_id,
            endpoint="recommend",
            client_ip=_client_ip(request),
            outcome=outcome,
            latency_ms=latency_ms,
            extra={
                "user_id": str(request_body.user_id),
                "top_k": request_body.top_k,
                "sla_met": guardrails.sla_met,
                "inference_cost_usd": guardrails.inference_cost_usd,
            },
        )

    items = [ExplainItem(**row) for row in raw.get("items", [])]
    return RecommendResponse(
        user_id=raw["user_id"],
        recommendations=raw["recommendations"],
        items=items,
        latency_ms=latency_ms,
        guardrails=guardrails,
    )


@app.post("/search", response_model=SearchResponse)
def search(request_body: SearchRequest, request: Request):
    _enforce_rate_limit(request)
    request_id = new_request_id()

    if search_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pii = redact_pii(request_body.query_text)
    query_text = pii.redacted_text if request_body.query_text else None
    redact_response = bool(_guardrails_cfg.get("pii_redact_responses", True))
    display_text = query_text if redact_response or not pii.had_pii else request_body.query_text

    try:
        raw = search_agent.search(
            query_text,
            request_body.query_image,
            request_body.top_k,
            explain=request_body.explain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = raw["latency_ms"]
    guardrails = _guardrail_meta(request_id, "search", latency_ms, pii_redacted=pii.had_pii)
    if _guardrails_cfg.get("audit_log_enabled", True):
        log_inference_event(
            request_id=request_id,
            endpoint="search",
            client_ip=_client_ip(request),
            outcome="ok",
            latency_ms=latency_ms,
            extra={
                "query_text_redacted": query_text,
                "pii_types": list(pii.types_found),
                "top_k": request_body.top_k,
                "sla_met": guardrails.sla_met,
                "inference_cost_usd": guardrails.inference_cost_usd,
            },
        )

    return SearchResponse(
        query_text=display_text,
        query_image=request_body.query_image,
        results=raw["results"],
        latency_ms=latency_ms,
        guardrails=guardrails,
    )


@app.post("/complete-the-look", response_model=CompleteTheLookResponse)
def complete_the_look(request_body: CompleteTheLookRequest, request: Request):
    _enforce_rate_limit(request)
    request_id = new_request_id()

    if complete_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = complete_agent.complete(
            request_body.seed_item_idx,
            request_body.top_k,
            explain=request_body.explain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = raw["latency_ms"]
    guardrails = _guardrail_meta(request_id, "complete_the_look", latency_ms, pii_redacted=False)
    if _guardrails_cfg.get("audit_log_enabled", True):
        log_inference_event(
            request_id=request_id,
            endpoint="complete_the_look",
            client_ip=_client_ip(request),
            outcome="ok",
            latency_ms=latency_ms,
            extra={
                "seed_item_idx": request_body.seed_item_idx,
                "diversity_audit": raw.get("diversity_audit"),
                "sla_met": guardrails.sla_met,
                "inference_cost_usd": guardrails.inference_cost_usd,
            },
        )

    items = [ExplainItem(**row) for row in raw.get("items", [])]
    return CompleteTheLookResponse(
        seed_item_idx=raw["seed_item_idx"],
        seed_category=raw["seed_category"],
        recommendations=raw["recommendations"],
        items=items,
        diversity_audit=raw.get("diversity_audit", {}),
        latency_ms=latency_ms,
        guardrails=guardrails,
    )
