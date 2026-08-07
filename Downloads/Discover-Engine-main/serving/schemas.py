"""Request/response schemas with Stage 2 validation."""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_QUERY_TEXT_LEN = 512
MAX_IMAGE_REF_LEN = 2048
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RecommendRequest(BaseModel):
    user_id: str | int
    top_k: int = Field(default=10, ge=1, le=100)
    explain: bool = Field(default=True, description="Include per-item explainability fields")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str | int) -> str | int:
        if isinstance(v, int):
            if v < 0:
                raise ValueError("user_id must be non-negative")
            return v
        s = v.strip()
        if not s or len(s) > 64:
            raise ValueError("user_id string must be 1–64 characters")
        if not _USER_ID_PATTERN.match(s):
            raise ValueError("user_id contains invalid characters")
        return s


class SearchRequest(BaseModel):
    query_text: Optional[str] = Field(default=None, max_length=MAX_QUERY_TEXT_LEN)
    query_image: Optional[str] = Field(default=None, max_length=MAX_IMAGE_REF_LEN)
    top_k: int = Field(default=10, ge=1, le=100)
    explain: bool = Field(default=True)

    @field_validator("query_text")
    @classmethod
    def strip_query_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s or None

    @field_validator("query_image")
    @classmethod
    def validate_image_ref(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        if s.startswith(("http://", "https://")):
            return s
        if re.match(r"^[A-Za-z0-9_./\\:-]+$", s) and ".." not in s:
            return s
        raise ValueError("query_image must be http(s) URL or safe local path")

    @model_validator(mode="after")
    def require_query(self) -> "SearchRequest":
        if not self.query_text and not self.query_image:
            raise ValueError("Provide query_text and/or query_image")
        return self


class CompleteTheLookRequest(BaseModel):
    seed_item_idx: int = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=100)
    explain: bool = Field(default=True)


class ExplainItem(BaseModel):
    item_idx: int
    score: Optional[float] = None
    category: Optional[str] = None
    reason: str


class GuardrailMeta(BaseModel):
    request_id: str
    latency_ms: float
    sla_met: bool
    inference_cost_usd: float
    pii_redacted: bool = False


class RecommendResponse(BaseModel):
    user_id: str | int
    recommendations: list[int]
    items: list[ExplainItem] = Field(default_factory=list)
    latency_ms: float
    guardrails: GuardrailMeta


class SearchResponse(BaseModel):
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    results: list[dict[str, Any]]
    latency_ms: float
    guardrails: GuardrailMeta


class CompleteTheLookResponse(BaseModel):
    seed_item_idx: int
    seed_category: str
    recommendations: list[int]
    items: list[ExplainItem] = Field(default_factory=list)
    diversity_audit: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    guardrails: GuardrailMeta
