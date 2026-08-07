"""
Standalone category diversity cap with auditable decision trail.

Used by CompleteTheLookAgent; logic mirrors Workstream A verified behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MAX_CATEGORY_FRACTION = 0.35


@dataclass
class DiversityDecision:
    item_idx: int
    category: str
    action: str  # "selected" | "skipped_cap"
    reason: str


@dataclass
class DiversityResult:
    selected: list[int]
    decisions: list[DiversityDecision] = field(default_factory=list)
    max_per_category: int = 0
    underfill: bool = False

    def to_audit_dict(self) -> dict:
        skipped = sum(1 for d in self.decisions if d.action == "skipped_cap")
        return {
            "max_per_category": self.max_per_category,
            "selected_count": len(self.selected),
            "skipped_diversity_cap": skipped,
            "underfill": self.underfill,
            "category_counts": dict(
                Counter(d.category for d in self.decisions if d.action == "selected")
            ),
        }


def apply_category_cap(
    candidates: list[int],
    top_k: int,
    idx_to_category: dict[int, str],
    max_category_fraction: float = DEFAULT_MAX_CATEGORY_FRACTION,
    default_category: str = "Unknown",
) -> DiversityResult:
    max_per_category = max(1, int(top_k * max_category_fraction + 0.999))
    selected: list[int] = []
    counts: Counter[str] = Counter()
    decisions: list[DiversityDecision] = []

    for idx in candidates:
        cat = idx_to_category.get(idx, default_category)
        if counts[cat] >= max_per_category:
            decisions.append(
                DiversityDecision(
                    item_idx=idx,
                    category=cat,
                    action="skipped_cap",
                    reason=f"category {cat!r} at cap {max_per_category}",
                )
            )
            continue
        selected.append(idx)
        counts[cat] += 1
        decisions.append(
            DiversityDecision(
                item_idx=idx,
                category=cat,
                action="selected",
                reason="within category cap",
            )
        )
        if len(selected) >= top_k:
            break

    return DiversityResult(
        selected=selected,
        decisions=decisions,
        max_per_category=max_per_category,
        underfill=len(selected) < top_k,
    )


def category_resolver_from_meta(item_meta: list[dict]) -> Callable[[int], str]:
    mapping = {m["item_idx"]: m.get("category", "Unknown") for m in item_meta}

    def resolve(idx: int) -> str:
        return mapping.get(idx, "Unknown")

    return resolve
