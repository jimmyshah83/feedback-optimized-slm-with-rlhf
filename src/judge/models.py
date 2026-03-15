"""Pydantic schemas for AI judge requests/responses and preference pairs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(ge=1, le=5)
    reason: str


class JudgeVerdict(BaseModel):
    """Structured output from the AI judge."""

    winner: str = Field(
        description="'A' or 'B' — which response is better"
    )
    dimension_scores_a: list[DimensionScore]
    dimension_scores_b: list[DimensionScore]
    explanation: str


class PreferencePair(BaseModel):
    """A single DPO training example."""

    iteration: int
    question: str
    context: str
    gold_answer: str
    gold_decision: str
    chosen: str
    rejected: str
    chosen_label: str = Field(
        description="'A' or 'B' — which candidate was chosen"
    )
    score_chosen: float
    score_rejected: float
