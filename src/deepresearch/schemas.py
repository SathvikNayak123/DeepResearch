from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str
    question: str


class Plan(BaseModel):
    sub_questions: list[SubQuestion]


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float = 0.0


class FetchResult(BaseModel):
    url: str
    content: str


class Claim(BaseModel):
    text: str
    source_id: str
    quote: str
    confidence: float


class WorkerNotes(BaseModel):
    sub_question_id: str
    sub_question: str
    claims: list[Claim]
    open_gaps: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    coverage_score: float
    rationale: str
    should_replan: bool
    new_sub_questions: list[SubQuestion] = Field(default_factory=list)


class SourceRegistryEntry(BaseModel):
    source_id: str
    url: str
    title: str


class Report(BaseModel):
    text: str
    citations: list[SourceRegistryEntry]


class CacheStats(BaseModel):
    search_hits: int = 0
    search_misses: int = 0
    fetch_hits: int = 0
    fetch_misses: int = 0
    estimated_dollars_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.search_hits + self.search_misses + self.fetch_hits + self.fetch_misses
        hits = self.search_hits + self.fetch_hits
        return hits / total if total else 0.0


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    question: str
    plan: Plan
    worker_notes: list[WorkerNotes]
    reflections: list[ReflectionResult]
    report: Report | None = None
    budget_hit: str | None = None
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    iterations: int = 0
    cache_stats: CacheStats = Field(default_factory=CacheStats)
