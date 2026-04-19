from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repeat_penalty: float


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    image_absolute_path: str | None
    max_new_tokens: int
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    latency_ms: int
