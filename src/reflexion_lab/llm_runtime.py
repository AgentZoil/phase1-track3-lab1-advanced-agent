from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from .prompts import (
    build_actor_prompt,
    build_evaluator_prompt,
    build_reflector_prompt,
)
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")


FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class LLMClient:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API Key is missing. Please set the OPENAI_API_KEY environment variable.")
        
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        usage = response.usage
        return LLMResponse(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )


def _clean_final_answer(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    for marker in ("final answer:", "final:", "answer:"):
        idx = lower.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker) :].strip()
            break
    return cleaned or "I don't know."


def _parse_json_block(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start == -1:
            return {}
        try:
            return json.loads(candidate[start:])
        except json.JSONDecodeError:
            return {}


class Runtime:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, int, float]:
        prompt = build_actor_prompt(example, reflection_memory)
        response = self._client.generate(prompt=prompt)
        return _clean_final_answer(response.text), int(response.total_tokens), float(response.latency_ms)

    def evaluator(self, example: QAExample, answer: str) -> JudgeResult:
        prompt = build_evaluator_prompt(example, answer)
        response = self._client.generate(prompt=prompt)
        payload = _parse_json_block(response.text)
        score = int(payload.get("score", 0))
        reason = payload.get("reason", "Judging failed to parse the model output.")
        failure_mode = payload.get("failure_mode", "wrong_final_answer")
        missing = list(payload.get("missing_evidence", []))
        spurious = list(payload.get("spurious_claims", []))
        if not payload:
            score = 1 if normalize_answer(answer) == normalize_answer(example.gold_answer) else 0
            reason = "Fallback judgment because the model response was not JSON." if score == 0 else "Fallback awarded success on normalization match."
            failure_mode = "wrong_final_answer"
        return JudgeResult(
            score=score,
            reason=reason,
            failure_mode=failure_mode,
            missing_evidence=missing,
            spurious_claims=spurious,
        )

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
        answer: str,
        reflection_memory: list[str],
    ) -> ReflectionEntry:
        prompt = build_reflector_prompt(example, answer, judge, reflection_memory, attempt_id)
        response = self._client.generate(prompt=prompt)
        payload = _parse_json_block(response.text)
        lesson = payload.get("lesson", "Review the missed evidence and try to ground every hop.")
        next_strategy = payload.get("next_strategy", "Compare each paragraph before answering.")
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=lesson,
            next_strategy=next_strategy,
        )


RUNTIME = Runtime(LLMClient(OPENAI_API_KEY, MODEL_NAME))


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, int, float]:
    return RUNTIME.actor_answer(example, attempt_id, agent_type, reflection_memory)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return RUNTIME.evaluator(example, answer)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult, answer: str, reflection_memory: list[str]) -> ReflectionEntry:
    return RUNTIME.reflector(example, attempt_id, judge, answer, reflection_memory)
