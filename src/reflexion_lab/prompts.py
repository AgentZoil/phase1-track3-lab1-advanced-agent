from __future__ import annotations

from textwrap import dedent

from .schemas import QAExample, JudgeResult

ACTOR_SYSTEM = dedent(
    """\
    You are the Actor for a multi-hop question answering benchmark.
    Use only the provided context. If the answer cannot be determined, admit uncertainty.
    Respond with a concise final answer only; do not include reasoning steps or JSON.
    """
)

EVALUATOR_SYSTEM = dedent(
    """\
    You are a deterministic judge for a QA agent.
    Each response must be valid JSON with keys score (0 or 1), reason (string),
    missing_evidence (list), and spurious_claims (list).
    Do not add extra text outside the JSON object.
    """
)

REFLECTOR_SYSTEM = dedent(
    """\
    You are an internal reflection assistant.
    Given what failed during the last attempt, explain the lesson learned and craft a next strategy.
    Reply with JSON containing `lesson` and `next_strategy` strings only.
    """
)


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"{chunk.title}: {chunk.text}" for chunk in example.context)


def _format_memory(reflection_memory: list[str]) -> str:
    if not reflection_memory:
        return "No reflections yet."
    return "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(reflection_memory))


def build_actor_prompt(example: QAExample, reflection_memory: list[str]) -> str:
    return dedent(
        f"""\
        {ACTOR_SYSTEM}

        Question: {example.question}
        Context:
        {_format_context(example)}

        Reflection memory:
        {_format_memory(reflection_memory)}

        Final answer (short, one sentence):
        """
    )


def build_evaluator_prompt(example: QAExample, answer: str) -> str:
    return dedent(
        f"""\
        {EVALUATOR_SYSTEM}

        Question: {example.question}
        Context:
        {_format_context(example)}

        Gold answer: {example.gold_answer}
        Candidate answer: {answer}
        """
    )


def build_reflector_prompt(
    example: QAExample,
    answer: str,
    judge: JudgeResult,
    reflection_memory: list[str],
    attempt_id: int,
) -> str:
    missing = judge.missing_evidence or ["Nothing noted"]
    spurious = judge.spurious_claims or ["None"]
    return dedent(
        f"""\
        {REFLECTOR_SYSTEM}

        Question: {example.question}
        Context:
        {_format_context(example)}

        Candidate answer: {answer}
        Failure reason: {judge.reason}
        Missing evidence: {', '.join(missing)}
        Spurious claims: {', '.join(spurious)}
        Previous reflections:
        {_format_memory(reflection_memory)}
        Attempt number: {attempt_id}
        """
    )
