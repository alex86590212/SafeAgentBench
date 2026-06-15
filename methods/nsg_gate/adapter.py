from __future__ import annotations

from .plan_parser import TypedPrimitive

_MULTI_WORD_VERBS = ("turn on", "turn off", "fillLiquid", "emptyLiquid")


def plan_to_primitives(steps: list[str]) -> list[TypedPrimitive]:
    """Convert SafeAgentBench low-level plan strings to TypedPrimitive objects.

    Each step like "fillLiquid Mug wine" becomes:
      TypedPrimitive(name="fillLiquid", raw="Mug wine", text_blob="fillliquid mug wine")

    Compatible with NSG rule_engine and asp_backend field expectations:
      - name  → matched against `primitive:` in structural/confirm rules
      - raw   → matched against structural rule patterns
      - text_blob → matched against lexical rule patterns + ASP keyword extraction
    """
    primitives: list[TypedPrimitive] = []
    for step in steps:
        step = step.strip()
        if not step:
            continue
        name, raw = _split_step(step)
        text_blob = step.lower()
        primitives.append(TypedPrimitive(name=name, raw=raw, text_blob=text_blob))
    return primitives


def _split_step(step: str) -> tuple[str, str]:
    for verb in _MULTI_WORD_VERBS:
        if step.startswith(verb + " "):
            return verb, step[len(verb) + 1:]
        if step == verb:
            return verb, ""
    parts = step.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
