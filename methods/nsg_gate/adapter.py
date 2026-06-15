from __future__ import annotations

from .plan_parser import TypedPrimitive

_MULTI_WORD_VERBS = ("turn on", "turn off", "fillLiquid", "emptyLiquid")

# Dataset uses both "turn on" (space) and "turn_on" (underscore) — normalise to space form.
_UNDERSCORE_NORM = {"turn_on": "turn on", "turn_off": "turn off"}


_BARE_VERBS = {"throw", "drop", "pour"}


def plan_to_primitives(steps: list[str]) -> list[TypedPrimitive]:
    """Convert SafeAgentBench low-level plan strings to TypedPrimitive objects.

    Each step like "fillLiquid Mug wine" becomes:
      TypedPrimitive(name="fillLiquid", raw="Mug wine", text_blob="fillliquid mug wine")

    Compatible with NSG rule_engine and asp_backend field expectations:
      - name  → matched against `primitive:` in structural/confirm rules
      - raw   → matched against structural rule patterns
      - text_blob → matched against lexical rule patterns + ASP keyword extraction

    Context enrichment for regex / ASP keyword matching:
      - put: text_blob includes last-picked object (destination-only steps)
      - throw/drop/pour (bare): text_blob includes last-picked object
      - pour (bare): also includes last-found destination target
      - drop/throw (bare): also includes last-found target when it differs
    """
    primitives: list[TypedPrimitive] = []
    last_picked: str = ""
    last_found: str = ""
    for step in steps:
        step = step.strip()
        if not step:
            continue
        # Normalise underscore variants before splitting
        for under, spaced in _UNDERSCORE_NORM.items():
            if step == under or step.startswith(under + " "):
                step = spaced + step[len(under):]
                break
        name, raw = _split_step(step)
        text_blob = step.lower()
        # Track context for enriching bare verb steps
        if name == "pick" and raw:
            last_picked = raw.lower()
        if name == "find" and raw:
            last_found = raw.lower()
        extras: list[str] = []
        if name == "put" and raw and last_picked:
            extras.append(last_picked)
        elif name in _BARE_VERBS and not raw:
            if last_picked:
                extras.append(last_picked)
            if last_found and last_found != last_picked:
                extras.append(last_found)
        if extras:
            text_blob = f"{text_blob} {' '.join(extras)}"
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
