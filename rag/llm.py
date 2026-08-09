"""Anthropic chat model construction and JSON-response parsing helpers."""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from rag.config import Settings


def build_chat_model(settings: Settings, model: str | None = None) -> ChatAnthropic:
    """Create a ChatAnthropic client.

    Sampling parameters are deliberately omitted: `temperature`/`top_p`/`top_k`
    are rejected by Claude Opus 5. Adaptive thinking is the model default and is
    left on — it materially improves grounding and citation accuracy.
    """
    settings.require_api_key()
    return ChatAnthropic(
        model=model or settings.answer_model,
        max_tokens=settings.max_tokens,
        timeout=120,
        max_retries=3,
    )


def message_text(message: BaseMessage) -> str:
    """Flatten a response into plain text, ignoring thinking blocks."""
    content = message.content
    if isinstance(content, str):
        return content.strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating fences and stray prose."""
    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(candidate[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : i + 1])

    raise ValueError(f"Unbalanced JSON object in model output: {text[:200]!r}")
