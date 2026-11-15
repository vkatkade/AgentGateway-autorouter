"""Vertex AI generateContent helpers: path parsing/rewriting, request features,
the redacted excerpt sent to the classifier, and thinkingConfig mutation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# /v1/projects/P/locations/L/publishers/google/models/M:generateContent
# /v1beta1/... and express-mode /v1/publishers/google/models/M:... are accepted too.
_PATH_RE = re.compile(
    r"^(?P<prefix>/v1(?:beta1)?/(?:projects/[^/]+/locations/[^/]+/)?publishers/google/models/)"
    r"(?P<model>[^/:?]+)"
    r"(?P<method>:(?:generateContent|streamGenerateContent))"
    r"(?P<query>\?.*)?$"
)


@dataclass(frozen=True)
class VertexPath:
  prefix: str
  model: str
  method: str
  query: str

  @property
  def streaming(self) -> bool:
    return self.method == ":streamGenerateContent"

  def with_model(self, model: str) -> str:
    return f"{self.prefix}{model}{self.method}{self.query}"


def parse_path(path: str) -> VertexPath | None:
  m = _PATH_RE.match(path or "")
  if not m:
    return None
  return VertexPath(m["prefix"], m["model"], m["method"], m["query"] or "")


# --------------------------------------------------------------------------
# Request features
# --------------------------------------------------------------------------

def _get(d: dict, camel: str, snake: str, default: Any = None) -> Any:
  """Proto-JSON accepts both camelCase and snake_case field names."""
  if camel in d:
    return d[camel]
  return d.get(snake, default)


def _key(d: dict, camel: str, snake: str) -> str:
  return snake if snake in d and camel not in d else camel


def _parts_text(parts: list) -> str:
  return "\n".join(p["text"] for p in parts or [] if isinstance(p, dict) and isinstance(p.get("text"), str))


@dataclass(frozen=True)
class Features:
  turns: int
  tool_declarations: int
  media_parts: int
  est_input_tokens: int
  last_turn_kind: str  # "user_text" | "tool_result" | "model" | "empty"
  system_text: str
  last_user_text: str


def extract_features(body: dict) -> Features:
  contents = body.get("contents") or []
  if isinstance(contents, dict):  # single Content is tolerated by some clients
    contents = [contents]

  total_chars = 0
  media = 0
  for c in contents:
    for p in (c or {}).get("parts") or []:
      if not isinstance(p, dict):
        continue
      if isinstance(p.get("text"), str):
        total_chars += len(p["text"])
      elif any(k in p for k in ("inlineData", "inline_data", "fileData", "file_data")):
        media += 1
      else:  # functionCall / functionResponse / executableCode ...
        total_chars += len(str(p))

  system = _get(body, "systemInstruction", "system_instruction") or {}
  system_text = _parts_text(system.get("parts") if isinstance(system, dict) else [])
  total_chars += len(system_text)

  tools = body.get("tools") or []
  decls = 0
  for t in tools:
    if isinstance(t, dict):
      decls += len(_get(t, "functionDeclarations", "function_declarations") or [])
      total_chars += len(str(t))

  last_user_text, kind = "", "empty"
  if contents:
    last = contents[-1] or {}
    parts = last.get("parts") or []
    if last.get("role") == "model":
      kind = "model"
    elif any(isinstance(p, dict) and ("functionResponse" in p or "function_response" in p) for p in parts):
      kind = "tool_result"
    else:
      kind = "user_text"
  for c in reversed(contents):
    if (c or {}).get("role", "user") == "user":
      text = _parts_text(c.get("parts"))
      if text:
        last_user_text = text
        break

  return Features(
      turns=len(contents),
      tool_declarations=decls,
      media_parts=media,
      est_input_tokens=total_chars // 4,
      last_turn_kind=kind,
      system_text=system_text,
      last_user_text=last_user_text,
  )


# --------------------------------------------------------------------------
# Redaction + excerpt
# --------------------------------------------------------------------------

_REDACTIONS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|ya29\.[0-9A-Za-z_\-]+)"), "[SECRET]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"), "[JWT]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[NUMBER]"),
    (re.compile(r"(?<!\w)\+?\d[\d ().-]{8,}\d\b"), "[PHONE]"),
]


def redact(text: str) -> str:
  for pattern, repl in _REDACTIONS:
    text = pattern.sub(repl, text)
  return text


def _head_tail(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  if limit < 40:
    return text[:limit]
  head = int(limit * 0.6)
  tail = limit - head - 7
  return f"{text[:head]}\n[...]\n{text[-tail:]}"


def build_excerpt(f: Features, max_chars: int, do_redact: bool = True) -> str:
  sys_budget = max_chars // 4
  user_budget = max_chars - sys_budget
  system = _head_tail(f.system_text, sys_budget) or "(none)"
  user = _head_tail(f.last_user_text, user_budget) or "(none)"
  if do_redact:
    system, user = redact(system), redact(user)
  return (
      "[Request metadata]\n"
      f"conversation_turns: {f.turns}\n"
      f"tool_declarations: {f.tool_declarations}\n"
      f"media_parts: {f.media_parts}\n"
      f"estimated_input_tokens: {f.est_input_tokens}\n"
      f"latest_turn: {f.last_turn_kind}\n\n"
      "[System instruction, truncated]\n"
      f"{system}\n\n"
      "[Latest user message, truncated]\n"
      f"{user}\n"
  )


# --------------------------------------------------------------------------
# thinkingConfig mutation
# --------------------------------------------------------------------------

def set_thinking_level(body: dict, level: str, case: str = "upper") -> None:
  """Sets generationConfig.thinkingConfig.thinkingLevel in place.

  Removes thinkingBudget: Gemini 3+ rejects requests that set both.
  """
  gc_key = _key(body, "generationConfig", "generation_config")
  gc = body.setdefault(gc_key, {})
  tc_key = _key(gc, "thinkingConfig", "thinking_config")
  tc = gc.setdefault(tc_key, {})
  tc.pop("thinkingBudget", None)
  tc.pop("thinking_budget", None)
  tc.pop("thinking_level", None)
  tc["thinkingLevel"] = level.lower() if case == "lower" else level.upper()
