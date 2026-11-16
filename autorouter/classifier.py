"""Gemini Decisions API classifier (Choice questions only).

One POST to /v1beta/decisions scores all routing questions against the same
excerpt in a single forward pass, so adding questions barely moves latency.

Every question is a `choice`:
  * one two-option (yes/no) question per Argon criterion -> P(yes) per criterion;
  * one three-option `thinking_level` question (LOW / MEDIUM / HIGH), used for Hydrogen;
  * one two-option `argon_depth` question (deep / moderate), used for Argon, where only
    MEDIUM vs HIGH is in play and two options give more decisive probabilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from autorouter.config import Config

# Each two-option question is one Argon criterion. Keys are never shown to the model.
ARGON_CRITERIA: dict[str, dict] = {
    "multistep_reasoning": {
        "instructions": (
            "Does answering the latest user message require multi-step reasoning, planning, "
            "mathematics, or careful logical deduction, rather than a direct lookup, a short "
            "rewrite, simple extraction, or small talk?"
        ),
        "yes": "Several dependent reasoning, planning or calculation steps.",
        "no": "A direct answer, lookup, rewrite, summary or chat.",
    },
    "code_work": {
        "instructions": (
            "Does the latest user message require writing non-trivial code, debugging, "
            "root-causing an error or stack trace, or reviewing or refactoring code?"
        ),
        "yes": "Writing, fixing, refactoring or reviewing real code.",
        "no": "No code, or only a trivial snippet.",
    },
    "high_stakes": {
        "instructions": (
            "Is this a high-stakes legal, financial, security, or compliance analysis where a "
            "subtle mistake could cause material harm?"
        ),
        "yes": "Legal, financial, security or similar consequential analysis.",
        "no": "Everyday, low-consequence content.",
    },
    "agentic_tool_heavy": {
        "instructions": (
            "Is this an agent step that must orchestrate several tools or reason over long, "
            "complex tool results to decide what to do next?"
        ),
        "yes": "Coordinating several tools or chained tool calls.",
        "no": "No tools, or a single simple tool call.",
    },
}

THINKING_LEVELS = ("LOW", "MEDIUM", "HIGH")
THINKING_KEY = "thinking_level"
THINKING_QUESTION = {
    "type": "choice",
    "instructions": (
        "How much deliberate reasoning does a strong model need before it answers the "
        "latest user message well?"
    ),
    "criteria": {
        "LOW": "Little: the answer is direct, routine, or a lookup.",
        "MEDIUM": "Some: a few reasoning steps or moderate analysis.",
        "HIGH": "Deep: hard, multi-step, or long-horizon reasoning.",
    },
}


ARGON_DEPTH_KEY = "argon_depth"
ARGON_DEPTH_QUESTION = {
    "type": "choice",
    "instructions": (
        "Does a strong answer to the latest user message need deep, multi-step reasoning "
        "beyond a moderate amount?"
    ),
    "criteria": {
        "deep": "Hard, multi-step or long-horizon reasoning, or careful verification of a subtle result.",
        "moderate": "A few reasoning steps or moderate analysis is enough.",
    },
}


@dataclass
class Classification:
  criteria: dict[str, float] = field(default_factory=dict)  # P(yes) per Argon criterion
  thinking_level: str | None = None  # LOW / MEDIUM / HIGH, as chosen by the model
  thinking_confidence: float = 0.0
  thinking_probs: dict[str, float] = field(default_factory=dict)  # LOW/MEDIUM/HIGH distribution
  p_deep: float | None = None  # P("deep") from argon_depth
  latency_ms: float = 0.0


class ClassifierError(Exception):
  pass


class Classifier(Protocol):
  def classify(self, excerpt: str, ask_criteria: bool = True) -> Classification: ...


def build_questions(ask_criteria: bool = True) -> dict:
  questions = {THINKING_KEY: THINKING_QUESTION, ARGON_DEPTH_KEY: ARGON_DEPTH_QUESTION}
  if ask_criteria:
    for key, q in ARGON_CRITERIA.items():
      questions[key] = {
          "type": "choice",
          "instructions": q["instructions"],
          "criteria": {"yes": q["yes"], "no": q["no"]},
      }
  return questions


class DecisionsClassifier:
  """Thread-safe REST client; keeps a pooled HTTP/2 connection to cut handshake time."""

  def __init__(self, cfg: Config, client: httpx.Client | None = None):
    if not cfg.gemini_api_key and client is None:
      logging.warning("GEMINI_API_KEY is unset; every request will pass through.")
    self._cfg = cfg
    timeout = cfg.decisions_timeout_ms / 1000.0
    self._client = client or httpx.Client(
        http2=True,
        timeout=httpx.Timeout(timeout, connect=timeout),
        headers={"x-goog-api-key": cfg.gemini_api_key, "content-type": "application/json"},
        limits=httpx.Limits(max_keepalive_connections=32, keepalive_expiry=120),
    )

  def classify(self, excerpt: str, ask_criteria: bool = True) -> Classification:
    if not self._client.headers.get("x-goog-api-key"):
      raise ClassifierError("no API key")
    payload = {
        "model": self._cfg.decisions_model,
        "input": excerpt,
        "questions": build_questions(ask_criteria),
    }
    start = time.perf_counter()
    try:
      resp = self._client.post(self._cfg.decisions_endpoint, json=payload)
    except httpx.HTTPError as e:  # timeouts included
      raise ClassifierError(f"{type(e).__name__}: {e}") from e
    latency = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
      code = ""
      try:
        code = resp.json().get("error", {}).get("code", "")
      except ValueError:
        pass
      raise ClassifierError(f"HTTP {resp.status_code} {code}".strip())
    return parse_answers(resp.json(), latency)


def parse_answers(data: dict, latency_ms: float = 0.0) -> Classification:
  """Choice answers: {"choice": ..., "confidence": ..., "probabilities": {option: p}}."""
  answers = data.get("answers") or {}
  out = Classification(latency_ms=latency_ms)
  for key in ARGON_CRITERIA:
    probs = (answers.get(key) or {}).get("probabilities") or {}
    if "yes" in probs:
      out.criteria[key] = float(probs["yes"])
  t = answers.get(THINKING_KEY) or {}
  level = str(t.get("choice", "")).upper()
  if level in THINKING_LEVELS:
    out.thinking_level = level
    out.thinking_confidence = float(t.get("confidence", 0.0))
  probs = {str(k).upper(): float(v) for k, v in (t.get("probabilities") or {}).items()
           if str(k).upper() in THINKING_LEVELS}
  if not probs and out.thinking_level:  # tolerate answers without a distribution
    probs = {out.thinking_level: out.thinking_confidence}
  out.thinking_probs = probs
  deep = ((answers.get(ARGON_DEPTH_KEY) or {}).get("probabilities") or {}).get("deep")
  if deep is not None:
    out.p_deep = float(deep)
  return out
