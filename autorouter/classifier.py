"""Gemini Decisions API classifier.

One POST to /v1beta/decisions scores all routing questions against the same
excerpt in a single forward pass, so adding questions barely moves latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from autorouter.config import Config

# Each yes/no question is one Argon criterion. Keys are never shown to the model.
ARGON_CRITERIA: dict[str, str] = {
    "multistep_reasoning": (
        "Does answering the latest user message require multi-step reasoning, planning, "
        "mathematics, or careful logical deduction, rather than a direct lookup, a short "
        "rewrite, simple extraction, or small talk?"
    ),
    "code_work": (
        "Does the latest user message require writing non-trivial code, debugging, "
        "root-causing an error or stack trace, or reviewing or refactoring code?"
    ),
    "high_stakes": (
        "Is this a high-stakes legal, financial, security, or compliance analysis where a "
        "subtle mistake could cause material harm?"
    ),
    "agentic_tool_heavy": (
        "Is this an agent step that must orchestrate several tools or reason over long, "
        "complex tool results to decide what to do next?"
    ),
}

THINKING_LEVELS = ("LOW", "MEDIUM", "HIGH")
THINKING_QUESTION = {
    "type": "score",
    "instructions": (
        "How much deliberate reasoning does a strong model need before it answers the "
        "latest user message well?"
    ),
    "criteria": [
        "Little: the answer is direct, routine, or a lookup.",
        "Some: a few reasoning steps or moderate analysis.",
        "Deep: hard, multi-step, or long-horizon reasoning.",
    ],
}


@dataclass
class Classification:
  criteria: dict[str, float] = field(default_factory=dict)  # P(yes) per Argon criterion
  thinking_score: float | None = None  # 0..2, may fall between levels
  thinking_confidence: float = 0.0
  latency_ms: float = 0.0


class ClassifierError(Exception):
  pass


class Classifier(Protocol):
  def classify(self, excerpt: str, ask_criteria: bool = True) -> Classification: ...


def build_questions(ask_criteria: bool = True) -> dict:
  questions = {"thinking_depth": THINKING_QUESTION}
  if ask_criteria:
    for key, text in ARGON_CRITERIA.items():
      questions[key] = {"type": "noul", "instructions": text}
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
  answers = data.get("answers") or {}
  out = Classification(latency_ms=latency_ms)
  for key in ARGON_CRITERIA:
    a = answers.get(key)
    if a and "noul" in a:
      out.criteria[key] = float(a["noul"])
  t = answers.get("thinking_depth")
  if t and "score" in t:
    out.thinking_score = float(t["score"])
    out.thinking_confidence = float(t.get("confidence", 0.0))
  return out
