"""Runtime configuration, read from environment variables (Cloud Run env / secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

VALID_LEVELS = ("LOW", "MEDIUM", "HIGH")


def _env(name: str, default: str) -> str:
  return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
  return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
  return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
  return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


def _env_levels(name: str, default: str) -> tuple[str, ...]:
  levels = tuple(l.strip().upper() for l in _env(name, default).split(",") if l.strip())
  bad = [l for l in levels if l not in VALID_LEVELS]
  if not levels:
    raise ValueError(f"{name}: at least one thinking level is required")
  if bad:
    raise ValueError(f"{name}: unsupported thinking levels {bad}; allowed {VALID_LEVELS}")
  return levels


@dataclass(frozen=True)
class Config:
  # --- Models (Vertex model IDs as they appear in the generateContent path) ---
  argon_model: str = "gemini-4-argon"
  # Stand-in until Gemini Hydrogen has a public Vertex model ID.
  hydrogen_model: str = "gemini-3.8-flash"
  # Only requests whose path already names one of these models are re-routed.
  # Anything else passes through untouched.
  routable_models: frozenset[str] = field(default_factory=frozenset)
  argon_levels: tuple[str, ...] = VALID_LEVELS
  hydrogen_levels: tuple[str, ...] = VALID_LEVELS

  # --- Decisions API ---
  decisions_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/decisions"
  decisions_model: str = "gemini-decision-1"
  gemini_api_key: str = ""
  # Leaves ~50 ms of the 300 ms budget for parsing, rewriting and the gateway hop.
  decisions_timeout_ms: int = 250

  # --- Policy ---
  argon_threshold: float = 0.65     # max P(argon criterion) >= this -> Argon
  hydrogen_threshold: float = 0.35  # max P(argon criterion) <= this -> Hydrogen
  thinking_min_confidence: float = 0.5
  long_context_tokens: int = 32_000  # deterministic Argon rule
  many_tools: int = 8                # deterministic Argon rule

  # --- Excerpt sent to the classifier ---
  excerpt_max_chars: int = 6_000
  redact: bool = True

  # --- Request rewriting ---
  # Proto-JSON enum spelling for thinkingLevel ("upper" -> "HIGH", "lower" -> "high").
  thinking_level_case: str = "upper"
  # Log the decision but do not mutate the request (rollout / evaluation).
  shadow_mode: bool = False

  # --- Server ---
  port: int = 8080
  # Each in-flight ext_proc stream holds a thread for its lifetime.
  server_threads: int = 64

  @property
  def all_routable(self) -> frozenset[str]:
    return self.routable_models or frozenset({self.argon_model, self.hydrogen_model})

  def levels_for(self, model: str) -> tuple[str, ...]:
    return self.argon_levels if model == self.argon_model else self.hydrogen_levels

  @classmethod
  def from_env(cls) -> "Config":
    routable = _env("ROUTABLE_MODELS", "")
    return cls(
        argon_model=_env("ARGON_MODEL", cls.argon_model),
        hydrogen_model=_env("HYDROGEN_MODEL", cls.hydrogen_model),
        routable_models=frozenset(m.strip() for m in routable.split(",") if m.strip()),
        argon_levels=_env_levels("ARGON_THINKING_LEVELS", "LOW,MEDIUM,HIGH"),
        hydrogen_levels=_env_levels("HYDROGEN_THINKING_LEVELS", "LOW,MEDIUM,HIGH"),
        decisions_endpoint=_env("DECISIONS_ENDPOINT", cls.decisions_endpoint),
        decisions_model=_env("DECISIONS_MODEL", cls.decisions_model),
        gemini_api_key=_env("GEMINI_API_KEY", ""),
        decisions_timeout_ms=_env_int("DECISIONS_TIMEOUT_MS", cls.decisions_timeout_ms),
        argon_threshold=_env_float("ARGON_THRESHOLD", cls.argon_threshold),
        hydrogen_threshold=_env_float("HYDROGEN_THRESHOLD", cls.hydrogen_threshold),
        thinking_min_confidence=_env_float("THINKING_MIN_CONFIDENCE", cls.thinking_min_confidence),
        long_context_tokens=_env_int("LONG_CONTEXT_TOKENS", cls.long_context_tokens),
        many_tools=_env_int("MANY_TOOLS", cls.many_tools),
        excerpt_max_chars=_env_int("EXCERPT_MAX_CHARS", cls.excerpt_max_chars),
        redact=_env_bool("REDACT", cls.redact),
        thinking_level_case=_env("THINKING_LEVEL_CASE", cls.thinking_level_case).lower(),
        shadow_mode=_env_bool("SHADOW_MODE", cls.shadow_mode),
        port=_env_int("PORT", cls.port),
        server_threads=_env_int("SERVER_THREADS", cls.server_threads),
    )
