"""Turns request features + classifier output into a routing decision.

Rules, in order:
  1. Opt-out header, unknown path, or a model outside ROUTABLE_MODELS -> pass through.
  2. Long context or many tool declarations -> Argon (deterministic, no model call needed).
  3. Otherwise ask the Decisions API. max P(Argon criterion):
       >= ARGON_THRESHOLD    -> Argon
       <= HYDROGEN_THRESHOLD -> Hydrogen
       in between            -> pass through unchanged (not confident)
     A classifier error or timeout -> pass through unchanged.
  4. Thinking level = the `thinking_level` choice (LOW / MEDIUM / HIGH) if its confidence
     >= THINKING_MIN_CONFIDENCE, clamped to the levels the chosen model supports (ties
     round up); otherwise left as the agent sent it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autorouter.classifier import THINKING_LEVELS, Classification, Classifier, ClassifierError
from autorouter.config import Config
from autorouter.vertex import Features, build_excerpt


@dataclass
class Decision:
  model: str | None = None           # None -> keep the requested model
  thinking_level: str | None = None  # None -> keep the agent's thinkingConfig
  reason: str = "pass_through"
  signals: dict = field(default_factory=dict)

  @property
  def changes_request(self) -> bool:
    return self.model is not None or self.thinking_level is not None


def _clamp_level(level: str, supported: tuple[str, ...]) -> str:
  if level in supported:
    return level
  order = list(THINKING_LEVELS)
  want = order.index(level)
  # Nearest supported level; ties go to the higher level (favor quality).
  return min(supported, key=lambda l: (abs(order.index(l) - want), -order.index(l)))


def _thinking(cfg: Config, c: Classification, model: str) -> str | None:
  if c.thinking_level not in THINKING_LEVELS or c.thinking_confidence < cfg.thinking_min_confidence:
    return None
  return _clamp_level(c.thinking_level, cfg.levels_for(model))


def decide(cfg: Config, requested_model: str, f: Features, classifier: Classifier) -> Decision:
  if requested_model not in cfg.all_routable:
    return Decision(reason="model_not_routable")

  forced = None
  if f.est_input_tokens >= cfg.long_context_tokens:
    forced = "long_context"
  elif f.tool_declarations >= cfg.many_tools:
    forced = "many_tools"

  excerpt = build_excerpt(f, cfg.excerpt_max_chars, cfg.redact)
  try:
    c = classifier.classify(excerpt, ask_criteria=forced is None)
  except ClassifierError as e:
    if forced:  # model choice doesn't need the classifier; keep thinking as sent
      return Decision(model=cfg.argon_model, reason=forced, signals={"classifier_error": str(e)})
    return Decision(reason="classifier_error", signals={"classifier_error": str(e)})

  signals = {
      "criteria": {k: round(v, 3) for k, v in c.criteria.items()},
      "thinking_choice": c.thinking_level,
      "thinking_confidence": round(c.thinking_confidence, 3),
      "classifier_ms": round(c.latency_ms, 1),
  }

  if forced:
    model, reason = cfg.argon_model, forced
  else:
    p_max = max(c.criteria.values(), default=None)
    if p_max is None:
      return Decision(reason="classifier_empty", signals=signals)
    signals["p_argon"] = round(p_max, 3)
    if p_max >= cfg.argon_threshold:
      top = max(c.criteria, key=c.criteria.get)
      model, reason = cfg.argon_model, f"argon:{top}"
    elif p_max <= cfg.hydrogen_threshold:
      model, reason = cfg.hydrogen_model, "hydrogen:simple"
    else:
      return Decision(reason="uncertain", signals=signals)

  return Decision(
      model=None if model == requested_model else model,
      thinking_level=_thinking(cfg, c, model),
      reason=reason,
      signals=signals | {"target_model": model},
  )
