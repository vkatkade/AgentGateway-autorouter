"""Turns request features + classifier output into a routing decision.

Rules, in order:
  1. Opt-out header, unknown path, or a model outside ROUTABLE_MODELS -> pass through.
  2. Long context or many tool declarations -> Argon (deterministic, no model call needed).
  3. Otherwise ask the Decisions API. max P(Argon criterion):
       >= ARGON_THRESHOLD    -> Argon
       <= HYDROGEN_THRESHOLD -> Hydrogen
       in between            -> pass through unchanged (not confident)
     A classifier error or timeout -> pass through unchanged.
  4. Thinking level (see choose_level):
       Argon    -> always explicit: HIGH if P(deep) >= ARGON_HIGH_AT, else MEDIUM; raised to HIGH
                   when multistep_reasoning or high_stakes P(yes) >= THINKING_FLOOR_HIGH_AT.
       Hydrogen -> LOW / HIGH when that option's probability >= THINKING_MIN_CONFIDENCE, else MEDIUM.
     With THINKING_RESPECT_AGENT=raise_only the level is never below the agent's own
     thinkingLevel. The result is clamped to the levels the chosen model supports.
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


_RANK = {l: i for i, l in enumerate(THINKING_LEVELS)}
FLOOR_HIGH_CRITERIA = ("multistep_reasoning", "high_stakes")


def choose_level(cfg: Config, c: Classification | None, model: str,
                 agent: str | None) -> tuple[str | None, str | None]:
  """Thinking level for the routed model. Returns (level, source)."""
  if model == cfg.argon_model:
    level, source = "MEDIUM", "argon_default"  # Argon always gets an explicit level
    if c is not None and c.p_deep is not None:
      source = "argon_depth"
      if c.p_deep >= cfg.argon_high_at:
        level = "HIGH"
    if level != "HIGH" and c is not None:
      for k in FLOOR_HIGH_CRITERIA:
        if c.criteria.get(k, 0.0) >= cfg.thinking_floor_high_at:
          level, source = "HIGH", f"floor:{k}"
          break
  else:
    probs = c.thinking_probs if c is not None else {}
    if not probs:
      return None, None  # no signal: keep the agent's thinkingConfig
    if probs.get("LOW", 0.0) >= cfg.thinking_min_confidence:
      level = "LOW"
    elif probs.get("HIGH", 0.0) >= cfg.thinking_min_confidence:
      level = "HIGH"
    else:
      level = "MEDIUM"
    source = "hydrogen_choice"
  if agent in _RANK and cfg.thinking_respect_agent == "raise_only" and _RANK[agent] > _RANK[level]:
    level, source = agent, "agent"
  return _clamp_level(level, cfg.levels_for(model)), source


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
    if forced:  # model choice doesn't need the classifier; Argon gets its MEDIUM floor
      level, source = choose_level(cfg, None, cfg.argon_model, f.agent_thinking_level)
      return Decision(model=None if requested_model == cfg.argon_model else cfg.argon_model,
                      thinking_level=level, reason=forced,
                      signals={"classifier_error": str(e), "thinking_source": source,
                               "target_model": cfg.argon_model})
    return Decision(reason="classifier_error", signals={"classifier_error": str(e)})

  signals = {
      "criteria": {k: round(v, 3) for k, v in c.criteria.items()},
      "thinking_choice": c.thinking_level,
      "p_deep": None if c.p_deep is None else round(c.p_deep, 3),
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

  level, source = choose_level(cfg, c, model, f.agent_thinking_level)
  return Decision(
      model=None if model == requested_model else model,
      thinking_level=level,
      reason=reason,
      signals=signals | {"target_model": model, "thinking_source": source},
  )
