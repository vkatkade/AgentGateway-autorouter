"""The experiment arms and how each one turns a request into the request sent to Vertex.

  argon               every call -> Argon, the harness's own thinking settings   (quality ceiling)
  hydrogen            every call -> Hydrogen                                      (cost floor)
  argon_medium        every call -> Argon with thinkingLevel MEDIUM               (fixed-thinking baseline)
  router              the deployed router decides model + thinking level          (system under test)
  router_no_thinking  the router picks the model; the harness's own thinking
                      settings are restored                                       (ablation of the thinking logic)

All arms share the same prompts, tools, sampler settings and (outside the proxy) user simulator.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Protocol

from autorouter.config import Config
from autorouter.policy import decide
from autorouter.vertex import extract_features, parse_path, set_thinking_level
from eval.agentic.extproc_client import RouteResult


@dataclass(frozen=True)
class ArmSpec:
  name: str
  description: str
  model: str | None = None         # "argon" | "hydrogen" | None (router decides / passthrough)
  thinking: str | None = None      # fixed thinkingLevel
  use_router: bool = False
  keep_agent_thinking: bool = False


ARMS: dict[str, ArmSpec] = {a.name: a for a in (
    ArmSpec("argon", "Always Argon, harness thinking settings", model="argon"),
    ArmSpec("hydrogen", "Always Hydrogen, harness thinking settings", model="hydrogen"),
    ArmSpec("argon_medium", "Always Argon, thinkingLevel MEDIUM", model="argon", thinking="MEDIUM"),
    ArmSpec("router", "Router: model + thinking level", use_router=True),
    ArmSpec("router_no_thinking", "Router model choice, harness thinking settings restored",
            use_router=True, keep_agent_thinking=True),
)}
BASELINE_ARM = "argon"


class Backend(Protocol):
  mode: str

  def route(self, path: str, body: bytes, opt_out: bool = False) -> RouteResult: ...


class InProcessRouter:
  """Runs the policy in this process (no gRPC hop). For fast iteration and offline tests; the
  headline numbers should come from the deployed router (RemoteRouter)."""
  mode = "inprocess"

  def __init__(self, cfg: Config, classifier=None):
    from autorouter.classifier import DecisionsClassifier
    self.cfg = cfg
    self.classifier = classifier or DecisionsClassifier(cfg)

  def route(self, path: str, body: bytes, opt_out: bool = False) -> RouteResult:
    import time
    vp = parse_path(path)
    if vp is None or opt_out:
      return RouteResult(path, body, {"x-autorouter-reason": "skipped"}, 0.0)
    try:
      req = json.loads(body)
    except ValueError:
      return RouteResult(path, body, {"x-autorouter-reason": "bad_json"}, 0.0)
    t0 = time.perf_counter()
    d = decide(self.cfg, vp.model, extract_features(req), self.classifier)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    headers = {"x-autorouter-reason": d.reason}
    if self.cfg.shadow_mode or not d.changes_request:
      return RouteResult(path, body, headers, ms, signals=d.signals)
    if d.thinking_level:
      set_thinking_level(req, d.thinking_level, self.cfg.thinking_level_case)
      headers["x-autorouter-thinking"] = d.thinking_level
    new_path = vp.with_model(d.model) if d.model else path
    headers["x-autorouter-model"] = d.model or vp.model
    return RouteResult(new_path, json.dumps(req, separators=(",", ":"), ensure_ascii=False).encode(),
                       headers, ms, signals=d.signals)


@dataclass
class Outcome:
  path: str
  body: dict
  route: RouteResult | None = None


def _gc_key(d: dict) -> str:
  return "generation_config" if "generation_config" in d and "generationConfig" not in d else "generationConfig"


def restore_thinking(new: dict, orig: dict) -> None:
  """Puts the harness's own thinkingConfig back (drops whatever the router set)."""
  o_gc = orig.get(_gc_key(orig)) or {}
  o_tc = o_gc.get("thinkingConfig", o_gc.get("thinking_config"))
  key = _gc_key(new)
  n_gc = new.setdefault(key, {})
  n_gc.pop("thinkingConfig", None)
  n_gc.pop("thinking_config", None)
  if o_tc is not None:
    n_gc["thinkingConfig"] = copy.deepcopy(o_tc)
  if not n_gc:
    new.pop(key, None)


def apply_arm(arm: ArmSpec, cfg: Config, backend: Backend | None, path: str, body: dict) -> Outcome:
  vp = parse_path(path)
  if vp is None:
    raise ValueError(f"not a Vertex generateContent path: {path}")

  if not arm.use_router:
    target = {"argon": cfg.argon_model, "hydrogen": cfg.hydrogen_model}.get(arm.model or "", vp.model)
    new = copy.deepcopy(body)
    if arm.thinking:
      set_thinking_level(new, arm.thinking, cfg.thinking_level_case)
    return Outcome(vp.with_model(target), new)

  if backend is None:
    raise ValueError(f"arm {arm.name} needs a router backend")
  raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
  res = backend.route(path, raw)
  new = json.loads(res.body)
  if arm.keep_agent_thinking:
    restore_thinking(new, body)
  return Outcome(res.path, new, res)
