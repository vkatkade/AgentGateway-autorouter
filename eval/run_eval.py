"""Offline eval: run the labeled prompts through the real Decisions API + routing policy.

  GEMINI_API_KEY=... python -m eval.run_eval [--sweep] [--file eval/prompts.jsonl]

Prints routing accuracy, the pass-through (uncertain) rate, thinking-level accuracy,
classifier latency p50/p95, and with --sweep, how accuracy and coverage move across
ARGON_THRESHOLD / HYDROGEN_THRESHOLD pairs (one API call per prompt, re-scored offline).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace

from autorouter.classifier import ClassifierError, DecisionsClassifier
from autorouter.config import Config
from autorouter.policy import decide
from autorouter.vertex import extract_features


class _Replay:
  """Feeds a cached Classification back into decide() for threshold sweeps."""

  def __init__(self, c):
    self.c = c

  def classify(self, excerpt, ask_criteria=True):
    if isinstance(self.c, ClassifierError):
      raise self.c
    return self.c


def to_body(row: dict) -> dict:
  body = {"contents": [{"role": "user", "parts": [{"text": row["prompt"]}]}]}
  if row.get("system"):
    body["systemInstruction"] = {"parts": [{"text": row["system"]}]}
  if row.get("tools"):
    body["tools"] = [{"functionDeclarations": [{"name": f"tool_{i}"} for i in range(row["tools"])]}]
  return body


def label(cfg: Config, model: str | None) -> str:
  return {cfg.argon_model: "argon", cfg.hydrogen_model: "hydrogen"}.get(model or "", "pass")


def score(cfg: Config, rows, cached):
  hits = routed = think_hits = think_n = 0
  details = []
  for row, c in zip(rows, cached):
    d = decide(cfg, cfg.hydrogen_model, extract_features(to_body(row)), _Replay(c))
    got = label(cfg, d.signals.get("target_model")) if d.reason not in ("uncertain", "classifier_error", "classifier_empty") else "pass"
    want = row["expect_model"]
    if got != "pass":
      routed += 1
      hits += want in (got, "either")
    if d.thinking_level and row.get("expect_thinking"):
      think_n += 1
      think_hits += d.thinking_level == row["expect_thinking"]
    details.append((row["id"], want, got, d.thinking_level, row.get("expect_thinking"), d.reason, d.signals.get("p_argon")))
  return {
      "routing_acc": hits / routed if routed else 0.0,
      "coverage": routed / len(rows),
      "thinking_acc": think_hits / think_n if think_n else 0.0,
      "details": details,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--file", default="eval/prompts.jsonl")
  ap.add_argument("--sweep", action="store_true")
  args = ap.parse_args()

  cfg = Config.from_env()
  if not cfg.gemini_api_key:
    print("Set GEMINI_API_KEY", file=sys.stderr)
    return 2
  # Eval isn't latency-bound; give the API room so timeouts don't skew accuracy.
  clf = DecisionsClassifier(replace(cfg, decisions_timeout_ms=max(cfg.decisions_timeout_ms, 5000)))
  rows = [json.loads(l) for l in open(args.file) if l.strip()]

  from autorouter.vertex import build_excerpt
  cached, lat = [], []
  for row in rows:
    f = extract_features(to_body(row))
    forced = f.est_input_tokens >= cfg.long_context_tokens or f.tool_declarations >= cfg.many_tools
    try:
      c = clf.classify(build_excerpt(f, cfg.excerpt_max_chars, cfg.redact), ask_criteria=not forced)
      lat.append(c.latency_ms)
    except ClassifierError as e:
      c = e
    cached.append(c)

  res = score(cfg, rows, cached)
  print(f"{'id':<14}{'want':<10}{'got':<10}{'think':<8}{'want':<8}{'p_argon':<9}reason")
  for i, want, got, t, wt, reason, p in res["details"]:
    flag = "" if want in (got, "either") or got == "pass" else "  <-- miss"
    print(f"{i:<14}{want:<10}{got:<10}{t or '-':<8}{wt or '-':<8}{p if p is not None else '-':<9}{reason}{flag}")
  print(f"\nrouting accuracy (routed only): {res['routing_acc']:.0%}   coverage: {res['coverage']:.0%}   "
        f"thinking accuracy: {res['thinking_acc']:.0%}")
  if lat:
    q = statistics.quantiles(lat, n=20) if len(lat) >= 2 else lat * 19
    print(f"classifier latency ms: p50={statistics.median(lat):.0f} p95={q[18]:.0f} "
          f"(budget {cfg.decisions_timeout_ms} ms)")

  if args.sweep:
    print("\nargon_thr hydro_thr  acc   coverage")
    for hi in (0.55, 0.6, 0.65, 0.7, 0.8):
      for lo in (0.2, 0.3, 0.35, 0.45):
        if lo >= hi:
          continue
        r = score(replace(cfg, argon_threshold=hi, hydrogen_threshold=lo), rows, cached)
        print(f"{hi:<10}{lo:<10} {r['routing_acc']:.0%}  {r['coverage']:.0%}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
