"""Interactive check of the auto-router's Decisions logic against the live API.

Sends the exact Choice questions the service extension sends
(autorouter.classifier.build_questions) and applies the same thresholds and
thinking-level rules (autorouter.config / autorouter.policy), so what you see
here is what the gateway would do for a single-turn request.

    pip install -r requirements-dev.txt      # google-genai needs Python 3.11+
    export GEMINI_API_KEY=...                # never hard-code the key
    python test_ir.py                        # prompts for queries until you quit
    python test_ir.py "your prompt here"     # one-shot
"""

from __future__ import annotations

import sys

from google import genai

from autorouter.classifier import build_questions, parse_answers
from autorouter.config import Config
from autorouter.policy import choose_level


def _plain(obj):
  """SDK answer objects -> plain dicts, so the extension's parser can read them."""
  if isinstance(obj, dict):
    return {k: _plain(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_plain(v) for v in obj]
  if hasattr(obj, "model_dump"):
    return _plain(obj.model_dump(exclude_none=True))
  if hasattr(obj, "__dict__"):
    return _plain({k: v for k, v in vars(obj).items() if not k.startswith("_")})
  return obj


def decide_and_print(client, cfg: Config, prompt: str) -> None:
  result = client.decisions.create(model=cfg.decisions_model, input=prompt,
                                   questions=build_questions())
  answers = _plain(result.answers)

  print("\nRaw answers:")
  for key, answer in answers.items():
    print(f"  {key}: {answer}")

  c = parse_answers({"answers": answers})
  if not c.criteria:
    print("  -> PASS THROUGH (no criterion probabilities in the response)")
    return
  top = max(c.criteria, key=c.criteria.get)
  p_max = c.criteria[top]

  print("\nDecision:")
  for q, p in c.criteria.items():
    print(f"  {q:22s} P(yes)={p:.3f}{'  <- top' if q == top else ''}")
  print(f"  p_max={p_max:.3f}  (Argon >= {cfg.argon_threshold}, Hydrogen <= {cfg.hydrogen_threshold})")
  if p_max >= cfg.argon_threshold:
    model = cfg.argon_model
  elif p_max <= cfg.hydrogen_threshold:
    model = cfg.hydrogen_model
  else:
    print("  -> PASS THROUGH (not confident; request left unchanged)")
    return
  print(f"  -> {'ARGON' if model == cfg.argon_model else 'HYDROGEN'}: {model}")

  if c.p_deep is not None:
    print(f"  argon_depth P(deep)={c.p_deep:.3f}  (HIGH on Argon >= {cfg.argon_high_at})")
  level, source = choose_level(cfg, c, model, agent=None)
  if level:
    print(f"  -> thinkingLevel={level} (source: {source})")
  else:
    print("  -> thinkingLevel unchanged (no thinking signal)")


def main() -> None:
  cfg = Config.from_env()  # same env vars as the Cloud Run service
  # Decisions is only served on v1beta; the SDK default version returns 404.
  client = genai.Client(http_options={"api_version": "v1beta"})  # reads GEMINI_API_KEY

  if len(sys.argv) > 1:
    decide_and_print(client, cfg, " ".join(sys.argv[1:]))
    return

  print("Autorouter test. Type a query and press Enter; empty line, 'exit' or Ctrl-D to quit.")
  while True:
    try:
      prompt = input("\nQuery> ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      break
    if not prompt or prompt.lower() in ("exit", "quit", "q"):
      break
    try:
      decide_and_print(client, cfg, prompt)
    except Exception as e:  # keep the loop alive on API errors
      print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
  main()
