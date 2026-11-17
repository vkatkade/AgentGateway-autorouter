"""Compare arms: quality, cost, routing behaviour and reliability, with paired statistics.

  python -m eval.agentic.analyze runs/tau3_banking-* runs/swebench_verified-* \\
      --baseline argon --prices eval/agentic/prices.json --out runs/report.md

Quality is compared PAIRED, on the tasks that ran cleanly in both arms (tasks with a harness error in
either arm are dropped and counted). With ~100 tasks a single success rate carries about +-10 points of
noise, so read the paired confidence intervals, not the raw rates.

Prices are never baked in: pass a JSON with per-million-token prices (see prices.example.json). Without
one, tokens are reported and dollars are skipped.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from eval.agentic.schema import CallRecord, TaskResult, read_jsonl

BOOTSTRAP_N = 10_000


@dataclass
class Run:
  dir: Path
  manifest: dict
  tasks: list[TaskResult]
  calls: list[CallRecord]

  @property
  def arm(self) -> str:
    return self.manifest["arm"]

  @property
  def benchmark(self) -> str:
    return self.manifest["benchmark"]


def load_run(path: Path) -> Run:
  return Run(path, json.loads((path / "manifest.json").read_text()),
             read_jsonl(path / "tasks.jsonl", TaskResult), read_jsonl(path / "calls.jsonl", CallRecord))


# ------------------------------------------------------------------- stats

def pass_hat_k(per_task: list[tuple[int, int]], k: int) -> float | None:
  """tau-bench pass^k: P(all k sampled trials succeed), averaged over tasks. per_task = [(n, c)]."""
  vals = [math.comb(c, k) / math.comb(n, k) for n, c in per_task if n >= k]
  return statistics.fmean(vals) if vals else None


def percentile(vals: list[float], q: float) -> float | None:
  if not vals:
    return None
  s = sorted(vals)
  i = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
  return s[i]


def sign_test_p(wins: int, losses: int) -> float:
  """Exact two-sided sign test (McNemar's exact test when outcomes are binary)."""
  n = wins + losses
  if n == 0:
    return 1.0
  tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n
  return min(1.0, 2 * tail)


def paired_bootstrap(a: list[float], b: list[float], stat, n: int = BOOTSTRAP_N, seed: int = 0):
  """stat(a, b) on the sample, with a 95% percentile CI from resampling tasks (pairs) together."""
  point = stat(a, b)
  if not a or len(a) != len(b):
    return point, None, None
  rng = random.Random(seed)
  idx = range(len(a))
  draws = []
  for _ in range(n):
    s = [rng.choice(idx) for _ in idx]
    try:
      draws.append(stat([a[i] for i in s], [b[i] for i in s]))
    except ZeroDivisionError:
      continue
  if not draws:
    return point, None, None
  draws.sort()
  return point, draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]


def _mean(x):
  return statistics.fmean(x) if x else 0.0


def _diff(a, b):
  return _mean(a) - _mean(b)


def _ratio(a, b):
  return _mean(a) / _mean(b)


def _saving(a, b):  # a = this arm's cost, b = baseline cost
  return 1 - _mean(a) / _mean(b)


# -------------------------------------------------------------------- cost

def call_cost(c: CallRecord, prices: dict) -> float | None:
  p = prices.get(c.routed_model)
  if not p:
    return None
  cached = min(c.cached_tokens, c.prompt_tokens)
  return ((c.prompt_tokens - cached) * p["input_per_m"] + cached * p.get("cached_input_per_m", p["input_per_m"])
          + (c.output_tokens + c.thoughts_tokens) * p["output_per_m"]) / 1e6


def per_task_series(run: Run, prices: dict) -> dict:
  """task -> {success, tokens, cost, calls}, averaged over that task's clean trials."""
  by_trial = collections.defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0, "priced": True})
  for c in run.calls:
    d = by_trial[(c.task_id, c.trial or 0)]
    d["tokens"] += c.total_tokens
    d["calls"] += 1
    cost = call_cost(c, prices) if prices else None
    if cost is None:
      d["priced"] = False
    else:
      d["cost"] += cost
  per: dict[str, list] = collections.defaultdict(list)
  for t in run.tasks:
    if t.error is None:
      per[t.task_id].append((t, by_trial.get((t.task_id, t.trial), {"tokens": 0, "cost": 0.0, "calls": 0, "priced": bool(prices)})))
  out = {}
  for task, items in per.items():
    out[task] = {"success": _mean([float(t.success) for t, _ in items]), "trials": len(items),
                 "successes": sum(t.success for t, _ in items),
                 "tokens": _mean([d["tokens"] for _, d in items]), "calls": _mean([d["calls"] for _, d in items]),
                 "cost": _mean([d["cost"] for _, d in items]) if prices and all(d["priced"] for _, d in items) else None}
  return out


# ----------------------------------------------------------------- summary

def routing_stats(run: Run) -> dict:
  calls = run.calls
  n = len(calls) or 1
  hydrogen = run.manifest.get("hydrogen_model")
  levels = collections.Counter((c.thinking_level or "none") for c in calls)
  reasons = collections.Counter((c.router_reason or "none").split(":")[0] for c in calls)
  rms = [c.router_ms for c in calls if c.router_ms is not None]
  cms = [c.classifier_ms for c in calls if c.classifier_ms is not None]
  return {
      "calls": len(calls),
      "share_hydrogen": sum(c.routed_model == hydrogen for c in calls) / n,
      "thinking_levels": {k: v / n for k, v in sorted(levels.items())},
      "reasons": {k: v / n for k, v in sorted(reasons.items())},
      "fail_open_rate": sum(c.router_ok is False for c in calls) / n,
      "classifier_error_rate": sum((c.router_reason or "") == "classifier_error" for c in calls) / n,
      "router_ms_p50": percentile(rms, 0.5), "router_ms_p95": percentile(rms, 0.95),
      "classifier_ms_p50": percentile(cms, 0.5), "classifier_ms_p95": percentile(cms, 0.95),
      "vertex_error_rate": sum((c.vertex_status or 0) != 200 for c in calls) / n,
      "signature_retries": sum(c.signature_retry for c in calls),
  }


def summarize(run: Run, prices: dict) -> dict:
  series = per_task_series(run, prices)
  errored = sorted({t.task_id for t in run.tasks if t.error is not None})
  trials = [(s["trials"], s["successes"]) for s in series.values()]
  max_k = max((n for n, _ in trials), default=0)
  costs = [s["cost"] for s in series.values() if s["cost"] is not None]
  return {
      "arm": run.arm, "benchmark": run.benchmark, "tasks_clean": len(series), "tasks_errored": len(errored),
      "errored_tasks": errored,
      "success_rate": _mean([s["success"] for s in series.values()]) if series else None,
      "pass_hat_k": {k: pass_hat_k(trials, k) for k in range(1, max_k + 1)},
      "calls_per_task": _mean([s["calls"] for s in series.values()]),
      "tokens_per_task": _mean([s["tokens"] for s in series.values()]),
      "cost_per_task": _mean(costs) if prices and len(costs) == len(series) and series else None,
      "routing": routing_stats(run),
  }


def compare(run: Run, base: Run, prices: dict, drop_errors: bool = True) -> dict | None:
  a, b = per_task_series(run, prices), per_task_series(base, prices)
  common = sorted(set(a) & set(b)) if drop_errors else sorted(set(a) | set(b))
  common = [t for t in common if t in a and t in b]
  if not common:
    return None
  qa, qb = [a[t]["success"] for t in common], [b[t]["success"] for t in common]
  wins = sum(x > y for x, y in zip(qa, qb))
  losses = sum(x < y for x, y in zip(qa, qb))
  out = {"tasks_paired": len(common), "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
         "sign_test_p": sign_test_p(wins, losses),
         "success_diff": paired_bootstrap(qa, qb, _diff),
         "quality_retained": paired_bootstrap(qa, qb, _ratio) if _mean(qb) else (None, None, None)}
  ca, cb = [a[t]["cost"] for t in common], [b[t]["cost"] for t in common]
  if all(x is not None for x in ca + cb) and _mean(cb):
    out["cost_saving"] = paired_bootstrap(ca, cb, _saving)
  ta, tb = [a[t]["tokens"] for t in common], [b[t]["tokens"] for t in common]
  if _mean(tb):
    out["token_saving"] = paired_bootstrap(ta, tb, _saving)
  return out


# ------------------------------------------------------------------ report

def _pct(x, d=1):
  return "–" if x is None else f"{100 * x:.{d}f}%"


def _ci(t, pct=True):
  if not t or t[0] is None:
    return "–"
  f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.2f}")
  unit = "%" if pct else ""
  if t[1] is None:
    return f"{f(t[0])}{unit}"
  return f"{f(t[0])}{unit} [{f(t[1])}, {f(t[2])}]"


def _p(c):
  return "–" if "sign_test_p" not in c else "%.3f" % c["sign_test_p"]


def _usd(x):
  return "–" if x is None else "$%.3f" % x


def render(by_bench: dict[str, list[tuple[dict, dict | None]]], baseline: str, have_prices: bool) -> str:
  lines = ["# Agentic eval report", "",
           f"Baseline arm: `{baseline}`. Intervals are 95% paired-bootstrap CIs over tasks; "
           "quality is compared on tasks that ran cleanly in both arms.", ""]
  if not have_prices:
    lines += ["> No price file given: dollar costs are omitted (tokens are shown).", ""]
  for bench, rows in by_bench.items():
    lines += [f"## {bench}", "",
              "| arm | tasks (errored) | success | Δ success vs base | quality retained | sign-test p | "
              "calls/task | tokens/task | $/task | cost saving | token saving |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in rows:
      c = c or {}
      lines.append(
          f"| {s['arm']} | {s['tasks_clean']} ({s['tasks_errored']}) | {_pct(s['success_rate'])} | "
          f"{_ci(c.get('success_diff'))} | {_ci(c.get('quality_retained'))} | "
          f"{_p(c)} | {s['calls_per_task']:.1f} | "
          f"{s['tokens_per_task']:,.0f} | {_usd(s['cost_per_task'])} | "
          f"{_ci(c.get('cost_saving'))} | {_ci(c.get('token_saving'))} |")
    k_rows = [(s["arm"], s["pass_hat_k"]) for s, _ in rows if len(s["pass_hat_k"]) > 1]
    if k_rows:
      ks = sorted({k for _, d in k_rows for k in d})
      lines += ["", "| arm | " + " | ".join(f"pass^{k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
      lines += [f"| {arm} | " + " | ".join(_pct(d.get(k)) for k in ks) + " |" for arm, d in k_rows]
    lines += ["", "| arm | calls | → Hydrogen | thinking levels sent | fail-open | classifier errors | "
              "router ms p50/p95 | vertex errors | sig. retries |", "|---|---|---|---|---|---|---|---|---|"]
    for s, _ in rows:
      r = s["routing"]
      lv = ", ".join(f"{k} {_pct(v, 0)}" for k, v in r["thinking_levels"].items())
      ms = "–" if r["router_ms_p50"] is None else f"{r['router_ms_p50']:.0f}/{r['router_ms_p95']:.0f}"
      lines.append(f"| {s['arm']} | {r['calls']} | {_pct(r['share_hydrogen'])} | {lv or '–'} | {_pct(r['fail_open_rate'])} | "
                   f"{_pct(r['classifier_error_rate'])} | {ms} | {_pct(r['vertex_error_rate'])} | {r['signature_retries']} |")
    errs = {s["arm"]: s["errored_tasks"] for s, _ in rows if s["errored_tasks"]}
    if errs:
      lines += ["", "Harness errors (excluded from quality): " + "; ".join(f"{a}: {len(t)}" for a, t in errs.items())]
    lines.append("")
  return "\n".join(lines)


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("runs", nargs="+", type=Path, help="run directories (each has manifest.json)")
  p.add_argument("--baseline", default="argon")
  p.add_argument("--prices", type=Path, help="JSON: model -> {input_per_m, output_per_m, cached_input_per_m}")
  p.add_argument("--out", type=Path, help="write the markdown report here (also prints it)")
  p.add_argument("--json", type=Path, help="write the full numbers as JSON")
  p.add_argument("--keep-errors", action="store_true", help="pair on all tasks, not only clean ones")
  args = p.parse_args(argv)

  prices = {k: v for k, v in json.loads(args.prices.read_text()).items() if not k.startswith("_")} if args.prices else {}
  runs = [load_run(d) for d in args.runs if (d / "manifest.json").exists()]
  if not runs:
    raise SystemExit("no run directories with a manifest.json")
  by_bench: dict[str, list[Run]] = collections.defaultdict(list)
  for r in runs:
    by_bench[r.benchmark].append(r)

  report, full = {}, {}
  for bench, rs in by_bench.items():
    base = next((r for r in rs if r.arm == args.baseline), None)
    rows = []
    for r in sorted(rs, key=lambda r: (r.arm != args.baseline, r.arm)):
      rows.append((summarize(r, prices), None if r is base or base is None else compare(r, base, prices, not args.keep_errors)))
    report[bench], full[bench] = rows, [{"summary": s, "vs_baseline": c} for s, c in rows]
    if base is None:
      print(f"warning: no '{args.baseline}' run for {bench}; paired columns are empty", flush=True)
  text = render(report, args.baseline, bool(prices))
  print(text)
  if args.out:
    args.out.write_text(text)
  if args.json:
    args.json.write_text(json.dumps(full, indent=2, default=str))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
