"""Run tau3-banking (banking_knowledge, all 97 tasks) through the proxy.

One `tau2 run` process per (task, trial). The agent LLM is the proxy (OpenAI-compatible); the
user simulator is a separate, fixed model that is NOT routed, so it is identical in every arm.
Each process sends x-eval-task / x-eval-trial headers, which is how calls are attributed.

  # install tau2 in its own environment (Python 3.12-3.13; it cannot share one with tau2-bench v0.x)
  git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench && uv sync --extra knowledge

  python -m eval.agentic.run_tau3 --arm router --out runs --start-proxy \\
      --router-target agw-autorouter-xxxx-uc.a.run.app:443 --router-tls --router-auth id_token \\
      --tau2-repo ../tau2-bench --tau2-cmd "uv run --project ../tau2-bench tau2" \\
      --user-llm vertex_ai/gemini-3.8-flash --user-llm-args '{"temperature":0}' --trials 3

Results: <out>/<run-id>/{manifest.json, calls.jsonl, tasks.jsonl, harness/}.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from eval.agentic.common import add_proxy_args, proxy_session, write_manifest
from eval.agentic.schema import TaskResult, append_jsonl, read_jsonl

BENCHMARK = "tau3_banking"
DOMAIN = "banking_knowledge"
SUCCESS_REWARD = 1.0 - 1e-6


def load_task_ids(tau2_repo: Path) -> list[str]:
  tasks = json.loads((tau2_repo / "data/tau2/domains" / DOMAIN / "tasks.json").read_text())
  return [t["id"] for t in tasks]


def agent_llm_args(proxy_url: str, task: str, trial: int, temperature: float | None, extra: dict) -> dict:
  args = {"api_base": f"{proxy_url}/v1", "api_key": "eval",
          "extra_headers": {"x-eval-task": task, "x-eval-trial": str(trial), "x-eval-benchmark": BENCHMARK}}
  if temperature is not None:
    args["temperature"] = temperature
  return {**args, **extra}


def build_command(args, task: str, trial: int, proxy_url: str, save_to: str) -> list[str]:
  llm_args = agent_llm_args(proxy_url, task, trial, args.agent_temperature, json.loads(args.agent_llm_args or "{}"))
  cmd = shlex.split(args.tau2_cmd) + [
      "run", "--domain", DOMAIN, "--task-ids", task, "--num-trials", "1", "--seed", str(args.seed + trial),
      "--agent-llm", "openai/agent", "--agent-llm-args", json.dumps(llm_args),
      "--user-llm", args.user_llm, "--retrieval-config", args.retrieval_config,
      "--max-concurrency", "1", "--save-to", save_to, "--log-level", "ERROR"]
  if args.user_llm_args:
    cmd += ["--user-llm-args", args.user_llm_args]
  if args.max_steps:
    cmd += ["--max-steps", str(args.max_steps)]
  if args.tau2_extra:
    cmd += shlex.split(args.tau2_extra)
  return cmd


def parse_results(path: Path, run_id: str, arm: str, task: str, trial: int) -> TaskResult:
  """One tau2 results.json (one task, one trial) -> TaskResult."""
  data = json.loads(path.read_text())
  sims = data.get("simulations") or []
  if not sims:
    return TaskResult(run_id, arm, BENCHMARK, task, False, trial, error="tau2 produced no simulation")
  sim = sims[0]
  reward = (sim.get("reward_info") or {}).get("reward")
  term = sim.get("termination_reason")
  term = getattr(term, "value", term)
  return TaskResult(
      run_id, arm, BENCHMARK, task, success=reward is not None and reward >= SUCCESS_REWARD, trial=trial,
      score=reward, termination=str(term) if term is not None else None, duration_s=sim.get("duration"),
      error=None if reward is not None else "no reward_info",
      extra={"messages": len(sim.get("messages") or []), "tau2_sim_id": sim.get("id")})


def run_one(args, run_dir: Path, run_id: str, proxy_url: str, task: str, trial: int, data_dir: Path) -> TaskResult:
  save_to = f"agentic-{run_id}-{task}-t{trial}"
  cmd = build_command(args, task, trial, proxy_url, save_to)
  log = run_dir / "harness" / f"{task}-t{trial}.log"
  log.parent.mkdir(parents=True, exist_ok=True)
  env = {**os.environ, "TAU2_DATA_DIR": str(data_dir)}
  t0 = time.time()
  with open(log, "wb") as f:
    f.write((" ".join(shlex.quote(c) for c in cmd) + "\n\n").encode())
    f.flush()
    try:
      proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=args.task_timeout)
      code = proc.returncode
    except subprocess.TimeoutExpired:
      code = -9
  results = data_dir / "simulations" / save_to / "results.json"
  if results.exists():
    try:
      return parse_results(results, run_id, args.arm, task, trial)
    except (ValueError, KeyError) as e:
      return TaskResult(run_id, args.arm, BENCHMARK, task, False, trial, error=f"unreadable results.json: {e}")
  return TaskResult(run_id, args.arm, BENCHMARK, task, False, trial, duration_s=round(time.time() - t0, 1),
                    error=f"tau2 exited {code} without results (see {log.name})")


def parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  add_proxy_args(p)
  g = p.add_argument_group("tau3-banking")
  g.add_argument("--tau2-repo", type=Path, default=Path(os.getenv("TAU2_REPO", "tau2-bench")),
                 help="checkout of sierra-research/tau2-bench (task list + data dir)")
  g.add_argument("--tau2-cmd", default="tau2", help='how to invoke tau2, e.g. "uv run --project ../tau2-bench tau2"')
  g.add_argument("--tau2-extra", default="", help="extra flags appended to every `tau2 run`")
  g.add_argument("--user-llm", required=True, help="fixed, un-routed user simulator model (LiteLLM name)")
  g.add_argument("--user-llm-args", default="", help="JSON for the user simulator (temperature, project, ...)")
  g.add_argument("--agent-temperature", type=float, default=None, help="omit to leave the provider default")
  g.add_argument("--agent-llm-args", default="", help="extra JSON merged into the agent's LiteLLM args")
  g.add_argument("--retrieval-config", default="bm25")
  g.add_argument("--trials", type=int, default=3)
  g.add_argument("--seed", type=int, default=300, help="base seed; trial k uses seed+k")
  g.add_argument("--limit", type=int, default=0, help="first N tasks only (pilot)")
  g.add_argument("--task-ids", nargs="*", default=None)
  g.add_argument("--max-steps", type=int, default=0)
  g.add_argument("--task-timeout", type=float, default=3600.0, help="seconds per (task, trial) process")
  g.add_argument("--redo-errors", action="store_true", help="re-run (task, trial) pairs that errored")
  return p


def main(argv=None) -> int:
  args = parser().parse_args(argv)
  run_id = args.run_id or f"{BENCHMARK}-{args.arm}"
  run_dir = Path(args.out) / run_id
  data_dir = Path(os.getenv("TAU2_DATA_DIR") or args.tau2_repo / "data").resolve()

  all_ids = args.task_ids or load_task_ids(args.tau2_repo)
  if args.limit:
    all_ids = all_ids[:args.limit]
  tasks_path = run_dir / "tasks.jsonl"
  done = {(r.task_id, r.trial) for r in read_jsonl(tasks_path, TaskResult) if r.error is None or not args.redo_errors}
  todo = [(t, k) for t in all_ids for k in range(args.trials) if (t, k) not in done]
  print(f"{run_id}: {len(all_ids)} tasks x {args.trials} trials, {len(todo)} to run, arm={args.arm}", file=sys.stderr)

  if args.dry_run:
    if todo:
      print(" ".join(shlex.quote(c) for c in build_command(
          args, todo[0][0], todo[0][1], "http://127.0.0.1:<proxy-port>", f"agentic-{run_id}-{todo[0][0]}-t{todo[0][1]}")))
    return 0

  run_dir.mkdir(parents=True, exist_ok=True)
  with proxy_session(args, run_dir, run_id, BENCHMARK) as proxy_url:
    write_manifest(run_dir, run_id, args, BENCHMARK, all_ids, args.trials, seed=args.seed, harness={
        "tau2_cmd": args.tau2_cmd, "retrieval_config": args.retrieval_config, "user_llm": args.user_llm,
        "user_llm_args": args.user_llm_args, "agent_temperature": args.agent_temperature,
        "max_steps": args.max_steps or None, "proxy_url": proxy_url})
    ok = bad = 0
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
      futs = {pool.submit(run_one, args, run_dir, run_id, proxy_url, t, k, data_dir): (t, k) for t, k in todo}
      for fut in cf.as_completed(futs):
        res = fut.result()
        append_jsonl(tasks_path, res)
        ok += res.success
        bad += res.error is not None
        print(f"  {res.task_id} t{res.trial}: {'PASS' if res.success else 'fail'}"
              f"{' [ERROR: ' + res.error + ']' if res.error else ''}", file=sys.stderr)
  print(f"done: {ok} passed, {bad} errored of {len(todo)}", file=sys.stderr)
  return 1 if bad else 0


if __name__ == "__main__":
  raise SystemExit(main())
