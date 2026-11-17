"""Run the pinned 100-task SWE-bench Verified sample through mini-swe-agent and the proxy, then grade.

One mini-swe-agent process per instance (its LLM base URL is the proxy, with an x-eval-task header),
then the official SWE-bench harness grades all patches at once. Docker is required for both the
agent's sandbox and the grader, so run this on a VM, not on Cloud Run.

  pip install mini-swe-agent swebench      # plus Docker

  python -m eval.agentic.run_swebench --arm router --out runs --start-proxy \\
      --router-target agw-autorouter-xxxx-uc.a.run.app:443 --router-tls --router-auth id_token

Results: <out>/<run-id>/{manifest.json, calls.jsonl, tasks.jsonl, predictions.json, harness/, grading/}.
Use --skip-grade to stop after the agent runs, --grade-only to (re)grade existing predictions.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from eval.agentic.common import add_proxy_args, proxy_session, write_manifest
from eval.agentic.sample_swebench import DEFAULT_OUT as DEFAULT_SAMPLE
from eval.agentic.schema import TaskResult, append_jsonl

BENCHMARK = "swebench_verified"
DATASET = "princeton-nlp/SWE-bench_Verified"
BENIGN_EXIT = {"Submitted", "LimitsExceeded"}


def load_ids(path: Path) -> list[str]:
  return json.loads(path.read_text())["instance_ids"]


def override_yaml(proxy_url: str, task: str, step_limit: int | None, temperature: float | None) -> str:
  """Config merged over mini-swe-agent's swebench.yaml: model -> the proxy, attributed to `task`."""
  kw = {"api_base": f"{proxy_url}/v1", "api_key": "eval", "drop_params": True,
        "extra_headers": {"x-eval-task": task, "x-eval-trial": "0", "x-eval-benchmark": BENCHMARK}}
  if temperature is not None:
    kw["temperature"] = temperature
  doc: dict = {"model": {"model_name": "openai/agent", "model_kwargs": kw}}
  if step_limit:
    doc["agent"] = {"step_limit": step_limit}
  return json.dumps(doc, indent=2)  # JSON is valid YAML


def build_command(args, task: str, out_dir: Path, cfg_path: Path) -> list[str]:
  return [args.agent_python, "-m", "minisweagent.run.benchmarks.swebench", "--subset", "verified", "--split", "test",
          "--filter", f"^{task}$", "-o", str(out_dir), "-w", "1", "-c", "swebench.yaml", "-c", str(cfg_path)]


def read_agent_outcome(out_dir: Path, task: str) -> dict:
  """exit_status / api_calls / error from the instance trajectory, and the patch from preds.json."""
  info, patch = {}, None
  traj = out_dir / task / f"{task}.traj.json"
  if traj.exists():
    info = json.loads(traj.read_text()).get("info", {})
  preds = out_dir / "preds.json"
  if preds.exists():
    patch = (json.loads(preds.read_text()).get(task) or {}).get("model_patch")
  return {"exit_status": info.get("exit_status"), "api_calls": (info.get("model_stats") or {}).get("api_calls"),
          "exception": info.get("exception_str"), "patch": patch, "has_traj": traj.exists()}


_PULL_LOCK = threading.Lock()


def image_name(task: str) -> str:
  return f"docker.io/swebench/sweb.eval.x86_64.{task.replace('__', '_1776_').lower()}:latest"


def ensure_image(task: str, retries: int = 3) -> str | None:
  """Pull the instance image once, serialised: parallel `docker run` auto-pulls race and exit 125."""
  img = image_name(task)
  with _PULL_LOCK:
    if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0:
      return None
    err = ""
    for i in range(retries):
      r = subprocess.run(["docker", "pull", "-q", img], capture_output=True, text=True)
      if r.returncode == 0:
        return None
      err = (r.stderr or r.stdout)[-300:]
      time.sleep(5 * (i + 1))
    return f"docker pull failed for {img}: {err}"


def run_one(args, run_dir: Path, proxy_url: str, task: str) -> dict:
  out_dir = run_dir / "harness" / task
  out_dir.mkdir(parents=True, exist_ok=True)
  cfg = out_dir / "override.yaml"
  cfg.write_text(override_yaml(proxy_url, task, args.step_limit, args.agent_temperature))
  pull_err = ensure_image(task)
  if pull_err:
    return {"task": task, "code": -1, "exit_status": "ImagePullError", "api_calls": 0, "exception": pull_err,
            "patch": None, "has_traj": False, "duration_s": 0.0}
  cmd = build_command(args, task, out_dir, cfg)
  env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_SILENT_STARTUP": "1",
         "MSWEA_CONFIGURED": "true"}
  t0 = time.time()
  with open(out_dir / "agent.log", "wb") as f:
    f.write((" ".join(shlex.quote(c) for c in cmd) + "\n\n").encode())
    f.flush()
    try:
      code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=args.task_timeout).returncode
    except subprocess.TimeoutExpired:
      code = -9
  out = read_agent_outcome(out_dir, task)
  out.update(task=task, code=code, duration_s=round(time.time() - t0, 1))
  return out


def merge_predictions(run_dir: Path, ids: list[str]) -> Path:
  merged = {}
  for task in ids:
    p = run_dir / "harness" / task / "preds.json"
    if p.exists():
      merged.update({k: v for k, v in json.loads(p.read_text()).items() if k == task})
  path = run_dir / "predictions.json"
  path.write_text(json.dumps(merged, indent=2))
  return path


def grade(args, run_dir: Path, run_id: str, preds: Path, ids: list[str]) -> dict:
  """Official SWE-bench harness; returns its report (resolved_ids, error_ids, ...)."""
  gdir = run_dir / "grading"
  gdir.mkdir(exist_ok=True)
  cmd = [args.grader_python, "-m", "swebench.harness.run_evaluation", "--dataset_name", DATASET,
         "--predictions_path", str(preds.resolve()), "--max_workers", str(args.grade_workers),
         "--run_id", run_id, "--instance_ids", *ids]
  with open(gdir / "grade.log", "wb") as f:
    subprocess.run(cmd, cwd=gdir, stdout=f, stderr=subprocess.STDOUT, check=False)
  reports = sorted(glob.glob(str(gdir / f"*.{run_id}.json")))
  if not reports:
    raise RuntimeError(f"SWE-bench grading produced no report (see {gdir / 'grade.log'})")
  return json.loads(Path(reports[-1]).read_text())


def to_task_results(run_id: str, arm: str, ids: list[str], outcomes: dict[str, dict], report: dict | None) -> list[TaskResult]:
  resolved = set((report or {}).get("resolved_ids", []))
  graded_err = set((report or {}).get("error_ids", []))
  out = []
  for task in ids:
    o = outcomes.get(task) or {}
    err = None
    if o.get("exception"):
      err = f"agent exception: {str(o['exception'])[:300]}"      # e.g. the proxy/Vertex failed mid-run
    elif not o.get("has_traj") and not o.get("patch"):
      err = f"agent produced no trajectory (exit {o.get('code')})"
    elif report is None:
      err = "not graded"
    elif task in graded_err:
      err = "grading error"
    out.append(TaskResult(
        run_id, arm, BENCHMARK, task, success=task in resolved and err is None, trial=0,
        score=1.0 if task in resolved else 0.0, termination=o.get("exit_status"), duration_s=o.get("duration_s"),
        error=err, extra={"api_calls": o.get("api_calls"), "empty_patch": not (o.get("patch") or "").strip()}))
  return out


def parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  add_proxy_args(p)
  g = p.add_argument_group("swe-bench")
  g.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="pinned instance-id list")
  g.add_argument("--limit", type=int, default=0, help="first N ids only (pilot)")
  g.add_argument("--ids", nargs="*", default=None, help="explicit instance ids (overrides --sample)")
  g.add_argument("--agent-python", default=sys.executable, help="interpreter with mini-swe-agent installed")
  g.add_argument("--grader-python", default=sys.executable, help="interpreter with swebench installed")
  g.add_argument("--step-limit", type=int, default=0, help="mini-swe-agent agent.step_limit")
  g.add_argument("--agent-temperature", type=float, default=None, help="omit to leave the provider default")
  g.add_argument("--task-timeout", type=float, default=3600.0)
  g.add_argument("--grade-workers", type=int, default=4)
  g.add_argument("--skip-grade", action="store_true")
  g.add_argument("--grade-only", action="store_true")
  return p


def main(argv=None) -> int:
  args = parser().parse_args(argv)
  run_id = args.run_id or f"{BENCHMARK}-{args.arm}"
  run_dir = Path(args.out) / run_id
  ids = args.ids or load_ids(args.sample)
  if args.limit:
    ids = ids[:args.limit]
  print(f"{run_id}: {len(ids)} instances, arm={args.arm}", file=sys.stderr)

  def finished(task: str) -> bool:
    p = run_dir / "harness" / task / "preds.json"
    return p.exists() and task in json.loads(p.read_text())

  todo = [] if args.grade_only else [t for t in ids if not finished(t)]
  if args.dry_run:
    if todo:
      print(" ".join(shlex.quote(c) for c in build_command(args, todo[0], run_dir / "harness" / todo[0], Path("override.yaml"))))
      print(override_yaml("http://127.0.0.1:<proxy-port>", todo[0], args.step_limit, args.agent_temperature))
    print(f"{len(todo)} instances to run", file=sys.stderr)
    return 0

  run_dir.mkdir(parents=True, exist_ok=True)
  outcomes: dict[str, dict] = {}
  with proxy_session(args, run_dir, run_id, BENCHMARK) as proxy_url:
    write_manifest(run_dir, run_id, args, BENCHMARK, ids, 1, harness={
        "agent": "mini-swe-agent", "dataset": DATASET, "step_limit": args.step_limit or None,
        "agent_temperature": args.agent_temperature, "sample": str(args.sample), "proxy_url": proxy_url})
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
      for fut in cf.as_completed([pool.submit(run_one, args, run_dir, proxy_url, t) for t in todo]):
        o = fut.result()
        outcomes[o["task"]] = o
        print(f"  {o['task']}: exit={o['exit_status']} calls={o['api_calls']} {o['duration_s']}s", file=sys.stderr)
  for t in ids:  # instances finished in an earlier invocation
    outcomes.setdefault(t, {**read_agent_outcome(run_dir / "harness" / t, t), "task": t})

  if args.skip_grade:
    return 0
  preds = merge_predictions(run_dir, ids)
  report = grade(args, run_dir, run_id, preds, ids)
  results = to_task_results(run_id, args.arm, ids, outcomes, report)
  tasks_path = run_dir / "tasks.jsonl"
  tasks_path.unlink(missing_ok=True)
  for r in results:
    append_jsonl(tasks_path, r)
  print(f"resolved {sum(r.success for r in results)}/{len(results)}; "
        f"{sum(r.error is not None for r in results)} errored", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
