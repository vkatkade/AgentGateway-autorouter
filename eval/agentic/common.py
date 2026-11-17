"""Helpers shared by the runners: run directories, the proxy subprocess, manifests."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from eval.agentic.arms import ARMS
from eval.agentic.schema import RunManifest, to_dict

ROUTER_ENV_KEYS = ("ARGON_MODEL", "HYDROGEN_MODEL", "ARGON_THRESHOLD", "HYDROGEN_THRESHOLD",
                   "ARGON_HIGH_AT", "THINKING_FLOOR_HIGH_AT", "THINKING_MIN_CONFIDENCE",
                   "THINKING_RESPECT_AGENT", "DECISIONS_TIMEOUT_MS", "LONG_CONTEXT_TOKENS", "MANY_TOOLS")


def now_iso() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git_sha(cwd: str | None = None) -> str | None:
  try:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=cwd, timeout=10).stdout.strip() or None
  except (OSError, subprocess.SubprocessError):
    return None


def free_port() -> int:
  with contextlib.closing(socket.socket()) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def add_proxy_args(p: argparse.ArgumentParser) -> None:
  """Flags shared by both runners; they map onto eval.agentic.proxy's flags."""
  g = p.add_argument_group("proxy / router")
  g.add_argument("--arm", required=True, choices=sorted(ARMS))
  g.add_argument("--out", required=True, help="parent directory; results go to <out>/<run-id>/")
  g.add_argument("--run-id", default="", help="default: <benchmark>-<arm>")
  g.add_argument("--proxy-url", default="", help="use an already running proxy, e.g. http://127.0.0.1:8088")
  g.add_argument("--start-proxy", action="store_true", help="start the proxy for this run (stopped afterwards)")
  g.add_argument("--router", choices=["remote", "inprocess", "none"], default=os.getenv("ROUTER_MODE", "remote"))
  g.add_argument("--router-target", default=os.getenv("ROUTER_TARGET", ""))
  g.add_argument("--router-tls", action="store_true", default=os.getenv("ROUTER_TLS", "").lower() in ("1", "true", "yes"))
  g.add_argument("--router-auth", choices=["none", "id_token"], default=os.getenv("ROUTER_AUTH", "none"))
  g.add_argument("--router-audience", default=os.getenv("ROUTER_AUDIENCE", ""))
  g.add_argument("--signature-policy", choices=["echo", "drop", "dummy"], default="echo")
  g.add_argument("--parallel", type=int, default=4, help="harness processes in flight")
  g.add_argument("--dry-run", action="store_true", help="print the plan and the commands; run nothing")


def proxy_cmd(args, run_dir: Path, run_id: str, benchmark: str, port: int) -> list[str]:
  cmd = [sys.executable, "-m", "eval.agentic.proxy", "--arm", args.arm, "--out", str(run_dir),
         "--run-id", run_id, "--benchmark", benchmark, "--port", str(port), "--router", args.router,
         "--signature-policy", args.signature_policy]
  if args.router_target:
    cmd += ["--router-target", args.router_target]
  if args.router_tls:
    cmd.append("--router-tls")
  if args.router_auth != "none":
    cmd += ["--router-auth", args.router_auth]
  if args.router_audience:
    cmd += ["--router-audience", args.router_audience]
  return cmd


@contextlib.contextmanager
def proxy_session(args, run_dir: Path, run_id: str, benchmark: str):
  """Yields the proxy base URL (http://host:port); starts and stops the proxy with --start-proxy."""
  if args.proxy_url and not args.start_proxy:
    yield args.proxy_url.rstrip("/")
    return
  port = free_port()
  log = open(run_dir / "proxy.log", "ab")
  proc = subprocess.Popen(proxy_cmd(args, run_dir, run_id, benchmark, port), stdout=log, stderr=log)
  url = f"http://127.0.0.1:{port}"
  try:
    for _ in range(60):
      if proc.poll() is not None:
        raise RuntimeError(f"proxy exited early (see {run_dir / 'proxy.log'})")
      try:
        urllib.request.urlopen(f"{url}/healthz", timeout=1).read()
        break
      except OSError:
        time.sleep(0.5)
    else:
      raise RuntimeError("proxy did not become healthy")
    yield url
  finally:
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
      proc.wait(timeout=10)
    log.close()


def write_manifest(run_dir: Path, run_id: str, args, benchmark: str, task_ids: list, trials: int,
                   harness: dict, seed: int | None = None, notes: str = "") -> RunManifest:
  proxy_json = {}
  if (run_dir / "proxy.json").exists():
    proxy_json = json.loads((run_dir / "proxy.json").read_text())
  m = RunManifest(
      run_id=run_id, arm=args.arm, benchmark=benchmark, created_at=now_iso(),
      router_mode=proxy_json.get("router_mode", args.router if ARMS[args.arm].use_router else "none"),
      router_target=args.router_target or None,
      argon_model=proxy_json.get("argon_model") or os.getenv("ARGON_MODEL", "gemini-4-argon"),
      hydrogen_model=proxy_json.get("hydrogen_model") or os.getenv("HYDROGEN_MODEL", "gemini-3.8-flash"),
      requested_model=proxy_json.get("requested_model"), seed=seed, task_ids=list(task_ids), trials=trials,
      harness={"git_sha": git_sha(), **harness},
      router_env={k: os.environ[k] for k in ROUTER_ENV_KEYS if k in os.environ}, notes=notes)
  (run_dir / "manifest.json").write_text(json.dumps(to_dict(m), indent=2))
  return m
