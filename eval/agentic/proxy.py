"""Router-in-the-loop proxy: an OpenAI-compatible endpoint in front of Vertex AI.

  benchmark harness (LiteLLM)  --/v1/chat/completions-->  this proxy
      -> convert to Vertex generateContent
      -> apply the experiment arm (and, for router arms, ask the router exactly as Agent Gateway would)
      -> call Vertex with ADC
      -> convert back, and append a CallRecord to <out>/calls.jsonl

Point a harness at it with an `openai/<anything>` model name and api_base=http://HOST:PORT/v1.
Attribute calls to tasks with request headers (extra_headers):
  x-eval-task, x-eval-trial, x-eval-benchmark   (the runners set these; one harness process per task)

Run (the runners can also start it for you with --start-proxy):

  python -m eval.agentic.proxy --arm router --out runs/router --port 8088 \\
      --router remote --router-target agw-autorouter-xxxx-uc.a.run.app:443 --router-tls --router-auth id_token
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests
from google.auth.exceptions import GoogleAuthError
from flask import Flask, Response, jsonify, request

from autorouter.config import Config
from autorouter.vertex import extract_features, parse_path
from eval.agentic import convert
from eval.agentic.arms import ARMS, Backend, InProcessRouter, apply_arm
from eval.agentic.extproc_client import RemoteRouter
from eval.agentic.schema import CallRecord, append_jsonl

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class Settings:
  arm: str
  out_dir: Path
  run_id: str = "run"
  benchmark: str | None = None
  requested_model: str = ""                 # model named in the request path; default: Argon
  project: str = ""
  location: str = "global"
  signature_policy: str = "echo"            # echo | drop | dummy
  dummy_signature: str = convert.DUMMY_SIGNATURE
  vertex_timeout_s: float = 300.0
  max_attempts: int = 5
  cfg: Config = field(default_factory=Config)


class VertexClient:
  """generateContent over REST with ADC, retrying 429/5xx with backoff."""

  def __init__(self, project: str, location: str, timeout_s: float, max_attempts: int):
    self.project, self.location, self.timeout_s, self.max_attempts = project, location, timeout_s, max_attempts
    self.host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    self._creds, self._lock = None, threading.Lock()

  def _token(self) -> str:
    with self._lock:
      if self._creds is None:
        static = os.getenv("VERTEX_ACCESS_TOKEN")
        if static:  # a pasted ~1 hour access token instead of ADC (it will not refresh)
          from google.oauth2.credentials import Credentials
          self._creds = Credentials(token=static)
        else:
          self._creds, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
          self.project = self.project or proj or ""
      if not self._creds.valid:
        self._creds.refresh(google.auth.transport.requests.Request())
      return self._creds.token

  def project_id(self) -> str:
    if not self.project:
      self._token()
    return self.project

  def post(self, path: str, body: dict) -> tuple[int, dict, float, int]:
    """Returns (status, json, ms of the last attempt, attempts)."""
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    status, payload, ms = 0, {}, 0.0
    for attempt in range(1, self.max_attempts + 1):
      t0 = time.perf_counter()
      try:
        token = self._token()
      except GoogleAuthError as e:  # not transient: fail the call at once instead of backing off
        return 401, {"error": {"message": f"Vertex credentials: {e}"}}, 0.0, attempt
      try:
        r = requests.post(f"https://{self.host}{path}", data=data, timeout=self.timeout_s,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        status = r.status_code
        try:
          payload = r.json()
        except ValueError:
          payload = {"error": {"message": r.text[:2000]}}
      except requests.RequestException as e:
        status, payload = 599, {"error": {"message": f"{type(e).__name__}: {e}"}}
      ms = (time.perf_counter() - t0) * 1000
      if status not in RETRY_STATUS and status != 599:
        return status, payload, round(ms, 1), attempt
      if attempt < self.max_attempts:
        time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
    return status, payload, round(ms, 1), self.max_attempts


def _err_msg(payload: dict) -> str:
  e = payload.get("error")
  if isinstance(e, dict):
    return str(e.get("message") or e)[:600]
  return json.dumps(payload)[:600]


def _add_dummy_signatures(body: dict, sig: str) -> None:
  for c in body.get("contents") or []:
    if c.get("role") == "model":
      for p in c.get("parts") or []:
        if "functionCall" in p and not p.get("thoughtSignature"):
          p["thoughtSignature"] = sig


def create_app(settings: Settings, backend: Backend | None, vertex: VertexClient) -> Flask:
  app = Flask(__name__)
  arm = ARMS[settings.arm]
  cfg = settings.cfg
  sigs = convert.SignatureCache()
  seq: dict[tuple, int] = {}
  seq_lock = threading.Lock()
  calls_path = settings.out_dir / "calls.jsonl"
  tally = {"calls": 0, "errors": 0, "by_model": {}}

  def next_seq(task: str, trial) -> int:
    with seq_lock:
      seq[(task, trial)] = seq.get((task, trial), 0) + 1
      return seq[(task, trial)]

  @app.get("/healthz")
  def healthz():
    return jsonify(ok=True, arm=settings.arm)

  @app.get("/v1/models")
  def models():
    ids = sorted(cfg.all_routable | {settings.requested_model})
    return jsonify(object="list", data=[{"id": i, "object": "model"} for i in ids])

  @app.get("/v1/stats")
  def stats():
    return jsonify(arm=settings.arm, run_id=settings.run_id, signatures_cached=len(sigs), **tally)

  @app.post("/v1/chat/completions")
  def chat():
    req = request.get_json(force=True, silent=True)
    if not isinstance(req, dict) or not req.get("messages"):
      return jsonify(error={"message": "messages are required", "type": "invalid_request_error"}), 400
    h = request.headers
    task = h.get("x-eval-task") or "unknown"
    trial_raw = h.get("x-eval-trial")
    trial = int(trial_raw) if (trial_raw or "").lstrip("-").isdigit() else None
    benchmark = h.get("x-eval-benchmark") or settings.benchmark

    asked = (req.get("model") or "").split("/")[-1]
    model = asked if asked in cfg.all_routable else settings.requested_model
    path = (f"/v1/projects/{vertex.project_id()}/locations/{settings.location}"
            f"/publishers/google/models/{model}:generateContent")

    body = convert.to_gemini(req, sigs, settings.signature_policy, settings.dummy_signature)
    feats = extract_features(body)
    try:
      outcome = apply_arm(arm, cfg, backend, path, body)
    except Exception as e:  # a bug in routing must not silently change what the arm measures
      return jsonify(error={"message": f"arm {arm.name} failed: {e}", "type": "proxy_error"}), 500

    sent = parse_path(outcome.path)
    sent_model = sent.model if sent else model
    sent_feats = extract_features(outcome.body)
    res = outcome.route
    sig = res.signals if res else {}

    status, payload, vms, attempts = vertex.post(outcome.path, outcome.body)
    sig_retry = False
    if (status == 400 and settings.signature_policy == "echo"
            and "signature" in _err_msg(payload).lower()):
      _add_dummy_signatures(outcome.body, settings.dummy_signature)
      status, payload, vms, a2 = vertex.post(outcome.path, outcome.body)
      attempts += a2
      sig_retry = True

    ok = status == 200
    usage = convert.usage_of(payload) if ok else {}
    cands = payload.get("candidates") or [{}] if ok else [{}]
    rec = CallRecord(
        run_id=settings.run_id, arm=settings.arm, task_id=task, ts=time.time(),
        requested_model=model, routed_model=sent_model, benchmark=benchmark, trial=trial,
        seq=next_seq(task, trial),
        router_mode=getattr(backend, "mode", "none") if arm.use_router else "none",
        router_ok=res.ok if res else None, router_error=res.error if res else None,
        router_ms=res.ms if res else None, router_reason=res.reason if res else None,
        router_headers=dict(res.headers) if res else {},
        classifier_ms=sig.get("classifier_ms"), p_deep=sig.get("p_deep"),
        thinking_source=sig.get("thinking_source"),
        rewritten=bool(res and (res.path != path or res.body != json.dumps(
            body, separators=(",", ":"), ensure_ascii=False).encode())),
        agent_thinking_level=feats.agent_thinking_level, thinking_level=sent_feats.agent_thinking_level,
        n_tools=feats.tool_declarations, est_input_tokens=feats.est_input_tokens,
        vertex_status=status, vertex_ms=vms, attempts=attempts, signature_retry=sig_retry,
        finish_reason=cands[0].get("finishReason") if ok else None,
        error=None if ok else _err_msg(payload), **usage)
    append_jsonl(calls_path, rec)
    tally["calls"] += 1
    tally["by_model"][sent_model] = tally["by_model"].get(sent_model, 0) + 1
    if not ok:
      tally["errors"] += 1
      code = 429 if status == 429 else 400 if status == 400 else 502
      return jsonify(error={"message": _err_msg(payload), "type": "upstream_error", "code": status}), code

    completion = convert.from_gemini(payload, sent_model, sigs)
    completion["x_autorouter"] = {"routed_model": sent_model, "thinking_level": sent_feats.agent_thinking_level,
                                  "reason": res.reason if res else None, "arm": settings.arm}
    if req.get("stream"):
      return Response("".join(convert.to_sse(completion)), mimetype="text/event-stream")
    return jsonify(completion)

  return app


# --------------------------------------------------------------------- CLI

def build_backend(args, cfg: Config) -> Backend | None:
  if not ARMS[args.arm].use_router:
    return None
  if args.router == "inprocess":
    if not cfg.gemini_api_key:
      raise SystemExit("in-process routing needs GEMINI_API_KEY (the Decisions API key)")
    return InProcessRouter(cfg)
  if args.router == "remote":
    if not args.router_target:
      raise SystemExit("--router remote needs --router-target host:port (or ROUTER_TARGET)")
    return RemoteRouter(args.router_target, tls=args.router_tls, auth=args.router_auth,
                        audience=args.router_audience, timeout=args.router_timeout)
  raise SystemExit(f"arm {args.arm} needs --router remote|inprocess")


def parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--arm", required=True, choices=sorted(ARMS))
  p.add_argument("--out", required=True, help="run directory; calls.jsonl is appended here")
  p.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
  p.add_argument("--benchmark", default=None)
  p.add_argument("--host", default="127.0.0.1")
  p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8088")))
  p.add_argument("--router", choices=["remote", "inprocess", "none"], default=os.getenv("ROUTER_MODE", "remote"))
  p.add_argument("--router-target", default=os.getenv("ROUTER_TARGET", ""))
  p.add_argument("--router-tls", action="store_true", default=os.getenv("ROUTER_TLS", "").lower() in ("1", "true", "yes"))
  p.add_argument("--router-auth", choices=["none", "id_token"], default=os.getenv("ROUTER_AUTH", "none"))
  p.add_argument("--router-audience", default=os.getenv("ROUTER_AUDIENCE") or None)
  p.add_argument("--router-timeout", type=float, default=10.0)
  p.add_argument("--requested-model", default=os.getenv("REQUESTED_MODEL", ""),
                 help="model named in the request path before routing; default: Argon")
  p.add_argument("--project", default=os.getenv("PROJECT_ID", ""))
  p.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "global"))
  p.add_argument("--signature-policy", choices=["echo", "drop", "dummy"], default="echo")
  p.add_argument("--dummy-signature", default=os.getenv("DUMMY_THOUGHT_SIGNATURE", convert.DUMMY_SIGNATURE))
  p.add_argument("--vertex-timeout", type=float, default=300.0)
  p.add_argument("--max-attempts", type=int, default=5)
  return p


def main(argv=None) -> None:
  args = parser().parse_args(argv)
  cfg = Config.from_env()  # models, thresholds and the Decisions key (in-process mode) from env
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  settings = Settings(
      arm=args.arm, out_dir=out, run_id=args.run_id or out.name, benchmark=args.benchmark,
      requested_model=args.requested_model or cfg.argon_model, project=args.project, location=args.location,
      signature_policy=args.signature_policy, dummy_signature=args.dummy_signature,
      vertex_timeout_s=args.vertex_timeout, max_attempts=args.max_attempts, cfg=cfg)
  backend = build_backend(args, cfg)
  vertex = VertexClient(settings.project, settings.location, settings.vertex_timeout_s, settings.max_attempts)
  (out / "proxy.json").write_text(json.dumps({
      "arm": args.arm, "router_mode": getattr(backend, "mode", "none"), "router_target": args.router_target or None,
      "requested_model": settings.requested_model, "argon_model": cfg.argon_model,
      "hydrogen_model": cfg.hydrogen_model, "signature_policy": args.signature_policy,
      "location": args.location}, indent=2))
  app = create_app(settings, backend, vertex)
  app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
  main()
