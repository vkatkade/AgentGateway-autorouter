"""eval/agentic: schema, converters, arms, proxy (real ext_proc gRPC to a local router), sampler,
runner helpers and the analysis. No network and no real model calls."""

import json
import math
import random
from concurrent import futures

import grpc
import pytest
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as pb_grpc

from autorouter.classifier import Classification
from autorouter.config import Config
from autorouter.server import AutoRouterCallout
from eval.agentic import analyze, convert, run_swebench, run_tau3, sample_swebench, schema
from eval.agentic.arms import ARMS, InProcessRouter, apply_arm, restore_thinking
from eval.agentic.extproc_client import RemoteRouter, RouteResult
from eval.agentic.proxy import Settings, create_app

pytest.importorskip("flask")

CFG = Config(gemini_api_key="test")
PATH = "/v1/projects/p1/locations/global/publishers/google/models/gemini-4-argon:generateContent"


class _Fake:
  def __init__(self, criteria, level="HIGH", p_deep=0.9):
    self.criteria, self.level, self.p_deep = criteria, level, p_deep

  def classify(self, excerpt, ask_criteria=True):
    return Classification(criteria=self.criteria, thinking_level=self.level, thinking_confidence=0.9,
                          thinking_probs={self.level: 0.9}, p_deep=self.p_deep, latency_ms=3.0)


@pytest.fixture
def grpc_router():
  servers = []

  def start(criteria, **kw):
    srv = AutoRouterCallout(Config(gemini_api_key="test"), classifier=_Fake(criteria, **kw))
    g = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_ExternalProcessorServicer_to_server(srv._callout_server, g)
    port = g.add_insecure_port("127.0.0.1:0")
    g.start()
    servers.append(g)
    return RemoteRouter(f"127.0.0.1:{port}")
  yield start
  for g in servers:
    g.stop(0)


# ---------------------------------------------------------------- schema

def test_jsonl_roundtrip_and_required_fields(tmp_path):
  p = tmp_path / "t.jsonl"
  schema.append_jsonl(p, schema.TaskResult("r", "router", "tau3_banking", "task_001", True, trial=2, extra={"k": 1}))
  schema.append_jsonl(p, schema.TaskResult("r", "router", "tau3_banking", "task_002", False, error="boom"))
  rows = schema.read_jsonl(p, schema.TaskResult)
  assert [r.task_id for r in rows] == ["task_001", "task_002"] and rows[0].trial == 2 and rows[1].error == "boom"
  with pytest.raises(ValueError, match="missing required"):
    schema.from_dict(schema.TaskResult, {"run_id": "r"})


def test_committed_json_schemas_are_in_sync():
  for name, text in schema.schema_files().items():
    assert (schema.SCHEMA_DIR / name).read_text() == text, f"{name} is stale: python -m eval.agentic.schema --write"


def test_schema_marks_required_and_nullable():
  s = schema.json_schema(schema.TaskResult)
  assert {"run_id", "arm", "benchmark", "task_id", "success"} <= set(s["required"])
  assert s["properties"]["error"]["type"] == ["string", "null"]


# --------------------------------------------------------------- convert

def _openai_req():
  return {
      "model": "openai/agent", "temperature": 0.2, "max_tokens": 512, "reasoning_effort": "low",
      "messages": [
          {"role": "system", "content": "be brief"},
          {"role": "user", "content": "refund order 7"},
          {"role": "assistant", "content": None, "tool_calls": [
              {"id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\": 7}"}},
              {"id": "call_b", "type": "function", "function": {"name": "kb", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "call_a", "content": "{\"status\": \"shipped\"}"},
          {"role": "tool", "tool_call_id": "call_b", "content": "plain text"},
      ],
      "tools": [{"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {
          "type": "object", "additionalProperties": False, "required": ["id"],
          "properties": {"id": {"type": ["integer", "null"], "title": "Id"}, "tags": {"type": "array"}}}}},
                {"type": "function", "function": {"name": "noargs", "parameters": {"type": "object", "properties": {}}}}],
      "tool_choice": "required"}


def test_to_gemini_maps_messages_tools_and_config():
  sigs = convert.SignatureCache()
  sigs.put("call_a", "SIG-A")
  b = convert.to_gemini(_openai_req(), sigs)
  assert b["systemInstruction"]["parts"] == [{"text": "be brief"}]
  roles = [c["role"] for c in b["contents"]]
  assert roles == ["user", "model", "user"]  # both tool results merged into one user turn
  fc = b["contents"][1]["parts"]
  assert fc[0]["functionCall"] == {"name": "lookup", "args": {"id": 7}} and fc[0]["thoughtSignature"] == "SIG-A"
  assert "thoughtSignature" not in fc[1]
  resp = b["contents"][2]["parts"]
  assert resp[0]["functionResponse"] == {"name": "lookup", "response": {"status": "shipped"}}
  assert resp[1]["functionResponse"] == {"name": "kb", "response": {"output": "plain text"}}
  decls = b["tools"][0]["functionDeclarations"]
  params = decls[0]["parameters"]
  assert params["type"] == "OBJECT" and "additionalProperties" not in params
  assert params["properties"]["id"] == {"type": "INTEGER", "title": "Id", "nullable": True}
  assert params["properties"]["tags"]["items"] == {"type": "STRING"}
  assert "parameters" not in decls[1]  # empty OBJECT schemas are rejected by Vertex
  assert b["toolConfig"]["functionCallingConfig"] == {"mode": "ANY"}
  assert b["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 512, "thinkingConfig": {"thinkingLevel": "LOW"}}


def test_signature_policies():
  req = _openai_req()
  sigs = convert.SignatureCache()
  sigs.put("call_a", "SIG-A")
  drop = convert.to_gemini(req, sigs, "drop")["contents"][1]["parts"]
  dummy = convert.to_gemini(req, sigs, "dummy", "DUMMY")["contents"][1]["parts"]
  assert all("thoughtSignature" not in p for p in drop)
  assert [p["thoughtSignature"] for p in dummy] == ["DUMMY", "DUMMY"]


def test_from_gemini_text_toolcalls_usage_and_signature_cache():
  sigs = convert.SignatureCache()
  resp = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
      {"text": "thinking...", "thought": True},
      {"text": "Hello"},
      {"functionCall": {"name": "lookup", "args": {"id": 1}}, "thoughtSignature": "S1"}]}}],
      "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 40,
                        "cachedContentTokenCount": 20, "totalTokenCount": 150}}
  out = convert.from_gemini(resp, "gemini-4-argon", sigs)
  msg = out["choices"][0]["message"]
  assert msg["content"] == "Hello" and out["choices"][0]["finish_reason"] == "tool_calls"
  tc = msg["tool_calls"][0]
  assert json.loads(tc["function"]["arguments"]) == {"id": 1} and sigs.get(tc["id"]) == "S1"
  assert out["usage"]["prompt_tokens"] == 100 and out["usage"]["completion_tokens"] == 50
  assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 40
  chunks = convert.to_sse(out)
  assert chunks[-1] == "data: [DONE]\n\n" and "tool_calls" in chunks[0]


def test_from_gemini_blocked_response_is_content_filter():
  out = convert.from_gemini({"promptFeedback": {"blockReason": "SAFETY"}}, "m")
  assert out["choices"][0]["finish_reason"] == "content_filter"


# ------------------------------------------------------------------ arms

def _body(level=None):
  b = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
  if level:
    b["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
  return b


def test_fixed_arms_rewrite_model_and_thinking():
  r = apply_arm(ARMS["hydrogen"], CFG, None, PATH, _body("HIGH"))
  assert "models/gemini-3.8-flash:" in r.path and r.body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
  r = apply_arm(ARMS["argon_medium"], CFG, None, PATH.replace("gemini-4-argon", "gemini-3.8-flash"), _body("LOW"))
  assert "models/gemini-4-argon:" in r.path and r.body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MEDIUM"


def test_restore_thinking_puts_back_or_removes():
  new = {"generationConfig": {"temperature": 1, "thinkingConfig": {"thinkingLevel": "HIGH"}}}
  restore_thinking(new, _body())            # harness set no thinking -> router's is removed
  assert new["generationConfig"] == {"temperature": 1}
  new = {"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}
  restore_thinking(new, _body("LOW"))
  assert new["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
  only = {"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}
  restore_thinking(only, _body())
  assert "generationConfig" not in only


def test_router_arms_with_remote_router(grpc_router):
  r = grpc_router({"code_work": 0.97})
  full = apply_arm(ARMS["router"], CFG, r, PATH.replace("gemini-4-argon", "gemini-3.8-flash"), _body())
  assert "models/gemini-4-argon:" in full.path
  assert full.body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
  assert full.route.reason == "argon:code_work"
  abl = apply_arm(ARMS["router_no_thinking"], CFG, r, PATH.replace("gemini-4-argon", "gemini-3.8-flash"), _body())
  assert "models/gemini-4-argon:" in abl.path and "generationConfig" not in abl.body


def test_inprocess_router_matches_policy():
  r = InProcessRouter(CFG, classifier=_Fake({"code_work": 0.97}, p_deep=0.9))
  res = r.route(PATH.replace("gemini-4-argon", "gemini-3.8-flash"), json.dumps(_body()).encode())
  assert "models/gemini-4-argon:" in res.path and res.headers["x-autorouter-thinking"] == "HIGH"
  assert res.signals["thinking_source"] == "argon_depth" and res.signals["classifier_ms"] == 3.0
  skipped = r.route("/some/other/path", b"{}")
  assert skipped.reason == "skipped"


# ----------------------------------------------------------------- proxy

class FakeVertex:
  def __init__(self, responses=None):
    self.posts, self.responses, self.project = [], list(responses or []), "p1"

  def project_id(self):
    return self.project

  def post(self, path, body):
    self.posts.append((path, json.loads(json.dumps(body))))
    if self.responses:
      status, payload = self.responses.pop(0)
    else:
      status, payload = 200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}}],
                              "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2,
                                                "thoughtsTokenCount": 5, "totalTokenCount": 17}}
    return status, payload, 12.5, 1


def make_app(tmp_path, arm, backend, vertex, **kw):
  s = Settings(arm=arm, out_dir=tmp_path, run_id="r1", benchmark="tau3_banking", requested_model="gemini-4-argon",
               project="p1", cfg=CFG, **kw)
  return create_app(s, backend, vertex).test_client()


HDR = {"x-eval-task": "task_007", "x-eval-trial": "1"}
REQ = {"model": "openai/agent", "messages": [{"role": "user", "content": "fix this failing test"}]}


def test_proxy_router_arm_end_to_end(tmp_path, grpc_router):
  fv = FakeVertex()
  client = make_app(tmp_path, "router", grpc_router({"code_work": 0.97}), fv)
  r = client.post("/v1/chat/completions", json=REQ, headers=HDR)
  assert r.status_code == 200
  out = r.get_json()
  assert out["choices"][0]["message"]["content"] == "ok" and out["x_autorouter"]["reason"] == "argon:code_work"
  path, body = fv.posts[0]
  assert "models/gemini-4-argon:" in path and body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"

  rec = schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)[0]
  assert (rec.arm, rec.task_id, rec.trial, rec.seq, rec.benchmark) == ("router", "task_007", 1, 1, "tau3_banking")
  assert rec.routed_model == "gemini-4-argon" and rec.thinking_level == "HIGH" and rec.router_ok is True
  assert rec.router_reason == "argon:code_work" and rec.rewritten is True
  assert (rec.prompt_tokens, rec.output_tokens, rec.thoughts_tokens, rec.total_tokens) == (10, 2, 5, 17)
  assert client.post("/v1/chat/completions", json=REQ, headers=HDR).status_code == 200
  assert schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)[1].seq == 2


def test_proxy_requested_model_is_argon_unless_a_routable_model_is_named(tmp_path):
  fv = FakeVertex()
  client = make_app(tmp_path, "argon", None, fv)
  client.post("/v1/chat/completions", json={**REQ, "model": "gemini-3.8-flash"})
  client.post("/v1/chat/completions", json={**REQ, "model": "openai/whatever"})
  rec = schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)
  assert [x.requested_model for x in rec] == ["gemini-3.8-flash", "gemini-4-argon"]
  assert [x.routed_model for x in rec] == ["gemini-4-argon", "gemini-4-argon"]  # the arm decides


def test_proxy_fail_open_when_router_is_down(tmp_path):
  fv = FakeVertex()
  client = make_app(tmp_path, "router", RemoteRouter("127.0.0.1:1", timeout=1.0), fv)
  assert client.post("/v1/chat/completions", json=REQ, headers=HDR).status_code == 200
  rec = schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)[0]
  assert rec.router_ok is False and rec.router_error and rec.routed_model == "gemini-4-argon"
  assert not rec.rewritten and fv.posts[0][0] == PATH


def test_proxy_stream_and_error_mapping(tmp_path):
  fv = FakeVertex()
  client = make_app(tmp_path, "argon", None, fv)
  r = client.post("/v1/chat/completions", json={**REQ, "stream": True})
  assert r.mimetype == "text/event-stream" and r.get_data(as_text=True).endswith("data: [DONE]\n\n")
  fv.responses = [(429, {"error": {"message": "quota"}})]
  r = client.post("/v1/chat/completions", json=REQ)
  assert r.status_code == 429
  rec = schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)[-1]
  assert rec.vertex_status == 429 and rec.error == "quota"
  assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_vertex_client_credential_failure_is_not_retried(monkeypatch):
  from google.auth.exceptions import DefaultCredentialsError
  from eval.agentic.proxy import VertexClient
  vc = VertexClient("p1", "global", 5.0, max_attempts=5)
  monkeypatch.setattr(vc, "_token", lambda: (_ for _ in ()).throw(DefaultCredentialsError("no adc")))
  status, payload, ms, attempts = vc.post("/x", {})
  assert status == 401 and attempts == 1 and "no adc" in payload["error"]["message"]


def test_vertex_client_uses_static_access_token(monkeypatch):
  from eval.agentic.proxy import VertexClient
  monkeypatch.setenv("VERTEX_ACCESS_TOKEN", "ya29.static")
  vc = VertexClient("p1", "global", 5.0, 1)
  assert vc._token() == "ya29.static" and vc.project_id() == "p1"


def test_vertex_client_retries_transient_statuses(monkeypatch):
  import eval.agentic.proxy as px
  seen = []

  class R:
    def __init__(self, status):
      self.status_code, self.text = status, "x"

    def json(self):
      return {"candidates": []} if self.status_code == 200 else {"error": {"message": "busy"}}
  codes = iter([503, 429, 200])
  monkeypatch.setattr(px.requests, "post", lambda *a, **k: seen.append(1) or R(next(codes)))
  monkeypatch.setattr(px.time, "sleep", lambda s: None)
  vc = px.VertexClient("p1", "global", 5.0, max_attempts=5)
  monkeypatch.setattr(vc, "_token", lambda: "t")
  status, _, _, attempts = vc.post("/x", {})
  assert status == 200 and attempts == 3 and len(seen) == 3


def test_proxy_retries_once_with_dummy_signature(tmp_path):
  toolcall = {"role": "assistant", "content": None, "tool_calls": [
      {"id": "call_x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
  req = {"messages": [{"role": "user", "content": "go"}, toolcall,
                      {"role": "tool", "tool_call_id": "call_x", "content": "{}"}]}
  fv = FakeVertex([(400, {"error": {"message": "Function call is missing a thought_signature"}})])
  client = make_app(tmp_path, "argon", None, fv)
  assert client.post("/v1/chat/completions", json=req).status_code == 200
  assert len(fv.posts) == 2
  assert fv.posts[1][1]["contents"][1]["parts"][0]["thoughtSignature"] == convert.DUMMY_SIGNATURE
  assert schema.read_jsonl(tmp_path / "calls.jsonl", schema.CallRecord)[0].signature_retry is True


def test_proxy_echoes_signature_from_previous_response(tmp_path):
  first = (200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
      {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "SIG"}]}}]})
  fv = FakeVertex([first])
  client = make_app(tmp_path, "argon", None, fv)
  msg = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}]}).get_json()
  call = msg["choices"][0]["message"]["tool_calls"][0]
  follow = {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": call["id"], "content": "{}"}]}
  client.post("/v1/chat/completions", json=follow)
  assert fv.posts[1][1]["contents"][1]["parts"][0]["thoughtSignature"] == "SIG"


# --------------------------------------------------------------- sampler

def _population():
  rows = []
  for repo, n in (("a/a", 60), ("b/b", 25), ("c/c", 10), ("d/d", 5)):
    for i in range(n):
      rows.append({"instance_id": f"{repo.split('/')[0]}-{i:03d}", "repo": repo,
                   "difficulty": ("<15 min fix", "15 min - 1 hour", "1-4 hours")[i % 3]})
  return rows


def test_sample_is_deterministic_proportional_and_order_independent():
  pop = _population()
  a = sample_swebench.stratified_sample(pop, 20, seed=1)
  shuffled = pop[:]
  random.Random(9).shuffle(shuffled)
  assert a == sample_swebench.stratified_sample(shuffled, 20, seed=1)
  assert a != sample_swebench.stratified_sample(pop, 20, seed=2)
  assert len(a) == len(set(a)) == 20
  by_repo = {r: sum(i.startswith(r[0]) for i in a) for r in ("a/a", "b/b", "c/c", "d/d")}
  assert by_repo == {"a/a": 12, "b/b": 5, "c/c": 2, "d/d": 1}


def test_largest_remainder_sums_and_bounds():
  q = sample_swebench.largest_remainder({"x": 7, "y": 3, "z": 1}, 5)
  assert sum(q.values()) == 5 and all(q[k] <= v for k, v in {"x": 7, "y": 3, "z": 1}.items())
  with pytest.raises(ValueError):
    sample_swebench.largest_remainder({"x": 1}, 2)


def test_committed_sample_is_valid():
  doc = json.loads(sample_swebench.DEFAULT_OUT.read_text())
  ids = doc["instance_ids"]
  assert doc["n"] == len(ids) == len(set(ids)) == 100 and ids == sorted(ids)
  assert doc["composition"]["population"]["n"] == 500 and doc["composition"]["sample"]["n"] == 100


# --------------------------------------------------------------- runners

class _A:  # argparse stand-in
  tau2_cmd = "uv run tau2"
  agent_temperature = None
  agent_llm_args = '{"reasoning_effort": "low"}'
  user_llm = "vertex_ai/gemini-3.8-flash"
  user_llm_args = '{"temperature": 0}'
  retrieval_config = "bm25"
  seed = 300
  max_steps = 100
  tau2_extra = "--timeout 600"


def test_tau3_command_carries_attribution_and_fixed_user():
  cmd = run_tau3.build_command(_A, "task_042", 2, "http://127.0.0.1:9", "save")
  assert cmd[:3] == ["uv", "run", "tau2"] and cmd[cmd.index("--seed") + 1] == "302"
  llm = json.loads(cmd[cmd.index("--agent-llm-args") + 1])
  assert llm["api_base"] == "http://127.0.0.1:9/v1" and llm["reasoning_effort"] == "low"
  assert llm["extra_headers"] == {"x-eval-task": "task_042", "x-eval-trial": "2", "x-eval-benchmark": "tau3_banking"}
  assert "temperature" not in llm  # provider default unless --agent-temperature is given
  assert cmd[cmd.index("--user-llm") + 1] == "vertex_ai/gemini-3.8-flash" and cmd[-2:] == ["--timeout", "600"]


def test_tau3_parse_results(tmp_path):
  def write(sims):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"simulations": sims}))
    return p
  ok = run_tau3.parse_results(write([{"reward_info": {"reward": 1.0}, "termination_reason": "agent_stop",
                                      "duration": 12.0, "messages": [1, 2]}]), "r", "router", "task_001", 0)
  assert ok.success and ok.termination == "agent_stop" and ok.extra["messages"] == 2
  assert not run_tau3.parse_results(write([{"reward_info": {"reward": 0.0}}]), "r", "router", "t", 0).success
  assert run_tau3.parse_results(write([]), "r", "router", "t", 0).error
  assert run_tau3.parse_results(write([{"reward_info": None}]), "r", "router", "t", 0).error == "no reward_info"


def test_swebench_override_and_results():
  doc = json.loads(run_swebench.override_yaml("http://127.0.0.1:9", "django__django-1", 50, 0.5))
  assert doc["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:9/v1"
  assert doc["model"]["model_kwargs"]["extra_headers"]["x-eval-task"] == "django__django-1"
  assert doc["agent"] == {"step_limit": 50} and doc["model"]["model_kwargs"]["temperature"] == 0.5
  ids = ["a-1", "b-2", "c-3", "d-4"]
  outcomes = {"a-1": {"has_traj": True, "patch": "diff", "exit_status": "Submitted", "code": 0},
              "b-2": {"has_traj": True, "patch": "", "exit_status": "LimitsExceeded", "code": 0},
              "c-3": {"has_traj": True, "patch": "diff", "exception": "proxy 502", "exit_status": "RuntimeError"},
              "d-4": {"has_traj": False, "patch": None, "code": 1}}
  res = {r.task_id: r for r in run_swebench.to_task_results(
      "r", "router", ids, outcomes, {"resolved_ids": ["a-1"], "error_ids": []})}
  assert res["a-1"].success and res["a-1"].error is None
  assert not res["b-2"].success and res["b-2"].error is None and res["b-2"].extra["empty_patch"]
  assert res["c-3"].error.startswith("agent exception") and not res["c-3"].success
  assert "no trajectory" in res["d-4"].error
  assert run_swebench.to_task_results("r", "a", ["a-1"], {"a-1": {"has_traj": True, "patch": "x"}}, None)[0].error == "not graded"


# -------------------------------------------------------------- analysis

def test_pass_hat_k_and_sign_test():
  assert analyze.pass_hat_k([(3, 3), (3, 0)], 1) == pytest.approx(0.5)
  assert analyze.pass_hat_k([(3, 2)], 2) == pytest.approx(math.comb(2, 2) / math.comb(3, 2))
  assert analyze.pass_hat_k([(1, 1)], 2) is None
  assert analyze.sign_test_p(0, 0) == 1.0
  assert analyze.sign_test_p(5, 0) == pytest.approx(2 / 32)
  assert analyze.sign_test_p(3, 3) == 1.0


def test_paired_bootstrap_identical_series_has_zero_diff():
  a = [1.0, 0.0, 1.0, 1.0, 0.0] * 4
  point, lo, hi = analyze.paired_bootstrap(a, list(a), analyze._diff, n=300)
  assert point == 0 and lo == 0 and hi == 0
  point, lo, hi = analyze.paired_bootstrap([1.0] * 30, [0.0] * 30, analyze._diff, n=300)
  assert point == 1 and lo == 1


def _mk_run(tmp_path, arm, outcomes, model, prompt_tokens, error_task=None):
  d = tmp_path / arm
  d.mkdir()
  (d / "manifest.json").write_text(json.dumps({"arm": arm, "benchmark": "swebench_verified",
                                               "hydrogen_model": "gemini-3.8-flash", "argon_model": "gemini-4-argon"}))
  for i, ok in enumerate(outcomes):
    task = f"t{i}"
    schema.append_jsonl(d / "tasks.jsonl", schema.TaskResult(arm, arm, "swebench_verified", task, ok,
                                                             error="x" if task == error_task else None))
    schema.append_jsonl(d / "calls.jsonl", schema.CallRecord(
        run_id=arm, arm=arm, task_id=task, ts=0.0, requested_model="gemini-4-argon", routed_model=model,
        trial=0, prompt_tokens=prompt_tokens, output_tokens=100, total_tokens=prompt_tokens + 100,
        router_ok=True if arm == "router" else None, router_ms=40.0 if arm == "router" else None,
        thinking_level="MEDIUM", router_reason="hydrogen:simple"))
  return d


def test_end_to_end_analysis(tmp_path, capsys):
  base = _mk_run(tmp_path, "argon", [True] * 8 + [False] * 2, "gemini-4-argon", 1000)
  router = _mk_run(tmp_path, "router", [True] * 7 + [False] * 3, "gemini-3.8-flash", 1000, error_task="t9")
  prices = tmp_path / "prices.json"
  prices.write_text(json.dumps({"gemini-4-argon": {"input_per_m": 4.0, "output_per_m": 8.0},
                                "gemini-3.8-flash": {"input_per_m": 1.0, "output_per_m": 2.0}}))
  out = tmp_path / "report.md"
  assert analyze.main([str(base), str(router), "--prices", str(prices), "--out", str(out),
                       "--json", str(tmp_path / "r.json")]) == 0
  full = json.loads((tmp_path / "r.json").read_text())["swebench_verified"]
  argon, rt = full[0], full[1]
  assert argon["summary"]["arm"] == "argon" and argon["vs_baseline"] is None
  assert argon["summary"]["cost_per_task"] == pytest.approx((1000 * 4 + 100 * 8) / 1e6)
  assert rt["summary"]["tasks_errored"] == 1 and rt["summary"]["tasks_clean"] == 9
  assert rt["summary"]["routing"]["share_hydrogen"] == 1.0
  cmp_ = rt["vs_baseline"]
  assert cmp_["tasks_paired"] == 9                    # t9 dropped: it errored in the router arm
  assert cmp_["cost_saving"][0] == pytest.approx(0.75)  # all calls moved to a 4x cheaper model
  assert "Hydrogen" in out.read_text() and "| router |" in out.read_text()
