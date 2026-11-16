import json
from concurrent import futures

import grpc
import httpx
import pytest
from envoy.config.core.v3.base_pb2 import HeaderMap, HeaderValue
from envoy.extensions.filters.http.ext_proc.v3.processing_mode_pb2 import ProcessingMode
from envoy.service.ext_proc.v3 import external_processor_pb2 as pb
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as pb_grpc

from autorouter import vertex
from autorouter.classifier import (Classification, ClassifierError, DecisionsClassifier,
                                   build_questions, parse_answers)
from autorouter.config import Config
from autorouter.policy import decide
from autorouter.server import AutoRouterCallout

ARGON, HYDROGEN = "gemini-4-argon", "gemini-3.8-flash"
PATH = f"/v1/projects/p1/locations/global/publishers/google/models/{HYDROGEN}:generateContent"
CFG = Config(gemini_api_key="test")


class FakeClassifier:
  def __init__(self, criteria=None, level="MEDIUM", conf=0.9, error=None):
    self.criteria = criteria if criteria is not None else {"multistep_reasoning": 0.1}
    self.level, self.conf, self.error = level, conf, error
    self.calls = []

  def classify(self, excerpt, ask_criteria=True):
    self.calls.append((excerpt, ask_criteria))
    if self.error:
      raise ClassifierError(self.error)
    return Classification(criteria=self.criteria if ask_criteria else {},
                          thinking_level=self.level, thinking_confidence=self.conf, latency_ms=12)


class Ctx:  # stands in for grpc.ServicerContext
  pass


def body(text="hi", **extra):
  b = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
  b.update(extra)
  return json.dumps(b).encode()


def headers(path=PATH, extra=()):
  hm = HeaderMap(headers=[HeaderValue(key=":path", raw_value=path.encode()),
                          HeaderValue(key=":authority", raw_value=b"aiplatform.googleapis.com"),
                          *[HeaderValue(key=k, raw_value=v.encode()) for k, v in extra]])
  return pb.ProcessingRequest(request_headers=pb.HttpHeaders(headers=hm))


def run(server, path=PATH, raw=None, extra=()):
  ctx = Ctx()
  h = server.process(headers(path, extra), ctx)
  b = server.process(pb.ProcessingRequest(request_body=pb.HttpBody(body=raw or body(), end_of_stream=True)), ctx)
  r = server.process(pb.ProcessingRequest(response_headers=pb.HttpHeaders()), ctx)
  return h, b, r


def set_headers(mut):
  return {h.header.key: h.header.raw_value.decode() for h in mut.set_headers}


def make(cls=None, **cfg):
  return AutoRouterCallout(Config(gemini_api_key="test", **cfg), classifier=cls or FakeClassifier())


# ---------------------------------------------------------------- paths

@pytest.mark.parametrize("path,model,stream", [
    (PATH, HYDROGEN, False),
    (f"/v1beta1/projects/p/locations/us-central1/publishers/google/models/{ARGON}:streamGenerateContent?alt=sse", ARGON, True),
    (f"/v1/publishers/google/models/{ARGON}:generateContent", ARGON, False),
])
def test_parse_path(path, model, stream):
  p = vertex.parse_path(path)
  assert p.model == model and p.streaming == stream
  assert p.with_model("X").replace("X", model) == path


@pytest.mark.parametrize("path", ["/v1/projects/p/locations/l/publishers/google/models/m:countTokens",
                                  "/v1/projects/p/locations/l/endpoints/openapi/chat/completions", "/healthz"])
def test_non_generate_paths_ignored(path):
  assert vertex.parse_path(path) is None


# ---------------------------------------------------------------- policy

def feats(text="hi", **b):
  return vertex.extract_features(json.loads(body(text, **b)))


def test_policy_argon_when_criterion_high():
  d = decide(CFG, HYDROGEN, feats(), FakeClassifier({"code_work": 0.9, "high_stakes": 0.2}, level="HIGH"))
  assert d.model == ARGON and d.reason == "argon:code_work" and d.thinking_level == "HIGH"


def test_policy_hydrogen_when_all_low():
  d = decide(CFG, ARGON, feats(), FakeClassifier({"code_work": 0.1, "high_stakes": 0.05}, level="LOW"))
  assert d.model == HYDROGEN and d.thinking_level == "LOW"


def test_policy_same_model_only_sets_thinking():
  d = decide(CFG, HYDROGEN, feats(), FakeClassifier({"code_work": 0.1}, level="MEDIUM"))
  assert d.model is None and d.thinking_level == "MEDIUM" and d.changes_request


def test_policy_uncertain_passes_through():
  d = decide(CFG, HYDROGEN, feats(), FakeClassifier({"code_work": 0.5}))
  assert d.reason == "uncertain" and not d.changes_request


def test_policy_classifier_error_passes_through():
  d = decide(CFG, HYDROGEN, feats(), FakeClassifier(error="ReadTimeout"))
  assert d.reason == "classifier_error" and not d.changes_request


def test_policy_low_thinking_confidence_keeps_thinking():
  d = decide(CFG, HYDROGEN, feats(), FakeClassifier({"code_work": 0.9}, conf=0.3))
  assert d.model == ARGON and d.thinking_level is None


def test_policy_unroutable_model_untouched():
  cls = FakeClassifier()
  d = decide(CFG, "gemini-2.5-pro", feats(), cls)
  assert d.reason == "model_not_routable" and not cls.calls


def test_policy_long_context_forces_argon_and_skips_criteria():
  cls = FakeClassifier({"code_work": 0.0}, level="HIGH")
  d = decide(CFG, HYDROGEN, feats("x" * 4 * 40_000), cls)
  assert d.model == ARGON and d.reason == "long_context" and cls.calls[0][1] is False


def test_policy_many_tools_survives_classifier_error():
  tools = [{"functionDeclarations": [{"name": f"t{i}"} for i in range(9)]}]
  d = decide(CFG, HYDROGEN, feats(tools=tools), FakeClassifier(error="boom"))
  assert d.model == ARGON and d.reason == "many_tools" and d.thinking_level is None


def test_thinking_clamped_to_supported_levels():
  cfg = Config(gemini_api_key="t", hydrogen_levels=("LOW", "HIGH"))
  d = decide(cfg, ARGON, feats(), FakeClassifier({"code_work": 0.0}, level="MEDIUM"))
  assert d.model == HYDROGEN and d.thinking_level == "HIGH"


# ---------------------------------------------------------------- body + excerpt

def test_set_thinking_level_drops_budget_and_keeps_other_fields():
  b = {"generationConfig": {"temperature": 0.2, "thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": True}}}
  vertex.set_thinking_level(b, "HIGH")
  assert b["generationConfig"] == {"temperature": 0.2, "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "HIGH"}}


def test_set_thinking_level_snake_case_body():
  b = {"generation_config": {"thinking_config": {"thinking_budget": 5}}}
  vertex.set_thinking_level(b, "LOW", case="lower")
  assert b == {"generation_config": {"thinking_config": {"thinkingLevel": "low"}}}


def test_features_and_redacted_excerpt():
  b = {
      "systemInstruction": {"parts": [{"text": "You are a finance agent."}]},
      "contents": [
          {"role": "user", "parts": [{"text": "Email bob@acme.com, key AIzaSyA1234567890abcdefghijklmnopqrstu"}]},
          {"role": "model", "parts": [{"functionCall": {"name": "lookup"}}]},
          {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {}}}]},
      ],
      "tools": [{"functionDeclarations": [{"name": "lookup"}, {"name": "pay"}]}],
  }
  f = vertex.extract_features(b)
  assert (f.turns, f.tool_declarations, f.last_turn_kind) == (3, 2, "tool_result")
  ex = vertex.build_excerpt(f, 2000)
  assert "[EMAIL]" in ex and "[SECRET]" in ex and "bob@acme.com" not in ex
  assert "finance agent" in ex and "tool_declarations: 2" in ex


def test_excerpt_truncates():
  ex = vertex.build_excerpt(feats("a" * 50_000), 1000)
  assert len(ex) < 1400 and "[...]" in ex


# ---------------------------------------------------------------- server phases

def test_server_rewrites_path_body_and_headers():
  srv = make(FakeClassifier({"multistep_reasoning": 0.95}, level="HIGH"))
  h, b, r = run(srv, raw=body("prove it", generationConfig={"thinkingConfig": {"thinkingBudget": 99}}))
  assert h.mode_override.request_body_mode == ProcessingMode.BUFFERED
  resp = b.request_body.response
  new = json.loads(resp.body_mutation.body)
  assert new["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
  hs = set_headers(resp.header_mutation)
  assert hs[":path"] == PATH.replace(HYDROGEN, ARGON)
  assert hs["content-length"] == str(len(resp.body_mutation.body))
  assert hs["x-autorouter-reason"] == "argon:multistep_reasoning"
  rh = set_headers(r.response_headers.response.header_mutation)
  assert rh["x-autorouter-model"] == ARGON and rh["x-autorouter-thinking"] == "HIGH"


def test_server_pass_through_on_uncertain():
  srv = make(FakeClassifier({"code_work": 0.5}))
  _, b, r = run(srv)
  assert not b.request_body.HasField("response")
  assert set_headers(r.response_headers.response.header_mutation)["x-autorouter-reason"] == "uncertain"


def test_server_opt_out_header_skips_body():
  cls = FakeClassifier({"code_work": 0.99})
  h, b, _ = run(make(cls), extra=[("x-autorouter", "off")])
  assert h.mode_override.request_body_mode == ProcessingMode.NONE
  assert not cls.calls and not b.request_body.HasField("response")


def test_server_ignores_other_paths():
  cls = FakeClassifier()
  h, _, _ = run(make(cls), path="/v1/projects/p/locations/l/publishers/google/models/m:countTokens")
  assert h.mode_override.request_body_mode == ProcessingMode.NONE and not cls.calls


def test_server_shadow_mode_logs_only():
  _, b, r = run(make(FakeClassifier({"code_work": 0.99}), shadow_mode=True))
  assert not b.request_body.HasField("response")
  hs = set_headers(r.response_headers.response.header_mutation)
  assert hs["x-autorouter-model"] == HYDROGEN and hs["x-autorouter-reason"] == "shadow:argon:code_work"


def test_server_bad_json_passes_through():
  _, b, _ = run(make(), raw=b"{not json")
  assert not b.request_body.HasField("response")


def test_server_streamed_body_not_routed():
  srv, ctx = make(FakeClassifier({"code_work": 0.99})), Ctx()
  srv.process(headers(), ctx)
  first = srv.process(pb.ProcessingRequest(request_body=pb.HttpBody(body=b'{"contents"', end_of_stream=False)), ctx)
  last = srv.process(pb.ProcessingRequest(request_body=pb.HttpBody(body=b':[]}', end_of_stream=True)), ctx)
  assert not first.request_body.HasField("response") and not last.request_body.HasField("response")


# ---------------------------------------------------------------- Decisions API client

def test_decisions_client_request_and_parse():
  seen = {}

  def handler(request: httpx.Request):
    seen["key"] = request.headers["x-goog-api-key"]
    seen["json"] = json.loads(request.content)
    return httpx.Response(200, json={"answers": {
        "code_work": {"type": "choice", "choice": "yes", "confidence": 0.97,
                      "probabilities": {"yes": 0.97, "no": 0.03}},
        "multistep_reasoning": {"type": "choice", "choice": "no", "confidence": 0.8,
                                "probabilities": {"no": 0.8, "yes": 0.2}},
        "thinking_level": {"type": "choice", "choice": "MEDIUM", "confidence": 0.7,
                           "probabilities": {"LOW": 0.1, "MEDIUM": 0.7, "HIGH": 0.2}},
    }})

  client = httpx.Client(transport=httpx.MockTransport(handler), headers={"x-goog-api-key": "k"})
  c = DecisionsClassifier(CFG, client=client).classify("excerpt")
  assert seen["key"] == "k" and seen["json"]["model"] == "gemini-decision-1"
  questions = seen["json"]["questions"]
  assert set(questions) == set(build_questions())
  assert all(q["type"] == "choice" for q in questions.values())  # Choice questions only
  assert set(questions["thinking_level"]["criteria"]) == {"LOW", "MEDIUM", "HIGH"}
  assert c.criteria["code_work"] == 0.97 and c.criteria["multistep_reasoning"] == 0.2
  assert c.thinking_level == "MEDIUM" and c.thinking_confidence == 0.7


def test_parse_answers_ignores_bad_shapes():
  c = parse_answers({"answers": {"code_work": {"choice": "yes"},  # no probabilities
                                 "thinking_level": {"choice": "MINIMAL", "confidence": 0.9}}})
  assert c.criteria == {} and c.thinking_level is None


def test_decisions_client_errors_become_classifier_error():
  client = httpx.Client(transport=httpx.MockTransport(
      lambda r: httpx.Response(400, json={"error": {"code": "safety", "message": "blocked"}})),
      headers={"x-goog-api-key": "k"})
  with pytest.raises(ClassifierError, match="400 safety"):
    DecisionsClassifier(CFG, client=client).classify("x")


def test_decisions_client_without_key_skips_network():
  def boom(request):
    raise AssertionError("should not call the API")
  client = httpx.Client(transport=httpx.MockTransport(boom))
  with pytest.raises(ClassifierError, match="no API key"):
    DecisionsClassifier(CFG, client=client).classify("x")


# ---------------------------------------------------------------- real gRPC round trip

def test_grpc_round_trip():
  srv = make(FakeClassifier({"code_work": 0.99}, level="LOW"), port=0)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
  pb_grpc.add_ExternalProcessorServicer_to_server(srv._callout_server, server)
  port = server.add_insecure_port("127.0.0.1:0")
  server.start()
  try:
    stub = pb_grpc.ExternalProcessorStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
    msgs = [headers(), pb.ProcessingRequest(request_body=pb.HttpBody(body=body(), end_of_stream=True))]
    out = list(stub.Process(iter(msgs)))
    hs = set_headers(out[1].request_body.response.header_mutation)
    assert hs[":path"].endswith(f"{ARGON}:generateContent")
    assert json.loads(out[1].request_body.response.body_mutation.body)["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "LOW"
  finally:
    server.stop(0)
