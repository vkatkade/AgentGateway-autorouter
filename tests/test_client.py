"""Reference test client <-> router over real ext_proc gRPC (no network, fake classifier)."""

import importlib.util
import json
import pathlib
from concurrent import futures

import grpc
import pytest
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as pb_grpc

from autorouter.classifier import Classification
from autorouter.config import Config
from autorouter.server import AutoRouterCallout

pytest.importorskip("flask")


class _Fake:
  def __init__(self, criteria, level="HIGH"):
    self.criteria, self.level = criteria, level

  def classify(self, excerpt, ask_criteria=True):
    return Classification(criteria=self.criteria, thinking_level=self.level, thinking_confidence=0.9,
                          thinking_probs={self.level: 0.9}, p_deep=0.9, latency_ms=1)


@pytest.fixture
def router():
  def start(criteria, shadow=False):
    srv = AutoRouterCallout(Config(gemini_api_key="test", shadow_mode=shadow), classifier=_Fake(criteria))
    g = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_ExternalProcessorServicer_to_server(srv._callout_server, g)
    port = g.add_insecure_port("127.0.0.1:0")
    g.start()
    servers.append(g)
    return f"127.0.0.1:{port}"
  servers = []
  yield start
  for g in servers:
    g.stop(0)


def load_client(target, monkeypatch):
  monkeypatch.setenv("ROUTER_TARGET", target)
  monkeypatch.setenv("PROJECT_ID", "p1")
  path = pathlib.Path(__file__).resolve().parents[1] / "client" / "app.py"
  spec = importlib.util.spec_from_file_location("agw_test_client", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_client_applies_router_rewrite(router, monkeypatch):
  client = load_client(router({"code_work": 0.97}), monkeypatch)
  r = client.app.test_client().post("/api/prompt", json={"prompt": "debug my code", "route_only": True})
  d = r.get_json()
  assert d["router_ok"] and d["rewritten"]
  assert d["sent_model"] == "gemini-4-argon"
  assert d["sent_generation_config"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
  assert d["router_headers"]["x-autorouter-reason"] == "argon:code_work"


def test_client_shadow_mode_sends_original(router, monkeypatch):
  client = load_client(router({"code_work": 0.97}, shadow=True), monkeypatch)
  d = client.app.test_client().post("/api/prompt", json={"prompt": "debug", "route_only": True}).get_json()
  assert not d["rewritten"] and d["sent_model"] == "gemini-3.8-flash"
  assert d["router_headers"]["x-autorouter-reason"] == "shadow:argon:code_work"


def test_client_opt_out_and_fail_open(router, monkeypatch):
  client = load_client(router({"code_work": 0.97}), monkeypatch)
  d = client.app.test_client().post("/api/prompt", json={"prompt": "x", "route_only": True,
                                                        "opt_out": True}).get_json()
  assert not d["rewritten"] and d["router_headers"] == {}

  down = load_client("127.0.0.1:1", monkeypatch)  # nothing listening -> fail open
  d = down.app.test_client().post("/api/prompt", json={"prompt": "x", "route_only": True}).get_json()
  assert d["router_ok"] is False and d["sent_model"] == "gemini-3.8-flash"
