"""Plays Agent Gateway's part against the deployed router (Envoy ext_proc v3 over gRPC).

Same exchange as client/app.py (itself modelled on the upstream reference client):
request_headers -> one buffered request_body (end_of_stream) -> response_headers, then the
router's :path / body mutations are applied by the caller. The router must be reachable:

  * from inside the VPC: ROUTER_TARGET=10.120.0.2:80 (internal ALB over PSC), plaintext h2c;
  * from anywhere else: the Cloud Run URL on :443 with TLS and an identity token
    (ROUTER_TLS=true, ROUTER_AUTH=id_token). The caller needs roles/run.invoker.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field

import grpc
from envoy.config.core.v3.base_pb2 import HeaderMap, HeaderValue
from envoy.service.ext_proc.v3 import external_processor_pb2 as pb
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as pb_grpc

VERTEX_AUTHORITY = "aiplatform.googleapis.com"


@dataclass
class RouteResult:
  path: str
  body: bytes
  headers: dict = field(default_factory=dict)   # x-autorouter-* the router stamped
  ms: float | None = None
  ok: bool = True                               # False: router failed, request goes out unchanged
  error: str | None = None
  signals: dict = field(default_factory=dict)   # only the in-process backend fills these

  @property
  def reason(self) -> str | None:
    return self.headers.get("x-autorouter-reason")


def _hm(pairs) -> HeaderMap:
  return HeaderMap(headers=[HeaderValue(key=k, raw_value=v.encode()) for k, v in pairs])


class IdTokenSource:
  """Identity token for the router's Cloud Run service (cached ~50 min)."""

  def __init__(self, audience: str | None):
    self.audience, self._tok, self._exp, self._lock = audience, None, 0.0, threading.Lock()

  def get(self) -> str:
    with self._lock:
      if self._tok and time.time() < self._exp:
        return self._tok
      self._tok, self._exp = self._fetch(), time.time() + 50 * 60
      return self._tok

  def _fetch(self) -> str:
    try:  # service account / GCE / Cloud Run identity
      import google.auth.transport.requests
      from google.oauth2 import id_token
      return id_token.fetch_id_token(google.auth.transport.requests.Request(), self.audience or "")
    except Exception:
      # user credentials: fall back to the gcloud CLI
      out = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True, timeout=30)
      if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"cannot mint an identity token for the router: {out.stderr.strip()[:300]}")
      return out.stdout.strip()


class RemoteRouter:
  mode = "remote"

  def __init__(self, target: str, tls: bool = False, auth: str = "none", audience: str | None = None,
               timeout: float = 10.0, authority: str = VERTEX_AUTHORITY):
    self.target, self.tls, self.timeout, self.authority = target, tls, timeout, authority
    self._tokens = IdTokenSource(audience) if auth == "id_token" else None
    self._stub = None
    self._lock = threading.Lock()

  def _get_stub(self):
    with self._lock:
      if self._stub is None:  # one pooled HTTP/2 channel per process
        ch = (grpc.secure_channel(self.target, grpc.ssl_channel_credentials()) if self.tls
              else grpc.insecure_channel(self.target))
        self._stub = pb_grpc.ExternalProcessorStub(ch)
      return self._stub

  def route(self, path: str, body: bytes, opt_out: bool = False) -> RouteResult:
    req_headers = [(":method", "POST"), (":path", path), (":authority", self.authority),
                   (":scheme", "https"), ("content-type", "application/json"),
                   ("content-length", str(len(body)))]
    if opt_out:
      req_headers.append(("x-autorouter", "off"))
    msgs = [
        pb.ProcessingRequest(request_headers=pb.HttpHeaders(headers=_hm(req_headers))),
        pb.ProcessingRequest(request_body=pb.HttpBody(body=body, end_of_stream=True)),
        pb.ProcessingRequest(response_headers=pb.HttpHeaders(
            headers=_hm([(":status", "200"), ("content-type", "application/json")]))),
    ]
    start = time.perf_counter()
    try:
      metadata = [("authorization", f"Bearer {self._tokens.get()}")] if self._tokens else None
      responses = list(self._get_stub().Process(iter(msgs), timeout=self.timeout, metadata=metadata))
    except (grpc.RpcError, RuntimeError) as e:  # fail open, exactly like failOpen: true on the binding
      err = f"{e.code().name}: {e.details()}" if isinstance(e, grpc.RpcError) else str(e)
      return RouteResult(path, body, ms=round((time.perf_counter() - start) * 1000, 1), ok=False, error=err)
    ms = round((time.perf_counter() - start) * 1000, 1)

    new_path, new_body, seen = path, body, {}
    for r in responses:
      for phase in ("request_headers", "request_body", "response_headers"):
        if not r.HasField(phase):
          continue
        common = getattr(r, phase).response
        for h in common.header_mutation.set_headers:
          k, v = h.header.key, (h.header.raw_value or h.header.value.encode()).decode()
          if k == ":path":
            new_path = v
          elif k.startswith("x-autorouter"):
            seen[k] = v
        if common.body_mutation.body:
          new_body = common.body_mutation.body
    return RouteResult(new_path, new_body, seen, ms)
