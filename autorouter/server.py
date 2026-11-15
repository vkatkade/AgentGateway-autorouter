"""ext_proc callout: Agent Gateway -> (this) -> Vertex AI generateContent.

Phases:
  REQUEST_HEADERS  match the Vertex path, read opt-out header, ask for a BUFFERED body.
  REQUEST_BODY     classify, then rewrite :path (model) + thinkingConfig + content-length.
  RESPONSE_HEADERS stamp x-autorouter-* so the calling agent can see what happened.

Both models are served from the same aiplatform.googleapis.com host, so routing is
a path + body rewrite. No backend switch (and no clear_route_cache) is needed.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field

from envoy.config.core.v3.base_pb2 import HeaderValue, HeaderValueOption
from envoy.extensions.filters.http.ext_proc.v3.processing_mode_pb2 import ProcessingMode
from envoy.service.ext_proc.v3 import external_processor_pb2 as pb
from grpc import ServicerContext

from autorouter import vertex
from autorouter.classifier import Classifier, DecisionsClassifier
from autorouter.config import Config
from autorouter.policy import Decision, decide
from extproc.service import callout_server

OPT_OUT_HEADER = "x-autorouter"  # "off" -> never touch this request
H_MODEL = "x-autorouter-model"
H_THINKING = "x-autorouter-thinking"
H_REASON = "x-autorouter-reason"
_OVERWRITE = HeaderValueOption.OVERWRITE_IF_EXISTS_OR_ADD


@dataclass
class _State:
  path: vertex.VertexPath | None = None
  opted_out: bool = False
  buf: bytearray = field(default_factory=bytearray)
  decision: Decision | None = None
  final_model: str = ""


def _state(ctx: ServicerContext) -> _State:
  s = getattr(ctx, "_autorouter_state", None)
  if s is None:
    s = _State()
    setattr(ctx, "_autorouter_state", s)
  return s


def _header(headers: pb.HttpHeaders, name: str) -> str:
  for h in headers.headers.headers:
    if h.key.lower() == name:
      return (h.raw_value or h.value.encode()).decode("utf-8", "replace")
  return ""


def _set(mutation, key: str, value: str) -> None:
  mutation.set_headers.append(
      HeaderValueOption(header=HeaderValue(key=key, raw_value=value.encode()), append_action=_OVERWRITE))


def log_event(severity: str, **fields) -> None:
  """One JSON line on stdout -> jsonPayload in Cloud Logging."""
  sys.stdout.write(json.dumps({"severity": severity, "component": "autorouter", **fields}) + "\n")
  sys.stdout.flush()


class AutoRouterCallout(callout_server.CalloutServer):

  def __init__(self, cfg: Config, classifier: Classifier | None = None, **kwargs):
    kwargs.setdefault("disable_tls", True)  # Agent Gateway producers speak plaintext gRPC (h2c)
    kwargs.setdefault("combined_health_check", True)
    kwargs.setdefault("plaintext_address", ("0.0.0.0", cfg.port))
    kwargs.setdefault("server_thread_count", cfg.server_threads)
    kwargs.setdefault("cert_chain_path", None)
    kwargs.setdefault("private_key_path", None)
    super().__init__(**kwargs)
    self.cfg = cfg
    self.classifier = classifier or DecisionsClassifier(cfg)

  # ------------------------------------------------------------------ phases

  def on_request_headers(self, headers: pb.HttpHeaders, context: ServicerContext):
    s = _state(context)
    s.path = vertex.parse_path(_header(headers, ":path"))
    s.opted_out = _header(headers, OPT_OUT_HEADER).lower() in ("off", "false", "0")

    resp = pb.ProcessingResponse(request_headers=pb.HeadersResponse())
    mode = ProcessingMode()
    if s.path is None or s.opted_out:
      # Not ours: skip the body and the response (honored if the gateway allows mode override).
      mode.request_body_mode = ProcessingMode.NONE
      mode.response_header_mode = ProcessingMode.SKIP
    else:
      mode.request_body_mode = ProcessingMode.BUFFERED
      mode.response_header_mode = ProcessingMode.SEND
    mode.response_body_mode = ProcessingMode.NONE
    resp.mode_override.CopyFrom(mode)
    return resp

  def on_request_body(self, body: pb.HttpBody, context: ServicerContext):
    s = _state(context)
    if s.path is None or s.opted_out:
      return None
    s.buf.extend(body.body)
    if not body.end_of_stream:
      # STREAMED delivery: :path can no longer change once chunks flow, so don't route.
      if s.decision is None:
        s.decision = Decision(reason="body_not_buffered")
        log_event("WARNING", event="skip", reason="body_not_buffered", path_model=s.path.model)
      return None
    if s.decision is not None:
      return None

    start = time.perf_counter()
    try:
      req = json.loads(bytes(s.buf))
      if not isinstance(req, dict):
        raise ValueError("body is not a JSON object")
    except ValueError as e:
      s.decision = Decision(reason="bad_json")
      log_event("WARNING", event="skip", reason="bad_json", error=str(e))
      return None

    features = vertex.extract_features(req)
    d = decide(self.cfg, s.path.model, features, self.classifier)
    s.decision = d
    s.final_model = d.model or s.path.model

    log_event(
        "INFO", event="decision", shadow=self.cfg.shadow_mode,
        requested_model=s.path.model, routed_model=s.final_model,
        thinking_level=d.thinking_level, reason=d.reason, signals=d.signals,
        streaming=s.path.streaming, est_input_tokens=features.est_input_tokens,
        tool_declarations=features.tool_declarations,
        total_ms=round((time.perf_counter() - start) * 1000, 1),
    )

    if self.cfg.shadow_mode or not d.changes_request:
      s.final_model = s.path.model
      return None

    if d.thinking_level:
      vertex.set_thinking_level(req, d.thinking_level, self.cfg.thinking_level_case)
    new_body = json.dumps(req, separators=(",", ":"), ensure_ascii=False).encode()

    out = pb.BodyResponse()
    out.response.body_mutation.body = new_body
    hm = out.response.header_mutation
    if d.model:
      _set(hm, ":path", s.path.with_model(d.model))
    _set(hm, "content-length", str(len(new_body)))
    _set(hm, H_MODEL, s.final_model)
    _set(hm, H_REASON, d.reason)
    if d.thinking_level:
      _set(hm, H_THINKING, d.thinking_level)
    return out

  def on_response_headers(self, headers: pb.HttpHeaders, context: ServicerContext):
    s = _state(context)
    resp = pb.HeadersResponse()
    if s.decision is None:
      return resp
    hm = resp.response.header_mutation
    _set(hm, H_MODEL, s.final_model or (s.path.model if s.path else ""))
    _set(hm, H_REASON, ("shadow:" if self.cfg.shadow_mode else "") + s.decision.reason)
    if s.decision.thinking_level and not self.cfg.shadow_mode:
      _set(hm, H_THINKING, s.decision.thinking_level)
    return resp


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy
  cfg = Config.from_env()
  log_event("INFO", event="startup", argon=cfg.argon_model, hydrogen=cfg.hydrogen_model,
            routable=sorted(cfg.all_routable), shadow=cfg.shadow_mode, port=cfg.port)
  AutoRouterCallout(cfg).run()


if __name__ == "__main__":
  main()
