"""Result records shared by the proxy, the runners and the analysis.

Three files per run directory (one directory per arm):

  manifest.json   one RunManifest: what was run and how
  calls.jsonl     one CallRecord per LLM call that went through the proxy
  tasks.jsonl     one TaskResult per (task, trial), written by the runner

The JSON Schemas in eval/agentic/schemas/ are generated from these dataclasses
(`python -m eval.agentic.schema --write`); a test keeps them in sync.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import types
import typing
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
BENCHMARKS = ("tau3_banking", "swebench_verified")
SCHEMA_DIR = pathlib.Path(__file__).parent / "schemas"


@dataclass
class CallRecord:
  """One generateContent call made on behalf of the agent."""
  run_id: str
  arm: str
  task_id: str
  ts: float
  requested_model: str
  routed_model: str
  benchmark: str | None = None
  trial: int | None = None
  seq: int | None = None                      # 1-based call number within (task, trial)
  schema_version: int = SCHEMA_VERSION
  # --- routing ---
  router_mode: str = "none"                   # remote | inprocess | none
  router_ok: bool | None = None               # False: router unreachable/errored, request failed open
  router_error: str | None = None
  router_ms: float | None = None
  router_reason: str | None = None            # x-autorouter-reason / Decision.reason
  router_headers: dict = field(default_factory=dict)
  classifier_ms: float | None = None
  p_deep: float | None = None
  thinking_source: str | None = None
  rewritten: bool = False                     # model or body changed by the router
  # --- what was sent ---
  agent_thinking_level: str | None = None     # thinkingLevel the harness asked for, if any
  thinking_level: str | None = None           # thinkingLevel actually sent to Vertex
  n_tools: int = 0
  est_input_tokens: int = 0
  # --- Vertex ---
  vertex_status: int | None = None
  vertex_ms: float | None = None
  attempts: int = 1
  signature_retry: bool = False               # retried with a dummy thought signature
  finish_reason: str | None = None
  error: str | None = None
  # --- usage (Vertex usageMetadata) ---
  prompt_tokens: int = 0
  output_tokens: int = 0                      # candidatesTokenCount
  thoughts_tokens: int = 0                    # thoughtsTokenCount (billed as output)
  cached_tokens: int = 0
  total_tokens: int = 0


@dataclass
class TaskResult:
  """The outcome of one (task, trial)."""
  run_id: str
  arm: str
  benchmark: str
  task_id: str
  success: bool
  trial: int = 0
  score: float | None = None                  # benchmark reward when it is not binary
  termination: str | None = None              # benchmark's own termination reason
  duration_s: float | None = None
  error: str | None = None                    # harness failure: the task did not really run
  extra: dict = field(default_factory=dict)
  schema_version: int = SCHEMA_VERSION


@dataclass
class RunManifest:
  run_id: str
  arm: str
  benchmark: str
  created_at: str
  router_mode: str = "none"
  router_target: str | None = None
  argon_model: str | None = None
  hydrogen_model: str | None = None
  requested_model: str | None = None
  seed: int | None = None
  task_ids: list = field(default_factory=list)
  trials: int = 1
  harness: dict = field(default_factory=dict)     # versions, commands, flags
  router_env: dict = field(default_factory=dict)  # thresholds etc. the router ran with (no secrets)
  notes: str = ""
  schema_version: int = SCHEMA_VERSION


# ----------------------------------------------------------------------- io

def to_dict(obj) -> dict:
  return dataclasses.asdict(obj)


def from_dict(cls, d: dict):
  """Builds a record, ignoring unknown keys; raises ValueError if a required field is missing."""
  names = {f.name for f in dataclasses.fields(cls)}
  required = {f.name for f in dataclasses.fields(cls)
              if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
  missing = sorted(required - d.keys())
  if missing:
    raise ValueError(f"{cls.__name__}: missing required field(s) {missing}")
  return cls(**{k: v for k, v in d.items() if k in names})


def append_jsonl(path: pathlib.Path, obj) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "a", encoding="utf-8") as f:  # one write() per line: safe across threads/processes
    f.write(json.dumps(to_dict(obj), ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: pathlib.Path, cls) -> list:
  out = []
  if not path.exists():
    return out
  with open(path, encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      try:
        out.append(from_dict(cls, json.loads(line)))
      except (ValueError, TypeError) as e:
        raise ValueError(f"{path}:{n}: {e}") from e
  return out


# -------------------------------------------------------------- JSON Schema

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _type_schema(tp) -> dict:
  origin = typing.get_origin(tp)
  if origin in (typing.Union, types.UnionType):
    args = typing.get_args(tp)
    non_null = [a for a in args if a is not type(None)]
    inner = _type_schema(non_null[0]) if len(non_null) == 1 else {}
    if type(None) in args and "type" in inner:
      inner = {**inner, "type": [inner["type"], "null"]}
    return inner
  if tp in _SCALARS:
    return {"type": _SCALARS[tp]}
  return {}


def json_schema(cls) -> dict:
  hints = typing.get_type_hints(cls)
  props, required = {}, []
  for f in dataclasses.fields(cls):
    props[f.name] = _type_schema(hints[f.name])
    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
      required.append(f.name)
  return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": cls.__name__,
          "type": "object", "properties": props, "required": required, "additionalProperties": True}


SCHEMAS = {"call_record": CallRecord, "task_result": TaskResult, "run_manifest": RunManifest}


def schema_files() -> dict[str, str]:
  return {f"{name}.schema.json": json.dumps(json_schema(cls), indent=2) + "\n" for name, cls in SCHEMAS.items()}


if __name__ == "__main__":
  import sys
  files = schema_files()
  if "--write" in sys.argv:
    SCHEMA_DIR.mkdir(exist_ok=True)
    for name, text in files.items():
      (SCHEMA_DIR / name).write_text(text)
      print("wrote", SCHEMA_DIR / name)
  else:
    for name, text in files.items():
      print(f"// {name}\n{text}")
