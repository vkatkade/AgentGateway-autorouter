"""OpenAI chat-completions <-> Vertex generateContent.

The benchmark harnesses (tau2, mini-swe-agent) talk to LLMs through LiteLLM, i.e. the
OpenAI chat-completions shape. The router only understands Vertex generateContent, so
the proxy converts at the edge. Only what the two benchmarks use is supported: text,
function tools and tool results.

Thought signatures: Gemini 3+ returns a `thoughtSignature` on function-call parts and
expects it echoed back on the next request. LiteLLM-based harnesses rebuild assistant
messages from their own types and drop unknown fields, so the signature is kept in a
server-side cache keyed by the tool-call id we minted, and re-attached here.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from typing import Any

DUMMY_SIGNATURE = "skip_thought_signature_validator"

_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
           "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
           "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter",
           "IMAGE_SAFETY": "content_filter"}

_REASONING_TO_LEVEL = {"minimal": "LOW", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH"}

# Fields Vertex accepts in a function parameter Schema.
_SCHEMA_KEYS = {"type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
                "properties", "required", "minProperties", "maxProperties", "minimum", "maximum",
                "minLength", "maxLength", "pattern", "example", "anyOf", "propertyOrdering",
                "default", "items"}


class SignatureCache:
  """tool_call id -> thoughtSignature (bounded, thread-safe)."""

  def __init__(self, max_items: int = 200_000):
    self._d: OrderedDict[str, str] = OrderedDict()
    self._lock = threading.Lock()
    self._max = max_items

  def put(self, call_id: str, sig: str) -> None:
    with self._lock:
      self._d[call_id] = sig
      self._d.move_to_end(call_id)
      while len(self._d) > self._max:
        self._d.popitem(last=False)

  def get(self, call_id: str) -> str | None:
    with self._lock:
      return self._d.get(call_id)

  def __len__(self) -> int:
    return len(self._d)


# ------------------------------------------------------------ request side

def clean_schema(node: Any) -> Any:
  """JSON Schema -> the OpenAPI subset Vertex accepts for function parameters."""
  if isinstance(node, list):
    return [clean_schema(n) for n in node]
  if not isinstance(node, dict):
    return node
  out: dict = {}
  for k, v in node.items():
    if k not in _SCHEMA_KEYS:
      continue
    if k == "properties" and isinstance(v, dict):
      out[k] = {pk: clean_schema(pv) for pk, pv in v.items()}
    elif k == "type":
      if isinstance(v, list):
        non_null = [t for t in v if t != "null"]
        if "null" in v:
          out["nullable"] = True
        v = non_null[0] if non_null else "string"
      out["type"] = str(v).upper()
    elif k in ("items", "anyOf"):
      out[k] = clean_schema(v)
    else:
      out[k] = v
  if out.get("type") == "ARRAY" and "items" not in out:
    out["items"] = {"type": "STRING"}
  return out


def _text(content: Any) -> str:
  if content is None:
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
  return str(content)


def _tool_response(content: Any) -> dict:
  text = _text(content)
  try:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
      return parsed
  except ValueError:
    pass
  return {"output": text}


def _push(contents: list, role: str, parts: list) -> None:
  if not parts:
    return
  if contents and contents[-1]["role"] == role:  # Vertex wants alternating turns
    contents[-1]["parts"].extend(parts)
  else:
    contents.append({"role": role, "parts": parts})


def to_gemini(req: dict, sigs: SignatureCache | None = None, signature_policy: str = "echo",
              dummy_signature: str = DUMMY_SIGNATURE) -> dict:
  """OpenAI chat-completions request -> Vertex generateContent body.

  signature_policy: echo (re-attach cached signatures), drop (never send one),
                    dummy (always send `dummy_signature`).
  """
  system_parts: list[dict] = []
  contents: list[dict] = []
  call_names: dict[str, str] = {}

  for m in req.get("messages") or []:
    role = m.get("role")
    if role in ("system", "developer"):
      t = _text(m.get("content"))
      if t:
        system_parts.append({"text": t})
    elif role == "user":
      _push(contents, "user", [{"text": _text(m.get("content"))}])
    elif role == "assistant":
      parts: list[dict] = []
      t = _text(m.get("content"))
      if t:
        parts.append({"text": t})
      for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name, cid = fn.get("name", ""), tc.get("id", "")
        call_names[cid] = name
        try:
          args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
          args = {}
        part: dict = {"functionCall": {"name": name, "args": args if isinstance(args, dict) else {}}}
        sig = None
        if signature_policy == "dummy":
          sig = dummy_signature
        elif signature_policy == "echo" and sigs is not None:
          sig = sigs.get(cid)
        if sig:
          part["thoughtSignature"] = sig
        parts.append(part)
      _push(contents, "model", parts)
    elif role == "tool":
      name = m.get("name") or call_names.get(m.get("tool_call_id", ""), "")
      _push(contents, "user", [{"functionResponse": {"name": name, "response": _tool_response(m.get("content"))}}])

  body: dict = {"contents": contents}
  if system_parts:
    body["systemInstruction"] = {"parts": system_parts}

  decls = []
  for t in req.get("tools") or []:
    fn = t.get("function") if isinstance(t, dict) else None
    if not fn:
      continue
    d = {"name": fn["name"], "description": fn.get("description", "")}
    params = clean_schema(fn.get("parameters") or {})
    if params.get("properties"):  # Vertex rejects an OBJECT schema with no properties
      d["parameters"] = params
    decls.append(d)
  if decls:
    body["tools"] = [{"functionDeclarations": decls}]
    choice = req.get("tool_choice")
    cfg = None
    if choice == "none":
      cfg = {"mode": "NONE"}
    elif choice == "required":
      cfg = {"mode": "ANY"}
    elif isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
      cfg = {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}
    if cfg:
      body["toolConfig"] = {"functionCallingConfig": cfg}

  gc: dict = {}
  if req.get("temperature") is not None:
    gc["temperature"] = req["temperature"]
  if req.get("top_p") is not None:
    gc["topP"] = req["top_p"]
  mt = req.get("max_completion_tokens") or req.get("max_tokens")
  if mt:
    gc["maxOutputTokens"] = int(mt)
  stop = req.get("stop")
  if stop:
    gc["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
  if req.get("seed") is not None:
    gc["seed"] = req["seed"]
  rf = req.get("response_format")
  if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
    gc["responseMimeType"] = "application/json"
  level = _REASONING_TO_LEVEL.get(str(req.get("reasoning_effort") or "").lower())
  if level:
    gc["thinkingConfig"] = {"thinkingLevel": level}
  if gc:
    body["generationConfig"] = gc
  return body


# ----------------------------------------------------------- response side

def usage_of(resp: dict) -> dict:
  u = resp.get("usageMetadata") or {}
  prompt = int(u.get("promptTokenCount") or 0)
  out = int(u.get("candidatesTokenCount") or 0)
  thoughts = int(u.get("thoughtsTokenCount") or 0)
  return {"prompt_tokens": prompt, "output_tokens": out, "thoughts_tokens": thoughts,
          "cached_tokens": int(u.get("cachedContentTokenCount") or 0),
          "total_tokens": int(u.get("totalTokenCount") or (prompt + out + thoughts))}


def from_gemini(resp: dict, model: str, sigs: SignatureCache | None = None) -> dict:
  """Vertex generateContent response -> OpenAI chat.completion."""
  cands = resp.get("candidates") or []
  text, tool_calls, finish = "", [], None
  if cands:
    cand = cands[0]
    finish = cand.get("finishReason")
    for p in (cand.get("content") or {}).get("parts") or []:
      if p.get("thought"):  # thought summaries are not part of the answer
        continue
      if "functionCall" in p:
        fc = p["functionCall"]
        cid = f"call_{uuid.uuid4().hex[:24]}"
        if p.get("thoughtSignature") and sigs is not None:
          sigs.put(cid, p["thoughtSignature"])
        tool_calls.append({"id": cid, "type": "function",
                           "function": {"name": fc.get("name", ""),
                                        "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}})
      elif isinstance(p.get("text"), str):
        text += p["text"]
  else:
    finish = (resp.get("promptFeedback") or {}).get("blockReason") or "SAFETY"

  reason = "tool_calls" if tool_calls else _FINISH.get(finish or "STOP", "stop")
  msg: dict = {"role": "assistant", "content": text or None}
  if tool_calls:
    msg["tool_calls"] = tool_calls
  u = usage_of(resp)
  return {
      "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
      "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
      "usage": {"prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["output_tokens"] + u["thoughts_tokens"],
                "total_tokens": u["total_tokens"],
                "prompt_tokens_details": {"cached_tokens": u["cached_tokens"]},
                "completion_tokens_details": {"reasoning_tokens": u["thoughts_tokens"]}},
  }


def to_sse(completion: dict) -> list[str]:
  """A non-streaming completion replayed as a one-chunk SSE stream (for stream=true callers)."""
  ch = completion["choices"][0]
  delta = dict(ch["message"])
  if delta.get("tool_calls"):
    delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(delta["tool_calls"])]
  base = {"id": completion["id"], "object": "chat.completion.chunk", "model": completion["model"]}
  first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
  last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": ch["finish_reason"]}],
          "usage": completion["usage"]}
  return [f"data: {json.dumps(first)}\n\n", f"data: {json.dumps(last)}\n\n", "data: [DONE]\n\n"]
