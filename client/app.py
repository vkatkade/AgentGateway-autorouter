"""Test client for the auto-router: submit a prompt, see the routing, get the answer.

There is no Agent Gateway in front of the router yet, so this client plays the
gateway's part over the same protocol (Envoy ext_proc v3, gRPC), following the
upstream reference client (service-extensions callouts/python/extproc/example/
ext_proc_client.py):

  1. request_headers   :method, :path (Vertex generateContent), :authority, content-type
  2. request_body      the whole JSON body in ONE message with end_of_stream=true
                       (the router asks for BUFFERED request bodies)
  3. response_headers  so the router can return its x-autorouter-* headers
  4. apply the router's :path / body mutations, then call Vertex AI with this
     service's own identity (ADC).

Env:
  ROUTER_TARGET     host:port of the router. Default "10.120.0.2:80" (internal LB, h2c).
  ROUTER_TLS        "true" to use TLS (e.g. the Cloud Run URL on :443). Default false.
  PROJECT_ID        project billed for Vertex calls.
  VERTEX_LOCATION   default "global".
  DEFAULT_MODEL     model the "agent" asks for. Default gemini-3.8-flash.
"""

from __future__ import annotations

import json
import os
import time

import google.auth
import google.auth.transport.requests
import grpc
import requests
from envoy.config.core.v3.base_pb2 import HeaderMap, HeaderValue
from envoy.service.ext_proc.v3 import external_processor_pb2 as pb
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as pb_grpc
from flask import Flask, jsonify, request

ROUTER_TARGET = os.getenv("ROUTER_TARGET", "10.120.0.2:80")
ROUTER_TLS = os.getenv("ROUTER_TLS", "false").lower() in ("1", "true", "yes")
PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCATION = os.getenv("VERTEX_LOCATION", "global")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")
VERTEX_HOST = "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"

app = Flask(__name__)
_creds = None
_stub = None


def _project() -> str:
  global PROJECT_ID
  if not PROJECT_ID:
    PROJECT_ID = google.auth.default()[1] or ""
  return PROJECT_ID


def _channel() -> grpc.Channel:
  if ROUTER_TLS:
    return grpc.secure_channel(ROUTER_TARGET, grpc.ssl_channel_credentials())
  return grpc.insecure_channel(ROUTER_TARGET)  # h2c, as Agent Gateway producers speak


def _router():
  global _stub
  if _stub is None:  # one pooled HTTP/2 channel per process
    _stub = pb_grpc.ExternalProcessorStub(_channel())
  return _stub


def _hm(pairs) -> HeaderMap:
  return HeaderMap(headers=[HeaderValue(key=k, raw_value=v.encode()) for k, v in pairs])


def ask_router(path: str, body: bytes, opt_out: bool):
  """Run one request through the router exactly as the gateway would."""
  req_headers = [(":method", "POST"), (":path", path), (":authority", VERTEX_HOST),
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
  responses = list(_router().Process(iter(msgs), timeout=10))
  ms = (time.perf_counter() - start) * 1000

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
  return new_path, new_body, seen, round(ms, 1)


def call_vertex(path: str, body: bytes):
  global _creds
  if _creds is None:
    _creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  if not _creds.valid:
    _creds.refresh(google.auth.transport.requests.Request())
  r = requests.post(f"https://{VERTEX_HOST}{path}", data=body, timeout=120,
                    headers={"Authorization": f"Bearer {_creds.token}",
                             "Content-Type": "application/json"})
  try:
    data = r.json()
  except ValueError:
    data = {"raw": r.text[:2000]}
  text = ""
  if r.ok:
    for c in data.get("candidates", []):
      text += "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []))
  return r.status_code, text, data


@app.post("/api/prompt")
def prompt():
  j = request.get_json(force=True)
  model = j.get("model") or DEFAULT_MODEL
  path = f"/v1/projects/{_project()}/locations/{LOCATION}/publishers/google/models/{model}:generateContent"
  body_obj = {"contents": [{"role": "user", "parts": [{"text": j["prompt"]}]}]}
  if j.get("system"):
    body_obj["systemInstruction"] = {"parts": [{"text": j["system"]}]}
  body = json.dumps(body_obj).encode()

  out = {"requested_model": model, "router_target": ROUTER_TARGET}
  try:
    new_path, new_body, headers, ms = ask_router(path, body, bool(j.get("opt_out")))
    out.update(router_ok=True, router_ms=ms, router_headers=headers,
               rewritten=new_path != path or new_body != body)
  except grpc.RpcError as e:  # fail open, like failOpen: true on the binding
    new_path, new_body = path, body
    out.update(router_ok=False, router_error=f"{e.code().name}: {e.details()}")
  out["sent_model"] = new_path.split("/models/")[1].split(":")[0]
  out["sent_generation_config"] = json.loads(new_body).get("generationConfig")

  if j.get("route_only"):
    return jsonify(out)
  t0 = time.perf_counter()
  status, text, raw = call_vertex(new_path, new_body)
  out.update(vertex_status=status, vertex_ms=round((time.perf_counter() - t0) * 1000, 1),
             answer=text, vertex_error=None if status == 200 else raw)
  return jsonify(out)


@app.get("/healthz")
def healthz():
  return "ok"


@app.get("/")
def index():
  return PAGE


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Auto-router test client</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#1f2328}
h1{font-size:20px;margin:0 0 4px}p.sub{color:#59636e;margin:0 0 20px}
textarea{width:100%;min-height:110px;font:inherit;padding:10px;border:1px solid #d1d9e0;border-radius:8px;box-sizing:border-box}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
select,button{font:inherit;padding:7px 12px;border-radius:8px;border:1px solid #d1d9e0;background:#fff}
button{background:#1f6feb;color:#fff;border-color:#1f6feb;cursor:pointer}button:disabled{opacity:.6}
.card{border:1px solid #d1d9e0;border-radius:8px;padding:14px 16px;margin-top:16px}
.kv{display:grid;grid-template-columns:200px 1fr;gap:4px 12px;font-size:14px}.kv div:nth-child(odd){color:#59636e}
.answer{white-space:pre-wrap}code{font-size:13px}.err{color:#cf222e}
</style></head><body>
<h1>Auto-router test client</h1>
<p class="sub">Sends your prompt through the router over ext_proc (as Agent Gateway would), then to Vertex AI.</p>
<textarea id="p" placeholder="Ask something... e.g. 'What is the capital of France?' or 'Debug this race condition in my Go worker pool'"></textarea>
<div class="row">
 <label>Requested model <select id="m"><option>gemini-3.8-flash</option><option>gemini-4-argon</option></select></label>
 <label><input type="checkbox" id="o"> send <code>x-autorouter: off</code></label>
 <label><input type="checkbox" id="r"> routing only (skip Vertex)</label>
 <button id="b">Send</button>
</div>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
$('b').onclick=async()=>{
 const prompt=$('p').value.trim(); if(!prompt)return;
 $('b').disabled=true; $('out').innerHTML='<div class="card">Working...</div>';
 try{
  const r=await fetch('/api/prompt',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({prompt,model:$('m').value,opt_out:$('o').checked,route_only:$('r').checked})});
  const d=await r.json(); const h=d.router_headers||{};
  let rows=[['Requested model',d.requested_model],['Router',d.router_ok?('reached in '+d.router_ms+' ms'):'<span class="err">unreachable: '+esc(d.router_error)+' (failed open)</span>'],
   ['Router reason',h['x-autorouter-reason']],['Router model header',h['x-autorouter-model']],['Thinking header',h['x-autorouter-thinking']],
   ['Request rewritten',d.rewritten?'yes':'no'],['Model sent to Vertex',d.sent_model],['generationConfig sent',d.sent_generation_config?'<code>'+esc(JSON.stringify(d.sent_generation_config))+'</code>':'']];
  if(d.vertex_status!==undefined)rows.push(['Vertex',d.vertex_status+' in '+d.vertex_ms+' ms']);
  let html='<div class="card"><div class="kv">'+rows.filter(x=>x[1]!==undefined&&x[1]!=='').map(([k,v])=>'<div>'+k+'</div><div>'+(String(v).startsWith('<')?v:esc(v))+'</div>').join('')+'</div></div>';
  if(d.answer)html+='<div class="card answer">'+esc(d.answer)+'</div>';
  if(d.vertex_error)html+='<div class="card err"><code>'+esc(JSON.stringify(d.vertex_error).slice(0,1500))+'</code></div>';
  $('out').innerHTML=html;
 }catch(e){$('out').innerHTML='<div class="card err">'+esc(e)+'</div>'}
 $('b').disabled=false;
};
</script></body></html>"""


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
